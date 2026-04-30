"""Runs the in-game capture/track/record loop. Assumes the minigame is
already started — main.py drives lobby first, then calls `run()` here.

`--mode debug` adds a preview window with tracker overlays; `--mode play`
runs headless. Both produce the same throws.jsonl output.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lobby
from basketball_tracker import BasketballSample, BasketballTracker
from game_over_detector import is_game_over
from preview_window import LabeledBox, PreviewWindow
from regions import EXIT_BUTTON_REGION, GAME_OVER_REGION, SCORE_REGION, THROW_ZONE
from rim_motion_tracker import RimMotionTracker
from rim_tracker import RimSample, RimTracker
from score_reader import ScoreReader
from screen_capture import Region, ScreenCapture, primary_monitor_region
from simple_rim_strategy import SimpleRimStrategy, classify_outcome
from throw_handler import ThrowRecorder
from utils.mouse import click


SCORE_READ_INTERVAL_S = 1.0  # OCR is slow (~30-80 ms); throttle in the loop
THROWS_LOG_PATH = Path(__file__).parent / "throws.jsonl"
# If this many throws in a row get dropped (score never resolved), the
# bot is almost certainly not actually in a game — the score region
# isn't displaying a number to read. Force a lobby pass to recover.
MAX_CONSECUTIVE_DROPS = 3


def _basketball_overlay(sample: BasketballSample, frame_origin: Region) -> LabeledBox:
    return LabeledBox(
        name=f"basketball ({sample.confidence:.2f})",
        top=sample.top - frame_origin["top"],
        left=sample.left - frame_origin["left"],
        width=sample.width,
        height=sample.height,
    )


def _rim_overlay(sample: RimSample, frame_origin: Region) -> LabeledBox:
    return LabeledBox(
        name=f"rim ({sample.confidence:.2f})",
        top=sample.top - frame_origin["top"],
        left=sample.left - frame_origin["left"],
        width=sample.width,
        height=sample.height,
    )


def _score_overlay(score_region: Region, frame_origin: Region) -> LabeledBox:
    return LabeledBox(
        name="score",
        top=score_region["top"] - frame_origin["top"],
        left=score_region["left"] - frame_origin["left"],
        width=score_region["width"],
        height=score_region["height"],
    )


def _throw_zone_overlay(zone: Region, frame_origin: Region) -> LabeledBox:
    return LabeledBox(
        name="throw zone",
        top=zone["top"] - frame_origin["top"],
        left=zone["left"] - frame_origin["left"],
        width=zone["width"],
        height=zone["height"],
    )


PREVIEW_MODES = ("off", "full", "light")


def run(
    preview: PreviewWindow | None = None,
    hotkey_listener=None,
) -> None:
    """Run the in-game loop (capture + trackers + score reader + throw
    recorder). `preview` is an optional pre-constructed window — pass None
    for headless. `hotkey_listener` gates the strategy's auto-throwing —
    if None, the strategy runs unconditionally (useful for direct game.py
    invocation); if provided, only fires while `listener.auto_enabled` is
    True (toggled with F1/F2). Assumes the game has already been started."""
    capture_region = primary_monitor_region()
    basketball_tracker = BasketballTracker()
    rim_tracker = RimTracker()
    score_reader = ScoreReader()
    score_region = SCORE_REGION._asdict()
    throw_zone = THROW_ZONE._asdict()
    game_over_region = GAME_OVER_REGION._asdict()
    strategy = SimpleRimStrategy(THROWS_LOG_PATH)
    rim_motion = RimMotionTracker()
    # Counter for consecutive dropped throws. Reset on any successful
    # finalize. When it crosses MAX_CONSECUTIVE_DROPS, the loop forces
    # a lobby recovery on the next iteration.
    consecutive_drops = [0]
    needs_lobby_recovery = [False]

    def _on_throw_finalized(record: dict) -> None:
        consecutive_drops[0] = 0
        rx = record.get("rim_x")
        ry = record.get("rim_y")
        traj = record.get("trajectory") or []
        if rx is None or ry is None:
            return
        outcome = classify_outcome(
            traj, rx, ry, scored=record.get("scored")
        )
        strategy.notify_outcome(outcome)
        print(f"[strategy] throw outcome: {outcome}")

    def _on_throw_dropped() -> None:
        consecutive_drops[0] += 1
        print(
            f"[game] throw drop #{consecutive_drops[0]} of "
            f"{MAX_CONSECUTIVE_DROPS} consecutive"
        )
        if consecutive_drops[0] >= MAX_CONSECUTIVE_DROPS:
            needs_lobby_recovery[0] = True

    recorder = ThrowRecorder(
        log_path=THROWS_LOG_PATH,
        zone=throw_zone,
        score_reader=score_reader,
        score_region=score_region,
        capture_region=capture_region,
        on_finalize=_on_throw_finalized,
        on_drop=_on_throw_dropped,
    )
    recorder.start()

    last_score: int | None = None
    last_score_at: float = 0.0

    try:
        with ScreenCapture(capture_region) as capture:
            while True:
                frame = capture.grab()
                ball = basketball_tracker.read(frame, capture_region)
                rim = rim_tracker.read(frame, capture_region)
                rim_motion.observe(rim)

                recorder.on_frame(frame, ball, rim)

                if is_game_over(frame, game_over_region, capture_region):
                    print("[game] game over screen detected — waiting 2s, clicking exit")
                    time.sleep(2.0)
                    click(
                        EXIT_BUTTON_REGION.left + EXIT_BUTTON_REGION.width // 2,
                        EXIT_BUTTON_REGION.top + EXIT_BUTTON_REGION.height // 2,
                    )
                    recorder.reset_score_state()
                    lobby.start_game(preview=preview)
                    consecutive_drops[0] = 0
                    needs_lobby_recovery[0] = False
                    continue

                if needs_lobby_recovery[0]:
                    print(
                        f"[game] {consecutive_drops[0]} throws in a row dropped "
                        f"(score never resolved) — bot likely not in a game; "
                        f"running lobby recovery"
                    )
                    recorder.reset_score_state()
                    lobby.start_game(preview=preview)
                    consecutive_drops[0] = 0
                    needs_lobby_recovery[0] = False
                    continue

                auto_enabled = (
                    hotkey_listener is None or hotkey_listener.auto_enabled
                )
                if auto_enabled and strategy.should_throw(ball, rim, rim_motion):
                    cx = throw_zone["left"] + throw_zone["width"] // 2
                    cy = throw_zone["top"] + throw_zone["height"] // 2
                    click(cx, cy)
                    strategy.mark_thrown()
                    predicted = rim_motion.predict(strategy.BALL_FLIGHT_S)
                    print(
                        f"[strategy] throw clicked at ({cx}, {cy}) — "
                        f"ball={ball.center} rim={rim.center} "
                        f"predicted_rim={predicted}"
                    )

                now = time.perf_counter()
                if now - last_score_at >= SCORE_READ_INTERVAL_S:
                    last_score = score_reader.read(frame, score_region, capture_region)
                    last_score_at = now
                    strategy.notify_score(last_score)

                if preview is not None:
                    overlays = [
                        _score_overlay(score_region, capture_region),
                        _throw_zone_overlay(throw_zone, capture_region),
                    ]
                    if ball is not None:
                        overlays.append(_basketball_overlay(ball, capture_region))
                    if rim is not None:
                        overlays.append(_rim_overlay(rim, capture_region))

                    extra = {
                        "Basketball": (
                            f"({ball.center[0]}, {ball.center[1]})"
                            if ball is not None else "?"
                        ),
                        "Rim": (
                            f"({rim.center[0]}, {rim.center[1]})"
                            if rim is not None else "?"
                        ),
                        "Score": str(last_score) if last_score is not None else "?",
                    }
                    if not preview.show(frame, overlays=overlays, extra_debug=extra):
                        break
    finally:
        recorder.stop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Idleon basketball bot")
    parser.add_argument(
        "--preview",
        choices=PREVIEW_MODES,
        default="off",
        help="off = no window; full = stream + overlays; light = debug text only",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    preview = (
        None if args.preview == "off"
        else PreviewWindow(lightweight=(args.preview == "light"))
    )
    try:
        run(preview=preview)
    finally:
        if preview is not None:
            preview.close()


if __name__ == "__main__":
    main()
