"""Game entry point — runs the basketball minigame loop.

`--mode debug` opens a preview window streaming the captured screen with
the basketball + backboard + rim tracker detections overlaid, plus a
periodic Tesseract OCR readout of the score region.
"""

from __future__ import annotations

import argparse
import time

from pathlib import Path

from backboard_tracker import BackboardSample, BackboardTracker
from basketball_tracker import BasketballSample, BasketballTracker
from preview_window import LabeledBox, PreviewWindow
from regions import SCORE_REGION, THROW_ZONE
from rim_tracker import RimSample, RimTracker
from score_reader import ScoreReader
from screen_capture import Region, ScreenCapture, primary_monitor_region
from throw_handler import ThrowRecorder


SCORE_READ_INTERVAL_S = 1.0  # OCR is slow (~30-80 ms); throttle in debug mode
THROWS_LOG_PATH = Path(__file__).parent / "throws.jsonl"


def _basketball_overlay(sample: BasketballSample, frame_origin: Region) -> LabeledBox:
    return LabeledBox(
        name=f"basketball ({sample.confidence:.2f})",
        top=sample.top - frame_origin["top"],
        left=sample.left - frame_origin["left"],
        width=sample.width,
        height=sample.height,
    )


def _backboard_overlay(sample: BackboardSample, frame_origin: Region) -> LabeledBox:
    return LabeledBox(
        name=f"backboard ({sample.confidence:.2f})",
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


def run_debug() -> None:
    capture_region = primary_monitor_region()
    basketball_tracker = BasketballTracker()
    backboard_tracker = BackboardTracker()
    rim_tracker = RimTracker()
    score_reader = ScoreReader()
    score_region = SCORE_REGION._asdict()
    throw_zone = THROW_ZONE._asdict()
    recorder = ThrowRecorder(
        log_path=THROWS_LOG_PATH,
        zone=throw_zone,
        score_reader=score_reader,
        score_region=score_region,
        capture_region=capture_region,
    )
    recorder.start()
    preview = PreviewWindow()

    last_score: int | None = None
    last_score_at: float = 0.0

    try:
        with ScreenCapture(capture_region) as capture:
            while True:
                frame = capture.grab()
                ball = basketball_tracker.read(frame, capture_region)
                backboard = backboard_tracker.read(frame, capture_region)
                rim = rim_tracker.read(frame, capture_region)

                recorder.on_frame(frame, ball)

                now = time.perf_counter()
                if now - last_score_at >= SCORE_READ_INTERVAL_S:
                    last_score = score_reader.read(frame, score_region, capture_region)
                    last_score_at = now

                overlays = [
                    _score_overlay(score_region, capture_region),
                    _throw_zone_overlay(throw_zone, capture_region),
                ]
                if ball is not None:
                    overlays.append(_basketball_overlay(ball, capture_region))
                if backboard is not None:
                    overlays.append(_backboard_overlay(backboard, capture_region))
                if rim is not None:
                    overlays.append(_rim_overlay(rim, capture_region))

                extra = {
                    "Basketball": (
                        f"({ball.center[0]}, {ball.center[1]})" if ball is not None else "?"
                    ),
                    "Backboard": (
                        f"({backboard.center[0]}, {backboard.center[1]})"
                        if backboard is not None else "?"
                    ),
                    "Rim": (
                        f"({rim.center[0]}, {rim.center[1]})" if rim is not None else "?"
                    ),
                    "Score": str(last_score) if last_score is not None else "?",
                }
                if not preview.show(frame, overlays=overlays, extra_debug=extra):
                    break
    finally:
        recorder.stop()
        preview.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Idleon basketball bot")
    parser.add_argument(
        "--mode",
        choices=("debug", "play"),
        default="play",
        help="debug = preview window with tracker overlays; play = run the strategy (not yet implemented)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "debug":
        run_debug()
    else:
        print("play mode not yet implemented")


if __name__ == "__main__":
    main()
