"""Listens for clicks in the throw zone, snapshots the basketball + rim
positions at click time, then ~3 s later reads the score and appends a
record to ``throws.jsonl``.

Score reading uses **multi-sample voting + plausibility filtering**:
each finalize attempt grabs ``SCORE_VOTE_SAMPLES`` fresh frames spaced
``SCORE_VOTE_INTERVAL_S`` apart, OCRs each, drops readings that violate
the score-progression rule (``last <= candidate <= last + 2`` — a shot
either misses, scores 1, or scores 2), and returns the mode of the
survivors. If no plausible reading is found, the throw is requeued and
retried up to ``MAX_SCORE_ATTEMPTS`` times. Only successful reads are
written — null scores are never logged.

Modeled on the dart project's throw_detector + throw_logger split: a
pynput Listener runs on its own thread and just enqueues clicks; the
main game loop drives state transitions (snapshot → wait → finalize)
each frame so all OCR/file I/O stays on the loop thread.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import mss
import numpy as np
from pynput import mouse

from basketball_tracker import BasketballSample
from rim_motion_tracker import positions_indicate_motion
from rim_tracker import RimSample
from score_reader import ScoreReader
from screen_capture import Region


def _new_game_id() -> str:
    """Time-ordered UUIDv7 (Python 3.14+); falls back to uuid4 on older
    interpreters so the field is always populated."""
    if hasattr(uuid, "uuid7"):
        return str(uuid.uuid7())
    return str(uuid.uuid4())


@dataclass(frozen=True)
class _PendingThrow:
    ts: str           # iso timestamp at click time
    click_at: float   # perf_counter at click time — for trajectory dt_ms
    ball_x: int | None
    ball_y: int | None
    rim_x: int | None
    rim_y: int | None
    # "up" if the ball was rising at click time, "down" if falling, None if
    # too few samples / motion below the noise floor.
    stroke: str | None
    finalize_at: float  # perf_counter time when score should next be read
    attempts: int       # how many times we've already tried to read the score
    # Ball positions sampled per frame after the click. Each entry is
    # [x, y, dt_ms] in screen coords. The list is mutated in place from
    # `on_frame` (frozen dataclass disallows reassigning fields, but the
    # list reference itself stays the same so .append works).
    trajectory: list[tuple[int, int, int]] = field(default_factory=list)
    # Rim positions sampled in the same window as `trajectory` — captures
    # rim motion during ball flight so post-hoc analysis can correlate the
    # ball arc with where the rim actually was at arrival time.
    rim_trajectory: list[tuple[int, int, int]] = field(default_factory=list)


class _ClickListener:
    """Buffers left-click presses inside `zone` for the main loop to consume."""

    def __init__(self, zone: Region) -> None:
        self.zone = zone
        self._queue: Queue[float] = Queue()
        self._listener = mouse.Listener(on_click=self._on_click)

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    def consume(self) -> float | None:
        try:
            return self._queue.get_nowait()
        except Empty:
            return None

    def _on_click(self, x: int, y: int, button, pressed: bool) -> None:
        if not pressed or button != mouse.Button.left:
            return
        z = self.zone
        if not (
            z["left"] <= x < z["left"] + z["width"]
            and z["top"] <= y < z["top"] + z["height"]
        ):
            return
        self._queue.put(time.perf_counter())


class ThrowRecorder:
    SCORE_DELAY_S = 3.0
    SCORE_RETRY_DELAY_S = 1.0
    MAX_SCORE_ATTEMPTS = 10
    # Per attempt: take this many fresh OCR samples and vote.
    SCORE_VOTE_SAMPLES = 5
    SCORE_VOTE_INTERVAL_S = 0.1
    # Collect ball + rim positions for this long after the click. Set just
    # under SCORE_DELAY_S so trajectory capture finishes before the score
    # read kicks in, and long enough that the ball reaches the rim level
    # (1.5 s was too short — trajectories were ending mid-flight, hundreds
    # of pixels short of the rim).
    TRAJECTORY_DURATION_S = 2.8
    # Pre-click ball-y history for stroke (up/down) detection.
    STROKE_HISTORY = 5
    STROKE_DELTA_PX = 3  # below this |delta|, motion is treated as noise

    def __init__(
        self,
        log_path: Path | str,
        zone: Region,
        score_reader: ScoreReader,
        score_region: Region,
        capture_region: Region,
        on_finalize=None,
        on_drop=None,
    ) -> None:
        self.log_path = Path(log_path)
        self.score_reader = score_reader
        self.score_region = score_region
        self.capture_region = capture_region
        # Optional callback `on_finalize(record_dict)` invoked after each
        # throw is logged — lets the caller (game.py) classify outcome
        # and feed it back to the strategy without coupling recorder
        # internals to the strategy.
        self._on_finalize = on_finalize
        # Optional callback `on_drop()` invoked when a throw is dropped
        # because the score never resolved after MAX_SCORE_ATTEMPTS.
        # Game.py uses this to detect "stuck reading score that doesn't
        # exist" — i.e. the bot is in lobby/menu state but doesn't know
        # — and force a lobby recovery.
        self._on_drop = on_drop
        self._listener = _ClickListener(zone)
        self._pending: list[_PendingThrow] = []
        # Sliding window of recent ball y positions for stroke detection.
        self._ball_y_history: deque[int] = deque(maxlen=self.STROKE_HISTORY)
        # Per-game identifier (UUIDv7 — time-ordered, sortable). Reset at
        # game restart so all throws from one game share an id.
        self._game_id: str = _new_game_id()
        print(f"recorder: starting game_id={self._game_id}")
        # Last successfully-logged score; used to filter out impossible OCR
        # spikes (e.g. "2" briefly read as "24"). Reset by reset_score_state()
        # at game restart.
        self._last_score: int | None = None

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    def reset_score_state(self) -> None:
        """Drop any pending throws, forget the last score, and rotate the
        game_id. Call this when the game restarts (new game starts at 0,
        so the prior session's last score would reject every plausible
        reading; subsequent throws need a fresh game_id too)."""
        if self._pending:
            print(
                f"recorder: dropping {len(self._pending)} pending throw(s) "
                f"on game reset"
            )
        self._pending = []
        self._last_score = None
        self._game_id = _new_game_id()
        print(f"recorder: new game_id={self._game_id}")

    def on_frame(
        self,
        frame: np.ndarray,
        ball: BasketballSample | None,
        rim: RimSample | None,
    ) -> None:
        """Drive the recorder once per game-loop frame. Snapshots any new
        clicks against `ball` + `rim`, finalizes (or retries) any pending
        throws whose timer has elapsed."""
        # Update the stroke-detection history first so a click consumed
        # this frame sees the latest ball-y in its delta.
        if ball is not None:
            self._ball_y_history.append(ball.center[1])

        while True:
            click_at = self._listener.consume()
            if click_at is None:
                break
            ball_x, ball_y = ball.center if ball is not None else (None, None)
            rim_x, rim_y = rim.center if rim is not None else (None, None)
            stroke = self._stroke()
            pending = _PendingThrow(
                ts=datetime.now().isoformat(timespec="milliseconds"),
                click_at=click_at,
                ball_x=ball_x,
                ball_y=ball_y,
                rim_x=rim_x,
                rim_y=rim_y,
                stroke=stroke,
                finalize_at=click_at + self.SCORE_DELAY_S,
                attempts=0,
            )
            self._pending.append(pending)
            print(
                f"throw click: ball=({ball_x}, {ball_y}) rim=({rim_x}, {rim_y}) "
                f"stroke={stroke or '?'}; reading score in {self.SCORE_DELAY_S:.0f}s"
            )

        now = time.perf_counter()
        # Sample ball + rim positions into each pending throw's trajectories
        # while we're still inside its TRAJECTORY_DURATION_S window. Skip
        # frames where the relevant tracker missed.
        for p in self._pending:
            dt_s = now - p.click_at
            if not (0.0 <= dt_s <= self.TRAJECTORY_DURATION_S):
                continue
            dt_ms = int(dt_s * 1000)
            if ball is not None:
                bx, by = ball.center
                p.trajectory.append((bx, by, dt_ms))
            if rim is not None:
                rx, ry = rim.center
                p.rim_trajectory.append((rx, ry, dt_ms))
        still_pending: list[_PendingThrow] = []
        for p in self._pending:
            if now < p.finalize_at:
                still_pending.append(p)
                continue
            score = self._read_score_consensus()
            attempts = p.attempts + 1
            if score is not None:
                # The "previous score" defaults to 0 on the first throw of a
                # game (last_score is None then) — game starts at 0, so any
                # increase means we scored.
                prev = self._last_score if self._last_score is not None else 0
                scored = score > prev
                rim_moving = positions_indicate_motion(p.rim_trajectory)
                record = {
                    "game_id": self._game_id,
                    "ts": p.ts,
                    "ball_x": p.ball_x,
                    "ball_y": p.ball_y,
                    "rim_x": p.rim_x,
                    "rim_y": p.rim_y,
                    "stroke": p.stroke,
                    "rim_moving": rim_moving,
                    "score": score,
                    "scored": scored,
                    "trajectory": [list(pt) for pt in p.trajectory],
                    "rim_trajectory": [list(pt) for pt in p.rim_trajectory],
                }
                self._append(record)
                print(
                    f"throw logged: ball=({p.ball_x}, {p.ball_y}) "
                    f"rim=({p.rim_x}, {p.rim_y}) score={score} "
                    f"prev={self._last_score} scored={scored} "
                    f"trajectory_pts={len(p.trajectory)} "
                    f"(attempts={attempts})"
                )
                self._last_score = score
                if self._on_finalize is not None:
                    try:
                        self._on_finalize(record)
                    except Exception as exc:
                        print(f"recorder: on_finalize callback failed: {exc}")
            elif attempts < self.MAX_SCORE_ATTEMPTS:
                still_pending.append(
                    replace(
                        p,
                        finalize_at=now + self.SCORE_RETRY_DELAY_S,
                        attempts=attempts,
                    )
                )
                print(
                    f"score read inconclusive "
                    f"({attempts}/{self.MAX_SCORE_ATTEMPTS}) — retrying in "
                    f"{self.SCORE_RETRY_DELAY_S:.1f}s"
                )
            else:
                print(
                    f"throw DROPPED: score never resolved after {attempts} "
                    f"attempts (ball=({p.ball_x}, {p.ball_y}) "
                    f"rim=({p.rim_x}, {p.rim_y}))"
                )
                if self._on_drop is not None:
                    try:
                        self._on_drop()
                    except Exception as exc:
                        print(f"recorder: on_drop callback failed: {exc}")
        self._pending = still_pending

    def _read_score_consensus(self) -> int | None:
        """Take SCORE_VOTE_SAMPLES samples; filter to plausible values
        (last_score..last_score+2, or all if we have no prior score); return
        the mode of survivors, or None if nothing readable + plausible."""
        samples: list[int] = []
        with mss.MSS() as sct:
            for i in range(self.SCORE_VOTE_SAMPLES):
                raw = sct.grab(self.capture_region)
                frame = np.asarray(raw)[:, :, :3]
                s = self.score_reader.read(
                    frame, self.score_region, self.capture_region
                )
                if s is not None:
                    samples.append(s)
                if i < self.SCORE_VOTE_SAMPLES - 1:
                    time.sleep(self.SCORE_VOTE_INTERVAL_S)

        if self._last_score is None:
            plausible = samples
        else:
            lo, hi = self._last_score, self._last_score + 2
            plausible = [s for s in samples if lo <= s <= hi]

        if not plausible:
            if samples:
                print(f"score samples {samples} — none plausible vs last={self._last_score}")
            return None
        return Counter(plausible).most_common(1)[0][0]

    def _stroke(self) -> str | None:
        """Classify the recent ball-y motion as 'up' or 'down' (or None
        when we have too few samples or the motion is below the noise
        floor). Screen y grows downward, so a positive delta = falling."""
        if len(self._ball_y_history) < 2:
            return None
        delta = self._ball_y_history[-1] - self._ball_y_history[0]
        if abs(delta) < self.STROKE_DELTA_PX:
            return None
        return "down" if delta > 0 else "up"

    def _append(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
