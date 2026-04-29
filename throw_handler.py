"""Listens for clicks in the throw zone, snapshots the basketball position
at click time, then 3 s later reads the score and appends a record to
``throws.jsonl``.

Modeled on the dart project's throw_detector + throw_logger split: a
pynput Listener runs on its own thread and just enqueues clicks; the
main game loop drives state transitions (snapshot → wait → finalize)
each frame so all OCR/file I/O stays on the loop thread.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import numpy as np
from pynput import mouse

from basketball_tracker import BasketballSample
from score_reader import ScoreReader
from screen_capture import Region


@dataclass(frozen=True)
class _PendingThrow:
    ts: str          # iso timestamp at click time
    ball_x: int | None
    ball_y: int | None
    finalize_at: float  # perf_counter time when score should be read


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

    def __init__(
        self,
        log_path: Path | str,
        zone: Region,
        score_reader: ScoreReader,
        score_region: Region,
        capture_region: Region,
    ) -> None:
        self.log_path = Path(log_path)
        self.score_reader = score_reader
        self.score_region = score_region
        self.capture_region = capture_region
        self._listener = _ClickListener(zone)
        self._pending: list[_PendingThrow] = []

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    def on_frame(self, frame: np.ndarray, ball: BasketballSample | None) -> None:
        """Drive the recorder once per game-loop frame. Snapshots any new
        clicks against `ball`, finalizes any pending throws whose 3 s timer
        has elapsed."""
        while True:
            click_at = self._listener.consume()
            if click_at is None:
                break
            ball_x, ball_y = (ball.center if ball is not None else (None, None))
            pending = _PendingThrow(
                ts=datetime.now().isoformat(timespec="milliseconds"),
                ball_x=ball_x,
                ball_y=ball_y,
                finalize_at=click_at + self.SCORE_DELAY_S,
            )
            self._pending.append(pending)
            print(
                f"throw click: ball=({ball_x}, {ball_y}); reading score in "
                f"{self.SCORE_DELAY_S:.0f}s"
            )

        now = time.perf_counter()
        ready = [p for p in self._pending if now >= p.finalize_at]
        if not ready:
            return
        self._pending = [p for p in self._pending if now < p.finalize_at]
        for p in ready:
            score = self.score_reader.read(frame, self.score_region, self.capture_region)
            self._append({
                "ts": p.ts,
                "ball_x": p.ball_x,
                "ball_y": p.ball_y,
                "score": score,
            })
            print(f"throw logged: ball=({p.ball_x}, {p.ball_y}) score={score}")

    def _append(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
