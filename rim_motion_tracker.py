"""Tracks the rim's motion over time so the strategy can lead a moving rim.

Each frame we feed `observe(rim)` with the live RimSample (or None when
the tracker missed a frame). The class keeps a sliding window of recent
positions and exposes:

- `predict(dt_seconds)` — linear-extrapolated rim position dt_seconds
  from now. Returns None when there's too little history to estimate
  velocity. Linear is fine for the ~1.5 s ball-flight horizon even if
  the rim is actually moving sinusoidally — over a window that short,
  the rim's path is well-approximated by its instantaneous velocity.
- `bounds()` — observed (min_x, min_y, max_x, max_y) within the
  history window. Useful for telling whether the rim is moving at all.
- `velocity()` — px/sec along x and y, averaged over the full window.

We don't try to fit the rim's motion model (period, amplitude). Most
shots happen with the rim either stationary or in a slow predictable
drift; over the ball-flight horizon, linear extrapolation captures
99% of what we need.
"""

from __future__ import annotations

import time
from collections import deque

from rim_tracker import RimSample


class RimMotionTracker:
    HISTORY_WINDOW_S = 2.0  # how far back the velocity estimate looks

    def __init__(self) -> None:
        # Each entry is (perf_counter_t, x, y).
        self._history: deque[tuple[float, int, int]] = deque()

    def observe(self, rim: RimSample | None) -> None:
        """Record the current frame's rim position. Pass None if the rim
        wasn't detected this frame (skipped, doesn't reset history)."""
        now = time.perf_counter()
        cutoff = now - self.HISTORY_WINDOW_S
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        if rim is not None:
            x, y = rim.center
            self._history.append((now, x, y))

    def predict(self, dt_seconds: float) -> tuple[int, int] | None:
        """Linear-extrapolate where the rim will be `dt_seconds` from now.
        Returns None when too few samples to estimate velocity."""
        if len(self._history) < 2:
            return None
        t0, x0, y0 = self._history[0]
        t1, x1, y1 = self._history[-1]
        dt = t1 - t0
        if dt <= 0:
            return (x1, y1)
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        return (int(round(x1 + vx * dt_seconds)), int(round(y1 + vy * dt_seconds)))

    def velocity(self) -> tuple[float, float]:
        """Average velocity (px/sec) over the history window. (0, 0) when
        too few samples."""
        if len(self._history) < 2:
            return (0.0, 0.0)
        t0, x0, y0 = self._history[0]
        t1, x1, y1 = self._history[-1]
        dt = t1 - t0
        if dt <= 0:
            return (0.0, 0.0)
        return ((x1 - x0) / dt, (y1 - y0) / dt)

    def bounds(self) -> tuple[int, int, int, int] | None:
        """Observed (min_x, min_y, max_x, max_y) in the window. None if
        history is empty."""
        if not self._history:
            return None
        xs = [x for _, x, _ in self._history]
        ys = [y for _, _, y in self._history]
        return (min(xs), min(ys), max(xs), max(ys))
