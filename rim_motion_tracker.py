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
    # The rim is considered "moving" when its observed bounding-box span
    # over the history window exceeds this many pixels in either axis.
    # Smaller than this is treated as tracker jitter on a stationary rim.
    MOTION_THRESHOLD_PX = 10

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

    def is_moving(self) -> bool:
        """True if the rim's positions over the history window span more
        than MOTION_THRESHOLD_PX in either axis. Returns False when too
        few samples (treated as stationary by default)."""
        b = self.bounds()
        if b is None or len(self._history) < 2:
            return False
        min_x, min_y, max_x, max_y = b
        return (
            max_x - min_x > self.MOTION_THRESHOLD_PX
            or max_y - min_y > self.MOTION_THRESHOLD_PX
        )


def positions_indicate_motion(
    positions, threshold_px: int = RimMotionTracker.MOTION_THRESHOLD_PX
) -> bool:
    """True if the (x, y) tuples span more than `threshold_px` in either
    axis. Used by the recorder to label a finalized throw's rim_trajectory
    as moving / stationary, with the same threshold the live tracker uses."""
    coords = [(p[0], p[1]) for p in positions if len(p) >= 2]
    if len(coords) < 2:
        return False
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (
        max(xs) - min(xs) > threshold_px
        or max(ys) - min(ys) > threshold_px
    )
