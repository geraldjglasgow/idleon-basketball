"""Real-time screen capture via mss. Returns BGR numpy frames."""

from __future__ import annotations

from typing import Optional

import mss
import numpy as np


Region = dict[str, int]  # keys: top, left, width, height


class ScreenCapture:
    """Captures a fixed screen region and yields BGR frames.

    Usage:
        with ScreenCapture({"top": 0, "left": 0, "width": 1920, "height": 1080}) as cap:
            frame = cap.grab()
    """

    def __init__(self, region: Region) -> None:
        self.region = region
        self._sct: Optional[mss.base.MSSBase] = None

    def __enter__(self) -> "ScreenCapture":
        self._sct = mss.mss().__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._sct is not None:
            self._sct.__exit__(exc_type, exc, tb)
            self._sct = None

    def grab(self) -> np.ndarray:
        if self._sct is None:
            raise RuntimeError("ScreenCapture must be used as a context manager")
        raw = self._sct.grab(self.region)
        return np.asarray(raw)[:, :, :3]


def primary_monitor_region() -> Region:
    """Return a Region covering the full primary monitor."""
    with mss.mss() as sct:
        # monitors[0] is the combined virtual screen; monitors[1:] are individual
        # displays. The primary isn't always at index 1.
        primary = next(
            (m for m in sct.monitors[1:] if m.get("is_primary")),
            sct.monitors[1],
        )
    return {
        "top": primary["top"],
        "left": primary["left"],
        "width": primary["width"],
        "height": primary["height"],
    }
