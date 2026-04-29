"""Tracks the basketball's position via OpenCV template matching.

The template at `assets/middle_basketball.png` is a small crop of the middle
of the ball (the orange-with-seams pattern). We match that template against
each captured frame; the best-match location is reported as a screen-space
bounding box matching the template's dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from screen_capture import Region


TEMPLATE_PATH = Path(__file__).parent / "assets" / "middle_basketball.png"


@dataclass(frozen=True)
class BasketballSample:
    """Per-frame basketball detection. Coordinates are in screen space."""

    left: int
    top: int
    width: int
    height: int
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        return self.left + self.width // 2, self.top + self.height // 2


class BasketballTracker:
    # cv2.matchTemplate with TM_CCOEFF_NORMED returns values in [-1, 1];
    # 0.7 is a conservative threshold for "actually visible".
    MIN_CONFIDENCE = 0.7
    # Match against grayscale at half resolution — ~12× cheaper than full-res
    # BGR matching against a full-monitor frame. Reported coordinates are
    # rescaled back to screen space.
    DOWNSCALE = 2

    def __init__(self, template_path: Path | str = TEMPLATE_PATH) -> None:
        path = Path(template_path)
        template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"basketball template not found: {path}")
        scaled = cv2.resize(
            template,
            None,
            fx=1 / self.DOWNSCALE,
            fy=1 / self.DOWNSCALE,
            interpolation=cv2.INTER_AREA,
        )
        self.template = scaled
        self.h, self.w = scaled.shape[:2]
        # Full-resolution dims for the bbox we report back to callers.
        self.full_h = self.h * self.DOWNSCALE
        self.full_w = self.w * self.DOWNSCALE

    def read(self, frame: np.ndarray, frame_origin: Region) -> BasketballSample | None:
        """Locate the basketball in `frame`. Returns None if not confidently found."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(
            gray,
            None,
            fx=1 / self.DOWNSCALE,
            fy=1 / self.DOWNSCALE,
            interpolation=cv2.INTER_AREA,
        )
        if scaled.shape[0] < self.h or scaled.shape[1] < self.w:
            return None
        result = cv2.matchTemplate(scaled, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(result)
        if max_v < self.MIN_CONFIDENCE:
            return None
        return BasketballSample(
            left=max_loc[0] * self.DOWNSCALE + frame_origin["left"],
            top=max_loc[1] * self.DOWNSCALE + frame_origin["top"],
            width=self.full_w,
            height=self.full_h,
            confidence=float(max_v),
        )
