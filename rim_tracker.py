"""Tracks the rim's position via OpenCV template matching.

The template at `assets/rim.png` is the orange rim with its lighter center
band — the ball needs to fall through this, so the rim's position is the
key target landmark for the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from screen_capture import Region


TEMPLATE_PATH = Path(__file__).parent / "assets" / "rim.png"


@dataclass(frozen=True)
class RimSample:
    """Per-frame rim detection. Coordinates are in screen space."""

    left: int
    top: int
    width: int
    height: int
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        return self.left + self.width // 2, self.top + self.height // 2


class RimTracker:
    MIN_CONFIDENCE = 0.7
    DOWNSCALE = 2

    def __init__(self, template_path: Path | str = TEMPLATE_PATH) -> None:
        path = Path(template_path)
        template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"rim template not found: {path}")
        scaled = cv2.resize(
            template,
            None,
            fx=1 / self.DOWNSCALE,
            fy=1 / self.DOWNSCALE,
            interpolation=cv2.INTER_AREA,
        )
        self.template = scaled
        self.h, self.w = scaled.shape[:2]
        self.full_h = self.h * self.DOWNSCALE
        self.full_w = self.w * self.DOWNSCALE

    def read(self, frame: np.ndarray, frame_origin: Region) -> RimSample | None:
        """Locate the rim in `frame`. Returns None if not confidently found."""
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
        return RimSample(
            left=max_loc[0] * self.DOWNSCALE + frame_origin["left"],
            top=max_loc[1] * self.DOWNSCALE + frame_origin["top"],
            width=self.full_w,
            height=self.full_h,
            confidence=float(max_v),
        )
