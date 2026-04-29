"""Tracks the basketball's position via HSV color masking.

The basketball is a bright orange ball on a dark blue starry sky — its
hue is distinctive enough that masking orange pixels and taking the
largest blob is far more robust than template matching against a sharp
ball.png. In particular this survives motion blur during a shot (orange
pixels stay orange when smeared) and partial occlusion when the
character is holding the ball (some orange is still visible).

We filter blobs by area + aspect ratio so other orange shapes on the
HUD/court (rim, backboard, UI text) don't get picked up.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from screen_capture import Region


@dataclass(frozen=True)
class BasketballSample:
    """Per-frame basketball detection. Coordinates are in screen space."""

    left: int
    top: int
    width: int
    height: int
    confidence: float  # blob area in pixels — higher = more orange detected

    @property
    def center(self) -> tuple[int, int]:
        return self.left + self.width // 2, self.top + self.height // 2


class BasketballTracker:
    # Bright basketball orange in OpenCV HSV (H: 0-179). Centered around 10-15;
    # high saturation + value to avoid muted UI orange.
    ORANGE_HSV_LOW = np.array([5, 120, 100], dtype=np.uint8)
    ORANGE_HSV_HIGH = np.array([20, 255, 255], dtype=np.uint8)

    # Filter blobs to ball-like shapes. The rim is wide-thin so its aspect
    # falls outside the tolerance; the backboard column is tall-thin and
    # huge area. UI orange is small.
    MIN_BLOB_AREA = 400
    MAX_BLOB_AREA = 12000
    MIN_ASPECT = 0.5  # height / width
    MAX_ASPECT = 2.0

    def __init__(self) -> None:
        # No template needed — pure color detector.
        pass

    def read(self, frame: np.ndarray, frame_origin: Region) -> BasketballSample | None:
        """Find the largest orange blob meeting size + aspect filters.
        Returns None if no candidate qualifies."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.ORANGE_HSV_LOW, self.ORANGE_HSV_HIGH)
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            return None

        best: tuple[int, int, int, int, int] | None = None  # (area, x, y, w, h)
        for i in range(1, n_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            if area < self.MIN_BLOB_AREA or area > self.MAX_BLOB_AREA:
                continue
            if w == 0 or h == 0:
                continue
            aspect = h / w
            if aspect < self.MIN_ASPECT or aspect > self.MAX_ASPECT:
                continue

            if best is None or area > best[0]:
                best = (area, x, y, w, h)

        if best is None:
            return None

        area, x, y, w, h = best
        return BasketballSample(
            left=x + frame_origin["left"],
            top=y + frame_origin["top"],
            width=w,
            height=h,
            confidence=float(area),
        )
