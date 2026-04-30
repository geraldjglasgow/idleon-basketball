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

    # Once we have a known position, prefer the candidate closest to it
    # over the merely-largest blob — guards against the tracker flipping
    # to UI elements (e.g. lives-indicator basketballs in the HUD) that
    # also pass the size + color gates. 250 px comfortably accommodates
    # real ball motion at ~30 fps (a thrown ball moves <200 px per frame)
    # while rejecting 400+ px jumps to remote UI blobs.
    POSITION_CONTINUITY_PX = 250

    def __init__(self) -> None:
        # Last accepted ball center, in screen coords. Used to filter
        # frame-to-frame jumps that would indicate a wrong-blob lock-on.
        self._last_center: tuple[int, int] | None = None

    def read(self, frame: np.ndarray, frame_origin: Region) -> BasketballSample | None:
        """Find an orange blob that's both ball-shaped AND consistent with
        the previous frame's ball position. Returns None if nothing
        plausible is found."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.ORANGE_HSV_LOW, self.ORANGE_HSV_HIGH)
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            return None

        # Collect every candidate that passes the size + aspect filters.
        candidates: list[tuple[int, int, int, int, int]] = []  # (area, x, y, w, h)
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
            candidates.append((area, x, y, w, h))

        if not candidates:
            return None

        # No prior position → pick the largest candidate (initial acquisition).
        # Prior position → pick the candidate whose center is closest to it,
        # but only if it's within POSITION_CONTINUITY_PX. If everything is too
        # far away, the ball was likely lost — fall back to largest to
        # re-acquire rather than freeze on a stale position.
        if self._last_center is None:
            best = max(candidates, key=lambda c: c[0])
        else:
            lx, ly = self._last_center
            within: list[tuple[int, tuple[int, int, int, int, int]]] = []
            for c in candidates:
                _, x, y, w, h = c
                cx = x + w // 2
                cy = y + h // 2
                dist_sq = (cx - lx) ** 2 + (cy - ly) ** 2
                if dist_sq <= self.POSITION_CONTINUITY_PX ** 2:
                    within.append((dist_sq, c))
            if within:
                within.sort(key=lambda t: t[0])
                best = within[0][1]
            else:
                # Re-acquire — last position is no longer near any candidate.
                best = max(candidates, key=lambda c: c[0])

        area, x, y, w, h = best
        self._last_center = (x + w // 2, y + h // 2)
        return BasketballSample(
            left=x + frame_origin["left"],
            top=y + frame_origin["top"],
            width=w,
            height=h,
            confidence=float(area),
        )
