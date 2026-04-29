"""Detects the "Game over!" screen by counting bright pixels.

The "Game over! / Exit to claim your points... come back soon!" message
spans TWO lines of white text on the dark starry sky. The start-of-game
prompt "Make a shot to start the game!" looks similar but is only ONE
line — it lives in the top half of the region with the bottom half empty
(except scattered stars). So we discriminate on the bottom half alone:
game-over has ~1700 white pixels there, start-prompt has 0, idle gameplay
has just a few star pixels. 400 is a safe threshold between them.
"""

from __future__ import annotations

import cv2
import numpy as np

from screen_capture import Region


WHITE_LUMINANCE = 200          # pixels brighter than this count as text
MIN_BOTTOM_WHITE_PIXELS = 400  # bottom-half text count required for game-over


def is_game_over(frame: np.ndarray, region: Region, frame_origin: Region) -> bool:
    """True if the GAME_OVER_REGION currently shows the 'Game over!' text."""
    crop = _crop(frame, region, frame_origin)
    if crop is None or crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, WHITE_LUMINANCE, 255)
    bottom_half = mask[mask.shape[0] // 2:]
    return int(np.count_nonzero(bottom_half)) >= MIN_BOTTOM_WHITE_PIXELS


def _crop(
    frame: np.ndarray, region: Region, frame_origin: Region
) -> np.ndarray | None:
    top = region["top"] - frame_origin["top"]
    left = region["left"] - frame_origin["left"]
    bottom = top + region["height"]
    right = left + region["width"]
    h, w = frame.shape[:2]
    if top < 0 or left < 0 or bottom > h or right > w:
        return None
    return frame[top:bottom, left:right]
