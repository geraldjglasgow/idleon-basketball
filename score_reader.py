"""Reads the in-game score (an integer) via Tesseract OCR.

Tesseract is invoked through pytesseract, which calls the `tesseract`
binary. On Windows the installer doesn't always add the binary to PATH —
we look in the default install locations so users don't have to set the
path manually.

If the binary isn't found, `read()` returns None and a one-time warning
is printed at startup, so debug mode still runs without OCR available.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from screen_capture import Region


_WINDOWS_TESSERACT_CANDIDATES = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
)


def _resolve_tesseract() -> str | None:
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for candidate in _WINDOWS_TESSERACT_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


class ScoreReader:
    # PSM 7 = treat the image as a single text line.
    # OEM 3 = default LSTM engine.
    # Whitelist digits to avoid Tesseract guessing letters from messy pixels.
    CONFIG = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
    # Tesseract handles small fonts poorly; upscale the crop before OCR.
    UPSCALE = 3
    # BGR thresholds for the digit color. Anything darker (including the
    # scrolling background) is masked out before OCR, so only the white-ish
    # number pixels reach Tesseract.
    WHITE_LOW = np.array([200, 200, 200], dtype=np.uint8)
    WHITE_HIGH = np.array([255, 255, 255], dtype=np.uint8)

    def __init__(self) -> None:
        path = _resolve_tesseract()
        self.available = path is not None
        if path is not None:
            pytesseract.pytesseract.tesseract_cmd = path
        else:
            print(
                "WARNING: Tesseract binary not found — score reader disabled. "
                "Install from https://github.com/UB-Mannheim/tesseract/wiki "
                "(default location: C:\\Program Files\\Tesseract-OCR)."
            )

    def read(
        self, frame: np.ndarray, region: Region, frame_origin: Region
    ) -> int | None:
        """Return the parsed score, or None if unreadable / Tesseract missing."""
        if not self.available:
            return None
        crop = self._crop(frame, region, frame_origin)
        if crop is None or crop.size == 0:
            return None

        mask = cv2.inRange(crop, self.WHITE_LOW, self.WHITE_HIGH)
        mask = cv2.resize(
            mask, None, fx=self.UPSCALE, fy=self.UPSCALE, interpolation=cv2.INTER_CUBIC
        )
        # Tesseract prefers black text on white background — invert the mask.
        ocr_input = cv2.bitwise_not(mask)

        text = pytesseract.image_to_string(ocr_input, config=self.CONFIG).strip()
        if not text or not text.isdigit():
            return None
        return int(text)

    @staticmethod
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
