"""Reads the in-game score (an integer) using a per-digit template atlas
that auto-bootstraps from Tesseract.

Reads do **template matching first** against `assets/digits/0.png` …
`9.png`. Each saved template is the binary mask of one digit,
height-normalized. At read time we extract the live mask's connected
components (one per digit, sorted left-to-right), normalize each to the
same height, and correlate against every template — the best match
wins, gated by an absolute threshold and a margin between best and
second-best so ambiguous shapes (0/O, 6/8) abstain rather than guessing.

When the atlas can't classify (missing digit, low confidence), we fall
back to **Tesseract** with the existing two-PSM cascade plus the
pixel-font fallback dict. If Tesseract resolves to digits *and* the
component count matches, each component is auto-saved as a new template
— so the atlas grows itself as you play, and after a session or two
Tesseract is rarely consulted.

Tesseract is invoked through pytesseract; on Windows we look in the
default install locations so users don't have to set PATH manually. If
the binary isn't found we still run via the atlas alone (and only the
bootstrap path is disabled).
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
    # PSM 7 (single text line) is our primary mode; PSM 8 (single word) is
    # a fallback for cases PSM 7 mis-reads.
    # OEM 3 = default LSTM engine.
    # No tessedit_char_whitelist: it's silently broken with OEM 3.
    PSM_PRIMARY = 7
    PSM_FALLBACK = 8
    UPSCALE = 4
    PADDING = 24
    MIN_DIGIT_PIXELS = 30
    LUMINANCE_THRESHOLD = 180
    MIN_COMPONENT_AREA = 20
    # Per-digit atlas parameters. Live components and saved templates are
    # normalized to this height before correlating.
    DIGIT_NORM_HEIGHT = 40
    DIGIT_MATCH_THRESHOLD = 0.85
    # Required margin between best and second-best correlation — protects
    # against ambiguous shapes (e.g. 0 vs O vs 8 with light pattern noise).
    DIGIT_MATCH_MARGIN = 0.08
    DIGITS_DIR = Path(__file__).parent / "assets" / "digits"
    # Tesseract's LSTM is trained on conventional fonts and consistently
    # misreads certain pixel-art digits. We map the observed misreads to
    # the correct digit; this is also what feeds auto-bootstrap when
    # template-matching can't classify.
    PIXEL_FONT_FALLBACKS = {
        "Lt": "4",
        "lt": "4",
        "oO": "0",
        "Oo": "0",
        "O": "0",
        "o": "0",
        "U": "0",
    }

    def __init__(self) -> None:
        path = _resolve_tesseract()
        self.available = path is not None
        if path is not None:
            pytesseract.pytesseract.tesseract_cmd = path
        else:
            print(
                "WARNING: Tesseract binary not found — atlas-only mode. "
                "Install from https://github.com/UB-Mannheim/tesseract/wiki "
                "to enable auto-bootstrap of new digit templates."
            )
        self._digit_atlas: dict[str, np.ndarray] = self._load_digit_atlas()
        print(
            f"ScoreReader: digit atlas covers "
            f"{sorted(self._digit_atlas.keys()) or '<empty>'}"
        )

    def read(
        self, frame: np.ndarray, region: Region, frame_origin: Region
    ) -> int | None:
        """Return the parsed score, or None if unreadable."""
        crop = self._crop(frame, region, frame_origin)
        if crop is None or crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.LUMINANCE_THRESHOLD, 255, cv2.THRESH_BINARY)
        # Drop small connected components — stars typically light up just a
        # few pixels apiece while each digit stroke is much larger.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        cleaned = np.zeros_like(mask)
        for i in range(1, n_labels):
            if int(stats[i, cv2.CC_STAT_AREA]) >= self.MIN_COMPONENT_AREA:
                cleaned[labels == i] = 255
        mask = cleaned
        if not (self.MIN_DIGIT_PIXELS <= np.count_nonzero(mask) <= mask.size // 2):
            return None

        # Try the atlas first — fast and accurate when templates exist.
        atlas_digits = self._classify_via_atlas(mask)
        if atlas_digits is not None:
            return int(atlas_digits)

        # Atlas couldn't classify. Fall back to Tesseract.
        if not self.available:
            return None
        ocr_input = self._tesseract_input(mask)
        digits = self._ocr_digits(ocr_input)
        if not digits:
            return None

        # Auto-bootstrap: if Tesseract's read length matches the live
        # component count, save each component as its corresponding digit
        # template. Existing templates aren't overwritten.
        self._maybe_bootstrap(mask, digits)
        return int(digits)

    # ----- atlas classification ------------------------------------------------

    def _classify_via_atlas(self, mask: np.ndarray) -> str | None:
        """Read each component via correlation against the atlas. Returns
        the assembled digit string, or None if any component falls below
        the threshold/margin gates (or the atlas is empty)."""
        if not self._digit_atlas:
            return None
        components = self._extract_digit_components(mask)
        if not components:
            return None
        result_chars: list[str] = []
        for comp in components:
            # Score against every atlas digit; track best and second-best
            # to enforce an unambiguous-margin gate.
            best_score = -1.0
            best_char: str | None = None
            second_score = -1.0
            for char, ref in self._digit_atlas.items():
                s = self._correlate(comp, ref)
                if s > best_score:
                    second_score = best_score
                    best_score = s
                    best_char = char
                elif s > second_score:
                    second_score = s
            if best_char is None:
                return None
            if best_score < self.DIGIT_MATCH_THRESHOLD:
                return None
            if best_score - second_score < self.DIGIT_MATCH_MARGIN:
                return None
            result_chars.append(best_char)
        return "".join(result_chars)

    def _extract_digit_components(self, mask: np.ndarray) -> list[np.ndarray]:
        """Find connected components, sort left-to-right, normalize each
        to DIGIT_NORM_HEIGHT preserving aspect.

        Drops outlier components whose height is wildly different from
        the others — a phantom UI artifact (e.g. a 1-shaped sliver from
        a neighboring icon edge) creeping into the score region was the
        root cause of "2 read as 12" misreads. We use the *max* height
        as the reference because real digits in this font are uniform
        height, and any short-by-half component is almost certainly
        not a digit.
        """
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        raw: list[tuple[int, int, int, np.ndarray]] = []  # (x, w, h, crop)
        for i in range(1, n_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.MIN_COMPONENT_AREA or w == 0 or h == 0:
                continue
            raw.append((x, w, h, mask[y:y + h, x:x + w]))
        if not raw:
            return []
        # Filter by height: anything shorter than half the tallest
        # component is treated as an artifact, not a digit.
        max_h = max(item[2] for item in raw)
        height_threshold = max_h * 0.5
        filtered = [item for item in raw if item[2] >= height_threshold]
        comps: list[tuple[int, np.ndarray]] = []
        for x, w, h, crop in filtered:
            scale = self.DIGIT_NORM_HEIGHT / h
            new_w = max(1, int(round(w * scale)))
            normalized = cv2.resize(
                crop,
                (new_w, self.DIGIT_NORM_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            comps.append((x, normalized))
        comps.sort(key=lambda c: c[0])
        return [c for _, c in comps]

    @staticmethod
    def _correlate(a: np.ndarray, b: np.ndarray) -> float:
        """Normalized cross-correlation between two same-height binaries.
        Resizes the narrower to the wider before matching."""
        h_a, w_a = a.shape
        h_b, w_b = b.shape
        if w_a < w_b:
            a = cv2.resize(a, (w_b, h_a), interpolation=cv2.INTER_AREA)
        elif w_b < w_a:
            b = cv2.resize(b, (w_a, h_b), interpolation=cv2.INTER_AREA)
        return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED).max())

    # ----- atlas persistence ---------------------------------------------------

    def _load_digit_atlas(self) -> dict[str, np.ndarray]:
        atlas: dict[str, np.ndarray] = {}
        if not self.DIGITS_DIR.is_dir():
            return atlas
        for path in sorted(self.DIGITS_DIR.glob("*.png")):
            char = path.stem
            if not (len(char) == 1 and char.isdigit()):
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            atlas[char] = img
        return atlas

    def _maybe_bootstrap(self, mask: np.ndarray, digits: str) -> None:
        """If the live component count matches the digit string length,
        save any components whose digit doesn't already have a template.
        Reloads the atlas afterward so the next read benefits."""
        components = self._extract_digit_components(mask)
        if len(components) != len(digits):
            return  # ambiguous — Tesseract collapsed/expanded chars
        self.DIGITS_DIR.mkdir(parents=True, exist_ok=True)
        saved_any = False
        for char, comp in zip(digits, components):
            out = self.DIGITS_DIR / f"{char}.png"
            if out.exists():
                continue
            cv2.imwrite(str(out), comp)
            print(f"ScoreReader: bootstrapped digit '{char}' -> {out}")
            saved_any = True
        if saved_any:
            self._digit_atlas = self._load_digit_atlas()

    # ----- tesseract path ------------------------------------------------------

    def _tesseract_input(self, mask: np.ndarray) -> np.ndarray:
        upscaled = cv2.resize(
            mask,
            None,
            fx=self.UPSCALE,
            fy=self.UPSCALE,
            interpolation=cv2.INTER_NEAREST,
        )
        padded = cv2.copyMakeBorder(
            upscaled,
            self.PADDING, self.PADDING, self.PADDING, self.PADDING,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        return cv2.bitwise_not(padded)

    def _ocr_digits(self, ocr_input: np.ndarray) -> str:
        """Run Tesseract twice if needed (PSM 7 → PSM 8) and return any
        digits it produces. Empty string means neither PSM succeeded and
        no fallback dict entry applied either."""
        for psm in (self.PSM_PRIMARY, self.PSM_FALLBACK):
            text = pytesseract.image_to_string(
                ocr_input, config=f"--psm {psm} --oem 3"
            ).strip()
            digits = "".join(c for c in text if c.isdigit())
            if digits:
                return digits
            if text in self.PIXEL_FONT_FALLBACKS:
                return self.PIXEL_FONT_FALLBACKS[text]
        return ""

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
