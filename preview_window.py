"""Live debug preview — renders frames to an OpenCV window with FPS and labeled-box overlays.

Modeled on the dart project's preview_window. Returns False from `show()`
when the user presses q / Esc / closes the window, so the caller can break
out of its main loop cleanly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LabeledBox:
    """A named rectangle in frame (not screen) coordinates."""

    name: str
    top: int
    left: int
    width: int
    height: int


class PreviewWindow:
    def __init__(self, window_name: str = "idleon-basketball preview") -> None:
        self.window_name = window_name
        self._last_tick = time.perf_counter()
        self._frames_since_tick = 0
        self._fps = 0.0
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(
        self,
        frame: np.ndarray,
        overlays: list[LabeledBox] | None = None,
        extra_debug: dict[str, str] | None = None,
    ) -> bool:
        """Render one frame. Returns False if the user requested quit."""
        self._update_fps()

        annotated = frame.copy()
        for box in overlays or []:
            self._draw_overlay(annotated, box)

        lines = [f"FPS: {self._fps:.1f}"]
        if extra_debug:
            lines.extend(f"{k}: {v}" for k, v in extra_debug.items())
        self._draw_debug_box(annotated, lines)

        cv2.imshow(self.window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False
        return True

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

    def _update_fps(self) -> None:
        self._frames_since_tick += 1
        now = time.perf_counter()
        elapsed = now - self._last_tick
        if elapsed >= 0.5:
            self._fps = self._frames_since_tick / elapsed
            self._frames_since_tick = 0
            self._last_tick = now

    @staticmethod
    def _draw_overlay(frame: np.ndarray, box: LabeledBox) -> None:
        color = (0, 255, 255)  # cyan, BGR
        x, y = box.left, box.top
        x2, y2 = x + box.width, y + box.height
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(box.name, font, scale, thickness)
        label_y = y - 6 if y - text_h - 6 >= 0 else y + text_h + 6
        cv2.rectangle(
            frame,
            (x, label_y - text_h - 4),
            (x + text_w + 8, label_y + 4),
            color,
            -1,
        )
        cv2.putText(
            frame, box.name, (x + 4, label_y), font, scale, (0, 0, 0), thickness
        )

    @staticmethod
    def _draw_debug_box(frame: np.ndarray, lines: list[str]) -> None:
        if not lines:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        line_height = 28
        padding = 12

        widths = [cv2.getTextSize(t, font, scale, thickness)[0][0] for t in lines]
        box_w = max(widths) + 2 * padding
        box_h = line_height * len(lines) + 2 * padding

        x = 10
        y = (frame.shape[0] - box_h) // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 1)

        for i, text in enumerate(lines):
            baseline_y = y + padding + (i + 1) * line_height - 8
            cv2.putText(
                frame,
                text,
                (x + padding, baseline_y),
                font,
                scale,
                (0, 255, 0),
                thickness,
            )
