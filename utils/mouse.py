"""Mouse input via pynput.

Coordinates are physical pixels (per-monitor DPI aware), matching what
`tools/select_area.py` captures. Cursor position is saved before each
action and restored afterward.
"""

import time

from pynput import mouse

_controller = mouse.Controller()


def click(x: int, y: int) -> None:
    """Move the cursor to (x, y), left-click, and restore the previous position."""
    saved = _controller.position
    _controller.position = (int(x), int(y))
    _controller.click(mouse.Button.left)
    _controller.position = saved


def long_click(x: int, y: int, hold_ms: int = 250) -> None:
    """Move to (x, y), press the left button, hold for hold_ms, release,
    and restore the previous cursor position."""
    saved = _controller.position
    _controller.position = (int(x), int(y))
    _controller.press(mouse.Button.left)
    time.sleep(hold_ms / 1000)
    _controller.release(mouse.Button.left)
    _controller.position = saved
