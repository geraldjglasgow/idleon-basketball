"""Screen regions (physical pixels) for the game.

Each region is `(left, top, width, height)` — the same shape pyautogui's
`region=` and our own screenshot helpers expect. Capture new regions with
`tools/select_area.bat`.
"""

from typing import NamedTuple


class Region(NamedTuple):
    left: int
    top: int
    width: int
    height: int


ITEMS_BUTTON = Region(left=1096, top=923, width=112, height=98)
ITEM_BASKETBALL = Region(left=1538, top=425, width=122, height=124)
YOU_KNOW_IT_BUTTON = Region(left=177, top=600, width=372, height=80)
SCORE_REGION = Region(left=156, top=74, width=59, height=29)
THROW_ZONE = Region(left=1546, top=29, width=302, height=91)
