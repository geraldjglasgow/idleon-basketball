import time
from pathlib import Path

import cv2
import mss
import numpy as np

from regions import ITEM_BASKETBALL, ITEMS_BUTTON, YOU_KNOW_IT_BUTTON
from utils.mouse import click, long_click


_ITEM_BASKETBALL_TEMPLATE = (
    Path(__file__).parent / "assets" / "item_basketball.png"
)
# The icon has a small count badge in the corner ("1", "2", …); a forgiving
# threshold keeps detection stable as that digit changes between sessions.
_MATCH_THRESHOLD = 0.6
_COOLDOWN_S = 120
_LONG_PRESS_SETTLE_S = 0.75
_VERIFY_DELAY_S = 1.0
_INVENTORY_OPEN_DELAY_S = 0.45

_TEMPLATE_GRAY = cv2.imread(str(_ITEM_BASKETBALL_TEMPLATE), cv2.IMREAD_GRAYSCALE)
if _TEMPLATE_GRAY is None:
    raise FileNotFoundError(f"missing asset: {_ITEM_BASKETBALL_TEMPLATE}")


def _basketball_in_inventory() -> bool:
    """True if the basketball icon currently fills the ITEM_BASKETBALL region."""
    with mss.MSS() as sct:
        raw = sct.grab(ITEM_BASKETBALL._asdict())
    crop = np.asarray(raw)[:, :, :3]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.shape != _TEMPLATE_GRAY.shape:
        gray = cv2.resize(gray, (_TEMPLATE_GRAY.shape[1], _TEMPLATE_GRAY.shape[0]))
    result = cv2.matchTemplate(gray, _TEMPLATE_GRAY, cv2.TM_CCOEFF_NORMED)
    return float(result.max()) >= _MATCH_THRESHOLD


def _click_center(region) -> None:
    click(region.left + region.width // 2, region.top + region.height // 2)


def _long_click_center(region) -> None:
    long_click(region.left + region.width // 2, region.top + region.height // 2)


def start_game() -> None:
    """Open the inventory, long-press the basketball, and confirm. Retries on
    cooldown (basketball still visible after the long press) every 2 minutes
    until the basketball disappears, signalling the game started."""
    while True:
        if not _basketball_in_inventory():
            print("basketball not visible — opening inventory")
            _click_center(ITEMS_BUTTON)
            time.sleep(_INVENTORY_OPEN_DELAY_S)

        _long_click_center(ITEM_BASKETBALL)
        time.sleep(_LONG_PRESS_SETTLE_S)
        _click_center(YOU_KNOW_IT_BUTTON)
        time.sleep(_VERIFY_DELAY_S)

        if not _basketball_in_inventory():
            print("basketball consumed — game started")
            return

        print(
            f"basketball still in inventory (cooldown?) — "
            f"waiting {_COOLDOWN_S}s before retry"
        )
        time.sleep(_COOLDOWN_S)
