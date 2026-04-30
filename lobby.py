"""Get the basketball minigame started.

State machine, looping until either the exit button is visible (we're in
the game) or the inventory is open with no basketball (we're out of items
— exit the program). Each iteration:

  1. exit visible?           → return (in game)
  2. basketball in inventory? → long-press it, click YOU_KNOW_IT, loop
  3. otherwise               → click ITEMS to open inventory; if basketball
                                still missing after that, exit the program
"""

import time
from collections import Counter
from pathlib import Path

import cv2
import mss
import numpy as np

from preview_window import PreviewWindow
from regions import (
    COOLDOWN_TIMER_REGION,
    EXIT_BUTTON_REGION,
    ITEM_BASKETBALL,
    ITEMS_BUTTON,
    Region,
    YOU_KNOW_IT_BUTTON,
)
from score_reader import ScoreReader
from utils.mouse import click, long_click


_ASSETS = Path(__file__).parent / "assets"
# The basketball icon has a tiny count badge that varies between runs;
# the exit button is solid red and matches very strongly. One forgiving
# threshold works for both.
_MATCH_THRESHOLD = 0.6
_CONSUME_WAIT_S = 3.0
_INVENTORY_OPEN_DELAY_S = 1.0
_COOLDOWN_S = 120  # if consume fails (basketball still visible), wait this long
# Cooldown timer OCR sanity bounds + voting. Real cooldown is at most ~600s,
# so anything beyond 999 is a misread (e.g. Tesseract appending a phantom
# 4th digit, like reading "293" as "2933"). 0 is also nonsense — that means
# no cooldown but the consume failed somehow.
_COOLDOWN_MAX_S = 999
# Take 5 samples and vote — more chances to outvote a single OCR glitch
# like a dropped leading digit. A common failure mode is "843" being read
# as "43"; with majority voting we resist that as long as most reads are
# correct, AND length-filtering below biases toward the more-confident
# reading when length disagrees.
_COOLDOWN_VOTE_SAMPLES = 5
_COOLDOWN_VOTE_INTERVAL_S = 0.05


def _load_template(name: str) -> np.ndarray:
    path = _ASSETS / name
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"missing asset: {path}")
    return img


_BASKETBALL_TEMPLATE = _load_template("item_basketball.png")
_EXIT_TEMPLATE = _load_template("exit_button.png")
# Same OCR pipeline as the in-game score — works for any small white-on-dark
# digit readout, including the cooldown timer next to the inventory icon.
_NUMBER_READER = ScoreReader()


def _read_cooldown_timer() -> int | None:
    """Capture COOLDOWN_TIMER_REGION several times and OCR each. Returns
    the mode of plausible readings, or None if no plausible reading was
    seen.

    Multi-sample voting + an upper-bound cap (999) protect against:
      - "293 -> 2933" hallucinated extra digit (rejected by upper cap)
      - "843 -> 43" dropped leading digit (handled by length-filter:
        when sample digit-counts disagree, only the longest survive
        before mode voting, on the assumption that an OCR misread is
        far more likely to drop a digit than to invent one)
    """
    region_dict = COOLDOWN_TIMER_REGION._asdict()
    samples: list[int] = []
    with mss.MSS() as sct:
        for i in range(_COOLDOWN_VOTE_SAMPLES):
            raw = sct.grab(region_dict)
            frame = np.asarray(raw)[:, :, :3]
            v = _NUMBER_READER.read(frame, region_dict, region_dict)
            if v is not None and 0 < v <= _COOLDOWN_MAX_S:
                samples.append(v)
            if i < _COOLDOWN_VOTE_SAMPLES - 1:
                time.sleep(_COOLDOWN_VOTE_INTERVAL_S)
    if not samples:
        return None
    # Length-filter: cooldowns count down, so within a 250 ms sampling
    # window all *correct* reads should have the same digit count. If
    # one sample reads 2 digits and the others read 3, the short read
    # is almost certainly a dropped-leading-digit misread.
    longest = max(len(str(v)) for v in samples)
    samples = [v for v in samples if len(str(v)) == longest]
    return Counter(samples).most_common(1)[0][0]


def _template_present(template: np.ndarray, region: Region) -> bool:
    """True if the live screen content at `region` matches `template`."""
    with mss.MSS() as sct:
        raw = sct.grab(region._asdict())
    crop = np.asarray(raw)[:, :, :3]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.shape != template.shape:
        gray = cv2.resize(gray, (template.shape[1], template.shape[0]))
    score = float(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED).max())
    return score >= _MATCH_THRESHOLD


def _basketball_in_inventory() -> bool:
    return _template_present(_BASKETBALL_TEMPLATE, ITEM_BASKETBALL)


def _exit_button_visible() -> bool:
    return _template_present(_EXIT_TEMPLATE, EXIT_BUTTON_REGION)


def _click_center(region: Region) -> None:
    click(region.left + region.width // 2, region.top + region.height // 2)


def _long_click_center(region: Region) -> None:
    long_click(region.left + region.width // 2, region.top + region.height // 2)


def _show(preview: PreviewWindow | None, *lines: str) -> None:
    if preview is not None:
        preview.show_status(list(lines))


def _sleep_with_preview(
    seconds: float, preview: PreviewWindow | None, *status: str
) -> None:
    """Sleep for `seconds`, but pump the preview window every 50 ms so the
    OS doesn't mark it 'Not Responding' and the user sees a live countdown."""
    if preview is None:
        time.sleep(seconds)
        return
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        remaining = end - time.perf_counter()
        preview.show_status(list(status) + [f"waiting {remaining:.0f}s"])
        time.sleep(0.05)


def start_game(preview: PreviewWindow | None = None) -> None:
    while True:
        _show(preview, "LOBBY", "checking exit button")
        if _exit_button_visible():
            print("lobby: exit button visible — already in game")
            return

        _show(preview, "LOBBY", "checking inventory")
        if _basketball_in_inventory():
            print("lobby: consuming basketball")
            _show(preview, "LOBBY", "consuming basketball")
            _long_click_center(ITEM_BASKETBALL)
            _sleep_with_preview(_CONSUME_WAIT_S, preview, "LOBBY", "after long-press")
            cooldown = _read_cooldown_timer()
            print(f"lobby: cooldown timer reads: {cooldown}")
            _click_center(YOU_KNOW_IT_BUTTON)
            _sleep_with_preview(_CONSUME_WAIT_S, preview, "LOBBY", "after you-know-it")
            # Trust the cooldown timer read directly. The previous logic
            # gated the wait on `_basketball_in_inventory()` after the YKI
            # click, but inventory animations can briefly hide the icon and
            # cause us to skip the wait + immediately retry.
            if cooldown is not None:
                wait_s = cooldown + 2
                print(
                    f"lobby: cooldown {cooldown}s detected — "
                    f"waiting {wait_s}s before retry"
                )
                _sleep_with_preview(
                    wait_s, preview, "LOBBY", f"cooldown {cooldown}s"
                )
            elif _basketball_in_inventory():
                # Cooldown read failed but basketball is still there —
                # something went wrong with consume; fall back to fixed wait.
                print(
                    f"lobby: basketball still visible, cooldown read failed — "
                    f"falling back to {_COOLDOWN_S}s"
                )
                _sleep_with_preview(
                    _COOLDOWN_S, preview, "LOBBY", "cooldown (fallback)"
                )
            continue

        print("lobby: opening inventory")
        _show(preview, "LOBBY", "opening inventory")
        _click_center(ITEMS_BUTTON)
        _sleep_with_preview(_INVENTORY_OPEN_DELAY_S, preview, "LOBBY", "opening")

        if not _basketball_in_inventory():
            # Unexpected state — keep retrying. A small wait avoids
            # hammering the ITEMS button on a tight loop.
            print("lobby: inventory open but no basketball — retrying")
            _sleep_with_preview(3.0, preview, "LOBBY", "no basketball — retrying")
