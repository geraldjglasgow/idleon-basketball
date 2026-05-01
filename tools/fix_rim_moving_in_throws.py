"""One-shot fixer for the `rim_moving` field in throws.jsonl.

The basketball minigame's rim is stationary at scores 0-9 and only starts
moving at score 10. Any pre-10 entry tagged `rim_moving: true` is a
tracker false-positive and pollutes the strategy's make set with bogus
"moving rim" patterns.

This tool sets `rim_moving = false` on entries where the rim was
provably stationary at click time:

  - score < 10                      (still pre-motion)
  - score == 10 AND scored == True  (this throw made it 10; rim wasn't
                                     moving when the click happened)

Entries with `score == 10 AND scored == False` are ambiguous (could be a
miss thrown while already-at-10, when the rim *is* moving) and are left
alone. Entries with `score >= 11` are also left alone — the rim was
moving at click time, so whatever value `rim_moving` already holds is
trusted as observed.

Usage:
  python tools/fix_rim_moving_in_throws.py            # dry run
  python tools/fix_rim_moving_in_throws.py --apply    # write changes
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


THROWS_PATH = Path(__file__).resolve().parent.parent / "throws.jsonl"
BACKUP_PATH = THROWS_PATH.with_suffix(".jsonl.bak")
TMP_PATH = THROWS_PATH.with_suffix(".jsonl.tmp")
MOVING_RIM_MIN_SCORE = 10


def _was_stationary_at_click(score: int | None, scored: bool | None) -> bool:
    """True iff the rim was provably stationary when this throw was clicked."""
    if score is None:
        return False
    if score < MOVING_RIM_MIN_SCORE:
        return True
    # score == 10 with a make => went 9 -> 10 => rim was stationary at click.
    if score == MOVING_RIM_MIN_SCORE and scored is True:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write changes (default: dry run)",
    )
    args = parser.parse_args()

    if not THROWS_PATH.exists():
        print(f"error: {THROWS_PATH} not found", file=sys.stderr)
        return 1

    total = 0
    changed = 0
    fixed_lines: list[str] = []

    with THROWS_PATH.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                fixed_lines.append(line)
                continue
            total += 1
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                fixed_lines.append(line)
                continue

            score = rec.get("score")
            scored = rec.get("scored")
            rim_moving = rec.get("rim_moving")

            if (
                _was_stationary_at_click(score, scored)
                and rim_moving is not False
            ):
                rec["rim_moving"] = False
                changed += 1
                fixed_lines.append(json.dumps(rec) + "\n")
            else:
                fixed_lines.append(line if line.endswith("\n") else line + "\n")

    print(f"scanned {total} entries")
    print(f"would change {changed} entries (rim_moving -> false for stationary throws)")

    if not args.apply:
        print("dry run — pass --apply to write changes")
        return 0

    if changed == 0:
        print("no changes needed; not writing")
        return 0

    print(f"backing up {THROWS_PATH} -> {BACKUP_PATH}")
    shutil.copy2(THROWS_PATH, BACKUP_PATH)

    with TMP_PATH.open("w", encoding="utf-8") as f:
        f.writelines(fixed_lines)
    TMP_PATH.replace(THROWS_PATH)
    print(f"wrote {THROWS_PATH} ({changed} entries updated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
