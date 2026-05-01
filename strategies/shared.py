"""Shared helpers used by every strategy.

Owns:
  - `_Make` dataclass — projection of one past successful throw.
  - `load_makes(...)` — read throws.jsonl and reduce to a list of `_Make`s.
  - `classify_outcome(...)` / `_trajectory_definitively_missed(...)` — the
    outcome-classification helpers game.py uses post-throw.
  - `MOVING_RIM_MIN_SCORE` — the score below which the rim is provably
    stationary (game observation: rim starts moving at score 10).

Keeping these module-level (not class-level) lets every strategy reuse
them without inheritance gymnastics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rim_motion_tracker import positions_indicate_motion


# The rim is stationary at scores 0-9 and starts moving at score 10. Any
# entry with `score < 10` is treated as a stationary-rim throw regardless
# of what its `rim_moving` field claims (handles legacy false-positives).
# An entry with `score == 10 and scored is True` was clicked while still
# at score 9, so it was also stationary at click time.
MOVING_RIM_MIN_SCORE = 10

# Two consecutive throws further apart than this are treated as different
# game sessions when inferring `scored` from score progression.
SESSION_GAP_S = 60.0

# A throw's trajectory must cross rim_y going downward within this many
# pixels of rim_x to count as a real make. Aiming for the exact center of
# the rim — the actual mouth is ~30 px wide, so 15 keeps us firmly in the
# middle.
RIM_PASS_TOLERANCE_PX = 15


@dataclass(frozen=True)
class _Make:
    """A successful past throw, projected for nearest-rim lookup."""

    rim_x: int
    rim_y: int
    dy: int                  # ball_y - rim_y at click time
    stroke: str | None       # "up" / "down" / None (legacy makes with no info)
    rim_moving: bool | None  # True if rim was moving during this throw,
                             # False if stationary, None for legacy logs
                             # without enough rim_trajectory data


def _was_stationary_at_click(score: int | None, scored: bool | None) -> bool:
    """True iff the rim was provably stationary when this throw was clicked.

    Mirrors tools/fix_rim_moving_in_throws.py — any entry below the motion
    threshold, or right at the threshold via a make, was clicked while
    pre-motion. We don't trust derived/recorded rim_moving for these.
    """
    if score is None:
        return False
    if score < MOVING_RIM_MIN_SCORE:
        return True
    if score == MOVING_RIM_MIN_SCORE and scored is True:
        return True
    return False


def _parse_ts(raw) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def load_makes(path: Path) -> list[_Make]:
    """Read throws.jsonl and return one `_Make` per successful throw.

    A throw counts as a make when either `scored: true` is recorded, or
    score progression in this game session shows an increase. Throws
    rejected by `_trajectory_definitively_missed` are dropped — flukes
    that bumped the rim from the side don't seed transferable patterns.
    """
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue

    records.sort(key=lambda r: r.get("ts") or "")

    makes: list[_Make] = []
    prev_score: int | None = None
    prev_ts: datetime | None = None
    for r in records:
        ball_x = r.get("ball_x")
        ball_y = r.get("ball_y")
        rim_x = r.get("rim_x")
        rim_y = r.get("rim_y")
        score = r.get("score")
        scored_explicit = r.get("scored")
        stroke = r.get("stroke")

        # Score-gated rim_moving: pre-threshold throws are stationary by
        # definition, regardless of what the entry stored or what the
        # rim_trajectory derives to.
        if _was_stationary_at_click(score, scored_explicit):
            rim_moving: bool | None = False
        else:
            rim_moving = r.get("rim_moving")
            if rim_moving is None:
                rim_traj = r.get("rim_trajectory") or []
                if len(rim_traj) >= 2:
                    rim_moving = positions_indicate_motion(rim_traj)

        ts = _parse_ts(r.get("ts"))

        if prev_ts is not None and ts is not None:
            if (ts - prev_ts).total_seconds() > SESSION_GAP_S:
                prev_score = None
        if (
            prev_score is not None
            and score is not None
            and score < prev_score
        ):
            prev_score = None

        if None in (ball_x, ball_y, rim_x, rim_y):
            if score is not None:
                prev_score = score
            if ts is not None:
                prev_ts = ts
            continue

        if scored_explicit is True:
            scored = True
        elif scored_explicit is False:
            scored = False
        else:
            scored = (
                prev_score is not None
                and score is not None
                and score > prev_score
            )

        if scored:
            trajectory = r.get("trajectory") or []
            if trajectory and _trajectory_definitively_missed(
                trajectory, rim_x, rim_y, RIM_PASS_TOLERANCE_PX,
            ):
                scored = False

        if scored:
            makes.append(_Make(
                rim_x=rim_x,
                rim_y=rim_y,
                dy=ball_y - rim_y,
                stroke=stroke if stroke in ("up", "down") else None,
                rim_moving=(
                    rim_moving if isinstance(rim_moving, bool) else None
                ),
            ))

        if score is not None:
            prev_score = score
        if ts is not None:
            prev_ts = ts

    return makes


# Horizontal threshold below which we say the ball never actually
# launched — distinct from a real undershoot. When the click hits during
# a bad part of the swing, the game gives the ball almost no horizontal
# velocity; the trajectory's max_x stays within ~50 px of the click
# point. Treating these as plain "undershoot" pollutes the dy-bias
# correction, since the dy was probably fine — the click timing wasn't.
NO_LAUNCH_HORIZONTAL_THRESHOLD_PX = 200

# Total horizontal range (max_x - min_x) below which we declare no-launch
# even if the trajectory technically dips back below rim_y. Real shots
# travel many hundreds of pixels horizontally; a wasted click sees the
# ball go straight up and back down, so its x range is on the order of
# pixels of jitter (~30 in observed bad clicks). This catches the case
# where a no-launch trajectory happens to cross rim_y on its way back
# down — without it, those misclassify as "undershoot" and the dy
# correction churns ineffectively across repeated bad clicks.
NO_LAUNCH_X_SPAN_PX = 100


def classify_outcome(
    trajectory: list,
    rim_x: int,
    rim_y: int,
    scored: bool | None = None,
    tolerance_px: int = 30,
) -> str:
    """Classify a thrown shot's outcome.

    If `scored` is True (the recorder confirmed the score went up after
    this throw), the result is "make" regardless of what the trajectory
    looked like — a backboard rebound that falls through the rim might
    bounce back too late for our fixed-window trajectory capture, so we
    trust the score signal as ground truth.

    Otherwise the trajectory is inspected:
      "make"       — descent crossing of rim_y is within tolerance of rim_x
      "no_launch"  — ball never got within NO_LAUNCH_HORIZONTAL_THRESHOLD_PX
                     of the rim; click landed in a dead zone of the swing
                     and produced essentially no horizontal velocity.
                     Distinguished from undershoot because the dy was
                     probably fine — only the click timing was wrong.
      "undershoot" — ball reached toward the rim but fell short of rim_y
                     before getting there.
      "overshoot"  — ball crossed rim_y right of the rim, OR ball passed
                     the rim horizontally while above rim level.
      "unknown"    — trajectory empty.
    """
    if scored is True:
        return "make"
    if not trajectory:
        return "unknown"

    xs = [pt[0] for pt in trajectory if len(pt) >= 2]
    max_x_reached = max(xs) if xs else None
    min_x_reached = min(xs) if xs else None
    x_span = (
        max_x_reached - min_x_reached
        if max_x_reached is not None and min_x_reached is not None
        else None
    )
    passed_over_rim = max_x_reached is not None and max_x_reached > rim_x

    # No-launch detection runs first: a click during a dead phase of the
    # swing produces an almost-vertical trajectory (ball pops up, comes
    # straight back down). When that trajectory happens to dip below
    # rim_y, the descent-crossing branch below would otherwise label it
    # an "undershoot" — but the dy was fine, the click timing was wrong.
    # Mislabeling churns the dy-correction streak instead of the timing.
    if x_span is not None and x_span < NO_LAUNCH_X_SPAN_PX:
        return "no_launch"

    descent_x: float | None = None
    for i in range(1, len(trajectory)):
        prev = trajectory[i - 1]
        curr = trajectory[i]
        if len(prev) < 2 or len(curr) < 2:
            continue
        x0, y0 = prev[0], prev[1]
        x1, y1 = curr[0], curr[1]
        if y0 < rim_y <= y1:
            denom = y1 - y0
            if denom == 0:
                continue
            f = (rim_y - y0) / denom
            descent_x = x0 + f * (x1 - x0)
            break

    if descent_x is not None:
        if abs(descent_x - rim_x) <= tolerance_px:
            return "make"
        if descent_x > rim_x + tolerance_px:
            return "overshoot"
        if passed_over_rim:
            return "overshoot"
        return "undershoot"

    # No descent crossing observed. If the ball never even approached
    # the rim horizontally, it's a "no launch" — the click was wasted.
    if (
        max_x_reached is not None
        and max_x_reached < rim_x - NO_LAUNCH_HORIZONTAL_THRESHOLD_PX
    ):
        return "no_launch"

    return "overshoot" if passed_over_rim else "undershoot"


def measure_ball_flight_s(
    path: Path,
    default_s: float = 1.5,
    min_samples: int = 10,
) -> float:
    """Median time from click to ball-reaches-rim_y across past makes.

    Walks throws.jsonl looking at each scored entry's trajectory, finds
    the descent crossing of rim_y, and returns the median dt_ms across
    all such crossings (converted to seconds). Falls back to `default_s`
    when fewer than `min_samples` valid trajectories exist — too few
    samples and the median is unstable.

    A few makes' trajectories don't show a descent crossing (window cut
    off too early, or score-only makes via backboard rebound) — those
    are skipped. The remaining samples have noisy outliers in both
    directions; median handles that better than a mean.
    """
    if not path.exists():
        return default_s
    flight_times_ms: list[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not r.get("scored"):
                continue
            rx, ry = r.get("rim_x"), r.get("rim_y")
            traj = r.get("trajectory") or []
            if rx is None or ry is None or not traj:
                continue
            for i in range(1, len(traj)):
                prev = traj[i - 1]
                curr = traj[i]
                if len(prev) < 3 or len(curr) < 3:
                    continue
                x0, y0, dt0 = prev[0], prev[1], prev[2]
                x1, y1, dt1 = curr[0], curr[1], curr[2]
                if y0 < ry <= y1:
                    denom = y1 - y0
                    if denom == 0:
                        continue
                    frac = (ry - y0) / denom
                    cross_dt_ms = dt0 + frac * (dt1 - dt0)
                    flight_times_ms.append(cross_dt_ms)
                    break
    if len(flight_times_ms) < min_samples:
        return default_s
    flight_times_ms.sort()
    median_ms = flight_times_ms[len(flight_times_ms) // 2]
    return median_ms / 1000.0


def _trajectory_definitively_missed(
    trajectory: list, rim_x: int, rim_y: int, tolerance_px: int
) -> bool:
    """True only when we have *evidence* the ball missed the rim — the
    trajectory crossed `rim_y` going downward at an x outside the
    tolerance. False when the trajectory passed cleanly through, OR
    when it cut off before reaching `rim_y`."""
    reached_rim = False
    for i in range(1, len(trajectory)):
        prev = trajectory[i - 1]
        curr = trajectory[i]
        if len(prev) < 2 or len(curr) < 2:
            continue
        x0, y0 = prev[0], prev[1]
        x1, y1 = curr[0], curr[1]
        if y0 < rim_y <= y1:
            reached_rim = True
            denom = y1 - y0
            if denom == 0:
                continue
            frac = (rim_y - y0) / denom
            x_at_rim = x0 + frac * (x1 - x0)
            if abs(x_at_rim - rim_x) <= tolerance_px:
                return False
    return reached_rim
