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


@dataclass(frozen=True)
class _Miss:
    """A past throw that didn't score, classified as a directional miss.

    Only undershoot / overshoot misses are useful for negative learning
    — they're confirmed cases where launching at this rim position with
    this dy didn't drop the ball through the rim. `no_launch` and
    `unknown` outcomes are excluded because they reflect click-timing
    issues rather than launch-position issues.
    """

    rim_x: int
    rim_y: int
    dy: int                  # ball_y - rim_y at click time
    stroke: str | None
    rim_moving: bool | None
    outcome: str             # "undershoot" or "overshoot"


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


def load_misses(path: Path) -> list[_Miss]:
    """Read throws.jsonl and return one `_Miss` per directionally-failed
    throw (undershoot or overshoot only).

    Outcomes are re-classified at load time using the current
    `classify_outcome` so the historical data benefits from any
    classifier fixes since the throw was logged. no_launch / unknown
    outcomes are dropped — they describe click-timing failures, not
    launch-position failures, and would muddy negative-learning signals.
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

    misses: list[_Miss] = []
    for r in records:
        ball_x = r.get("ball_x")
        ball_y = r.get("ball_y")
        rim_x = r.get("rim_x")
        rim_y = r.get("rim_y")
        scored_explicit = r.get("scored")
        stroke = r.get("stroke")
        score = r.get("score")

        if None in (ball_x, ball_y, rim_x, rim_y):
            continue
        # Only score-confirmed misses qualify here. Inferred misses (from
        # score progression) carry too much ambiguity for negative
        # learning; we'd rather under-count misses than poison the model
        # with false negatives.
        if scored_explicit is not False:
            continue

        trajectory = r.get("trajectory") or []
        outcome = classify_outcome(trajectory, rim_x, rim_y, scored=False)
        if outcome not in ("undershoot", "overshoot"):
            continue

        if _was_stationary_at_click(score, scored_explicit):
            rim_moving: bool | None = False
        else:
            rim_moving = r.get("rim_moving")
            if rim_moving is None:
                rim_traj = r.get("rim_trajectory") or []
                if len(rim_traj) >= 2:
                    rim_moving = positions_indicate_motion(rim_traj)

        misses.append(_Miss(
            rim_x=rim_x,
            rim_y=rim_y,
            dy=ball_y - rim_y,
            stroke=stroke if stroke in ("up", "down") else None,
            rim_moving=(rim_moving if isinstance(rim_moving, bool) else None),
            outcome=outcome,
        ))

    return misses


class DyModel:
    """Quadratic regression of `dy` as a function of `(rim_x, rim_y)`.

    Fit once at strategy startup from successful stationary-rim throws.
    Used as a cold-start fallback when the live rim has wandered too
    far from any historical make for the nearest-make `dy` to be
    reliable — e.g. the first throw of a new game at an unfamiliar rim
    position.

    The model only explains ~26% of the variance in make `dy` (most of
    the variation is in dimensions we don't observe — animation phase,
    bounce timing, etc.), so it's not used when nearby makes exist.
    But for far-from-data rim positions, its prediction is on average
    ~30 px closer to a real make's `dy` than picking a distant make's
    recorded `dy` would be.
    """

    # Fit needs enough data points for a stable solve. Below this we
    # silently disable the model and the caller falls back to its prior
    # behavior.
    MIN_FIT_SAMPLES = 50

    def __init__(self, makes: list["_Make"]) -> None:
        import numpy as np
        # Stationary-rim makes only — moving-rim dy was measured under
        # different physics (the rim was at a different phase of its
        # swing at click time) and would skew the fit.
        stationary = [m for m in makes if m.rim_moving is False]
        if len(stationary) < self.MIN_FIT_SAMPLES:
            self._coef = None
            self.n_samples = len(stationary)
            return
        X = np.array([[m.rim_x, m.rim_y] for m in stationary])
        y = np.array([m.dy for m in stationary])
        A = self._features(X)
        self._coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.n_samples = len(stationary)

    @staticmethod
    def _features(X):
        import numpy as np
        rx, ry = X[:, 0], X[:, 1]
        return np.column_stack([rx, ry, rx * rx, ry * ry, rx * ry, np.ones(len(X))])

    def predict(self, rim_x: int, rim_y: int) -> int | None:
        """Predicted dy for this rim position, or None if no fit."""
        if self._coef is None:
            return None
        import numpy as np
        X = np.array([[rim_x, rim_y]])
        return int(round(float((self._features(X) @ self._coef)[0])))


class MissIndex:
    """O(1)-ish nearby-miss lookup for the make-selection penalty.

    Buckets historical misses by (rim_x_bucket, rim_y_bucket, dy_bucket)
    so the strategy can ask "how many misses landed near this candidate
    make's signature?" without scanning the full miss list each pick.
    """

    BUCKET_PX = 50  # bucket size for all three axes

    def __init__(self, misses: list[_Miss]) -> None:
        from collections import defaultdict
        self._buckets: dict[tuple[int, int, int], int] = defaultdict(int)
        self._undershoot_buckets: dict[tuple[int, int, int], int] = defaultdict(int)
        self._overshoot_buckets: dict[tuple[int, int, int], int] = defaultdict(int)
        for m in misses:
            key = (m.rim_x // self.BUCKET_PX, m.rim_y // self.BUCKET_PX, m.dy // self.BUCKET_PX)
            self._buckets[key] += 1
            if m.outcome == "undershoot":
                self._undershoot_buckets[key] += 1
            elif m.outcome == "overshoot":
                self._overshoot_buckets[key] += 1
        self.total = len(misses)

    def count_near(self, rim_x: int, rim_y: int, dy: int) -> int:
        """Number of historical misses in the same bucket as this signature."""
        key = (rim_x // self.BUCKET_PX, rim_y // self.BUCKET_PX, dy // self.BUCKET_PX)
        return self._buckets.get(key, 0)


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

    Otherwise the trajectory is inspected via two crossings:

      1. Vertical crossing (descent through rim_y): if the ball passes
         from above to below rim_y *near* rim_x, it's a "make".
      2. Horizontal crossing (first time the ball reaches rim_x): the
         ball's altitude at that moment classifies the miss kind.

    Verdicts:
      "make"       — descent crossing of rim_y within `tolerance_px` of rim_x.
      "no_launch"  — ball never traveled enough horizontally; click hit
                     a dead zone and gave the ball ~no horizontal velocity.
      "undershoot" — either the ball never reached rim_x, OR when it did
                     cross rim_x it was at an altitude below rim_y by more
                     than tolerance_px (passed *under* the rim — needs
                     more loft, not less).
      "overshoot"  — ball crossed rim_x at or above rim_y altitude but
                     missed (sailed past, hit backboard, etc.) Includes
                     the case where the ball bounces back toward the
                     character: the first rim_x crossing (going right)
                     was above the rim, so the throw was high enough.
      "unknown"    — trajectory empty.

    Why altitude-at-rim_x-crossing matters: the old classifier called
    any throw with `max_x > rim_x` an overshoot, but a ball that crossed
    rim_x 200 px below rim_y didn't really "overshoot" — it was a
    vertical undershoot that happened to have enough horizontal range.
    Calling that "overshoot" biases the next throw to *less* loft,
    which makes the next miss worse.
    """
    if scored is True:
        return "make"
    if not trajectory:
        return "unknown"

    xs = [pt[0] for pt in trajectory if len(pt) >= 2]
    if not xs:
        return "unknown"
    max_x_reached = max(xs)
    min_x_reached = min(xs)
    x_span = max_x_reached - min_x_reached

    # No-launch (variant 1): ball barely moved horizontally — click hit
    # a dead phase, ball went straight up and back down. Trajectory's
    # full x range is tiny.
    if x_span < NO_LAUNCH_X_SPAN_PX:
        return "no_launch"

    # Descent crossing of rim_y — used only for make detection. We don't
    # rely on it for overshoot/undershoot since it conflates "ball passed
    # below rim altitude before reaching rim_x" (undershoot) with "ball
    # passed below rim altitude after bouncing back" (overshoot).
    # Also capture the x-direction at the crossing so we can distinguish
    # a real make (ball heading right toward / through the rim) from a
    # backboard rebound (ball heading left, away from the rim) that
    # happens to cross rim_y at an x near rim_x.
    descent_x: float | None = None
    descent_dx: float | None = None
    for i in range(1, len(trajectory)):
        prev = trajectory[i - 1]
        curr = trajectory[i]
        if len(prev) < 2 or len(curr) < 2:
            continue
        y0, y1 = prev[1], curr[1]
        if y0 < rim_y <= y1:
            denom = y1 - y0
            if denom == 0:
                continue
            f = (rim_y - y0) / denom
            descent_x = prev[0] + f * (curr[0] - prev[0])
            descent_dx = curr[0] - prev[0]
            break

    if (
        descent_x is not None
        and abs(descent_x - rim_x) <= tolerance_px
        and (descent_dx is None or descent_dx >= 0)
    ):
        # Ball was descending through rim altitude near rim_x AND still
        # moving rightward (toward/through the rim). A leftward descent
        # at this position means the ball already passed rim_x and is
        # rebounding back from the backboard — that's an overshoot, not
        # a make, and the recorder's `scored` field would have caught a
        # real make above anyway.
        return "make"

    # No-launch (variant 2): ball never had a clean descent crossing AND
    # never got close to the rim horizontally — degenerate trajectory,
    # not a real attempt. Only applies when descent_x is None; a real
    # undershoot with a clean fall stays an undershoot even if it lands
    # hundreds of px short of the rim.
    if (
        descent_x is None
        and max_x_reached < rim_x - NO_LAUNCH_HORIZONTAL_THRESHOLD_PX
    ):
        return "no_launch"

    # Ball never reached rim_x: clear horizontal undershoot.
    if max_x_reached < rim_x - tolerance_px:
        return "undershoot"

    # Ball reached or passed rim_x — find its altitude at the first
    # crossing of rim_x (going either direction). The first crossing in
    # a typical throw is left-to-right; for backboard rebounds it's
    # still the right-going crossing (the ball had to pass rim_x going
    # right before it could bounce back).
    x_cross_y: float | None = None
    for i in range(1, len(trajectory)):
        prev = trajectory[i - 1]
        curr = trajectory[i]
        if len(prev) < 2 or len(curr) < 2:
            continue
        x0, x1 = prev[0], curr[0]
        if (x0 < rim_x <= x1) or (x0 > rim_x >= x1):
            denom = x1 - x0
            if denom == 0:
                continue
            f = (rim_x - x0) / denom
            x_cross_y = prev[1] + f * (curr[1] - prev[1])
            break

    # Crossing rim_x while below rim_y altitude → the throw didn't have
    # enough loft. Direction at the crossing doesn't matter: whether ball
    # was still going right or already heading back left, it failed to
    # be at rim height when it had the horizontal reach. Treat as
    # undershoot so the next correction adds loft.
    if x_cross_y is not None and x_cross_y > rim_y + tolerance_px:
        return "undershoot"

    # Ball passed rim_x at or above rim altitude — a real overshoot
    # trajectory. The throw had enough loft and enough horizontal reach;
    # next correction should reduce one of them.
    return "overshoot"


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
