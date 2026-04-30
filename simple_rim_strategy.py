"""Simple rim strategy.

Replays past makes from throws.jsonl, adapting to the current rim
position and the current swing direction.

Approach:
  1. Load every throw whose `scored` field is true (or, if missing,
     inferred via score-progression). A throw only counts as a real
     make if its trajectory actually crosses rim_y going downward
     within ``RIM_PASS_TOLERANCE_PX`` of rim_x — flukes that bumped
     the rim from the side are rejected. Throws with no trajectory
     data (legacy logs) are kept on a permissive basis.
  2. For each make, store the rim center, the ball's vertical offset
     from the rim at click time (`dy = ball_y - rim_y`), and the stroke
     ("up"/"down"/None) the ball was on at click time. Horizontal offset
     is dropped — the player stands still, so `dx` just mirrors rim_x
     and tells us nothing transferable.
  3. At each frame, compute the live stroke from a sliding window of
     recent ball-y samples (same heuristic the recorder uses). If a
     RimMotionTracker is provided, predict the rim position
     ``BALL_FLIGHT_S`` seconds ahead and use that as the target rim;
     otherwise use the live rim. Filter makes to those compatible with
     the live stroke (legacy makes without a stroke field are kept as
     wildcards), pick the one whose rim was closest to the target rim,
     and click when the live ball's `dy` is within DY_TOLERANCE_PX of
     that make's `dy`.
"""

from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from basketball_tracker import BasketballSample
from rim_tracker import RimSample


@dataclass(frozen=True)
class _Make:
    """A successful past throw, projected for nearest-rim lookup."""

    rim_x: int
    rim_y: int
    dy: int            # ball_y - rim_y at click time
    stroke: str | None  # "up" / "down" / None (legacy makes with no info)


# Two consecutive throws further apart than this are treated as different
# game sessions when inferring `scored` from score progression.
_SESSION_GAP_S = 60.0


class SimpleRimStrategy:
    DY_TOLERANCE_PX = 25
    # Don't issue a second click until this long after the previous one.
    # Roughly matches the recorder's SCORE_DELAY_S so the previous throw's
    # score has time to update before plausibility filtering runs.
    COOLDOWN_S = 4.0
    # Live ball-y window for stroke detection (must match recorder's
    # heuristic so on-screen click decisions agree with what gets logged).
    STROKE_HISTORY = 5
    STROKE_DELTA_PX = 3
    # A throw's trajectory must cross rim_y going downward within this
    # many pixels of rim_x to count as a real make. We're aiming for the
    # exact center of the rim (clean swish) — anything more lenient lets
    # rim-graze bounces seed the strategy with patterns that don't
    # reproduce. The actual rim mouth is ~30 px wide, so 15 keeps us
    # firmly in the middle.
    RIM_PASS_TOLERANCE_PX = 15
    # Approximate ball flight time (click -> ball reaches rim level).
    # Used by the strategy to predict where the rim will be when the
    # ball arrives, so we can lead a moving rim.
    BALL_FLIGHT_S = 1.5
    # Random exploration window — used when we miss but don't yet have a
    # direction (e.g. trajectory data inconclusive). Biased negative for
    # the common front-of-rim under-throw failure mode.
    EXPLORATION_DY_LOW = -60
    EXPLORATION_DY_HIGH = 15
    # Directed correction step — when notify_outcome tells us we under- or
    # over-shot, shift the dy target this many pixels per consecutive
    # streak frame in the appropriate direction. Grows with the streak so
    # we converge on a working release within a handful of throws.
    DIRECTIONAL_CORRECTION_STEP_PX = 30
    # Cap on the cumulative directional correction. Without this, a long
    # streak of undershoots could push target_dy hundreds of pixels off,
    # right out of the ball's actual swing range — and then nothing
    # would ever match. ±90 keeps us within ~3 streak-steps' worth.
    DIRECTIONAL_CORRECTION_MAX_PX = 90

    def __init__(self, throws_log_path: Path | str) -> None:
        self.makes: list[_Make] = self._load(Path(throws_log_path))
        self.last_throw_at: float = 0.0
        self._ball_y_history: deque[int] = deque(maxlen=self.STROKE_HISTORY)
        # Score feedback for adaptive exploration: notify_score(...) tracks
        # whether throws are scoring; if not, should_throw() picks a
        # progressively-different make to break the pattern.
        self._last_observed_score: int | None = None
        self._misses_since_score: int = 0
        # Throttled "why I'm not throwing" diagnostic — logged at most
        # once per second so the console doesn't get spammed with
        # per-frame chatter.
        self._last_wait_log_at: float = 0.0
        self._last_wait_reason: str = ""
        # Directional outcome streaks — set by notify_outcome based on the
        # latest throw's trajectory analysis. Used to bias the dy target
        # in a meaningful direction (rather than random), so consecutive
        # under-throws push the release point earlier in the swing.
        self._undershoot_streak: int = 0
        self._overshoot_streak: int = 0
        with_stroke = sum(1 for m in self.makes if m.stroke is not None)
        print(
            f"SimpleRimStrategy: loaded {len(self.makes)} make(s) "
            f"({with_stroke} with stroke info) from {throws_log_path}"
        )

    def should_throw(
        self,
        ball: BasketballSample | None,
        rim: RimSample | None,
        rim_motion=None,
    ) -> bool:
        """True if a stroke-compatible past make's release pattern matches
        the live ball/predicted-rim, and we're past the cooldown.

        If `rim_motion` is provided (a RimMotionTracker), the target rim
        is the predicted position BALL_FLIGHT_S into the future — so a
        moving rim is led rather than aimed-at. Falls back to the live
        rim when prediction isn't available."""
        if ball is None or rim is None:
            return self._waiting(
                f"no detection (ball={ball is not None}, rim={rim is not None})"
            )
        if not self.makes:
            return self._waiting("no makes loaded")
        cooldown_remaining = self.COOLDOWN_S - (
            time.perf_counter() - self.last_throw_at
        )
        if cooldown_remaining > 0:
            return self._waiting(f"cooldown ({cooldown_remaining:.1f}s left)")

        # Update stroke history first so this frame's ball is in the window.
        self._ball_y_history.append(ball.center[1])
        live_stroke = self._stroke()
        if live_stroke is None:
            return self._waiting("stroke not yet classified (need more motion)")

        # Filter to makes whose stroke matches (or whose stroke is unknown,
        # which we keep as a wildcard so legacy log entries still count).
        candidates = [
            m for m in self.makes
            if m.stroke is None or m.stroke == live_stroke
        ]
        if not candidates:
            # No stroke-matching makes — better a worse-aimed shot than a
            # silent freeze. Fall back to every make and let nearest-rim +
            # exploration sort it out.
            candidates = list(self.makes)

        # Lead the rim if motion data is available — the ball takes ~1.5 s
        # to reach the rim, so we should match against where the rim will
        # be then, not where it is now.
        target = None
        if rim_motion is not None:
            target = rim_motion.predict(self.BALL_FLIGHT_S)
        if target is None:
            target = rim.center
        rx, ry = target

        # Sort candidates by distance to target rim. After a miss, advance
        # the index every throw AND perturb the dy target so we don't
        # repeat the same release pattern — the user's complaint that
        # three identical missed shots was the strategy "not adjusting".
        candidates_sorted = sorted(
            candidates,
            key=lambda m: (m.rim_x - rx) ** 2 + (m.rim_y - ry) ** 2,
        )
        idx = self._misses_since_score % len(candidates_sorted)
        nearest = candidates_sorted[idx]
        target_dy = nearest.dy
        # Directed correction takes precedence — if recent throws under-
        # or over-shot, shift the target *that direction* progressively
        # (capped so it can't walk outside the ball's swing range).
        if self._undershoot_streak > 0:
            target_dy -= min(
                self.DIRECTIONAL_CORRECTION_STEP_PX * self._undershoot_streak,
                self.DIRECTIONAL_CORRECTION_MAX_PX,
            )
        elif self._overshoot_streak > 0:
            target_dy += min(
                self.DIRECTIONAL_CORRECTION_STEP_PX * self._overshoot_streak,
                self.DIRECTIONAL_CORRECTION_MAX_PX,
            )
        elif self._misses_since_score > 0:
            # No directional signal yet — fall back to random exploration.
            target_dy += random.randint(
                self.EXPLORATION_DY_LOW,
                self.EXPLORATION_DY_HIGH,
            )
        bx, by = ball.center
        cur_dy = by - ry
        delta = abs(cur_dy - target_dy)
        if delta > self.DY_TOLERANCE_PX:
            return self._waiting(
                f"dy match: ball_dy={cur_dy} target={target_dy} "
                f"(off by {delta}, tol={self.DY_TOLERANCE_PX}, stroke={live_stroke})"
            )
        return True

    def mark_thrown(self) -> None:
        """Record that we just clicked a throw — gates the cooldown and
        bumps the miss counter (cleared by notify_score on a make)."""
        self.last_throw_at = time.perf_counter()
        self._misses_since_score += 1

    def notify_outcome(self, outcome: str) -> None:
        """Game.py hands us each finalized throw's classified outcome
        ('make' / 'undershoot' / 'overshoot' / 'unknown'). Make resets all
        streaks. Undershoot/overshoot bumps the matching directional
        streak, clears the opposite — so consecutive under-throws push
        the release earlier and earlier; an overshoot resets that bias."""
        if outcome == "make":
            if self._undershoot_streak or self._overshoot_streak or self._misses_since_score:
                print(
                    f"[strategy] make -> reset miss state "
                    f"(was misses={self._misses_since_score}, "
                    f"under={self._undershoot_streak}, over={self._overshoot_streak})"
                )
            self._misses_since_score = 0
            self._undershoot_streak = 0
            self._overshoot_streak = 0
        elif outcome == "undershoot":
            self._undershoot_streak += 1
            self._overshoot_streak = 0
            print(f"[strategy] undershoot streak = {self._undershoot_streak}")
        elif outcome == "overshoot":
            self._overshoot_streak += 1
            self._undershoot_streak = 0
            print(f"[strategy] overshoot streak = {self._overshoot_streak}")
        # "unknown": leave streaks alone — random exploration takes over.

    def notify_score(self, score: int | None) -> None:
        """Game.py hands us each fresh score read so we can detect makes
        and reset the miss counter. Only a strict increase counts — same
        score means the throw was a miss, score going down means the
        game restarted (counter still resets so we don't carry stale
        miss state across games)."""
        if score is None:
            return
        if self._last_observed_score is None:
            self._last_observed_score = score
            return
        if score > self._last_observed_score:
            if self._misses_since_score > 0:
                print(
                    f"[strategy] score {self._last_observed_score} -> {score}: "
                    f"resetting miss streak ({self._misses_since_score})"
                )
            self._misses_since_score = 0
        elif score < self._last_observed_score:
            # Game restart — score reset; clear miss state so the new
            # game's first attempt isn't already in exploration mode.
            self._misses_since_score = 0
        self._last_observed_score = score

    def _waiting(self, reason: str) -> bool:
        """Throttled diagnostic — log why we're not throwing. The first
        word of `reason` is the canonical key; we log on key change or
        when 5 s have passed since the last log of the same key. Without
        canonicalization, time-varying details (cooldown remaining, dy
        delta) would spam every frame."""
        key = reason.split(":", 1)[0].split()[0] if reason else ""
        now = time.perf_counter()
        if key != self._last_wait_reason or now - self._last_wait_log_at >= 5.0:
            print(f"[strategy] waiting: {reason}")
            self._last_wait_log_at = now
            self._last_wait_reason = key
        return False

    def _stroke(self) -> str | None:
        if len(self._ball_y_history) < 2:
            return None
        delta = self._ball_y_history[-1] - self._ball_y_history[0]
        if abs(delta) < self.STROKE_DELTA_PX:
            return None
        return "down" if delta > 0 else "up"

    @staticmethod
    def _load(path: Path) -> list[_Make]:
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
            stroke = r.get("stroke")  # may be missing on legacy entries
            ts = _parse_ts(r.get("ts"))

            if prev_ts is not None and ts is not None:
                if (ts - prev_ts).total_seconds() > _SESSION_GAP_S:
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
                # Reject only when trajectory data *proves* the ball missed
                # the rim laterally. If the trajectory cut off before the
                # ball even reached rim_y (window too short), we have no
                # evidence and fall back to trusting `scored`.
                if trajectory and _trajectory_definitively_missed(
                    trajectory,
                    rim_x,
                    rim_y,
                    SimpleRimStrategy.RIM_PASS_TOLERANCE_PX,
                ):
                    scored = False

            if scored:
                makes.append(_Make(
                    rim_x=rim_x,
                    rim_y=rim_y,
                    dy=ball_y - rim_y,
                    stroke=stroke if stroke in ("up", "down") else None,
                ))

            if score is not None:
                prev_score = score
            if ts is not None:
                prev_ts = ts

        return makes


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
      "undershoot" — ball fell short / crossed rim_y left of the rim
      "overshoot"  — ball crossed rim_y right of the rim, OR ball passed
                     the rim horizontally while above rim level (= it
                     soared over the rim — likely a backboard hit)
      "unknown"    — trajectory empty
    """
    if scored is True:
        return "make"
    if not trajectory:
        return "unknown"

    # Did the ball ever reach past the rim's center, at any altitude?
    # The HSV ball tracker tends to lose the ball at apex / backboard
    # impact (motion blur + color overlap), so post-rebound we often
    # only have samples *below* rim level — even though the ball
    # clearly hit the backboard. Using max_x (no y constraint) is more
    # robust than "above rim" since lateral position is what actually
    # distinguishes a backboard rebound from a fall-short undershoot.
    max_x_reached = max(
        (pt[0] for pt in trajectory if len(pt) >= 2),
        default=None,
    )
    passed_over_rim = max_x_reached is not None and max_x_reached > rim_x

    # Walk the trajectory looking for the first descent through rim_y.
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
        # Descent shows the ball coming down left of the rim — but if the
        # ball had been past the rim mid-flight, this is a backboard
        # rebound, not a fall-short.
        if passed_over_rim:
            return "overshoot"
        return "undershoot"

    # No descent crossing observed within the trajectory window. If the
    # ball had soared past the rim, that's an overshoot (it just hadn't
    # come down yet by the time the window closed). Otherwise the ball
    # never made it that far — undershoot.
    return "overshoot" if passed_over_rim else "undershoot"


def _trajectory_definitively_missed(
    trajectory: list, rim_x: int, rim_y: int, tolerance_px: int
) -> bool:
    """True only when we have *evidence* the ball missed the rim — the
    trajectory crossed `rim_y` going downward at an x outside the
    tolerance. False when the trajectory passed cleanly through, OR
    when it cut off before reaching `rim_y` (no evidence either way —
    trajectory window too short to judge, so we don't reject)."""
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
                return False  # clean pass-through
    # If we reached the rim and never matched cleanly -> definitely missed.
    # Otherwise the window cut off too early to judge — caller keeps scored.
    return reached_rim


def _parse_ts(raw) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None
