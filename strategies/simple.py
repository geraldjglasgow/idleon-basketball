"""Simple rim strategy.

Replays past makes from throws.jsonl, adapting to the current rim
position and the current swing direction. See module-level docstring on
the legacy simple_rim_strategy.py for the full algorithm description —
this file is the moved-and-trimmed version.

Approach (unchanged from the original):
  1. Load every successful past throw via shared.load_makes.
  2. For each make, store rim center, ball-y offset (`dy = ball_y - rim_y`),
     stroke, and whether the rim was moving at click time.
  3. Each frame, compute live stroke; lead the rim by BALL_FLIGHT_S; pick
     the make whose rim was closest to the predicted rim, filtered by
     stroke + motion compatibility. Click when ball-`dy` matches.
"""

from __future__ import annotations

import random
import time
from collections import deque
from pathlib import Path

from basketball_tracker import BasketballSample
from rim_tracker import RimSample

from .base import Strategy
from .shared import _Make, load_makes, measure_ball_flight_s


class SimpleRimStrategy(Strategy):
    DY_TOLERANCE_PX = 25
    # Don't issue a second click until this long after the previous one.
    COOLDOWN_S = 4.0
    # Wait for the predicted rim to be near a position we have a make for.
    MAX_PREDICTED_RIM_DIST_PX = 80
    # Live ball-y window for stroke detection. STROKE_HISTORY/_DELTA must
    # match the recorder's so the stroke field on logged makes uses the
    # same rule we filter on. The strategy's *click gate* is tighter:
    # STROKE_CONFIRM_FRAMES consecutive frames must all show motion in
    # the same direction with at least STROKE_CONFIRM_DELTA_PX between
    # them. That rejects transitional / about-to-reverse frames where
    # the loose 5-sample delta still reads "up" but the ball is actually
    # decelerating. Settings: 2 frames @ 3 px is much stricter than the
    # original (any 3 px over 5 samples) but still loose enough that
    # dropped tracker frames or slow ball segments don't lock the gate
    # closed forever.
    STROKE_HISTORY = 5
    STROKE_DELTA_PX = 3
    STROKE_CONFIRM_FRAMES = 2
    STROKE_CONFIRM_DELTA_PX = 3
    # Approximate ball flight time (click -> ball reaches rim level).
    # The class attribute is the fallback; __init__ replaces it on the
    # instance with the empirical median from past makes when enough
    # data is available.
    BALL_FLIGHT_S = 1.5
    # Random exploration window — used when we miss but don't yet have a
    # direction. Biased negative for the common front-of-rim under-throw.
    EXPLORATION_DY_LOW = -60
    EXPLORATION_DY_HIGH = 15
    # Force-throw timeout — if we've been waiting longer than this since
    # the last throw, take a blind/best-effort shot rather than freezing.
    WAIT_TIMEOUT_S = 60.0
    # Directed correction step + cap — when notify_outcome tells us we
    # under- or over-shot, shift dy by step*streak in that direction.
    # Step raised from 30 -> 50 and cap from 90 -> 200 because real
    # undershoots in the wild were 200+ px; the old cap couldn't recover.
    DIRECTIONAL_CORRECTION_STEP_PX = 50
    DIRECTIONAL_CORRECTION_MAX_PX = 200

    # A make gets quarantined for the rest of the session after this
    # many consecutive non-make outcomes attributed to it. Stops the
    # nearest-rim picker from getting stuck on a make whose recorded
    # release no longer reproduces (different in-game release physics,
    # stale calibration, etc.). On a make the counter resets, so a
    # genuinely-good make stays usable forever.
    MAX_CONSECUTIVE_FAILURES_PER_MAKE = 3

    # The simple strategy is happy with the tracker's default 2 s window —
    # linear extrapolation only needs a couple of seconds of velocity.
    REQUIRED_HISTORY_WINDOW_S: float | None = None

    def __init__(self, throws_log_path: Path | str) -> None:
        path = Path(throws_log_path)
        self.makes: list[_Make] = load_makes(path)
        # Override class default with empirical measurement when we have
        # enough scored throws to compute a reliable median.
        self.BALL_FLIGHT_S = measure_ball_flight_s(
            path, default_s=type(self).BALL_FLIGHT_S
        )
        self.last_throw_at: float = time.perf_counter()
        self._ball_y_history: deque[int] = deque(maxlen=self.STROKE_HISTORY)
        self._last_observed_score: int | None = None
        self._misses_since_score: int = 0
        self._last_wait_log_at: float = 0.0
        self._last_wait_reason: str = ""
        self._undershoot_streak: int = 0
        self._overshoot_streak: int = 0
        # Make quarantine: a make that fails MAX_CONSECUTIVE_FAILURES_PER_MAKE
        # times in a row gets dropped from the candidate set for the rest
        # of the session. Identity is by object, so the same _Make
        # instance picked twice will share state.
        self._make_failure_counts: dict[int, int] = {}
        self._quarantined_makes: set[int] = set()
        # The make whose dy/release pattern was used for the most recent
        # click. Set by should_throw, consumed by notify_outcome to credit
        # the correct make with the result.
        self._last_used_make: _Make | None = None
        with_stroke = sum(1 for m in self.makes if m.stroke is not None)
        print(
            f"SimpleRimStrategy: loaded {len(self.makes)} make(s) "
            f"({with_stroke} with stroke info) from {throws_log_path}; "
            f"BALL_FLIGHT_S={self.BALL_FLIGHT_S:.2f}s"
        )

    def should_throw(
        self,
        ball: BasketballSample | None,
        rim: RimSample | None,
        rim_motion=None,
    ) -> bool:
        # Feed ball history every frame the ball is detected — including
        # during cooldown — so that as soon as cooldown clears the stroke
        # gate has a populated window to evaluate against, instead of
        # bootstrapping from zero each post-cooldown.
        if ball is not None:
            self._ball_y_history.append(ball.center[1])

        if ball is None or rim is None:
            return self._waiting(
                f"no detection (ball={ball is not None}, rim={rim is not None})"
            )
        cooldown_remaining = self.COOLDOWN_S - (
            time.perf_counter() - self.last_throw_at
        )
        if cooldown_remaining > 0:
            return self._waiting(f"cooldown ({cooldown_remaining:.1f}s left)")

        waited_s = time.perf_counter() - self.last_throw_at
        if waited_s >= self.WAIT_TIMEOUT_S:
            print(
                f"[strategy] FORCING throw — waited {waited_s:.0f}s "
                f"(>= {self.WAIT_TIMEOUT_S:.0f}s timeout)"
            )
            return True

        if not self.makes:
            return self._waiting("no makes loaded")

        live_stroke = self._stroke()
        if live_stroke is None:
            return self._waiting("stroke not yet classified (need more motion)")

        stroke_matched = [
            m for m in self.makes
            if m.stroke is None or m.stroke == live_stroke
        ]
        if not stroke_matched:
            stroke_matched = list(self.makes)

        if rim_motion is not None:
            live_moving = rim_motion.is_moving()
            motion_matched = [
                m for m in stroke_matched
                if m.rim_moving is None or m.rim_moving == live_moving
            ]
            candidates = motion_matched if motion_matched else stroke_matched
        else:
            candidates = stroke_matched

        # Drop quarantined makes — they've failed too many times in a row
        # to be worth re-trying this session. If quarantining empties the
        # set, fall back to the un-filtered candidates (better an imperfect
        # shot than a freeze).
        non_quarantined = [
            m for m in candidates if id(m) not in self._quarantined_makes
        ]
        if non_quarantined:
            candidates = non_quarantined

        target = None
        if rim_motion is not None:
            target = rim_motion.predict(self.BALL_FLIGHT_S)
        if target is None:
            target = rim.center
        rx, ry = target

        candidates_sorted = sorted(
            candidates,
            key=lambda m: (m.rim_x - rx) ** 2 + (m.rim_y - ry) ** 2,
        )
        idx = self._misses_since_score % len(candidates_sorted)
        nearest = candidates_sorted[idx]
        rim_to_make_dist = (
            (nearest.rim_x - rx) ** 2 + (nearest.rim_y - ry) ** 2
        ) ** 0.5
        if rim_to_make_dist > self.MAX_PREDICTED_RIM_DIST_PX:
            min_dist_to_make = (
                rim_motion.min_distance_to(nearest.rim_x, nearest.rim_y)
                if rim_motion is not None
                else None
            )
            rim_passes_near_make = (
                min_dist_to_make is not None
                and min_dist_to_make <= self.MAX_PREDICTED_RIM_DIST_PX
            )
            if rim_passes_near_make:
                return self._waiting(
                    f"predicted rim ({rx}, {ry}) is {rim_to_make_dist:.0f}px "
                    f"from make ({nearest.rim_x}, {nearest.rim_y}); rim has "
                    f"been within {min_dist_to_make:.0f}px in recent history "
                    f"— waiting for next pass"
                )
            self._log_imperfect_match(rim_to_make_dist, nearest)

        target_dy = nearest.dy
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
        # Remember which make's pattern we just committed to — notify_outcome
        # will use this to credit/blame the make for the resulting outcome.
        self._last_used_make = nearest
        return True

    def mark_thrown(self) -> None:
        self.last_throw_at = time.perf_counter()
        self._misses_since_score += 1

    def notify_outcome(self, outcome: str) -> None:
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
            # The make's failure counter is also reset by this outcome —
            # tracked + cleared in _record_outcome_for_make below.
            self._record_outcome_for_make(outcome)
        elif outcome == "undershoot":
            self._undershoot_streak += 1
            self._overshoot_streak = 0
            print(f"[strategy] undershoot streak = {self._undershoot_streak}")
            self._record_outcome_for_make(outcome)
        elif outcome == "overshoot":
            self._overshoot_streak += 1
            self._undershoot_streak = 0
            print(f"[strategy] overshoot streak = {self._overshoot_streak}")
            self._record_outcome_for_make(outcome)
        elif outcome == "no_launch":
            # The dy was probably fine; the click hit a dead spot in the
            # swing and gave the ball no horizontal velocity. Don't bias
            # the dy correction streaks — instead, mark this make as
            # suspect so we rotate to a different one next throw.
            print(
                f"[strategy] no_launch — click timing missed; "
                f"NOT biasing dy, will rotate makes"
            )
            self._record_outcome_for_make(outcome)

    def notify_score(self, score: int | None) -> None:
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
            self._misses_since_score = 0
        self._last_observed_score = score

    def notify_game_reset(self) -> None:
        """Clear per-game state at game restart. The new game starts at
        score 0; without resetting `_last_observed_score`, the miss-streak
        accounting would treat the drop from previous score to 0 as a
        normal regression rather than a fresh game.

        Quarantines and per-make failure counts persist across games
        intentionally — a make that consistently fails is just bad data,
        and a new game doesn't change that."""
        self._last_observed_score = None
        self._misses_since_score = 0
        self._undershoot_streak = 0
        self._overshoot_streak = 0
        self._last_used_make = None
        self._last_wait_reason = ""
        self._last_wait_log_at = 0.0

    def _record_outcome_for_make(self, outcome: str) -> None:
        """Update the per-make failure counter for the make that produced
        the most recent throw, and quarantine it if its consecutive-failure
        count crosses the threshold. A make ('make') zeroes the counter."""
        m = self._last_used_make
        # Clear immediately so a future throw without an intervening
        # should_throw->return-True doesn't double-count.
        self._last_used_make = None
        if m is None:
            return
        key = id(m)
        if outcome == "make":
            if self._make_failure_counts.pop(key, 0):
                print(
                    f"[strategy] make at ({m.rim_x}, {m.rim_y}) — "
                    f"resetting its failure counter"
                )
            return
        # Any non-make outcome counts toward quarantine. We treat
        # no_launch the same as a real miss for this purpose: a make
        # whose recorded release keeps producing dead-zone clicks isn't
        # useful regardless of the cause.
        count = self._make_failure_counts.get(key, 0) + 1
        self._make_failure_counts[key] = count
        if count >= self.MAX_CONSECUTIVE_FAILURES_PER_MAKE:
            self._quarantined_makes.add(key)
            print(
                f"[strategy] QUARANTINING make at ({m.rim_x}, {m.rim_y}) "
                f"after {count} consecutive failures (last: {outcome}); "
                f"won't be picked again this session"
            )

    def _log_imperfect_match(self, dist: float, make: _Make) -> None:
        now = time.perf_counter()
        if now - self._last_wait_log_at >= 5.0 or self._last_wait_reason != "imperfect":
            print(
                f"[strategy] no near match — rim is {dist:.0f}px from "
                f"closest make ({make.rim_x}, {make.rim_y}); proceeding "
                f"with best-effort release"
            )
            self._last_wait_log_at = now
            self._last_wait_reason = "imperfect"

    def _waiting(self, reason: str) -> bool:
        key = reason.split(":", 1)[0].split()[0] if reason else ""
        now = time.perf_counter()
        if key != self._last_wait_reason or now - self._last_wait_log_at >= 5.0:
            print(f"[strategy] waiting: {reason}")
            self._last_wait_log_at = now
            self._last_wait_reason = key
        return False

    def _stroke(self) -> str | None:
        """Confident stroke direction or None.

        Tolerant confirmation rule: across the last STROKE_HISTORY-1
        frame-to-frame deltas, count how many are >= STROKE_CONFIRM_DELTA_PX
        in each direction. If one direction wins by at least
        STROKE_CONFIRM_FRAMES votes (and the other direction has fewer),
        return that direction. This handles dropped tracker frames and
        brief flat segments — earlier rules required *consecutive* frames,
        which locked the gate closed whenever the tracker missed a frame.

        With STROKE_HISTORY=5 we have up to 4 deltas; STROKE_CONFIRM_FRAMES=2
        means we need 2 same-direction confirming deltas with strictly
        more in that direction than the opposite.
        """
        history = list(self._ball_y_history)
        if len(history) < 2:
            return None
        deltas = [history[i + 1] - history[i] for i in range(len(history) - 1)]
        down_votes = sum(1 for d in deltas if d >= self.STROKE_CONFIRM_DELTA_PX)
        up_votes = sum(1 for d in deltas if -d >= self.STROKE_CONFIRM_DELTA_PX)
        if down_votes >= self.STROKE_CONFIRM_FRAMES and down_votes > up_votes:
            return "down"
        if up_votes >= self.STROKE_CONFIRM_FRAMES and up_votes > down_votes:
            return "up"
        # Bootstrap: not enough confirming votes yet — fall back to the
        # loose total-window-delta rule (matches the recorder's heuristic).
        delta = history[-1] - history[0]
        if abs(delta) < self.STROKE_DELTA_PX:
            return None
        return "down" if delta > 0 else "up"
