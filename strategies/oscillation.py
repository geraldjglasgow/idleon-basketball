"""Phase-aware moving-rim strategy.

When the rim is moving, this strategy fits a sinusoidal model of the
rim's oscillation (RimOscillationModel) and times each click so that —
after the ball's BALL_FLIGHT_S flight — the rim arrives at a position
that matches a known successful make.

Two phases of behavior:

  * Pre-motion phase (live score < MOVING_RIM_MIN_SCORE):
    Delegates to a SimpleRimStrategy instance. The rim is provably
    stationary at this score range, so the simple stationary-rim logic
    is the right tool — and reusing it avoids duplicating cooldown,
    stroke, and dy-match handling.

  * Oscillation phase (latched once we observe score >= 10 AND the
    motion tracker reports moving):
    Pure phase-locked timing. Wait until the model is `ready()`, then
    each frame:
      1. For every moving-rim make, solve for the soonest future time
         t* >= now where model.position_at(t* + BALL_FLIGHT_S) is
         within MAKE_MATCH_TOLERANCE_PX of (make.rim_x, make.rim_y).
      2. Pick the (make, t*) with smallest t* - now.
      3. Throw when (t* - now) <= ALIGNMENT_WINDOW_S AND the live ball
         dy matches the make's dy within DY_TOLERANCE_PX AND the live
         stroke matches the make's stroke.
    No force-throw timeout — the user prioritizes accuracy over
    throughput. We will sit and wait through full cycles until the
    model + dy + stroke all align.

The latch is sticky: once we're in oscillation mode, we never fall back
to stationary logic, even if `is_moving()` momentarily reads false (the
in-game rim never stops moving once it starts).
"""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

from basketball_tracker import BasketballSample
from rim_tracker import RimSample

from .base import Strategy
from .oscillation_model import RimOscillationModel
from .shared import (
    MissIndex,
    MOVING_RIM_MIN_SCORE,
    _Make,
    load_makes,
    load_misses,
    measure_ball_flight_s,
)
from .simple import SimpleRimStrategy


class OscillationStrategy(Strategy):
    DY_TOLERANCE_PX = 25
    # Matches SimpleRimStrategy's 2 s cooldown — see comment there.
    COOLDOWN_S = 2.0
    # Class default used when too few past makes exist to measure
    # empirically. __init__ replaces self.BALL_FLIGHT_S with the median
    # rim-crossing time from past makes when measurement is reliable.
    BALL_FLIGHT_S = 1.5

    # Need history long enough to span at least 1.5 oscillation cycles
    # for the model's period estimate to converge. 12 s comfortably
    # accommodates periods up to ~6 s.
    REQUIRED_HISTORY_WINDOW_S: float | None = 12.0

    # When searching for `t*` we sample the model at this resolution.
    # Smaller = more precise alignment, but more cosines per frame.
    # 25 ms is well below a typical rim period and ~1 frame at 30 fps.
    SEARCH_STEP_S = 0.025

    # How far ahead we'll look for an alignment opportunity. 1.5 periods
    # ensures we always find the soonest crossing if one exists.
    SEARCH_HORIZON_PERIODS = 1.5

    # A future rim_x counts as "matching" a known make's rim_x when
    # within this many pixels. Tighter than the simple strategy's
    # MAX_PREDICTED_RIM_DIST_PX because the model's x prediction is
    # accurate to a handful of pixels.
    MAKE_MATCH_TOLERANCE_PX = 25

    # Best-effort fallback tolerance — used when no make falls within
    # MAKE_MATCH_TOLERANCE_PX of any arrival point in the search horizon.
    # We pick the closest-miss (make, t*) within this larger window and
    # take the shot anyway, biasing the dy if the under/over streak
    # tells us which way to nudge. Better an imperfect shot that
    # generates trajectory data we can learn from than freezing
    # waiting for a perfect alignment that never comes.
    BEST_EFFORT_TOLERANCE_PX = 80

    # A make is considered relevant to the current rim only if its
    # recorded rim_y is within this tolerance of the live rim_y. Past
    # makes from a different level (different hoop height) have a very
    # different rim_y and would steer us toward release patterns that
    # don't apply here. The in-game rim oscillates only horizontally, so
    # a single level's rim_y is essentially constant — wide tolerance
    # would just admit cross-level noise.
    RIM_Y_MATCH_TOLERANCE_PX = 30

    # If oscillation mode can't find any alignment opportunity for this
    # long, defer per-frame decisions to the simple fallback. Two common
    # causes: this level has no moving-rim make data yet (we need to
    # generate some misses to learn from), or the rim_y is far enough
    # from any past make that y-filtering empties the candidate set.
    # The simple strategy's exploration + directional correction will
    # take shots and generate trajectory data we can mine next session.
    NO_ALIGNMENT_TIMEOUT_S = 30.0

    # Live ball-y window for stroke detection (must match recorder).
    # See SimpleRimStrategy for the rationale on the tighter
    # confirmation gate.
    STROKE_HISTORY = 5
    STROKE_DELTA_PX = 3
    STROKE_CONFIRM_FRAMES = 2
    STROKE_CONFIRM_DELTA_PX = 3

    # We only consider clicking when the model says t* is this close.
    # 1 frame at 30 fps ≈ 33 ms, so 50 ms gives the dy/stroke gates one
    # frame of slop on either side.
    ALIGNMENT_WINDOW_S = 0.05

    # Throttled "why I'm not throwing" log cadence — same idea as the
    # simple strategy, just one constant.
    WAIT_LOG_INTERVAL_S = 5.0

    # Latch hardening: number of consecutive frames where (score >=
    # MOVING_RIM_MIN_SCORE AND rim is moving) before we trust the
    # transition into oscillation mode. Defends against a single OCR
    # misread (e.g. "2" -> "24") flipping us permanently into a mode
    # the rim doesn't actually support.
    LATCH_CONFIRM_FRAMES = 5

    # After the latch flips, hold off on any throw until we've collected
    # at least this multiple of one rim period worth of motion history.
    # The rim only starts moving at score 10, so at latch time we have
    # ~0s of post-motion history; even if the model technically marks
    # itself ready, its amplitude estimate is still firming up. Waiting
    # 1.5 cycles ensures we've directly observed both extremes of the
    # swing before timing a throw against them.
    POST_LATCH_PERIODS = 1.5

    # If the ball fails to launch (click landed in a phase where the
    # game gives the ball no horizontal velocity) this many times in a
    # row, defer the next throw by NO_LAUNCH_BACKOFF_PERIODS rim periods
    # so the next click lands at a different phase of the swing. Two is
    # tight enough to react before we burn a life on a third dud.
    NO_LAUNCH_BACKOFF_THRESHOLD = 2
    NO_LAUNCH_BACKOFF_PERIODS = 1.0

    # A make is quarantined for the rest of the session after this many
    # consecutive non-make outcomes attributed to it. Set to 1 because
    # the rim's oscillation is deterministic — if make X produced a miss
    # this period, the rim returns to X's recorded position next period
    # and we'd reproduce the same miss. One failure is enough signal to
    # stop trusting that make and rotate to a different alignment.
    MAX_CONSECUTIVE_FAILURES_PER_MAKE = 1

    def __init__(self, throws_log_path: Path | str) -> None:
        path = Path(throws_log_path)
        self._fallback = SimpleRimStrategy(path)
        # Use the same empirical flight time the fallback already measured.
        # The fallback's BALL_FLIGHT_S is the data-driven median from
        # past makes; sharing it keeps both code paths consistent.
        self.BALL_FLIGHT_S = self._fallback.BALL_FLIGHT_S
        self._all_makes: list[_Make] = load_makes(path)
        # Only moving-rim makes are useful in oscillation mode. Stationary
        # makes might still be useful if the rim happens to swing through
        # their position, but their `dy` was recorded against a static
        # rim — for now, restrict.
        self._moving_makes: list[_Make] = [
            m for m in self._all_makes if m.rim_moving is True
        ]
        # Historical misses index for negative learning — same as the
        # fallback strategy uses, but consulted from oscillation's own
        # alignment search.
        self._misses_index = MissIndex(load_misses(path))
        self._oscillation = RimOscillationModel()

        self.last_throw_at: float = time.perf_counter()
        self._ball_y_history: deque[int] = deque(maxlen=self.STROKE_HISTORY)
        self._last_score: int | None = None
        self._in_oscillation_mode: bool = False
        # Counter of consecutive frames satisfying the latch condition.
        # Reset whenever the condition fails. Crosses LATCH_CONFIRM_FRAMES
        # to actually flip the latch.
        self._latch_confirm_count: int = 0
        # Wallclock when the latch flipped — used by POST_LATCH_PERIODS
        # to gate throws until we've observed enough of the rim's swing
        # to know its bounds.
        self._latched_at: float | None = None
        # When the plausibility gate rejects N consecutive identical
        # readings, eventually accept — that reading is most likely real
        # (game restart, big skip, OCR persistently right rather than
        # persistently wrong). Without this, a misread would freeze our
        # score state forever even if reality matched it.
        self._rejected_score_value: int | None = None
        self._rejected_score_count: int = 0
        # Wallclock of the last frame where _find_soonest_alignment
        # succeeded. If too long passes, oscillation mode bypasses to
        # simple — see NO_ALIGNMENT_TIMEOUT_S.
        self._last_alignment_at: float | None = None

        # Diagnostic streaks for adaptive dy bias (mirrors SimpleRimStrategy).
        self._undershoot_streak: int = 0
        self._overshoot_streak: int = 0
        # The dy the bot actually clicked at on its most recent throw.
        # When a streak is active, the correction anchors off this
        # value rather than the rotated make's recorded dy — rotation
        # alone can swing the base by hundreds of px and override the
        # streak's intended shift. Cleared on make / game-reset.
        self._last_click_dy: int | None = None
        # Per-make quarantine state. The make whose recorded rim_x we
        # aligned to for the most recent click — set inside
        # `_should_throw_oscillating` when it returns True, consumed by
        # `notify_outcome` to credit/blame that make. Failure counts and
        # quarantines persist across games (a bad make stays bad).
        self._last_used_make: _Make | None = None
        self._make_failure_counts: dict[int, int] = {}
        self._quarantined_makes: set[int] = set()
        # When the ball repeatedly fails to launch (click landed in a
        # dead phase of the swing), backing off a full period before the
        # next attempt shifts the click into a different phase. Without
        # this we can repeatedly click at the same bad alignment cycle
        # after cycle, hemorrhaging lives.
        self._no_launch_streak: int = 0
        self._defer_until: float = 0.0

        # Throttling state.
        self._last_wait_log_at: float = 0.0
        self._last_wait_reason: str = ""

        print(
            f"OscillationStrategy: loaded {len(self._all_makes)} make(s) "
            f"({len(self._moving_makes)} on a moving rim), "
            f"{self._misses_index.total} miss(es) for negative learning "
            f"from {path}; "
            f"BALL_FLIGHT_S={self.BALL_FLIGHT_S:.2f}s"
        )

    # --- Strategy protocol ---------------------------------------------------

    def should_throw(
        self,
        ball: BasketballSample | None,
        rim: RimSample | None,
        rim_motion=None,
    ) -> bool:
        # Latch into oscillation mode once we've observed motion at or
        # above the threshold score for LATCH_CONFIRM_FRAMES in a row.
        # Once latched, never unlatched (the in-game rim never stops).
        if not self._in_oscillation_mode:
            if self._should_enter_oscillation_mode(rim_motion):
                self._in_oscillation_mode = True
                self._latched_at = time.perf_counter()
                self._last_alignment_at = time.perf_counter()
                print(
                    f"[strategy] entering oscillation mode "
                    f"(score={self._last_score}, rim moving observed for "
                    f"{self.LATCH_CONFIRM_FRAMES}+ frames)"
                )

        if not self._in_oscillation_mode:
            # Pre-motion phase: defer to the stationary strategy.
            return self._fallback.should_throw(ball, rim, rim_motion)

        return self._should_throw_oscillating(ball, rim, rim_motion)

    def mark_thrown(self) -> None:
        self.last_throw_at = time.perf_counter()
        if self._in_oscillation_mode:
            # Keep the fallback's clock in sync so if we ever did fall
            # back, its cooldown wouldn't accidentally fire again.
            self._fallback.last_throw_at = self.last_throw_at
        else:
            self._fallback.mark_thrown()

    def notify_outcome(self, outcome: str) -> None:
        # Always pass through to fallback so its directional streaks stay
        # current — they're useful if we ever revert. The fallback has
        # its own quarantine state for shots taken through its
        # should_throw path (score < 10 or oscillation's no-alignment
        # bypass); we maintain a separate quarantine here for shots
        # picked by oscillation mode itself.
        self._fallback.notify_outcome(outcome)
        if outcome == "make":
            self._undershoot_streak = 0
            self._overshoot_streak = 0
            self._no_launch_streak = 0
            # Rim teleports after a make — last click's dy no longer
            # describes a relevant launch position.
            self._last_click_dy = None
        elif outcome == "undershoot":
            self._undershoot_streak += 1
            self._overshoot_streak = 0
            self._no_launch_streak = 0
        elif outcome == "overshoot":
            self._overshoot_streak += 1
            self._undershoot_streak = 0
            self._no_launch_streak = 0
        elif outcome == "no_launch":
            self._no_launch_streak += 1
            # Don't churn the dy streaks — dy was probably fine, the
            # click timing within the swing was wrong. After
            # NO_LAUNCH_BACKOFF_THRESHOLD consecutive no-launches, defer
            # the next throw by a full rim period to break out of the
            # bad phase.
            if self._no_launch_streak >= self.NO_LAUNCH_BACKOFF_THRESHOLD:
                period = self._oscillation.period_s()
                if period is not None:
                    self._defer_until = (
                        time.perf_counter() + period * self.NO_LAUNCH_BACKOFF_PERIODS
                    )
                    print(
                        f"[strategy] {self._no_launch_streak} consecutive no-launch "
                        f"throws — deferring next attempt by "
                        f"{period * self.NO_LAUNCH_BACKOFF_PERIODS:.1f}s "
                        f"({self.NO_LAUNCH_BACKOFF_PERIODS:.1f} period(s)) "
                        f"to land on a different phase"
                    )
        # unknown: don't bias the dy streaks.

        self._record_outcome_for_make(outcome)

    def _record_outcome_for_make(self, outcome: str) -> None:
        """Update the per-make failure counter for the make that produced
        the most recent oscillation-mode throw, and quarantine it once
        the count hits MAX_CONSECUTIVE_FAILURES_PER_MAKE. A make outcome
        clears the counter so a genuinely-good make stays usable."""
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
        count = self._make_failure_counts.get(key, 0) + 1
        self._make_failure_counts[key] = count
        if count >= self.MAX_CONSECUTIVE_FAILURES_PER_MAKE:
            self._quarantined_makes.add(key)
            print(
                f"[strategy] QUARANTINING make at ({m.rim_x}, {m.rim_y}) "
                f"after {count} consecutive failures (last: {outcome}); "
                f"won't be picked again this session"
            )

    # If the plausibility gate rejects the same value this many times in
    # a row, accept it on the next read. That value is too persistent to
    # be a one-off misread — it's either a real game state we missed
    # (restart, big skip we couldn't observe) or the OCR is stuck on a
    # consistent error. Either way, freezing our score forever is worse
    # than re-syncing once.
    REJECT_OVERRIDE_THRESHOLD = 5

    def notify_game_reset(self) -> None:
        """Drop all per-game state. The next game starts at score 0
        with a stationary rim, so the latch, streaks, score memory, and
        oscillation-model fit from the previous game must all clear —
        otherwise we sit forever in oscillation mode logging "learning
        rim oscillation" against a rim that isn't moving yet."""
        was_latched = self._in_oscillation_mode
        self._in_oscillation_mode = False
        self._latched_at = None
        self._latch_confirm_count = 0
        self._last_score = None
        self._rejected_score_value = None
        self._rejected_score_count = 0
        self._undershoot_streak = 0
        self._overshoot_streak = 0
        self._no_launch_streak = 0
        self._defer_until = 0.0
        self._last_alignment_at = None
        self._oscillation = RimOscillationModel()
        self._last_wait_reason = ""
        self._last_wait_log_at = 0.0
        # Drop the in-flight make pointer; quarantines + failure counts
        # persist (a make that consistently misses is bad data, and a
        # new game doesn't change that).
        self._last_used_make = None
        self._last_click_dy = None
        # Forward to fallback so its score-memory + streaks reset too —
        # otherwise the simple strategy's _last_observed_score from the
        # previous game would mis-evaluate the new game's progression.
        self._fallback.notify_game_reset()
        if was_latched:
            print(
                "[strategy] game reset — clearing oscillation latch and "
                "per-game state"
            )

    def notify_score(self, score: int | None) -> None:
        self._fallback.notify_score(score)
        if score is None:
            return
        # Match the recorder's SCORE_MAX_INCREASE: a real game can jump
        # by up to 5 points between live notify_score calls, especially
        # when consecutive made throws resolve in the same window or a
        # prior throw was dropped.
        is_implausible_jump = (
            self._last_score is not None and score - self._last_score > 5
        )
        is_implausible_drop = (
            self._last_score is not None
            and score < self._last_score
            and self._last_score - score < 3
        )
        if is_implausible_jump or is_implausible_drop:
            # If the same "implausible" value keeps coming back, accept it
            # after REJECT_OVERRIDE_THRESHOLD repeats — better to re-sync
            # to a possibly-wrong score than to ignore reality forever.
            if score == self._rejected_score_value:
                self._rejected_score_count += 1
            else:
                self._rejected_score_value = score
                self._rejected_score_count = 1
            if self._rejected_score_count >= self.REJECT_OVERRIDE_THRESHOLD:
                print(
                    f"[strategy] accepting persistent score {score} "
                    f"after {self._rejected_score_count} consistent reads "
                    f"(was {self._last_score}); re-syncing"
                )
                self._last_score = score
                self._rejected_score_value = None
                self._rejected_score_count = 0
                return
            kind = "jump" if is_implausible_jump else "drop"
            print(
                f"[strategy] ignoring implausible score {kind} "
                f"{self._last_score} -> {score} (likely OCR misread; "
                f"#{self._rejected_score_count}/{self.REJECT_OVERRIDE_THRESHOLD})"
            )
            return
        # Plausible reading — clear any lingering rejection streak and
        # update last_score normally.
        self._rejected_score_value = None
        self._rejected_score_count = 0
        self._last_score = score

    # --- Oscillation-mode logic ---------------------------------------------

    def _should_enter_oscillation_mode(self, rim_motion) -> bool:
        """Track consecutive frames meeting the latch condition; only
        return True once we've seen LATCH_CONFIRM_FRAMES of them in a
        row. A single fluke (OCR misread, transient `is_moving` true)
        won't latch us into a mode the rim can't actually support."""
        if (
            rim_motion is None
            or self._last_score is None
            or self._last_score < MOVING_RIM_MIN_SCORE
            or not rim_motion.is_moving()
        ):
            self._latch_confirm_count = 0
            return False
        self._latch_confirm_count += 1
        return self._latch_confirm_count >= self.LATCH_CONFIRM_FRAMES

    def _should_throw_oscillating(
        self,
        ball: BasketballSample | None,
        rim: RimSample | None,
        rim_motion,
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

        # No-launch backoff: keep clicks out of the swing phase that
        # produced the recent dud(s) until enough time has passed to
        # land on a different phase.
        defer_remaining = self._defer_until - time.perf_counter()
        if defer_remaining > 0:
            return self._waiting(
                f"no-launch backoff ({defer_remaining:.1f}s left, "
                f"streak={self._no_launch_streak})"
            )

        if rim_motion is None:
            return self._waiting("no rim_motion tracker provided")

        # Refresh the oscillation fit from the tracker's history.
        self._oscillation.update(rim_motion.samples())
        if not self._oscillation.ready():
            # Wait — never fall back to a stationary-rim strategy here.
            # Throwing without a phase-locked model on a moving rim costs
            # lives; accuracy beats throughput.
            return self._waiting(
                f"learning rim oscillation ({self._oscillation.diagnostics()})"
            )

        # Post-latch warm-up: even after ready(), insist on POST_LATCH_PERIODS
        # of observed motion before throwing, so the amplitude estimate
        # reflects the rim's true bounds rather than whatever fragment of
        # a swing was captured immediately after the rim started moving.
        if self._latched_at is not None:
            since_latch = time.perf_counter() - self._latched_at
            period = self._oscillation.period_s()
            if period is not None:
                required = period * self.POST_LATCH_PERIODS
                if since_latch < required:
                    return self._waiting(
                        f"post-latch warm-up: {since_latch:.1f}s of "
                        f"{required:.1f}s "
                        f"({self.POST_LATCH_PERIODS:.1f}× period={period:.2f}s) "
                        f"observed; waiting to confirm rim bounds"
                    )

        if not self._moving_makes:
            return self._waiting(
                "no moving-rim makes in throws.jsonl — need data first"
            )

        live_stroke = self._stroke()
        if live_stroke is None:
            return self._waiting("stroke not yet classified (need more motion)")

        # First filter: makes whose recorded rim_y matches the live rim's
        # y. This selects "makes from the same in-game level" — different
        # levels have different hoop heights, and a make from a different
        # level's release pattern is irrelevant here. Without this filter
        # we'd sit waiting for the rim_x to align with a make whose
        # rim_y is hundreds of pixels away.
        live_rim_y = rim.center[1]
        same_level_makes = [
            m for m in self._moving_makes
            if abs(m.rim_y - live_rim_y) <= self.RIM_Y_MATCH_TOLERANCE_PX
        ]
        if not same_level_makes:
            # No moving-rim make data for this level. The simple strategy
            # will take exploratory shots that generate trajectory data
            # we can mine next session. Bump the alignment timer so the
            # NO_ALIGNMENT_TIMEOUT_S log pathway also fires if this state
            # persists.
            self._waiting(
                f"no moving-rim makes near rim_y={live_rim_y} "
                f"(±{self.RIM_Y_MATCH_TOLERANCE_PX}px) — using simple fallback"
            )
            return self._fallback.should_throw(ball, rim, rim_motion)

        # Then filter by stroke compatibility (legacy stroke=None makes
        # act as wildcards, like the simple strategy).
        candidates = [
            m for m in same_level_makes
            if m.stroke is None or m.stroke == live_stroke
        ]
        if not candidates:
            candidates = same_level_makes

        # Drop quarantined makes — each missed once already this session,
        # and the rim's deterministic oscillation means we'd reproduce
        # those misses if we picked them again. Fall back to the
        # un-filtered set only if quarantining empties everything.
        non_quarantined = [
            m for m in candidates if id(m) not in self._quarantined_makes
        ]
        if non_quarantined:
            candidates = non_quarantined

        # Find the soonest t* where the rim's x at arrival matches a
        # candidate make's recorded rim_x. y is already accounted for by
        # the same-level filter above, and the in-game rim doesn't move
        # vertically — so we only need to time the horizontal phase.
        best = self._find_soonest_alignment(candidates)
        if best is None:
            # If we've gone too long without finding an alignment, even
            # within the same-level candidate set, the simple fallback
            # gets this frame's decision so we don't sit forever.
            stuck = (
                time.perf_counter() - self._last_alignment_at
                if self._last_alignment_at is not None
                else 0.0
            )
            if stuck > self.NO_ALIGNMENT_TIMEOUT_S:
                self._waiting(
                    f"no alignment in {stuck:.0f}s "
                    f"(>= {self.NO_ALIGNMENT_TIMEOUT_S:.0f}s) — using simple fallback"
                )
                return self._fallback.should_throw(ball, rim, rim_motion)
            return self._waiting(
                f"no make alignment within next "
                f"{self.SEARCH_HORIZON_PERIODS:.1f} cycles "
                f"[{self._oscillation.diagnostics()}]"
            )
        # We found an alignment — reset the no-alignment timer.
        self._last_alignment_at = time.perf_counter()
        target_make, t_star, predicted, miss_px, exact = best
        now = time.perf_counter()
        time_until = t_star - now

        if not exact:
            # Best-effort match: log it once per windowed waiting throttle
            # so we know when the strategy is reaching beyond perfect
            # alignment. Throttler dedup'd by reason key so we won't spam.
            self._waiting(
                f"best-effort alignment: closest make rim_x={target_make.rim_x} "
                f"is {miss_px:.0f}px from rim's nearest approach; "
                f"shooting anyway (tol={self.BEST_EFFORT_TOLERANCE_PX})"
            )

        if time_until > self.ALIGNMENT_WINDOW_S:
            return self._waiting(
                f"waiting {time_until*1000:.0f}ms for rim to align with make "
                f"({target_make.rim_x}, {target_make.rim_y}); "
                f"predicted arrival rim={predicted} miss={miss_px:.0f}px"
            )

        # We're inside the alignment window — last gate is dy + stroke.
        target_dy = self._adjusted_target_dy(target_make.dy)
        rx, ry = predicted
        bx, by = ball.center
        cur_dy = by - ry
        delta = abs(cur_dy - target_dy)
        if delta > self.DY_TOLERANCE_PX:
            return self._waiting(
                f"dy match: ball_dy={cur_dy} target={target_dy} "
                f"(off by {delta}, tol={self.DY_TOLERANCE_PX}, "
                f"stroke={live_stroke})"
            )
        # Commit — remember which make's pattern produced this throw so
        # notify_outcome can credit the right one. Record the click dy
        # too so the next streak correction anchors against it.
        self._last_used_make = target_make
        self._last_click_dy = cur_dy
        return True

    def _find_soonest_alignment(
        self, candidates: list[_Make]
    ) -> tuple[_Make, float, tuple[int, int], float, bool] | None:
        """Return (make, t*, predicted_rim_at_t*+flight, miss_px, exact)
        for the soonest future moment we can release.

        Match criterion is rim_x only — candidates have already been
        filtered by rim_y elsewhere (same-level constraint), and the
        in-game rim oscillates only horizontally so y is essentially
        invariant during the swing.

        Two passes through the search horizon:
          1. Look for a perfect match (rim_x within MAKE_MATCH_TOLERANCE_PX
             of a candidate). First one found wins (time-ordered walk).
             Returns (..., miss_px, exact=True).
          2. If no perfect match, return the closest miss within
             BEST_EFFORT_TOLERANCE_PX. Returns (..., miss_px, exact=False).
          3. If even the closest miss is worse than that, return None.
        """
        period = self._oscillation.period_s()
        if period is None:
            return None
        now = time.perf_counter()
        horizon_s = period * self.SEARCH_HORIZON_PERIODS
        steps = max(1, int(math.ceil(horizon_s / self.SEARCH_STEP_S)))
        # Pre-sort candidates by historical miss density at their
        # signature (fewer misses near = checked earlier). Within a
        # given time step, multiple candidates may be within
        # MAKE_MATCH_TOLERANCE_PX — by ordering ascending on miss count,
        # the first perfect match found is the one with the fewest
        # historical failures at its signature.
        candidates = sorted(
            candidates,
            key=lambda m: self._misses_index.count_near(m.rim_x, m.rim_y, m.dy),
        )
        # Track the smallest |x_arrival - make.rim_x| across all
        # (k, make) pairs as a fallback. We can't return early during
        # the perfect-match pass without giving up on best-effort.
        best_perfect: tuple[_Make, float, tuple[int, int], float] | None = None
        best_miss: tuple[_Make, float, tuple[int, int], float] | None = None
        for k in range(steps + 1):
            t_star = now + k * self.SEARCH_STEP_S
            arrival = self._oscillation.position_at(t_star + self.BALL_FLIGHT_S)
            if arrival is None:
                return None
            ax, _ay = arrival
            for m in candidates:
                d = abs(m.rim_x - ax)
                if d <= self.MAKE_MATCH_TOLERANCE_PX:
                    # First perfect match wins — earlier is sooner.
                    if best_perfect is None:
                        best_perfect = (m, t_star, arrival, d)
                if best_miss is None or d < best_miss[3]:
                    best_miss = (m, t_star, arrival, d)
            if best_perfect is not None:
                m, t_s, arr, d = best_perfect
                return (m, t_s, arr, d, True)
        if best_miss is not None and best_miss[3] <= self.BEST_EFFORT_TOLERANCE_PX:
            m, t_s, arr, d = best_miss
            return (m, t_s, arr, d, False)
        return None

    # Match SimpleRimStrategy's directional-correction tuning. Step
    # bumped to 80 (from 50) so a single miss produces a visibly
    # different launch position on the next throw.
    DIRECTIONAL_CORRECTION_STEP_PX = 80
    DIRECTIONAL_CORRECTION_MAX_PX = 200

    def _adjusted_target_dy(self, base_dy: int) -> int:
        """Apply directional correction (under/overshoot streaks) to the
        target dy. When a streak is active and we have a previous
        click's dy on record, anchor the correction to that — otherwise
        rotation through different makes can override the streak shift
        and even reverse direction. Cold-start (no last click) falls
        back to base_dy. Result is clamped to ±DIRECTIONAL_CORRECTION_MAX_PX
        of `base_dy` so the walked anchor can't drift into physically
        unreachable values where the bot stalls waiting for the ball."""
        if self._undershoot_streak > 0 and self._last_click_dy is not None:
            target = self._last_click_dy - self.DIRECTIONAL_CORRECTION_STEP_PX
        elif self._overshoot_streak > 0 and self._last_click_dy is not None:
            target = self._last_click_dy + self.DIRECTIONAL_CORRECTION_STEP_PX
        elif self._undershoot_streak > 0:
            target = base_dy - min(
                self.DIRECTIONAL_CORRECTION_STEP_PX * self._undershoot_streak,
                self.DIRECTIONAL_CORRECTION_MAX_PX,
            )
        elif self._overshoot_streak > 0:
            target = base_dy + min(
                self.DIRECTIONAL_CORRECTION_STEP_PX * self._overshoot_streak,
                self.DIRECTIONAL_CORRECTION_MAX_PX,
            )
        else:
            return base_dy
        return max(
            base_dy - self.DIRECTIONAL_CORRECTION_MAX_PX,
            min(base_dy + self.DIRECTIONAL_CORRECTION_MAX_PX, target),
        )

    def _stroke(self) -> str | None:
        history = list(self._ball_y_history)
        if len(history) < 2:
            return None
        if len(history) > self.STROKE_CONFIRM_FRAMES:
            recent = history[-self.STROKE_CONFIRM_FRAMES - 1 :]
            deltas = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
            if all(d >= self.STROKE_CONFIRM_DELTA_PX for d in deltas):
                return "down"
            if all(-d >= self.STROKE_CONFIRM_DELTA_PX for d in deltas):
                return "up"
            return None
        delta = history[-1] - history[0]
        if abs(delta) < self.STROKE_DELTA_PX:
            return None
        return "down" if delta > 0 else "up"

    def _waiting(self, reason: str) -> bool:
        key = reason.split(":", 1)[0].split()[0] if reason else ""
        now = time.perf_counter()
        if (
            key != self._last_wait_reason
            or now - self._last_wait_log_at >= self.WAIT_LOG_INTERVAL_S
        ):
            print(f"[strategy] waiting: {reason}")
            self._last_wait_log_at = now
            self._last_wait_reason = key
        return False
