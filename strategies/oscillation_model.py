"""Sinusoidal model for the rim's oscillation.

Given a list of `(t, x, y)` samples (the rim's observed position over
time), fit a per-axis sine model so we can predict where the rim will
be at any future time. The simple linear-extrapolation in
RimMotionTracker.predict() is accurate over short horizons but breaks
down near the rim's turnaround points — exactly where we'd want to time
a throw if we're aiming for the swing extremes.

Model per axis:

    x(t) = cx + Ax * cos(2π * (t - t_ref) / T + φx)
    y(t) = cy + Ay * cos(2π * (t - t_ref) / T + φy)

Period `T` is shared (rim oscillates as one rigid body). Amplitudes and
phases are estimated independently per axis so a rim moving only on x,
or moving in a circle, fits naturally.

Estimation pipeline (called from `update`):
  1. Smooth each axis with a 3-sample moving average.
  2. Walk the smoothed series and collect indices where the first
     derivative changes sign — those are turnaround events (extrema).
  3. Period = mean spacing between *every other* extremum (peak→peak
     and trough→trough are the same period).
  4. Centers and amplitudes from observed min/max over the window.
  5. Phase per axis fit by minimizing sum-of-squares residual against
     the smoothed samples (closed-form using the observed value +
     velocity at t_ref to disambiguate the cosine quadrant).

We need at least 1.5 cycles of history (3 turnarounds across both
axes combined, or per axis) before declaring `ready()`. With a typical
in-game oscillation period and a 12 s history window that's well within
reach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class _AxisFit:
    center: float
    amplitude: float
    phase: float    # radians, evaluated at t_ref
    rms_error: float


# Minimum samples before we even try to fit. Below this the period
# estimate is too noisy.
_MIN_SAMPLES = 8

# Minimum amplitude (px) before we treat an axis as "actually
# oscillating". Below this we report center as the prediction and
# treat the axis as effectively static. Matches the rim-motion jitter
# threshold.
_MIN_AMPLITUDE_PX = 6.0

# Minimum number of turnarounds (extrema) we need on the *driving*
# axis (whichever has bigger amplitude) before period is trusted.
# 3 = 1.5 full cycles; 4 = 2 cycles. We use 3 to keep the warm-up short.
_MIN_TURNAROUNDS = 3


class RimOscillationModel:
    """Fits and evaluates a sinusoidal model of the rim's motion."""

    def __init__(self) -> None:
        self._t_ref: float | None = None
        self._period_s: float | None = None
        self._x_fit: _AxisFit | None = None
        self._y_fit: _AxisFit | None = None
        self._sample_count: int = 0
        # Number of extrema observed on the driving (higher-amplitude)
        # axis at the last update. We require at least one full swing
        # peak→trough→peak (3 extrema) before declaring ready, so the
        # observed amplitude reflects the rim's true bounds, not just
        # one half of the swing.
        self._driving_extrema_count: int = 0

    def update(self, samples: list[tuple[float, int, int]]) -> None:
        """Refresh fit from a list of (t, x, y) samples (oldest first)."""
        self._sample_count = len(samples)
        if len(samples) < _MIN_SAMPLES:
            self._invalidate()
            return

        ts = [s[0] for s in samples]
        xs = [float(s[1]) for s in samples]
        ys = [float(s[2]) for s in samples]

        xs_s = _smooth(xs)
        ys_s = _smooth(ys)

        x_extrema = _extrema_indices(xs_s)
        y_extrema = _extrema_indices(ys_s)

        x_amp = (max(xs_s) - min(xs_s)) / 2.0 if xs_s else 0.0
        y_amp = (max(ys_s) - min(ys_s)) / 2.0 if ys_s else 0.0

        # Use the higher-amplitude axis to estimate period — it has
        # the cleanest extrema.
        if x_amp >= y_amp:
            period = _period_from_extrema(ts, x_extrema)
            self._driving_extrema_count = len(x_extrema)
        else:
            period = _period_from_extrema(ts, y_extrema)
            self._driving_extrema_count = len(y_extrema)

        if period is None:
            self._invalidate()
            return

        self._t_ref = ts[-1]
        self._period_s = period
        self._x_fit = _fit_axis(ts, xs_s, period, self._t_ref)
        self._y_fit = _fit_axis(ts, ys_s, period, self._t_ref)

    def ready(self) -> bool:
        """True once we have a usable fit. `position_at(t)` should only
        be trusted when this returns True.

        Requires that we've observed at least one full peak→trough→peak
        on the driving axis (3 extrema) before declaring ready. Without
        that, the amplitude reflects only what the rim has done so far
        — possibly only half a swing — and we'd time throws against
        bounds the rim hasn't actually shown us yet.
        """
        return (
            self._period_s is not None
            and self._x_fit is not None
            and self._y_fit is not None
            and self._driving_extrema_count >= _MIN_TURNAROUNDS
        )

    def period_s(self) -> float | None:
        return self._period_s

    def position_at(self, t: float) -> tuple[int, int] | None:
        """Predict the rim's (x, y) at absolute perf_counter time `t`.
        Returns None when the model isn't ready."""
        if not self.ready():
            return None
        assert self._t_ref is not None
        assert self._period_s is not None
        assert self._x_fit is not None
        assert self._y_fit is not None
        x = _evaluate(self._x_fit, self._period_s, t - self._t_ref)
        y = _evaluate(self._y_fit, self._period_s, t - self._t_ref)
        return (int(round(x)), int(round(y)))

    def amplitude_px(self) -> tuple[float, float] | None:
        """(x_amp, y_amp) of the fitted model, or None if not ready."""
        if not self.ready():
            return None
        assert self._x_fit is not None
        assert self._y_fit is not None
        return (self._x_fit.amplitude, self._y_fit.amplitude)

    def diagnostics(self) -> str:
        """One-line description of the current fit, for logging."""
        if not self.ready():
            return f"not ready (samples={self._sample_count})"
        assert self._period_s is not None
        assert self._x_fit is not None
        assert self._y_fit is not None
        return (
            f"T={self._period_s:.2f}s "
            f"x: c={self._x_fit.center:.0f} A={self._x_fit.amplitude:.0f} "
            f"y: c={self._y_fit.center:.0f} A={self._y_fit.amplitude:.0f}"
        )

    def _invalidate(self) -> None:
        self._period_s = None
        self._x_fit = None
        self._y_fit = None
        self._driving_extrema_count = 0


def _smooth(values: list[float]) -> list[float]:
    """3-sample moving average — endpoints duplicated for length."""
    n = len(values)
    if n < 3:
        return list(values)
    out = [0.0] * n
    out[0] = values[0]
    out[-1] = values[-1]
    for i in range(1, n - 1):
        out[i] = (values[i - 1] + values[i] + values[i + 1]) / 3.0
    return out


def _extrema_indices(values: list[float]) -> list[int]:
    """Indices where the first derivative changes sign — i.e. local
    minima/maxima. We carry the *last non-zero* delta across plateaus
    so e.g. a sequence going up, going flat for a few samples (pixel
    quantization), then going down still registers as one extremum at
    the start of the flat region."""
    extrema: list[int] = []
    n = len(values)
    if n < 3:
        return extrema
    last_nonzero_delta = 0.0
    for i in range(1, n):
        delta = values[i] - values[i - 1]
        if delta == 0:
            continue
        if last_nonzero_delta != 0 and (last_nonzero_delta > 0) != (delta > 0):
            extrema.append(i - 1)
        last_nonzero_delta = delta
    return extrema


def _period_from_extrema(ts: list[float], extrema_idx: list[int]) -> float | None:
    """Period from spacing between same-direction extrema (every other).
    Two consecutive extrema (peak→trough) span half a period; peak→peak
    or trough→trough spans a full period."""
    if len(extrema_idx) < _MIN_TURNAROUNDS:
        return None
    times = [ts[i] for i in extrema_idx]
    full_cycle_gaps: list[float] = []
    for i in range(2, len(times)):
        gap = times[i] - times[i - 2]
        if gap > 0:
            full_cycle_gaps.append(gap)
    if full_cycle_gaps:
        return sum(full_cycle_gaps) / len(full_cycle_gaps)
    half_cycle_gaps = [
        times[i] - times[i - 1]
        for i in range(1, len(times))
        if times[i] - times[i - 1] > 0
    ]
    if not half_cycle_gaps:
        return None
    return 2.0 * (sum(half_cycle_gaps) / len(half_cycle_gaps))


def _fit_axis(
    ts: list[float],
    values: list[float],
    period: float,
    t_ref: float,
) -> _AxisFit:
    """Fit `c + A * cos(ω(t - t_ref) + φ)` to (ts, values) with ω known.

    Linear least squares in (a, b) where the model is rewritten as
        v(t) = c + a * cos(ω*Δt) + b * sin(ω*Δt)
    with Δt = t - t_ref, and `a = A*cos(φ)`, `b = -A*sin(φ)`. Then
    A = sqrt(a^2 + b^2) and φ = atan2(-b, a).
    """
    omega = 2.0 * math.pi / period
    n = len(values)
    c = sum(values) / n
    sum_cc = 0.0
    sum_ss = 0.0
    sum_cs = 0.0
    sum_vc = 0.0
    sum_vs = 0.0
    for t, v in zip(ts, values):
        dt = t - t_ref
        ct = math.cos(omega * dt)
        st = math.sin(omega * dt)
        vc = v - c
        sum_cc += ct * ct
        sum_ss += st * st
        sum_cs += ct * st
        sum_vc += vc * ct
        sum_vs += vc * st
    det = sum_cc * sum_ss - sum_cs * sum_cs
    if abs(det) < 1e-9:
        # Degenerate (all samples at one phase); fall back to a flat fit.
        return _AxisFit(center=c, amplitude=0.0, phase=0.0, rms_error=0.0)
    a = (sum_ss * sum_vc - sum_cs * sum_vs) / det
    b = (sum_cc * sum_vs - sum_cs * sum_vc) / det
    amplitude = math.sqrt(a * a + b * b)
    phase = math.atan2(-b, a)
    # RMS error against the original values for diagnostic / confidence.
    err_sq = 0.0
    for t, v in zip(ts, values):
        dt = t - t_ref
        pred = c + a * math.cos(omega * dt) + b * math.sin(omega * dt)
        err_sq += (v - pred) ** 2
    rms = math.sqrt(err_sq / n)
    return _AxisFit(center=c, amplitude=amplitude, phase=phase, rms_error=rms)


def _evaluate(fit: _AxisFit, period: float, dt: float) -> float:
    """Evaluate the fitted axis at relative time `dt = t - t_ref`."""
    if fit.amplitude < _MIN_AMPLITUDE_PX:
        return fit.center
    omega = 2.0 * math.pi / period
    return fit.center + fit.amplitude * math.cos(omega * dt + fit.phase)
