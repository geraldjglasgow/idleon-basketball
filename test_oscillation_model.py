"""Unit tests for RimOscillationModel.

Run: venv/Scripts/python -m pytest test_oscillation_model.py -v
(or: venv/Scripts/python test_oscillation_model.py for the cheap path).
"""

from __future__ import annotations

import math

from strategies.oscillation_model import RimOscillationModel


def _synthetic_samples(
    *,
    period_s: float,
    cx: float,
    ax: float,
    phix: float,
    cy: float,
    ay: float,
    phiy: float,
    duration_s: float,
    fps: float = 30.0,
    t0: float = 1000.0,
) -> list[tuple[float, int, int]]:
    omega = 2.0 * math.pi / period_s
    n = int(duration_s * fps)
    out: list[tuple[float, int, int]] = []
    for i in range(n):
        t = t0 + i / fps
        x = cx + ax * math.cos(omega * (t - t0) + phix)
        y = cy + ay * math.cos(omega * (t - t0) + phiy)
        out.append((t, int(round(x)), int(round(y))))
    return out


def test_period_within_5_pct_for_clean_sinusoid() -> None:
    samples = _synthetic_samples(
        period_s=4.0, cx=1200, ax=80, phix=0.0,
        cy=600, ay=20, phiy=math.pi / 2,
        duration_s=12.0,
    )
    m = RimOscillationModel()
    m.update(samples)
    assert m.ready(), m.diagnostics()
    p = m.period_s()
    assert p is not None
    assert abs(p - 4.0) / 4.0 < 0.05, f"period={p}, expected ~4.0"


def test_position_at_known_time_within_3px() -> None:
    samples = _synthetic_samples(
        period_s=5.0, cx=1000, ax=120, phix=1.2,
        cy=550, ay=40, phiy=-0.4,
        duration_s=15.0,
    )
    m = RimOscillationModel()
    m.update(samples)
    assert m.ready(), m.diagnostics()
    last_t = samples[-1][0]
    omega = 2.0 * math.pi / 5.0
    for ahead_s in (0.5, 1.0, 1.5, 2.5):
        target_t = last_t + ahead_s
        true_x = 1000 + 120 * math.cos(omega * (target_t - samples[0][0]) + 1.2)
        true_y = 550 + 40 * math.cos(omega * (target_t - samples[0][0]) - 0.4)
        pred = m.position_at(target_t)
        assert pred is not None
        dx = pred[0] - true_x
        dy = pred[1] - true_y
        assert abs(dx) < 3 and abs(dy) < 3, (
            f"ahead={ahead_s}s: pred={pred} expected~=({true_x:.0f}, {true_y:.0f})"
        )


def test_not_ready_with_too_few_samples() -> None:
    samples = _synthetic_samples(
        period_s=4.0, cx=1200, ax=80, phix=0.0,
        cy=600, ay=20, phiy=0.0,
        duration_s=0.2,  # ~6 samples at 30 fps
    )
    m = RimOscillationModel()
    m.update(samples)
    assert not m.ready()
    assert m.position_at(samples[-1][0] + 1.0) is None


def test_not_ready_for_stationary_input() -> None:
    # All samples at the same point — no extrema, no period.
    t0 = 1000.0
    samples = [(t0 + i / 30.0, 1200, 600) for i in range(int(12 * 30))]
    m = RimOscillationModel()
    m.update(samples)
    assert not m.ready()


def test_static_axis_returns_center_for_low_amplitude() -> None:
    # x oscillates strongly, y barely moves. Predicted y should snap to
    # center (no sub-jitter sinusoid extrapolated).
    samples = _synthetic_samples(
        period_s=3.0, cx=1000, ax=100, phix=0.0,
        cy=600, ay=2, phiy=0.0,  # below _MIN_AMPLITUDE_PX
        duration_s=10.0,
    )
    m = RimOscillationModel()
    m.update(samples)
    assert m.ready()
    last_t = samples[-1][0]
    pred = m.position_at(last_t + 1.5)
    assert pred is not None
    assert abs(pred[1] - 600) <= 1, f"y predicted as {pred[1]} (should be ~600)"


def _run_all() -> int:
    tests = [
        test_period_within_5_pct_for_clean_sinusoid,
        test_position_at_known_time_within_3px,
        test_not_ready_with_too_few_samples,
        test_not_ready_for_stationary_input,
        test_static_axis_returns_center_for_low_amplitude,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"ERROR {fn.__name__}: {exc!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
