import numpy as np

from src.soccer_winprob import (
    build_generator,
    ctmc_outcome,
    outcome_curve,
    simulate_remaining_match,
    skellam_outcome,
)


def test_generator_rows_sum_to_zero():
    q, _ = build_generator(0.016547, 0.008961)
    assert np.allclose(q.sum(axis=1), 0.0)


def test_outcome_probabilities_sum_to_one():
    p = skellam_outcome(0.016547, 0.008961, 90.0)
    assert np.isclose(p.win + p.draw + p.loss, 1.0)


def test_ctmc_matches_skellam_reference():
    exact = skellam_outcome(0.016547, 0.008961, 90.0)
    numerical = ctmc_outcome(
        0.016547,
        0.008961,
        horizon=90.0,
        dt=0.05,
        d_min=-10,
        d_max=10,
    )
    assert np.allclose(
        numerical.as_array(),
        exact.as_array(),
        atol=2e-3,
    )


def test_monte_carlo_is_reproducible_and_normalized():
    p = np.repeat(0.015, 30)
    out = simulate_remaining_match(p, p, n_sim=20_000, seed=7)
    assert np.isclose(out.win + out.draw + out.loss, 1.0)


def test_zero_remaining_time_is_deterministic():
    assert np.allclose(
        skellam_outcome(0.02, 0.01, 0.0, current_diff=0).as_array(),
        [0, 1, 0],
    )
    assert np.allclose(
        skellam_outcome(0.02, 0.01, 0.0, current_diff=1).as_array(),
        [1, 0, 0],
    )
    assert np.allclose(
        skellam_outcome(0.02, 0.01, 0.0, current_diff=-1).as_array(),
        [0, 0, 1],
    )


def test_live_curve_ends_at_current_state():
    elapsed, probs = outcome_curve(
        0.02,
        0.01,
        horizon=90,
        current_diff=0,
    )
    assert elapsed[0] == 0 and elapsed[-1] == 90
    assert np.allclose(probs[-1], [0, 1, 0])
    assert np.allclose(probs.sum(axis=1), 1.0)
