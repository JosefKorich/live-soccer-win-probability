"""Core stochastic-modeling utilities for live soccer win probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from scipy.stats import skellam


@dataclass(frozen=True)
class OutcomeProbabilities:
    """Win, draw, and loss probabilities."""

    win: float
    draw: float
    loss: float

    def as_array(self) -> np.ndarray:
        return np.array([self.win, self.draw, self.loss], dtype=float)


def mle_goal_rate(
    total_goals: float,
    n_matches: int,
    match_minutes: float = 90.0,
) -> float:
    """Maximum-likelihood estimate of a homogeneous Poisson goal rate."""
    if n_matches <= 0 or match_minutes <= 0:
        raise ValueError("n_matches and match_minutes must be positive")
    if total_goals < 0:
        raise ValueError("total_goals cannot be negative")
    return float(total_goals) / (float(n_matches) * float(match_minutes))


def build_generator(
    lambda_team: float,
    lambda_opp: float,
    d_min: int = -8,
    d_max: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a birth-death CTMC generator for score differential D(t)."""
    if lambda_team < 0 or lambda_opp < 0:
        raise ValueError("Poisson intensities must be non-negative")
    if d_min >= d_max:
        raise ValueError("d_min must be smaller than d_max")

    states = np.arange(d_min, d_max + 1)
    q = np.zeros((len(states), len(states)), dtype=float)

    for i, d in enumerate(states):
        if d < d_max:
            q[i, i + 1] = lambda_team
        if d > d_min:
            q[i, i - 1] = lambda_opp
        q[i, i] = -q[i].sum()

    return q, states


def forward_euler(
    q: np.ndarray,
    p0: np.ndarray,
    horizon: float,
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate p'(t)=Q^T p(t) by forward Euler."""
    if horizon < 0 or dt <= 0:
        raise ValueError("horizon must be non-negative and dt must be positive")

    q = np.asarray(q, dtype=float)
    p = np.asarray(p0, dtype=float).copy()

    if q.shape[0] != q.shape[1] or q.shape[0] != p.size:
        raise ValueError("incompatible Q and p0 dimensions")
    if not np.isclose(p.sum(), 1.0):
        raise ValueError("p0 must sum to one")

    n_steps = int(round(horizon / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    trajectory = np.empty((n_steps + 1, p.size), dtype=float)
    trajectory[0] = p

    for step in range(n_steps):
        p = p + dt * (q.T @ p)
        # Guard against tiny floating-point excursions.
        p = np.clip(p, 0.0, None)
        p /= p.sum()
        trajectory[step + 1] = p

    return times, trajectory


def outcome_from_distribution(
    p: np.ndarray,
    states: np.ndarray,
) -> OutcomeProbabilities:
    """Collapse a score-difference distribution into W/D/L probabilities."""
    p = np.asarray(p, dtype=float)
    states = np.asarray(states)
    return OutcomeProbabilities(
        win=float(p[states > 0].sum()),
        draw=float(p[states == 0].sum()),
        loss=float(p[states < 0].sum()),
    )


def skellam_outcome(
    lambda_team: float,
    lambda_opp: float,
    horizon: float,
    current_diff: int = 0,
) -> OutcomeProbabilities:
    """Exact final W/D/L probabilities under homogeneous Poisson scoring.

    horizon is the number of minutes remaining in the match.
    """
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if lambda_team < 0 or lambda_opp < 0:
        raise ValueError("Poisson intensities must be non-negative")

    if horizon == 0:
        return OutcomeProbabilities(
            win=float(current_diff > 0),
            draw=float(current_diff == 0),
            loss=float(current_diff < 0),
        )

    mu_team = lambda_team * horizon
    mu_opp = lambda_opp * horizon

    # Final differential = current_diff + K, where K is Skellam(mu_team, mu_opp).
    draw_k = -current_diff
    loss = float(skellam.cdf(draw_k - 1, mu_team, mu_opp))
    draw = float(skellam.pmf(draw_k, mu_team, mu_opp))
    win = float(1.0 - loss - draw)

    return OutcomeProbabilities(win=win, draw=draw, loss=loss)


def ctmc_outcome(
    lambda_team: float,
    lambda_opp: float,
    horizon: float = 90.0,
    dt: float = 0.1,
    current_diff: int = 0,
    d_min: int = -8,
    d_max: int = 8,
) -> OutcomeProbabilities:
    """Numerical CTMC W/D/L probabilities from a score differential."""
    q, states = build_generator(
        lambda_team,
        lambda_opp,
        d_min=d_min,
        d_max=d_max,
    )

    if current_diff not in states:
        raise ValueError("current_diff must lie inside the truncated state space")

    p0 = np.zeros(len(states), dtype=float)
    p0[np.where(states == current_diff)[0][0]] = 1.0

    _, trajectory = forward_euler(q, p0, horizon=horizon, dt=dt)
    return outcome_from_distribution(trajectory[-1], states)


def outcome_curve(
    lambda_team: float,
    lambda_opp: float,
    horizon: int = 90,
    current_diff: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Conditional final W/D/L curve as match time elapses.

    The current score differential is held fixed while the remaining horizon
    shrinks. With current_diff=0, this answers: if the match is still tied at
    minute t, what are the model-implied final probabilities?
    """
    elapsed = np.arange(0, horizon + 1, dtype=float)
    probs = np.array(
        [
            skellam_outcome(
                lambda_team,
                lambda_opp,
                horizon=float(horizon - t),
                current_diff=current_diff,
            ).as_array()
            for t in elapsed
        ]
    )
    return elapsed, probs


def estimate_lambda_kernel(
    goal_times: Iterable[float],
    total_matches: int,
    horizon: float = 90.0,
    bandwidth: float = 3.0,
    n_grid: int = 91,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a smooth non-homogeneous Poisson intensity by Gaussian KDE."""
    if total_matches <= 0 or bandwidth <= 0:
        raise ValueError("total_matches and bandwidth must be positive")

    goal_times = np.asarray(list(goal_times), dtype=float)
    grid = np.linspace(0.0, horizon, n_grid)

    if goal_times.size == 0:
        return grid, np.zeros_like(grid)

    z = (grid[:, None] - goal_times[None, :]) / bandwidth
    kernels = np.exp(-0.5 * z**2) / (
        np.sqrt(2.0 * np.pi) * bandwidth
    )
    intensity = kernels.sum(axis=1) / total_matches
    return grid, intensity


def simulate_remaining_match(
    p_team: Iterable[float],
    p_opp: Iterable[float],
    n_sim: int = 20_000,
    current_diff: int = 0,
    seed: int | None = 42,
) -> OutcomeProbabilities:
    """Monte Carlo final-outcome simulation from per-minute scoring probabilities."""
    p_team = np.asarray(list(p_team), dtype=float)
    p_opp = np.asarray(list(p_opp), dtype=float)

    if p_team.shape != p_opp.shape:
        raise ValueError("p_team and p_opp must have the same length")
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")
    if np.any((p_team < 0) | (p_team > 1)) or np.any(
        (p_opp < 0) | (p_opp > 1)
    ):
        raise ValueError("scoring probabilities must lie in [0, 1]")

    rng = np.random.default_rng(seed)
    team_goals = (
        rng.random((n_sim, len(p_team))) < p_team
    ).sum(axis=1)
    opp_goals = (
        rng.random((n_sim, len(p_opp))) < p_opp
    ).sum(axis=1)
    final_diff = current_diff + team_goals - opp_goals

    return OutcomeProbabilities(
        win=float(np.mean(final_diff > 0)),
        draw=float(np.mean(final_diff == 0)),
        loss=float(np.mean(final_diff < 0)),
    )
