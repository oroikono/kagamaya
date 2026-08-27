"""Nested-grid outcome producer for the temporal-quadrature bias contrast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..maintrack_suite_v41.numerics import integrate_etdrk4
from ..maintrack_suite_v41.simulator import Family, HiddenTruth, TRUE_SUPPORTS, Trajectory
from ..unified_agent_v2_1_scaled.simulator import BASELINE, Channel, _initial, _noise, _problem, simulate


@dataclass(frozen=True)
class TemporalCell:
    nodes: int
    subsample: int


CELLS = (
    TemporalCell(7, 8),
    TemporalCell(13, 4),
    TemporalCell(25, 2),
    TemporalCell(49, 1),
)
TEMPORAL_MODES = (2, 5)
ACTION_COST = 1.0
FINE_CHANNEL = Channel("gradient_space", 96, 0.12, 0.0025, ACTION_COST)


def finest_noisy_trajectory(
    family: Family,
    episode_seed: int,
    public_seed: int,
    noise: float,
    noise_model: str,
) -> tuple[Trajectory, HiddenTruth]:
    """Generate one 49-node trajectory and one latent fine-grid noise draw."""

    internal_dt = 0.00025
    steps = int(round(FINE_CHANNEL.t_final / internal_dt))
    stride = int(round(FINE_CHANNEL.save_dt / internal_dt))
    x = np.arange(FINE_CHANNEL.nx) * (2.0 * np.pi / FINE_CHANNEL.nx)
    k = 2.0 * np.pi * np.fft.fftfreq(FINE_CHANNEL.nx, d=2.0 * np.pi / FINE_CHANNEL.nx)
    linear, nonlinear, coefficients = _problem(
        family, k, np.random.default_rng(episode_seed)
    )
    clean = integrate_etdrk4(
        _initial(FINE_CHANNEL, x, public_seed),
        linear,
        nonlinear,
        internal_dt,
        steps,
        stride,
    )
    noise_seed = int(
        np.random.SeedSequence(
            [episode_seed, public_seed, 991, sum(map(ord, FINE_CHANNEL.name))]
        ).generate_state(1)[0]
    )
    observed = _noise(clean, noise, noise_model, noise_seed)
    t = np.linspace(0.0, FINE_CHANNEL.t_final, 49)
    trajectory = Trajectory(
        f"e{episode_seed}-gradient-space-49",
        FINE_CHANNEL.name,
        x,
        t,
        observed,
        noise_model,
        noise,
    )
    truth = HiddenTruth(
        family, TRUE_SUPPORTS[family], coefficients, episode_seed
    )
    return trajectory, truth


def subsample(fine: Trajectory, cell: TemporalCell) -> Trajectory:
    """Exact nested subsample, including identical values at shared nodes."""

    indices = np.arange(0, fine.t.size, cell.subsample)
    if indices.size != cell.nodes or indices[-1] != fine.t.size - 1:
        raise ValueError("invalid nested temporal cell")
    return Trajectory(
        f"{fine.sample_id}-n{cell.nodes}",
        fine.action,
        fine.x.copy(),
        fine.t[indices].copy(),
        fine.u[indices].copy(),
        fine.noise_model,
        fine.relative_noise,
    )


def matched_episode(
    family: Family,
    episode_seed: int,
    public_seed: int,
    noise: float,
    noise_model: str,
) -> tuple[Trajectory, dict[int, Trajectory], HiddenTruth]:
    """Return a fixed baseline and all four nested channel observations."""

    baseline, _ = simulate(
        family, BASELINE, episode_seed, public_seed, noise, noise_model
    )
    fine, truth = finest_noisy_trajectory(
        family, episode_seed, public_seed, noise, noise_model
    )
    return baseline, {cell.nodes: subsample(fine, cell) for cell in CELLS}, truth

