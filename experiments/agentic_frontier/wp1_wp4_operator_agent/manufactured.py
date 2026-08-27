"""Manufactured observed ``(u,f)`` pairs for the frozen recovery tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .serialization import array_digest, content_digest


@dataclass(frozen=True)
class ObservedField:
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray
    f: np.ndarray
    metadata: dict[str, Any]

    def digest(self) -> str:
        return content_digest(
            {
                "x": array_digest(self.x),
                "t": array_digest(self.t),
                "u": array_digest(self.u),
                "f": array_digest(self.f),
                "metadata": self.metadata,
            }
        )


def manufactured_reaction_diffusion(
    *,
    excitation: str,
    nx: int = 128,
    nt: int = 81,
    reaction: float = 0.7,
    diffusion: float = 0.15,
) -> ObservedField:
    """Return samples satisfying ``u_t = reaction*u + diffusion*u_xx + f``.

    Exact derivatives are confined to this generator. The discovery engine is
    passed only the sampled ``u`` and ``f`` arrays.
    """
    if excitation not in {"single_mode", "multi_mode"}:
        raise ValueError("excitation must be 'single_mode' or 'multi_mode'")
    if nx < 8 or nt < 9:
        raise ValueError("the manufactured grid is too small")

    x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False, dtype=float)
    t = np.linspace(0.0, 1.0, nt, endpoint=True, dtype=float)
    xx = x[None, :]
    tt = t[:, None]

    # The first frozen benchmark is stationary. This keeps the trapezoidal
    # temporal integration-by-parts identity exact up to roundoff while the
    # spatial excitation, rather than hidden analytic derivatives, controls
    # whether reaction and diffusion can be separated.
    amplitude_1 = np.ones_like(tt)
    amplitude_1_t = np.zeros_like(tt)
    u = amplitude_1 * np.sin(xx)
    u_t = amplitude_1_t * np.sin(xx)
    u_xx = -amplitude_1 * np.sin(xx)

    if excitation == "multi_mode":
        amplitude_2 = 0.6 * np.ones_like(tt)
        amplitude_2_t = np.zeros_like(tt)
        u = u + amplitude_2 * np.sin(2.0 * xx)
        u_t = u_t + amplitude_2_t * np.sin(2.0 * xx)
        u_xx = u_xx - 4.0 * amplitude_2 * np.sin(2.0 * xx)
        second_mode_min_amplitude = float(np.min(amplitude_2))
    else:
        second_mode_min_amplitude = 0.0

    f = u_t - reaction * u - diffusion * u_xx
    metadata = {
        "schema_version": "verisigs.wp1_wp4.observed_field.v1",
        "excitation": excitation,
        "domain": {"x": [0.0, 2.0 * np.pi], "t": [0.0, 1.0]},
        "periodic_unique_spatial_points": True,
        "second_mode_min_amplitude": second_mode_min_amplitude,
        "generator_truth_present_in_runtime_object": False,
    }
    return ObservedField(x=x, t=t, u=u, f=f, metadata=metadata)
