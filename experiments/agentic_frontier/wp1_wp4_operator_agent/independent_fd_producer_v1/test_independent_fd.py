from __future__ import annotations

import copy
import hashlib
import json
import unittest

import numpy as np

from .simulator import (
    BASELINE,
    GRADIENT,
    Family,
    add_noise,
    derivative_matrix,
    draw_parameters,
    initial_condition,
    observed_trajectory,
    solve_clean,
    subsample,
)
from .campaign import CampaignConfig, run
from .validate import validate_payload


def _rehash(payload: dict) -> None:
    body = dict(payload)
    body.pop("content_sha256", None)
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class IndependentFiniteDifferenceTests(unittest.TestCase):
    @staticmethod
    def tiny_payload() -> dict:
        return run(
            CampaignConfig(
                (9590,),
                families=("advection_diffusion",),
                noise_models=("iid",),
                relative_noises=(0.001,),
                split="test_development",
            )
        )

    def test_periodic_derivatives(self) -> None:
        nx = 96
        x = np.arange(nx) * (2.0 * np.pi / nx)
        field = np.sin(3.0 * x) + 0.2 * np.cos(5.0 * x)
        exact = {
            1: 3.0 * np.cos(3.0 * x) - 1.0 * np.sin(5.0 * x),
            2: -9.0 * np.sin(3.0 * x) - 5.0 * np.cos(5.0 * x),
            3: -27.0 * np.cos(3.0 * x) + 25.0 * np.sin(5.0 * x),
            4: 81.0 * np.sin(3.0 * x) + 125.0 * np.cos(5.0 * x),
        }
        tolerances = {1: 2e-6, 2: 2e-5, 3: 3e-4, 4: 3e-3}
        for derivative in range(1, 5):
            error = np.linalg.norm(derivative_matrix(nx, derivative) @ field - exact[derivative]) / np.linalg.norm(exact[derivative])
            self.assertLess(error, tolerances[derivative], (derivative, error))

    def test_advection_diffusion_against_analytic_solution(self) -> None:
        for config in (BASELINE, GRADIENT):
            x, t, numerical, _ = solve_clean(Family.ADVECTION_DIFFUSION, config, 9590, 8675309)
            parameters, _ = draw_parameters(Family.ADVECTION_DIFFUSION, 9590)
            initial = initial_condition(config, x, 8675309)
            k = np.fft.fftfreq(config.nx, d=1.0 / config.nx)
            initial_hat = np.fft.fft(initial)
            analytic = np.vstack([
                np.fft.ifft(initial_hat * np.exp((-1j * parameters["c"] * k - parameters["d"] * k**2) * time)).real
                for time in t
            ])
            error = np.linalg.norm(numerical - analytic) / np.linalg.norm(analytic)
            self.assertLess(error, 2e-4, (config.name, error))

    def test_nested_shared_nodes_and_noise(self) -> None:
        x, t, clean, _ = solve_clean(Family.ADVECTION_DIFFUSION, GRADIENT, 9590, 8675309)
        fine = observed_trajectory(GRADIENT, 9590, 8675309, clean, x, t, 0.005, "correlated")
        for nodes in (7, 13, 25, 49):
            coarse = subsample(fine, nodes)
            stride = 48 // (nodes - 1)
            np.testing.assert_array_equal(coarse.u, fine.u[::stride])
            np.testing.assert_array_equal(coarse.t, fine.t[::stride])
        same = add_noise(clean, 0.005, "correlated", 123)
        np.testing.assert_array_equal(same, add_noise(clean, 0.005, "correlated", 123))

    def test_advective_sign_mutation_is_detected(self) -> None:
        _, _, correct, _ = solve_clean(Family.ADVECTION_DIFFUSION, GRADIENT, 9590, 8675309)
        _, _, mutated, _ = solve_clean(
            Family.ADVECTION_DIFFUSION,
            GRADIENT,
            9590,
            8675309,
            advection_sign=-1.0,
        )
        relative_change = np.linalg.norm(correct[-1] - mutated[-1]) / np.linalg.norm(correct[-1])
        self.assertGreater(relative_change, 0.05)

    def test_payload_validator_recomputes_endpoints_and_summaries(self) -> None:
        payload = self.tiny_payload()
        validate_payload(payload)
        mutated = copy.deepcopy(payload)
        mutated["rows"][0]["correct"] = not mutated["rows"][0]["correct"]
        _rehash(mutated)
        with self.assertRaisesRegex(ValueError, "correct label"):
            validate_payload(mutated)

        stale = copy.deepcopy(payload)
        stale["summary"][next(iter(stale["summary"]))]["episodes"] += 1
        _rehash(stale)
        with self.assertRaisesRegex(ValueError, "summary"):
            validate_payload(stale)


if __name__ == "__main__":
    unittest.main()
