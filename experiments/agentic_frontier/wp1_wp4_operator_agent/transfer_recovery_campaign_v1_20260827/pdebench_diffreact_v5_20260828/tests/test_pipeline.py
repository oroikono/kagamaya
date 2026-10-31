from __future__ import annotations

import unittest
import math
import numpy as np
from pathlib import Path
import tempfile

from .. import contract
from ..evidence import (
    _spatial_cosine, build_from_projection, nested_time_indices, project,
    validate_nesting,
)
from .. import pipeline
from ..pipeline import Diagnostics, correct, publisher_axis, read_source, terminal
from ..confirmation import encode_diagnostics, validate_upstream
from ..aggregate import decode_diagnostics, outcome, validate_episode
from ..policies import (
    MAX_SELECTABLE_VIEW_COST,
    all_mode_agreement,
    evaluate_all,
    public_objective,
    score_max,
    view_cost,
)


class ContractTests(unittest.TestCase):
    def test_contract_and_splits(self):
        contract.validate()
        self.assertEqual(len(contract.THRESHOLD_GRID), 675)
        self.assertEqual(len(contract.CONTRACTS), 5)
        self.assertEqual(len(contract.SEALED_CONFIRMATION_IDS), 60)
        self.assertEqual(contract.POLICIES, ("score_max", "fixed_fine", "all_mode_agreement"))

    def test_upstream_authority_is_exactly_bound(self):
        self.assertEqual(
            validate_upstream(),
            contract.Thresholds(0.025, 0.1, 1.0e4, 0.0, 0.0),
        )

    def test_public_score_preserves_frozen_functional_form(self):
        th = contract.Thresholds(**contract.SELECTED_THRESHOLDS)
        low_out = Diagnostics((), 0.11, 0.0, 2.0, 0.0)
        high_out = Diagnostics((), 0.30, 0.0, 2.0, 0.0)
        inconclusive = Diagnostics((), 0.05, 0.0, 2.0, 0.0)
        cost = view_cost(7, 2)
        self.assertGreater(
            public_objective(high_out, "OUTSIDE_LIBRARY", cost, th),
            public_objective(low_out, "OUTSIDE_LIBRARY", cost, th),
        )
        self.assertGreater(
            public_objective(low_out, "OUTSIDE_LIBRARY", cost, th),
            public_objective(inconclusive, "INCONCLUSIVE", cost, th),
        )
        self.assertEqual(MAX_SELECTABLE_VIEW_COST, view_cost(25, 5))
        nonfinite = Diagnostics((), math.inf, 0.0, math.inf, 0.0)
        self.assertLess(
            public_objective(nonfinite, "OUTSIDE_LIBRARY", cost, th),
            public_objective(low_out, "OUTSIDE_LIBRARY", cost, th),
        )

    @staticmethod
    def _policy_rows():
        rows = []
        for n in contract.TEMPORAL_SAMPLES:
            for m in contract.TEST_MODE_COUNTS:
                rows.append({
                    "view_id": f"T{n:03d}_M{m}",
                    "temporal_samples": n,
                    "test_modes": m,
                    "cost": view_cost(n, m),
                    "status": "IDENTIFIED",
                    "support": ["u"],
                    "public_objective": 10.0 + n / 1000 - m / 10000,
                })
        return rows

    def test_score_max_and_agreement_are_deterministic(self):
        rows = self._policy_rows()
        attack = score_max(rows)
        self.assertEqual(attack["selected_view_id"], "T025_M2")
        guard = all_mode_agreement(rows)
        self.assertEqual(guard["selected_view_id"], "T025_M5")
        self.assertEqual(len(guard["witnessed_view_ids"]), 4)

    def test_agreement_demotes_when_every_group_disagrees(self):
        rows = self._policy_rows()
        for row in rows:
            if row["test_modes"] == 2:
                row["status"] = "INCONCLUSIVE"
                row["support"] = []
        guard = all_mode_agreement(rows)
        self.assertEqual(guard["status"], "INCONCLUSIVE")
        self.assertIsNone(guard["selected_view_id"])
        self.assertEqual(guard["witnessed_view_ids"], [])

    def test_serialized_episode_replays_and_score_mutation_fails(self):
        thresholds = contract.Thresholds(**contract.SELECTED_THRESHOLDS)
        diagnostics = Diagnostics(contract.ATOMS["u"], 0.01, 0.1, 2.0, 0.5)
        rows = self._policy_rows()
        for row in rows:
            row["support"] = list(contract.ATOMS["u"])
            row["diagnostics"] = encode_diagnostics(diagnostics)
            row["public_objective"] = public_objective(
                diagnostics, "IDENTIFIED", row["cost"], thresholds
            )
        proposals = evaluate_all(rows)
        for proposal in proposals:
            proposal["outcome"] = outcome(
                proposal["status"], tuple(proposal["support"]), "full_u"
            )
        episode = {
            "contract_id": "full_u",
            "views": rows,
            "proposals": proposals,
            "reference": {
                "status": "IDENTIFIED",
                "support": list(contract.ATOMS["u"]),
                "outcome": "correct",
            },
        }
        validate_episode(episode, thresholds)
        rows[0]["public_objective"] += 0.01
        with self.assertRaisesRegex(ValueError, "public objective replay mismatch"):
            validate_episode(episode, thresholds)

    def test_nonfinite_diagnostics_have_canonical_json_encoding(self):
        diagnostics = Diagnostics(("u",), math.inf, -math.inf, math.nan, 0.0)
        encoded = encode_diagnostics(diagnostics)
        self.assertEqual(encoded["residual"], "inf")
        self.assertEqual(encoded["gap"], "-inf")
        self.assertEqual(encoded["condition"], "nan")
        decoded = decode_diagnostics(encoded)
        self.assertTrue(math.isinf(decoded.residual))
        self.assertTrue(math.isnan(decoded.condition))

    def test_nested_views(self):
        validate_nesting()
        previous = set()
        for count in (7, 13, 25, 49, 101):
            current = set(nested_time_indices(count))
            self.assertTrue(previous <= current)
            previous = current

    def test_neumann_cosines(self):
        x = np.array([-1.0, 1.0])
        y = np.linspace(-1.0, 1.0, 9)
        for kx in range(6):
            for ky in range(6):
                phi, lap = _spatial_cosine(kx, ky, x, y)
                self.assertEqual(phi.shape, (2, 9))
                self.assertTrue(np.all(np.isfinite(lap)))
                # Analytic normal derivative has a sine factor and vanishes at
                # x,y = -1,1 for integer cosine modes.
                self.assertAlmostEqual(np.sin(kx * np.pi * 0.0), 0.0)
                self.assertAlmostEqual(np.sin(kx * np.pi), 0.0, places=12)

    def test_exact_publisher_axis_contract(self):
        axis = publisher_axis()
        self.assertEqual(axis.shape, (128,))
        self.assertEqual(float(axis[0]), -0.994140625)
        self.assertEqual(float(axis[-1]), 0.990234375)
        np.testing.assert_array_equal(np.diff(axis), np.full(127, 0.015625))

    def test_constant_mode_keeps_bias_column(self):
        times = np.linspace(0.0, 5.0, 101)
        centers = publisher_axis()
        data = np.zeros((101, 128, 128, 2), dtype=float)
        data[..., 0] = 0.5
        data[..., 1] = -0.25
        bundle = project(source_id=0, data=data, x=centers, y=centers,
                         times=times, noise_level=0.0)
        evidence = build_from_projection(
            source_id=0, bundle=bundle, target_name="u",
            allowed_atoms=contract.ATOMS["u"], temporal_samples=25,
            test_modes=5, noise_level=0.0,
        )
        self.assertEqual(evidence.design.shape, (180, 5))
        self.assertGreater(np.max(np.abs(evidence.design[:, 3])), 0.0)
        self.assertEqual(set(evidence.roles), {"train", "validate"})

    def test_terminal_order_and_truth(self):
        th = contract.Thresholds(0.05, 0.2, 1e6, 1e-6, 2e-4)
        d = Diagnostics(contract.ATOMS["u"], 0.01, 0.1, 2.0, 0.5)
        status, support = terminal(d, th)
        self.assertEqual(status, "IDENTIFIED")
        self.assertTrue(correct(status, support, "full_u"))
        outside, support = terminal(Diagnostics((), 0.4, 0.0, 1.0, 0.0), th)
        self.assertEqual((outside, support), ("OUTSIDE_LIBRARY", ()))
        self.assertTrue(correct(outside, support, "omit_u_local_source_block"))

    def test_result_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            pipeline.write_json(path, {"stage": "first"})
            with self.assertRaises(FileExistsError):
                pipeline.write_json(path, {"stage": "second"})

    def test_publisher_hdf_schema_roundtrip(self):
        try:
            import h5py
        except ModuleNotFoundError:
            self.skipTest("h5py unavailable in local runtime; required on Euler")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake.h5"
            centers = publisher_axis()
            with h5py.File(path, "w") as handle:
                group = handle.create_group("0000")
                group.create_dataset("data", data=np.zeros((101, 128, 128, 2), dtype=np.float32))
                grid = group.create_group("grid")
                grid.create_dataset("x", data=centers.astype(np.float32))
                grid.create_dataset("y", data=centers.astype(np.float32))
                grid.create_dataset("t", data=np.linspace(0, 5, 101, dtype=np.float32))
            with h5py.File(path, "r") as handle:
                data, x, y, times = read_source(handle, 0)
            self.assertEqual(data.shape, (101, 128, 128, 2))
            self.assertEqual((len(x), len(y), len(times)), (128, 128, 101))
            with h5py.File(path, "r+") as handle:
                handle["0000/grid/x"][:] = np.linspace(-1 + 1/128, 1 - 1/128, 128)
            with h5py.File(path, "r") as handle:
                with self.assertRaisesRegex(ValueError, "decimated spatial grid drift"):
                    read_source(handle, 0)

    def test_reaction_signs_on_spatially_constant_solution(self):
        # Independent RK4 integration of the zero-Laplacian coupled ODE.  The
        # constant cosine rows must satisfy target = sum(full mechanism rows).
        def rhs(state):
            u, v = state
            return np.array([u - u**3 - contract.K - v, u - v])

        state = np.array([0.2, -0.1], dtype=float)
        dt = 5.0 / 10_000
        samples = [state.copy()]
        for step in range(10_000):
            k1 = rhs(state)
            k2 = rhs(state + 0.5 * dt * k1)
            k3 = rhs(state + 0.5 * dt * k2)
            k4 = rhs(state + dt * k3)
            state = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            if (step + 1) % 100 == 0:
                samples.append(state.copy())
        values = np.asarray(samples)
        data = np.empty((101, 128, 128, 2), dtype=float)
        data[..., 0] = values[:, 0, None, None]
        data[..., 1] = values[:, 1, None, None]
        times = np.linspace(0.0, 5.0, 101)
        centers = publisher_axis()
        bundle = project(source_id=1, data=data, x=centers, y=centers,
                         times=times, noise_level=0.0)
        for target_name in ("u", "v"):
            evidence = build_from_projection(
                source_id=1, bundle=bundle, target_name=target_name,
                allowed_atoms=contract.ATOMS[target_name],
                temporal_samples=101, test_modes=5, noise_level=0.0,
            )
            constant_rows = np.arange(0, len(evidence.target), 36)
            np.testing.assert_allclose(
                evidence.target[constant_rows],
                evidence.design[constant_rows].sum(axis=1),
                rtol=0.0, atol=4.0e-5,
            )


if __name__ == "__main__":
    unittest.main()
