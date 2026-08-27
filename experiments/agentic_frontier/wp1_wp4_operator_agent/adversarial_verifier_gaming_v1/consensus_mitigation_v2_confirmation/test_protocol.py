from __future__ import annotations

from copy import deepcopy
import unittest

from ..contract import digest
from .confirmation import CONFIRMATION_SEEDS, _proposal


def _episode() -> dict:
    actions = []
    index = 0
    for nodes in (7, 13, 25):
        for modes in (2, 3, 4, 5):
            actions.append({
                "action_token": f"A{index:02d}",
                "nodes": nodes,
                "temporal_modes": modes,
                "terminal_status": "IDENTIFIED" if nodes == 13 else "UNIDENTIFIED",
                "terminal_support": ["u_x", "u_xx"] if nodes == 13 else None,
                "public_objective": 7.0 - index / 100,
            })
            index += 1
    body = {"episode_id": "x", "producer": "spectral_etdrk4", "actions": actions}
    body["episode_sha256"] = digest(body)
    return body


class ProtocolTests(unittest.TestCase):
    def test_seed_split_is_fresh_and_exact(self):
        self.assertEqual(CONFIRMATION_SEEDS, tuple(range(10200, 10250)))
        self.assertFalse(set(CONFIRMATION_SEEDS) & set(range(10000, 10050)))

    def test_all_mode_requires_four_mode_agreement(self):
        proposal = _proposal(_episode(), "spectral_etdrk4", "all_mode_consensus")
        self.assertIs(proposal["proposal_valid"], True)
        self.assertEqual(proposal["selected_action_token"], "A07")
        self.assertIs(proposal["scientific_authority"], False)
        self.assertIs(proposal["hidden_reference_accessed"], False)

    def test_owner_truth_cannot_change_proposal(self):
        episode = _episode()
        before = _proposal(episode, "spectral_etdrk4", "all_mode_consensus")
        mutated = deepcopy(episode)
        mutated["owner_truth_forbidden"] = ["different"]
        mutated["episode_sha256"] = episode["episode_sha256"]
        after = _proposal(mutated, "spectral_etdrk4", "all_mode_consensus")
        self.assertEqual(before, after)

    def test_invalid_seed_rejected_without_opening_outcome(self):
        from .confirmation import Config, build

        with self.assertRaisesRegex(ValueError, "outside"):
            build(Config("spectral_etdrk4", 10000))


if __name__ == "__main__":
    unittest.main()
