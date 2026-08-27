from __future__ import annotations

import unittest

from .screen import POLICIES, choose


def episode(status_by_config):
    actions = []
    index = 0
    for nodes in (7, 13, 25):
        for modes in (2, 3, 4, 5):
            status, support, objective = status_by_config.get(
                (nodes, modes), ("UNIDENTIFIED", None, -1.0)
            )
            actions.append(
                {
                    "action_token": f"A{index:02d}",
                    "nodes": nodes,
                    "temporal_modes": modes,
                    "terminal_status": status,
                    "terminal_support": support,
                    "public_objective": objective,
                }
            )
            index += 1
    return {"actions": actions}


class ConsensusTests(unittest.TestCase):
    def test_all_policies_abstain_without_terminal_consensus(self):
        item = episode({})
        for policy in POLICIES:
            self.assertEqual(choose(item, policy), (None, 25, []))

    def test_adjacent_consensus_selects_richer_view_and_accounts_nodes(self):
        support = ["u_x", "u_xx"]
        item = episode(
            {
                (7, 3): ("IDENTIFIED", support, 8.0),
                (13, 3): ("IDENTIFIED", support, 7.0),
            }
        )
        token, consulted, evidence = choose(item, "adjacent_node_consensus")
        self.assertEqual(token, "A05")
        self.assertEqual(consulted, 13)
        self.assertEqual(evidence, ["A01", "A05"])

    def test_two_axis_requires_two_distinct_nodes_and_modes(self):
        support = ["u_x"]
        one_axis = episode(
            {
                (7, 2): ("IDENTIFIED", support, 5.0),
                (7, 3): ("IDENTIFIED", support, 5.0),
            }
        )
        self.assertEqual(choose(one_axis, "two_axis_consensus"), (None, 25, []))
        two_axis = episode(
            {
                (7, 2): ("IDENTIFIED", support, 5.0),
                (7, 3): ("IDENTIFIED", support, 5.0),
                (13, 2): ("IDENTIFIED", support, 5.0),
            }
        )
        token, consulted, _ = choose(two_axis, "two_axis_consensus")
        self.assertEqual(token, "A00")
        self.assertEqual(consulted, 13)


if __name__ == "__main__":
    unittest.main()
