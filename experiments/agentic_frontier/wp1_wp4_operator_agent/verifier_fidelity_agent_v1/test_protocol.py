from __future__ import annotations

from copy import deepcopy

from ..independent_fd_producer_v1.simulator import Family as FDFamily
from ..maintrack_suite_v41.simulator import Family
from .campaign import NESTED_NODES, select, signature


def row(nodes, status, support=None, correct=False, false=False):
    return {"nodes": nodes, "final": {"status": status, "support": support},
            "correct": correct, "false_promotion": false}


def test_coarse_oog_cannot_stop_adaptive_controller():
    cells = {7: row(7, "OUT_OF_GRAMMAR"), 13: row(13, "UNIDENTIFIED"),
             25: row(25, "IDENTIFIED", ["u_x"]), 49: row(49, "IDENTIFIED", ["u_x"])}
    assert select(cells, "adaptive_identified")["nodes"] == 25
    assert select(cells, "unsafe_first_terminal")["nodes"] == 7


def test_adaptive_falls_back_to_fine():
    cells = {n: row(n, "UNIDENTIFIED") for n in NESTED_NODES}
    cells[49] = row(49, "OUT_OF_GRAMMAR")
    assert select(cells, "adaptive_identified")["nodes"] == 49


def test_policy_is_owner_truth_blind():
    cells = {7: row(7, "UNIDENTIFIED"), 13: row(13, "IDENTIFIED", ["u_x"]),
             25: row(25, "IDENTIFIED", ["u_x"]), 49: row(49, "IDENTIFIED", ["u_x"])}
    mutated = deepcopy(cells)
    for item in mutated.values(): item["truth_support"] = ["secret"]
    for policy in ("adaptive_identified", "unsafe_first_terminal", "two_resolution_stable", "oracle_match"):
        assert select(cells, policy)["nodes"] == select(mutated, policy)["nodes"]
        assert signature(select(cells, policy)) == signature(select(mutated, policy))


def test_oracle_is_cheapest_fine_match():
    cells = {7: row(7, "UNIDENTIFIED"), 13: row(13, "IDENTIFIED", ["a"]),
             25: row(25, "IDENTIFIED", ["b"]), 49: row(49, "IDENTIFIED", ["b"])}
    assert select(cells, "oracle_match")["nodes"] == 25


def test_distinct_family_enums_require_value_bridge():
    assert FDFamily is not Family
    for family in Family:
        assert FDFamily(family.value).value == family.value
