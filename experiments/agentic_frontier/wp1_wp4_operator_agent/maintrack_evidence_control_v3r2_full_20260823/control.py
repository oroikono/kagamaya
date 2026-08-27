"""Frozen DIAGNOSE -> SUBMIT(view)/ABSTAIN environment and owner scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

from .contract import ControlAction, VIEW_BANK


VIEW_MAP = {view.view_id: view for view in VIEW_BANK}
VIEW_IDS = frozenset(VIEW_MAP)


@dataclass(frozen=True)
class Terminal:
    status: str
    support: tuple[str, ...]
    reason: str
    diagnose_view_id: str
    submit_view_id: str | None
    registered_cost: float


@dataclass(frozen=True)
class Outcome:
    correct: bool
    false_promotion: bool
    inconclusive: bool


def score(terminal: Terminal, owner: dict) -> Outcome:
    if terminal.status == "INCONCLUSIVE":
        return Outcome(False, False, True)
    correct = ((owner["truth_kind"] == "in_library"
                and terminal.status == "IDENTIFIED"
                and tuple(owner["truth_support"]) == terminal.support)
               or (owner["truth_kind"] == "outside_library"
                   and terminal.status == "OUTSIDE_LIBRARY"))
    return Outcome(correct, not correct, False)


@dataclass
class EvidenceControl:
    episode: dict
    budget: float
    diagnose_view_id: str | None = None
    terminal: Terminal | None = None
    trace: list[dict] = field(default_factory=list)

    def _record(self, view_id: str) -> dict:
        records = {row["view_id"]: row for row in self.episode["views"]}
        return records[view_id]

    def public_state(self) -> dict:
        diagnosis = None
        if self.diagnose_view_id is not None:
            diagnosis = self._record(self.diagnose_view_id)
        return {
            "episode_nonce": self.episode["episode_id"],
            "budget": self.budget,
            "diagnosis": diagnosis,
            "available_views": [
                {
                    "view_id": view.view_id,
                    "spatial_nodes": view.spatial_nodes,
                    "temporal_samples": view.temporal_samples,
                    "test_mode_count": view.test_mode_count,
                    "window_start": view.window_start,
                    "window_end": view.window_end,
                    "registered_cost": view.registered_cost,
                }
                for view in VIEW_BANK
            ],
        }

    def step(self, action: ControlAction) -> Terminal | None:
        action.validate(VIEW_IDS)
        if self.terminal is not None:
            raise RuntimeError("episode already terminated")
        if self.diagnose_view_id is None:
            if action.kind != "DIAGNOSE":
                raise ValueError("first action must be DIAGNOSE")
            assert action.view_id is not None
            if VIEW_MAP[action.view_id].registered_cost > self.budget:
                raise ValueError("diagnosis exceeds budget")
            self.diagnose_view_id = action.view_id
            self.trace.append({"kind": action.kind, "view_id": action.view_id})
            return None
        if action.kind == "DIAGNOSE":
            raise ValueError("exactly one diagnosis is permitted")
        self.trace.append({"kind": action.kind, "view_id": action.view_id})
        if action.kind == "ABSTAIN":
            self.terminal = Terminal(
                "INCONCLUSIVE", (), "policy_abstained", self.diagnose_view_id,
                None, VIEW_MAP[self.diagnose_view_id].registered_cost)
            return self.terminal
        assert action.view_id is not None
        cost = VIEW_MAP[self.diagnose_view_id].registered_cost
        if action.view_id != self.diagnose_view_id:
            cost += VIEW_MAP[action.view_id].registered_cost
        if cost > self.budget:
            raise ValueError("submission exceeds budget")
        record = self._record(action.view_id)
        self.terminal = Terminal(
            record["status"], tuple(record["support"]),
            "submitted_frozen_authority_view", self.diagnose_view_id,
            action.view_id, cost)
        return self.terminal

    @property
    def trace_digest(self) -> str:
        return hashlib.sha256(json.dumps({
            "episode_id": self.episode["episode_id"],
            "trace": self.trace,
            "terminal": None if self.terminal is None else self.terminal.__dict__,
        }, sort_keys=True).encode()).hexdigest()
