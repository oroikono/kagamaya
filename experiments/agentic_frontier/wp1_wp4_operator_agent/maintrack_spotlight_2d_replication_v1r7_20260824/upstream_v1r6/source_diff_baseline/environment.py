"""Typed budgeted agent environment with owner-isolated scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Literal

from .authority import AuthorityRecord
from .contract import TERMINAL_STATUSES, VIEW_BANK


ActionKind = Literal["ACQUIRE", "CHALLENGE", "COMMIT", "ABSTAIN"]


@dataclass(frozen=True)
class AgentAction:
    kind: ActionKind
    view_id: str | None = None

    def validate(self) -> None:
        if self.kind in {"ACQUIRE", "CHALLENGE"} and self.view_id is None:
            raise ValueError("evidence action requires view_id")
        if self.kind in {"COMMIT", "ABSTAIN"} and self.view_id is not None:
            raise ValueError("terminal action must not include view_id")


@dataclass(frozen=True)
class PublicEpisode:
    episode_id: str
    origin: str
    noise_model: str
    noise_level: float
    view_records: dict[str, AuthorityRecord]

    def validate(self) -> None:
        expected = {view.view_id for view in VIEW_BANK}
        if set(self.view_records) != expected:
            raise ValueError("episode does not contain the frozen view manifest")
        if any(record.episode_id != self.episode_id for record in self.view_records.values()):
            raise ValueError("cross-episode record contamination")


@dataclass(frozen=True)
class OwnerLabel:
    episode_id: str
    family: str
    truth_kind: str
    truth_support: tuple[str, ...]
    omitted_mechanism: str | None


@dataclass(frozen=True)
class TerminalDecision:
    status: str
    support: tuple[str, ...]
    reason: str
    evidence_view_ids: tuple[str, ...]
    realized_cost: float


@dataclass(frozen=True)
class Score:
    correct: bool
    false_promotion: bool
    inconclusive: bool


def score_decision(decision: TerminalDecision, label: OwnerLabel) -> Score:
    if decision.status not in TERMINAL_STATUSES:
        raise ValueError("unknown terminal status")
    if decision.status == "INCONCLUSIVE":
        return Score(False, False, True)
    correct = ((label.truth_kind == "in_library"
                and decision.status == "IDENTIFIED"
                and decision.support == label.truth_support)
               or (label.truth_kind == "outside_library"
                   and decision.status == "OUTSIDE_LIBRARY"))
    return Score(correct=correct, false_promotion=not correct, inconclusive=False)


def _consensus(records: list[AuthorityRecord]) -> tuple[str, tuple[str, ...], str]:
    if not records:
        return "INCONCLUSIVE", (), "no_evidence"
    signatures = {(record.status, record.support) for record in records}
    if len(signatures) == 1:
        status, support = next(iter(signatures))
        return status, support, "exact_cross_view_agreement"
    return "INCONCLUSIVE", (), "cross_view_disagreement"


@dataclass
class EvidenceEnvironment:
    public_episode: PublicEpisode
    budget: float
    acquired: list[AuthorityRecord] = field(default_factory=list)
    action_trace: list[AgentAction] = field(default_factory=list)
    terminal: TerminalDecision | None = None

    def public_state(self) -> dict:
        return {
            "episode_id": self.public_episode.episode_id,
            "origin": self.public_episode.origin,
            "noise_model": self.public_episode.noise_model,
            "noise_level": self.public_episode.noise_level,
            "budget": self.budget,
            "spent": sum(record.registered_cost for record in self.acquired),
            "available_view_manifest": [
                {
                    "view_id": view.view_id,
                    "spatial_nodes": view.spatial_nodes,
                    "temporal_samples": view.temporal_samples,
                    "test_mode_count": view.test_mode_count,
                    "window": [view.window_start, view.window_end],
                    "registered_cost": view.registered_cost,
                }
                for view in VIEW_BANK
            ],
            "acquired": [record.public_dict() for record in self.acquired],
        }

    def step(self, action: AgentAction) -> TerminalDecision | None:
        action.validate()
        if self.terminal is not None:
            raise RuntimeError("episode already terminated")
        self.action_trace.append(action)
        if action.kind in {"ACQUIRE", "CHALLENGE"}:
            if action.view_id not in self.public_episode.view_records:
                raise ValueError("view outside frozen manifest")
            record = self.public_episode.view_records[action.view_id]
            if any(old.view_id == record.view_id for old in self.acquired):
                raise ValueError("repeat acquisitions are not admitted in v2")
            spent = sum(old.registered_cost for old in self.acquired)
            if spent + record.registered_cost > self.budget + 1e-9:
                raise ValueError("action exceeds registered budget")
            self.acquired.append(record)
            return None
        if action.kind == "ABSTAIN":
            self.terminal = TerminalDecision(
                "INCONCLUSIVE", (), "agent_abstained",
                tuple(record.view_id for record in self.acquired),
                sum(record.registered_cost for record in self.acquired),
            )
            return self.terminal
        status, support, reason = _consensus(self.acquired)
        self.terminal = TerminalDecision(
            status, support, reason,
            tuple(record.view_id for record in self.acquired),
            sum(record.registered_cost for record in self.acquired),
        )
        return self.terminal

    @property
    def trace_digest(self) -> str:
        payload = {
            "episode_id": self.public_episode.episode_id,
            "actions": [{"kind": action.kind, "view_id": action.view_id}
                        for action in self.action_trace],
            "terminal": None if self.terminal is None else {
                "status": self.terminal.status,
                "support": self.terminal.support,
                "reason": self.terminal.reason,
                "views": self.terminal.evidence_view_ids,
                "cost": self.terminal.realized_cost,
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

