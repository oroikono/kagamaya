"""Scoring for public development traces.

This scorer is intentionally public because v0.1.2 is a development/replay set.
A future hidden evaluation must keep the equivalent scorer and labels
server-side.
"""

from __future__ import annotations

from typing import Any, Mapping


DECISIVE = {"IDENTIFIED", "OUTSIDE_LIBRARY"}


def _last_observation(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    events = trace.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("trace has no events")
    event = events[-1]
    if not isinstance(event, Mapping) or not isinstance(event.get("observation"), Mapping):
        raise ValueError("trace terminal observation is malformed")
    return event["observation"]


def score_trace(trace: Mapping[str, Any], label: Mapping[str, Any]) -> dict[str, Any]:
    if trace.get("case_id") != label.get("case_id"):
        raise ValueError("trace/label case mismatch")
    observation = _last_observation(trace)
    termination = str(observation.get("termination"))
    status = "INCONCLUSIVE"
    support: list[str] | None = None
    if termination == "COMMIT":
        view = observation.get("committed_view")
        if not isinstance(view, Mapping):
            raise ValueError("COMMIT lacks committed_view")
        status = str(view.get("terminal"))
        raw_support = view.get("support")
        support = list(raw_support) if isinstance(raw_support, list) else None
    held_out = label.get("held_out")
    if not isinstance(held_out, Mapping):
        raise ValueError("label lacks held_out block")
    in_library = bool(held_out["in_library"])
    truth_support = list(held_out.get("truth_support") or [])
    correct = (
        in_library
        and status == "IDENTIFIED"
        and support == truth_support
    ) or ((not in_library) and status == "OUTSIDE_LIBRARY")
    false_promotion = status in DECISIVE and not correct
    inconclusive = status == "INCONCLUSIVE"
    return {
        "case_id": trace["case_id"],
        "status": status,
        "support": support,
        "in_library": in_library,
        "truth_support": truth_support,
        "correct": bool(correct),
        "false_promotion": bool(false_promotion),
        "inconclusive": bool(inconclusive),
        "utility": int(bool(correct)) - int(bool(false_promotion)),
    }
