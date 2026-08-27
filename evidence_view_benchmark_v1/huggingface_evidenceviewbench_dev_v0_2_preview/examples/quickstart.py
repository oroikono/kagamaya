"""Dependency-free EvidenceViewBench-Dev walkthrough."""

from __future__ import annotations

import json

from evidenceviewbench import (
    Action,
    EvidenceViewEnv,
    case_bundle,
    load_observations,
    score_trace,
)


case = load_observations()[0]
env = EvidenceViewEnv(case, budget=60, visibility="acquire_only")
reset = env.reset()
assert "family" not in reset and "noise_model" not in reset

env.step(Action.acquire_view(7, 2))
env.step(Action.acquire_view(49, 5))
terminal, done = env.step(Action.commit(49, 5))
assert done and terminal["termination"] == "COMMIT"

trace = env.export_trace()
label = case_bundle(case["case_id"])["label"]
score = score_trace(trace, label)

print(
    json.dumps(
        {
            "case_id": case["case_id"],
            "termination": trace["termination"],
            "spent": trace["spent"],
            "score": score,
        },
        indent=2,
        sort_keys=True,
    )
)
