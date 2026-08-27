"""Minimal 2-D/coupled agent-environment replay."""

from evidenceviewbench import (
    TwoDAction,
    TwoDEvidenceEnv,
    load_two_d_owner,
    load_two_d_public,
    score_two_d_terminal,
)


public = load_two_d_public()[0]
owner = next(
    row
    for row in load_two_d_owner()
    if row["origin_component"] == public["origin_component"]
    and row["episode_id"] == public["episode_id"]
)

costs = {row["view_id"]: row["registered_cost"] for row in public["views"]}
env = TwoDEvidenceEnv(public, budget=costs["V035"] + costs["V051"])
env.step(TwoDAction.acquire("V035"))
env.step(TwoDAction.challenge("V051"))
terminal = env.step(TwoDAction.commit())
assert terminal is not None

trace = env.export_trace()
assert TwoDEvidenceEnv.replay_trace(public, trace) == trace
score = score_two_d_terminal(terminal, owner)

print(
    {
        "episode_id": public["episode_id"],
        "origin_component": public["origin_component"],
        "terminal": terminal.status,
        "views": terminal.evidence_view_ids,
        "correct": score.correct,
        "false_promotion": score.false_promotion,
        "trace_sha256": trace["trace_sha256"],
    }
)
