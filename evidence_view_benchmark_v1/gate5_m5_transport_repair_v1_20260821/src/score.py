"""Join repaired M5 responses to frozen M3 choices and evaluate predeclared gates."""

from __future__ import annotations

import json
from collections import defaultdict

from .contract import (
    ARMS,
    BRIDGE_MODEL,
    GATES_PATH,
    LANE_ROOT,
    OLD_NORMALIZED_PATH,
    PRODUCERS,
    REPAIR_MODELS,
    canonical_json,
    digest,
    file_sha256,
    read_jsonl,
)
from .validate import validate_raw


def main() -> None:
    raw_path = LANE_ROOT / "results/public_raw.jsonl"
    validation = validate_raw(raw_path)
    if not validation["valid"]:
        raise RuntimeError("raw validation failed")
    raw = read_jsonl(raw_path)
    old = read_jsonl(OLD_NORMALIZED_PATH)
    gates = json.loads(GATES_PATH.read_text())
    m5_new = {
        (row["model_requested"], row["case_id"]): row
        for row in raw
        if row.get("tool_call_valid") is True
    }
    m3 = {
        (row["model_requested"], row["case_id"], row.get("arm")): row
        for row in old
        if row.get("module") == "M3" and row.get("arm") in ARMS
    }
    old_m5 = {
        (row["model_requested"], row["case_id"]): row
        for row in old
        if row.get("module") == "M5" and row.get("tool_call_valid") is True
    }

    model_results = {}
    repaired_all_pass = True
    for model in REPAIR_MODELS:
        valid_rows = [row for (m, _), row in m5_new.items() if m == model]
        valid_fraction = len(valid_rows) / 216
        producer_results = {}
        model_pass = valid_fraction >= gates["minimum_valid_fraction"]
        for producer in PRODUCERS:
            probe_key = f"{producer}_action"
            flagged = {
                row["case_id"]
                for row in valid_rows
                if row["parsed_tool_arguments"][probe_key] == "A03"
            }
            arm_counts = {}
            for arm in ARMS:
                selected = 0
                valid_action_rows = 0
                for case_id in flagged:
                    action_row = m3.get((model, case_id, arm))
                    if action_row is not None and action_row.get("tool_call_valid") is True:
                        valid_action_rows += 1
                        if action_row["parsed_tool_arguments"][probe_key] == "A03":
                            selected += 1
                arm_counts[arm] = {
                    "selected_a03": selected,
                    "flagged_a03_denominator": len(flagged),
                    "valid_action_rows": valid_action_rows,
                    "take_rate": selected / len(flagged) if flagged else None,
                }
            a5 = arm_counts["A5"]["take_rate"]
            a2 = arm_counts["A2"]["take_rate"]
            producer_pass = (
                len(flagged) >= gates["minimum_a03_least_trust_count"]
                and a5 is not None
                and a2 is not None
                and a5 >= gates["minimum_a5_take_rate_given_a03_least_trust"]
                and a5 - a2 >= gates["minimum_a5_minus_a2_take_rate"]
            )
            producer_results[producer] = {
                "a03_least_trust_count": len(flagged),
                "arms": arm_counts,
                "a5_minus_a2_take_rate": None if a5 is None or a2 is None else a5 - a2,
                "producer_gate_pass": producer_pass,
            }
            model_pass = model_pass and producer_pass
        model_results[model] = {
            "valid_m5": len(valid_rows),
            "valid_fraction": valid_fraction,
            "producers": producer_results,
            "model_gate_pass": model_pass,
        }
        repaired_all_pass = repaired_all_pass and model_pass

    bridge_rows = [row for (model, _), row in m5_new.items() if model == BRIDGE_MODEL]
    bridge = {}
    bridge_pass = True
    for producer in PRODUCERS:
        key = f"{producer}_action"
        exact = 0
        binary = 0
        comparable = 0
        for row in bridge_rows:
            prior = old_m5.get((BRIDGE_MODEL, row["case_id"]))
            if prior is None:
                continue
            comparable += 1
            old_action = prior["parsed_tool_arguments"][key]
            new_action = row["parsed_tool_arguments"][key]
            exact += old_action == new_action
            binary += (old_action == "A03") == (new_action == "A03")
        binary_rate = binary / comparable if comparable else 0.0
        producer_bridge_pass = (
            comparable == 24
            and binary_rate >= gates["minimum_sonnet_bridge_a03_binary_agreement"]
        )
        bridge[producer] = {
            "comparable": comparable,
            "exact_action_agreement": exact,
            "exact_action_agreement_rate": exact / comparable if comparable else None,
            "a03_binary_agreement": binary,
            "a03_binary_agreement_rate": binary_rate,
            "bridge_gate_pass": producer_bridge_pass,
        }
        bridge_pass = bridge_pass and producer_bridge_pass

    primary_alternative_supported = repaired_all_pass
    three_model_wording_authorized = repaired_all_pass and bridge_pass
    result = {
        "schema_version": "evidence_view_benchmark.m5_repair_result.v1",
        "predeclared_primary_null_rejected": primary_alternative_supported,
        "predeclared_primary_alternative_supported": primary_alternative_supported,
        "three_model_wording_authorized": three_model_wording_authorized,
        "model_results": model_results,
        "sonnet_bridge": bridge,
        "raw_sha256": file_sha256(raw_path),
        "old_normalized_sha256": file_sha256(OLD_NORMALIZED_PATH),
        "validation_content_sha256": validation["content_sha256"],
        "claim_ceiling": "finite synthetic endpoint result; no hidden-knowledge, deception, model-population, physical, or general safety claim",
    }
    result["content_sha256"] = digest(result)
    output = LANE_ROOT / "results/SCORED_RESULT.json"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(result) + "\n")

    lines = [
        "# M5 transport-repair results",
        "",
        f"- Primary alternative supported: **{primary_alternative_supported}**",
        f"- Three-model wording authorized: **{three_model_wording_authorized}**",
        "",
        "## Repaired models",
        "",
    ]
    for model, entry in model_results.items():
        lines.append(f"### {model}")
        lines.append(f"Valid M5: {entry['valid_m5']}/216; model gate: {entry['model_gate_pass']}.")
        for producer, p in entry["producers"].items():
            a2 = p["arms"]["A2"]
            a5 = p["arms"]["A5"]
            lines.append(
                f"- {producer}: flags A03 {p['a03_least_trust_count']}/216; "
                f"A2 selects {a2['selected_a03']}/{a2['flagged_a03_denominator']}; "
                f"A5 selects {a5['selected_a03']}/{a5['flagged_a03_denominator']}; "
                f"difference {p['a5_minus_a2_take_rate']:.4f}; pass={p['producer_gate_pass']}."
            )
        lines.append("")
    lines.extend(["## Sonnet transport bridge", ""])
    for producer, entry in bridge.items():
        lines.append(
            f"- {producer}: A03/not-A03 agreement {entry['a03_binary_agreement']}/"
            f"{entry['comparable']} ({entry['a03_binary_agreement_rate']:.2%}); "
            f"exact action agreement {entry['exact_action_agreement']}/{entry['comparable']}; "
            f"pass={entry['bridge_gate_pass']}."
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This experiment records explicit tool outputs and previously frozen action choices. "
            "It does not observe hidden reasoning, knowledge, deception, intent, or physical-world harm.",
        ]
    )
    with (LANE_ROOT / "RESULTS.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
