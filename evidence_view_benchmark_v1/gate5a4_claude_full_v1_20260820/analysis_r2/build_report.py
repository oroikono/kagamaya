"""Build exact machine-readable and human-readable campaign reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..src.contract import ARMS, MODELS, PRODUCERS, action_geometry, canonical_json, digest, file_sha256


SHORT = {
    "claude-opus-4-8": "Opus 4.8",
    "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-haiku-4-5": "Haiku 4.5",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    args = parser.parse_args()
    raw = [json.loads(line) for line in args.raw.read_text().splitlines() if line]
    score = json.loads(args.score.read_text())
    validation = json.loads(args.validation.read_text())
    if validation.get("valid") is not True or validation.get("raw_sha256") != file_sha256(args.raw):
        raise ValueError("raw validation binding failed")

    validity = {}
    accounting = {}
    for model in MODELS:
        model_rows = [row for row in raw if row.get("model_requested") == model]
        validity[model] = {}
        for module in ("M3", "M5"):
            rows = [row for row in model_rows if row.get("module") == module]
            errors = Counter(
                row.get("tool_parse_error") or row.get("error")
                for row in rows
                if row.get("tool_call_valid") is not True
            )
            validity[model][module] = {
                "scheduled": len(rows),
                "valid": sum(row.get("tool_call_valid") is True for row in rows),
                "invalid": sum(row.get("tool_call_valid") is not True for row in rows),
                "error_counts": dict(errors),
            }
        accounting[model] = {
            "input_tokens": sum(row.get("usage", {}).get("input_tokens", 0) for row in model_rows),
            "output_tokens": sum(row.get("usage", {}).get("output_tokens", 0) for row in model_rows),
            "observed_list_price_usd": sum(row.get("observed_list_price_usd", 0) for row in model_rows),
        }

    discordance = {}
    for model in MODELS:
        for producer in PRODUCERS:
            rows = [row for row in score["rows"] if row["model"] == model and row["producer"] == producer and row["in_library"]]
            index = {(row["case_id"], row["arm"]): row for row in rows}
            counts = Counter()
            case_ids = sorted({row["case_id"] for row in rows})
            for case_id in case_ids:
                a2 = index[(case_id, "A2")]["false_promotion_owner_diagnostic"]
                a3 = index[(case_id, "A3")]["false_promotion_owner_diagnostic"]
                counts[f"A2_{int(a2)}__A3_{int(a3)}"] += 1
            discordance[f"{model}::{producer}"] = dict(counts)

    full = {
        "schema_version": "evidence_view_benchmark.claude_full_report.v1",
        "raw_sha256": file_sha256(args.raw),
        "scored_sha256": file_sha256(args.score),
        "raw_validation_content_sha256": validation["content_sha256"],
        "schedule": {"models": list(MODELS), "cases": 216, "arms": list(ARMS), "m5_per_model": 216, "calls": 3240},
        "spend": {
            "smoke_usd": 0.107472,
            "campaign_model_sum_usd": sum(row["observed_list_price_usd"] for row in accounting.values()),
            "retained_timeout_upper_usd": 0.032495,
            "grant_consumption_usd": 57.93726200000004,
            "hard_cap_usd": 100.0,
            "note": "grant consumption includes smoke and the conservative retained reservation for the non-retried timeout",
        },
        "validity": validity,
        "accounting_by_model": accounting,
        "owner_summaries": score["summaries"],
        "A3_minus_A2": score["A3_minus_A2"],
        "A3_A2_paired_discordance": discordance,
        "m5_choice_joins": score["m5_choice_joins"],
        "pooled_paired_summaries": score["pooled_paired_summaries"],
    }
    full["content_sha256"] = digest(full)
    args.summary_json.write_text(canonical_json(full) + "\n", encoding="utf-8")

    lines = [
        "# Full Claude evidence-choice campaign — exact results",
        "",
        "Status: complete, raw schedule validated, owner join complete.",
        "",
        "## Scope and integrity",
        "",
        "- 216 public cases; 180 in-library and 36 hyperdiffusion cases.",
        "- Four M3 arms (A1, A2, A3, A5) plus M5 for three Claude models: 3,240 calls.",
        "- The 36 hyperdiffusion cases are excluded from every false-promotion denominator below.",
        f"- Raw SHA-256: `{full['raw_sha256']}`.",
        f"- Scored SHA-256: `{full['scored_sha256']}`.",
        f"- Total grant consumption: ${full['spend']['grant_consumption_usd']:.6f} of $100, including smoke and the retained timeout reservation.",
        "- One Opus M3/A5 request timed out and was not retried; it is an invalid proposal, not a fabricated response.",
        "",
        "## Forced-tool validity and accounting",
        "",
        "| Model | M3 valid | M5 valid | Input tokens | Output tokens | Campaign list-price cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        lines.append(
            f"| {SHORT[model]} | {validity[model]['M3']['valid']}/{validity[model]['M3']['scheduled']} | "
            f"{validity[model]['M5']['valid']}/{validity[model]['M5']['scheduled']} | "
            f"{accounting[model]['input_tokens']:,} | {accounting[model]['output_tokens']:,} | "
            f"${accounting[model]['observed_list_price_usd']:.6f} |"
        )
    lines += [
        "",
        "Opus M5 usually exhausted the 256-token cap before supplying the required explanation (214/216 invalid). Haiku returned the `signals` array as a serialized string (216/216 invalid). Sonnet satisfied both tool contracts on all 1,080 calls. No response was repaired or retried.",
        "",
        "## False promotion on the 180 in-library cases",
        "",
        "The two producer components have the same false-promotion counts in every model/arm cell; they are paired corroborative components, not independent replications.",
        "",
        "| Model | A1 | A2 scientific accuracy | A3 decisiveness | A5 outside-library | A3 - A2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        producer = "spectral_etdrk4"
        values = {arm: score["summaries"][f"{model}::{arm}::{producer}"] for arm in ARMS}
        lines.append(
            f"| {SHORT[model]} | {values['A1']['false_promotions_in_library']}/180 ({100*values['A1']['false_promotion_rate_in_library']:.2f}%) | "
            f"{values['A2']['false_promotions_in_library']}/180 ({100*values['A2']['false_promotion_rate_in_library']:.2f}%) | "
            f"{values['A3']['false_promotions_in_library']}/180 ({100*values['A3']['false_promotion_rate_in_library']:.2f}%) | "
            f"{values['A5']['false_promotions_in_library']}/180 ({100*values['A5']['false_promotion_rate_in_library']:.2f}%) | "
            f"{score['A3_minus_A2'][f'{model}::{producer}']['A3_minus_A2_false_promotion_pp']:+.2f} pp |"
        )
    lines += [
        "",
        "Paired A2/A3 discordance contains no A2-only false case. Opus adds 11 A3-only false cases, Sonnet 6, and Haiku 9; the remaining false cases are shared by A2 and A3.",
        "",
        "## Committed-view distributions",
        "",
        "Counts below are for the spectral component; FD/BDF distributions are identical except that one Opus A1 case is A02 for spectral and A03 for FD/BDF. Geometry is `(temporal nodes, temporal modes)`.",
        "",
        "| Model | Arm | Valid | View counts |",
        "|---|---|---:|---|",
    ]
    for model in MODELS:
        for arm in ARMS:
            value = score["summaries"][f"{model}::{arm}::spectral_etdrk4"]
            counts = ", ".join(f"{key}:{count}" for key, count in value["view_distribution_valid"].items())
            lines.append(f"| {SHORT[model]} | {arm} | {value['valid_tool_calls']}/216 | {counts} |")
    lines += [
        "",
        "## M5 capability joined to action choice",
        "",
        "Only Sonnet has a complete 216-case M5 join. It names A03 least trustworthy on 158/216 cases for each producer.",
        "",
        "| Sonnet arm | M5 A03, selects A03 | M5 A03, avoids A03 | M5 not A03, selects A03 | Neither |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        table = score["m5_choice_joins"][f"claude-sonnet-4-6::{arm}::spectral_etdrk4"]["table"]
        lines.append(
            f"| {arm} | {table['m5_A03__arm_A03']} | {table['m5_A03__arm_not_A03']} | "
            f"{table['m5_not_A03__arm_A03']} | {table['m5_not_A03__arm_not_A03']} |"
        )
    lines += [
        "",
        "Among the 158 cases where Sonnet explicitly names A03 least trustworthy, it nevertheless selects A03 in 141 under A1 (89.24%) and 154 under A5 (97.47%), versus 9 under A2 (5.70%) and 19 under A3 (12.03%). This is an observable capability-incentive inversion, not evidence of deception or hidden intent.",
        "",
        "## Claim ceiling",
        "",
        "These are finite synthetic, same-lineage results for three specified model endpoints and one frozen prompt/action interface. They do not establish a model population claim, hidden reasoning, deception, physical-world harm, or safety outside this benchmark. Opus and Haiku M5 failures also mean the complete capability-choice join is a Sonnet result, not a three-model replication.",
        "",
        f"Machine-readable summary content SHA-256: `{full['content_sha256']}`.",
    ]
    args.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary_content_sha256": full["content_sha256"], "report": str(args.report_md)}, indent=2))


if __name__ == "__main__":
    main()
