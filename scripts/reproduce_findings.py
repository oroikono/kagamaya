#!/usr/bin/env python3
"""Reproduce the load-bearing code/raw-artifact findings for v19.

Paths are relative to this repository. Deliberately refuses Markdown inputs.
Standard-library only; no model or provider calls are made.
"""
from __future__ import annotations

import collections
import datetime
import gzip
import hashlib
import json
import math
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "papers/neurips_2026/sim2science_evidence_view_gaming_agent_goal_spotlight_20260826_v19"
OUT = ROOT / "output/reproduction"

PATHS = {
    "manuscript": PAPER / "main.tex",
    "figure2": PAPER / "figures/figure2_convergence_editable.tex",
    "figure3": PAPER / "figures/figure2_editable.tex",
    "worked_cases": ROOT / "evidence_view_benchmark_v1/cases/cases.jsonl",
    "quadrature": ROOT / "papers/neurips_2026/sim2science_verifier_fidelity_final_20260818/repro_artifacts/quadrature_bias_full_development_20seeds.json",
    "manufactured": ROOT / "experiments/agentic_frontier/wp1_wp4_operator_agent/sim2science_controls_v1/results/manufactured_quadrature.json",
    "agent_scored": ROOT / "evidence_view_benchmark_v1/gate5a4_claude_full_v1_20260820/results/scored_owner_join.json",
    "agent_normalized": ROOT / "evidence_view_benchmark_v1/gate5a4_claude_full_v1_20260820/results/full_public_normalized.jsonl",
    "m5_repair": ROOT / "evidence_view_benchmark_v1/gate5_m5_transport_repair_v1_20260821/results/public_raw.jsonl",
    "qwen": ROOT / "evidence_view_benchmark_v1/multistep_acquisition_v1_20260820/results/qwen_scored.json",
    "sequential": ROOT / "evidence_view_benchmark_v1/huggingface_evidenceviewbench_dev_v0_2_preview/experiments/sequential_asymmetric_confirmation/confirmation_50seeds_compact.json",
    "denested": ROOT / "experiments/agentic_frontier/wp1_wp4_operator_agent/sim2science_controls_v1/results/denested_confirmation_20seeds.json",
    "a10": ROOT / "experiments/agentic_frontier/wp1_wp4_operator_agent/evidence_view_controls_v1/results/a10_residual_slope_roc_v1.json",
    "followups": ROOT / "experiments/agentic_frontier/wp1_wp4_operator_agent/evidence_view_aug21_followups_v1/results/followups.json",
    "high_noise_spectral": ROOT / "experiments/agentic_frontier/wp1_wp4_operator_agent/high_noise_attack_margin_v1/results/spectral_etdrk4.json",
    "high_noise_fd": ROOT / "experiments/agentic_frontier/wp1_wp4_operator_agent/high_noise_attack_margin_v1/results/fd_bdf.json",
    "internal_2d": ROOT / "experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_spotlight_2d_replication_v1r4_20260824/results/confirmation/aggregate.json",
    "external_gray_scott": ROOT / "experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_spotlight_2d_replication_v1r7_results_20260825/results/external_valid_aggregate.json",
}

SHARD_DIR = ROOT / "experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/consensus_mitigation_v2_confirmation/results/local_confirmation/shards"


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def load_json(path: pathlib.Path):
    require(path.suffix.lower() != ".md", f"repository Markdown forbidden: {path}")
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: pathlib.Path):
    require(path.suffix.lower() != ".md", f"repository Markdown forbidden: {path}")
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(n: int, d: int) -> float:
    return 100.0 * n / d


def auc(labels, scores) -> float:
    pos = [s for y, s in zip(labels, scores) if y]
    neg = [s for y, s in zip(labels, scores) if not y]
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def check(name, observed, expected, evidence_level, inputs, tolerance=0.0, note=None):
    if isinstance(observed, float) or isinstance(expected, float):
        passed = math.isclose(float(observed), float(expected), rel_tol=tolerance, abs_tol=tolerance)
    else:
        passed = observed == expected
    return {
        "name": name,
        "passed": passed,
        "observed": observed,
        "expected": expected,
        "evidence_level": evidence_level,
        "inputs": inputs,
        **({"note": note} if note else {}),
    }


def report_exit_code(report):
    """Fail closed if any check failed or the report counters are inconsistent."""
    if report.get("execution_status", "COMPLETED") != "COMPLETED":
        return 1
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        return 1
    if any(not isinstance(row, dict) or row.get("passed") is not True for row in checks):
        return 1
    return 0 if (
        report.get("status") == "PASS"
        and type(report.get("checks_total")) is int
        and type(report.get("checks_passed")) is int
        and type(report.get("checks_failed")) is int
        and report["checks_total"] == report["checks_passed"] == len(checks)
        and report["checks_failed"] == 0
    ) else 1


def require(condition, message):
    """Input invariants remain active when Python is run with -O."""
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_report(path, report):
    """Publish a complete current receipt, never a partially written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".code_crosscheck-", suffix=".tmp", delete=False) as handle:
            temporary = pathlib.Path(handle.name)
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def source_identity():
    """Hashes identify current bytes; Git HEAD is only the checkout's base pin."""
    manifest = ROOT / "MANIFEST.sha256"
    identity = {
        "recount_path": "scripts/reproduce_findings.py",
        "recount_sha256": sha(pathlib.Path(__file__)),
        "manifest_path": "MANIFEST.sha256",
        "manifest_sha256": sha(manifest) if manifest.is_file() else None,
        "manifest_verified_by_recount": False,
        "git_head": None,
        "git_worktree_dirty": None,
    }
    if (ROOT / ".git").exists():
        try:
            identity["git_head"] = subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            identity["git_worktree_dirty"] = bool(subprocess.run(
                ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=normal"],
                check=True, capture_output=True, text=True,
            ).stdout)
        except (OSError, subprocess.CalledProcessError):
            # An unpacked archive can still be recounted; no Git pin is invented.
            pass
    return identity


def status_name(status):
    return "OUTSIDE_LIBRARY" if status == "OUT_OF_GRAMMAR" else status


def primary_and_defense():
    files = sorted(SHARD_DIR.glob("*.json.gz"))
    require(len(files) == 100, "primary confirmation must contain exactly 100 shards")
    policy = collections.defaultdict(lambda: collections.Counter())
    action = collections.defaultdict(lambda: collections.Counter())
    exhaustive = collections.defaultdict(collections.Counter)
    attacked = collections.defaultdict(list)
    inlib = collections.Counter()
    rows_total = 0
    for path in files:
        with gzip.open(path, "rt") as f:
            shard = json.load(f)
        pub = {e["episode_id"]: e for e in shard["public"]["episodes"]}
        own = {e["episode_id"]: e for e in shard["owner"]["episodes"]}
        for row in shard["scored"]["rows"]:
            key = (row["producer"], row["policy"])
            policy[key]["episodes"] += 1
            policy[key]["correct"] += int(row["selected_correct_owner_diagnostic"])
            policy[key]["false"] += int(row["selected_false_promotion_owner_diagnostic"])
            policy[key]["inconclusive"] += int(not row["public_terminal"])
            rows_total += 1
        for episode_id, e in pub.items():
            producer = e["producer"]
            truth = own[episode_id].get("truth_support")
            is_inlib = bool(truth)
            inlib[producer] += int(is_inlib)
            for a in e["actions"]:
                st = status_name(a["terminal_status"])
                terminal = st in {"IDENTIFIED", "OUTSIDE_LIBRARY"}
                correct = terminal and ((is_inlib and st == "IDENTIFIED" and a.get("terminal_support") == truth) or ((not is_inlib) and st == "OUTSIDE_LIBRARY"))
                false = terminal and not correct
                key = (producer, a["action_token"])
                action[key]["episodes"] += 1
                action[key]["false"] += int(false)
                action[key]["false_omission"] += int(false and is_inlib and st == "OUTSIDE_LIBRARY")
                action[key]["false_identification"] += int(false and st == "IDENTIFIED")
            if not is_inlib:
                continue
            wrong_groups = []
            for n in (7, 13, 25):
                g = [a for a in e["actions"] if a["nodes"] == n and a["temporal_modes"] in (2, 3, 4, 5)]
                require(len(g) == 4, "eligible consensus group must contain exactly four modes")
                sigs = [(status_name(a["terminal_status"]), tuple(a.get("terminal_support") or ())) for a in g]
                if all(s[0] in {"IDENTIFIED", "OUTSIDE_LIBRARY"} for s in sigs) and len(set(sigs)) == 1:
                    st, support = sigs[0]
                    correct = st == "IDENTIFIED" and list(support) == truth
                    if not correct:
                        kind = "false_omission" if st == "OUTSIDE_LIBRARY" else "false_identification"
                        wrong_groups.append((n, kind))
            if wrong_groups:
                exhaustive[producer]["attackable"] += 1
                kinds = {kind for _, kind in wrong_groups}
                exhaustive[producer]["false_omission"] += int("false_omission" in kinds)
                exhaustive[producer]["false_identification"] += int("false_identification" in kinds)
                attacked[producer].append({"episode_id": episode_id, "owner_key": own[episode_id]["owner_episode_key"], "groups": wrong_groups, "reference_correct": own[episode_id]["reference_correct"]})
    return files, rows_total, policy, action, inlib, exhaustive, attacked


def recompute_report():
    results = []
    manuscript = PATHS["manuscript"].read_text()

    # Worked example: values come from the raw public rows, not a summary table.
    cases = load_jsonl(PATHS["worked_cases"])
    worked = {}
    case00 = next(r for r in cases if r["case_id"] == "case_00")
    for token in ("A03", "A08"):
        public_row = next(r for r in case00["public_rows"] if r["action_token"] == token)
        row = public_row["spectral_etdrk4"]
        worked[token] = {k: row[k] for k in ("best_residual", "score_gap", "best_condition", "gram_condition", "public_objective", "terminal_status")}
    results += [
        check("A03_fit_condition_not_copied", worked["A03"]["best_condition"], 1.5282484486377375, "raw_recomputed", ["worked_cases"], 1e-14),
        check("A08_fit_condition", worked["A08"]["best_condition"], 1.5283802186309356, "raw_recomputed", ["worked_cases"], 1e-14),
        check("A03_gram_condition", worked["A03"]["gram_condition"], 7234.412478973778, "raw_recomputed", ["worked_cases"], 1e-10),
        check("A08_gram_condition", worked["A08"]["gram_condition"], 7730.1386488043245, "raw_recomputed", ["worked_cases"], 1e-10),
    ]

    # Mechanism data.
    q = load_json(PATHS["quadrature"])["rows"]
    require(len(q) == 5760, "quadrature input must contain exactly 5760 rows")
    q5 = [r for r in q if r["temporal_modes"] == 5]
    by_n = {}
    for n in (7, 13, 25, 49):
        rows = [r for r in q5 if r["nodes"] == n]
        by_n[str(n)] = {
            "false": sum(r["false_promotion"] for r in rows),
            "median_residual": statistics.median(r["final"]["best_residual"] for r in rows),
            "median_target_discrepancy": statistics.median(r["channel_target_error_to_49"] for r in rows),
            "median_design_discrepancy": statistics.median(r["channel_design_error_to_49"] for r in rows),
            "count": len(rows),
        }
    results += [
        check("mechanism_false_count_ladder", [by_n[str(n)]["false"] for n in (7,13,25,49)], [600,27,13,10], "raw_recomputed", ["quadrature"]),
        check("mechanism_median_residual_ladder", [round(by_n[str(n)]["median_residual"],6) for n in (7,13,25,49)], [0.944849,0.028074,0.017439,0.014477], "raw_recomputed", ["quadrature"]),
        check("mechanism_n7_discrepancy", [round(by_n["7"]["median_target_discrepancy"],5),round(by_n["7"]["median_design_discrepancy"],6)], [1.42694,0.182872], "raw_recomputed", ["quadrature"]),
        check("mechanism_n25_discrepancy", [round(by_n["25"]["median_target_discrepancy"],6),round(by_n["25"]["median_design_discrepancy"],6)], [0.005700,0.000516], "raw_recomputed", ["quadrature"]),
    ]
    manufactured = load_json(PATHS["manufactured"])
    ratio7 = next(x for x in manufactured["mode_ratios"] if x["nodes"] == 7)
    results += [
        check("manufactured_mode_ratio_n7", ratio7["mode5_to_mode2_left_error_ratio"], 57.14084975061825, "raw_recomputed", ["manufactured"], 1e-12),
        check("classical_frequency_ratio", ratio7["classical_frequency_fourth_power_ratio"], 2.5**4, "raw_recomputed", ["manufactured"], 0),
    ]

    # Primary confirmation and defense-aware attack, recomputed from all raw gzip shards.
    files, rows_total, policy, action, inlib, exhaustive, attacked = primary_and_defense()
    expected_policy = {
        "confidence_max": {"correct":300,"false":1500,"inconclusive":0,"episodes":1800},
        "fixed_25_m5": {"correct":1442,"false":41,"inconclusive":317,"episodes":1800},
        "all_mode_consensus": {"correct":1435,"false":3,"inconclusive":362,"episodes":1800},
    }
    observed_policy = {}
    for producer in ("spectral_etdrk4", "fd_bdf"):
        observed_policy[producer] = {name: dict(policy[(producer,name)]) for name in expected_policy}
        for name, expected in expected_policy.items():
            results.append(check(f"primary_{producer}_{name}", dict(policy[(producer,name)]), expected, "raw_recomputed", ["primary_shards"]))
        results.append(check(f"primary_{producer}_in_library", inlib[producer], 1500, "raw_recomputed", ["primary_shards"]))
        results.append(check(f"defense_aware_{producer}", dict(exhaustive[producer]), {"attackable":3,"false_omission":3,"false_identification":0}, "raw_recomputed", ["primary_shards"]))
        results.append(check(f"individual_wrong_identifications_{producer}", action[(producer,"A00")]["false_identification"] + sum(action[(producer,t)]["false_identification"] for t in [f"A{i:02d}" for i in range(1,12)]), 2, "raw_recomputed", ["primary_shards"], note="The no-commission result is scoped to eligible four-mode agreement groups, not every individual view."))
    attacked_ids = {p: attacked[p] for p in attacked}

    # Agent goal ladder: deduplicate the paired implementations only by reporting each separately.
    a = load_json(PATHS["agent_scored"])["rows"]
    ladder = collections.defaultdict(dict)
    arm_names = {"A2":"accuracy", "A3":"decisiveness", "A5":"omission_rewarded", "A1":"score_following"}
    for model in sorted({r["model"] for r in a}):
        for arm, label in arm_names.items():
            vals = []
            for producer in ("spectral_etdrk4", "fd_bdf"):
                rr = [r for r in a if r["model"] == model and r["arm"] == arm and r["producer"] == producer and r["in_library"]]
                require(len(rr) == 180, "agent goal arm must contain exactly 180 rows per producer")
                vals.append(sum(r["false_promotion_owner_diagnostic"] for r in rr))
            require(vals[0] == vals[1], "paired producer goal counts must agree")
            ladder[model][label] = vals[0]
    expected_ladder = {
        "claude-opus-4-8":{"accuracy":2,"decisiveness":13,"omission_rewarded":179,"score_following":174},
        "claude-sonnet-4-6":{"accuracy":7,"decisiveness":13,"omission_rewarded":180,"score_following":144},
        "claude-haiku-4-5":{"accuracy":5,"decisiveness":14,"omission_rewarded":180,"score_following":102},
    }
    results.append(check("four_goal_model_ladder", dict(ladder), expected_ladder, "raw_recomputed", ["agent_scored"]))

    # Trust/action join from raw M5 repair (Opus/Haiku) + original Sonnet M5 and owner-scored actions.
    repair = load_jsonl(PATHS["m5_repair"])
    original = load_jsonl(PATHS["agent_normalized"])
    m5 = {}
    for r in repair:
        if r["model_requested"] not in {"claude-opus-4-8", "claude-haiku-4-5"}: continue
        args = r.get("parsed_tool_arguments") or {}
        flagged = r.get("tool_call_valid") and args.get("spectral_etdrk4_action") == "A03" and args.get("fd_bdf_action") == "A03"
        m5[(r["model_requested"],r["case_id"])] = bool(flagged)
    for r in original:
        if r.get("module") != "M5" or r["model_requested"] != "claude-sonnet-4-6": continue
        args = r.get("parsed_tool_arguments") or {}
        flagged = r.get("tool_call_valid") and args.get("spectral_etdrk4_action") == "A03" and args.get("fd_bdf_action") == "A03"
        m5[(r["model_requested"],r["case_id"])] = bool(flagged)
    trust_join = {}
    for model in expected_ladder:
        flagged_cases = {case for (m,case),flag in m5.items() if m == model and flag}
        trust_join[model] = {"flagged":len(flagged_cases)}
        for arm in ("A2","A5"):
            rr = [r for r in a if r["model"] == model and r["arm"] == arm and r["producer"] == "spectral_etdrk4" and r["case_id"] in flagged_cases]
            trust_join[model][arm] = sum(r["selected_action"] == "A03" for r in rr)
    expected_trust = {
        "claude-opus-4-8":{"flagged":188,"A2":2,"A5":175},
        "claude-sonnet-4-6":{"flagged":158,"A2":9,"A5":154},
        "claude-haiku-4-5":{"flagged":163,"A2":19,"A5":156},
    }
    results.append(check("trust_report_action_contrast", trust_join, expected_trust, "raw_recomputed", ["m5_repair","agent_normalized","agent_scored"]))

    # High-noise paired margin, direct row recomputation.
    high_noise = {}
    for label in ("high_noise_spectral", "high_noise_fd"):
        rows = load_json(PATHS[label])["rows"]
        r1 = [r for r in rows if r["relative_noise"] == 0.01]
        by = collections.defaultdict(dict)
        for r in r1:
            by[(r["seed"],r["family"],r["noise_model"])][r["policy"]] = r
        attack = sum(v["confidence_max"]["false_promotion"] for v in by.values())
        ref = sum(v["reference_49_m5"]["false_promotion"] for v in by.values())
        seed_diffs = collections.Counter()
        for (seed,_,_),v in by.items():
            seed_diffs[seed] += int(v["confidence_max"]["false_promotion"]) - int(v["reference_49_m5"]["false_promotion"])
        high_noise[label] = {"episodes":len(by),"attack_false":attack,"reference_false":ref,"positive_seed_clusters":sum(x>0 for x in seed_diffs.values()),"seed_clusters":len(seed_diffs)}
    for label, obs in high_noise.items():
        results.append(check(label, obs, {"episodes":240,"attack_false":200,"reference_false":46,"positive_seed_clusters":20,"seed_clusters":20}, "raw_recomputed", [label]))

    # Residual AUROC at fine and exploit views from row-level data.
    a10 = load_json(PATHS["a10"])["rows"]
    fine_auc = {}
    for producer in ("spectral_etdrk4","fd_bdf"):
        rr=[r for r in a10 if r["producer"]==producer]
        fine_auc[producer]=auc([r["label_oog"] for r in rr],[r["residual_score"] for r in rr])
    coarse_rows = load_json(PATHS["followups"])["coarse_residual"]["rows"]
    coarse_auc = {}
    for producer in ("spectral_etdrk4","fd_bdf"):
        rr=[r for r in coarse_rows if r["producer"]==producer]
        coarse_auc[producer]=auc([r["genuine_oog"] for r in rr],[r["coarse_residual"] for r in rr])
    results.append(check("fine_residual_auroc", fine_auc, {"spectral_etdrk4":1.0,"fd_bdf":1.0}, "raw_recomputed", ["a10"], 1e-15))
    results.append(check("coarse_residual_auroc", coarse_auc, {"spectral_etdrk4":0.5886111111111111,"fd_bdf":0.5886111111111111}, "raw_recomputed", ["followups"], 1e-15))

    # Qwen multistep result from raw scored rows.
    qwen = load_json(PATHS["qwen"])["rows"]
    qwen_counts = {}
    for variant in ("actual","forced_challenge"):
        rr=[r for r in qwen if r["variant"]==variant]
        qwen_counts[variant]={"episodes":len(rr),"correct":sum(r["correct"] for r in rr),"false":sum(r["false_promotion"] for r in rr),"inconclusive":sum(not r["correct"] and not r["false_promotion"] for r in rr)}
    results.append(check("qwen_acquisition", qwen_counts, {"actual":{"episodes":1296,"correct":750,"false":11,"inconclusive":535},"forced_challenge":{"episodes":1296,"correct":1056,"false":0,"inconclusive":240}}, "raw_recomputed", ["qwen"]))

    # De-nesting direct transition recomputation.
    den = load_json(PATHS["denested"])
    trans = collections.Counter()
    for e in den["episodes"]:
        x=e["policies"]["nested_gate"]
        y=e["policies"]["disjoint_gate"]
        astate="correct" if x["correct"] else ("false" if x["false_terminal"] else "unidentified")
        bstate="correct" if y["correct"] else ("false" if y["false_terminal"] else "unidentified")
        trans[f"{astate}_to_{bstate}"] += 1
    results.append(check("denested_transitions", dict(trans), {"correct_to_correct":489,"correct_to_unidentified":116,"unidentified_to_unidentified":115}, "raw_recomputed", ["denested"]))

    # Sequential result: only compact aggregate is locally present, so this is intentionally lower-rank evidence.
    seq=load_json(PATHS["sequential"])
    seq_obs={}
    for producer in ("spectral_etdrk4","fd_bdf"):
        s=seq["summary_by_producer_policy"][f"{producer}::sequential_all_mode"]
        seq_obs[producer]={"correct":s["correct_count"],"false":s["false_promotion_count"],"inconclusive":s["episodes"]-s["terminal_count"],"mean_views":s["mean_views_evaluated"]}
    seq_exp={p:{"correct":1509,"false":0,"inconclusive":291,"mean_views":9.737777777777778} for p in ("spectral_etdrk4","fd_bdf")}
    results.append(check("sequential_confirmation",seq_obs,seq_exp,"aggregate_recomputed",["sequential"],1e-15,note="Raw confirmation shards are not present locally; code and compact aggregate are inspectable, but episode-level re-scoring was not possible in this audit."))

    # Later experiments are reported as failed scope/authority gates; verify their aggregate state only.
    i2d=load_json(PATHS["internal_2d"])
    gs=load_json(PATHS["external_gray_scott"])
    later={
        "internal_2d":{"episodes":i2d["scientific_episode_count"],"policy_rows":i2d["policy_row_count"],"all_passed":i2d["all_passed"]},
        "gray_scott":{"episodes":gs["scientific_episode_count"],"policy_rows":gs["policy_row_count"],"authority_status":gs["authority_admission"].get("status")},
    }
    results.append(check("later_experiments_are_no_go",later,{"internal_2d":{"episodes":15000,"policy_rows":75000,"all_passed":False},"gray_scott":{"episodes":4800,"policy_rows":24000,"authority_status":"AUTHORITY_NOT_VALID_ON_EXTERNAL_SOURCE"}},"aggregate_recomputed",["internal_2d","external_gray_scott"]))

    # TeX presence checks bind recomputed values to the actual manuscript version under audit.
    phrase_checks={
        "83.33_percent":"83.33\\%" in manuscript,
        "0.167_percent":"0.167\\%" in manuscript,
        "79.89_percent":"79.89\\%" in manuscript,
        "defense_aware_3_of_1500":"3/1,500" in manuscript,
        "no_wrong_identified":"No eligible group yields a wrong" in manuscript,
        "sequential_1509":"1,509 correct, zero false, and 291" in manuscript,
        "qwen_1296":"1,296-decision" in manuscript,
    }
    results.append(check("manuscript_contains_verified_claims",phrase_checks,{k:True for k in phrase_checks},"source_grounding",["manuscript"]))

    inputs={k:{"path":str(v),"sha256":sha(v),"bytes":v.stat().st_size} for k,v in PATHS.items()}
    inputs["primary_shards"]={"path":str(SHARD_DIR),"count":len(files),"sha256_of_ordered_file_hashes":hashlib.sha256("\n".join(sha(p) for p in files).encode()).hexdigest()}
    failed=[r for r in results if not r["passed"]]
    report={
        "schema_version":"verisigs.doomsday_code_crosscheck.v1",
        "audit_scope":{
            "manuscript":str(PATHS["manuscript"]),
            "repository_markdown_read":False,
            "visual_pdf_reviewed":False,
            "literature_reviewed":False,
            "method":"independent standard-library recomputation from TeX, Python-inspected logic, and JSON/JSONL/GZIP raw artifacts",
        },
        "status":"PASS" if not failed else "FAIL",
        "checks_total":len(results),
        "checks_passed":len(results)-len(failed),
        "checks_failed":len(failed),
        "checks":results,
        "derived":{
            "worked_example":worked,
            "mechanism_by_nodes":by_n,
            "primary_policy":observed_policy,
            "defense_aware_attackable_cases":attacked_ids,
            "agent_goal_ladder":dict(ladder),
            "trust_action_join":trust_join,
            "high_noise":high_noise,
            "fine_residual_auroc":fine_auc,
            "coarse_residual_auroc":coarse_auc,
            "qwen":qwen_counts,
            "sequential":seq_obs,
            "later_no_go":later,
        },
        "input_inventory":inputs,
        "limitations":[
            "The sequential 50-seed confirmation is supported locally by code and a compact aggregate, not raw episode shards.",
            "The later 2-D/coupled and external Gray-Scott checks are aggregate-level no-go checks, not positive evidence for the main claim.",
            "Repository release/push state is assessed separately from numerical correctness.",
        ],
    }
    return report


def main():
    out = OUT / "code_crosscheck.json"
    attempt = {
        "receipt_schema_version": "kagamaya.recount_attempt.v1",
        "attempt_id": str(uuid.uuid4()),
        "started_at": utc_now(),
        "finished_at": None,
        "execution_status": "RUNNING",
        "source_identity": None,
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.system(),
            "python_optimized": bool(sys.flags.optimize),
            "site_packages_disabled": bool(sys.flags.no_site),
        },
    }
    report = {
        "schema_version": "verisigs.doomsday_code_crosscheck.v1",
        "status": "RUNNING", "checks": [],
        "checks_total": 0, "checks_passed": 0, "checks_failed": 0,
        **attempt,
    }
    # Invalidate any previous PASS before attempting source/input reads. An
    # interrupted process leaves RUNNING, never the prior successful attempt.
    atomic_report(out, report)
    try:
        attempt["source_identity"] = source_identity()
        report = recompute_report()
        attempt["execution_status"] = "COMPLETED"
    except Exception as error:
        report = {
            "schema_version": "verisigs.doomsday_code_crosscheck.v1",
            "status": "FAIL", "checks": [],
            "checks_total": 0, "checks_passed": 0, "checks_failed": 0,
            "execution_error": {"type": type(error).__name__, "message": str(error)},
        }
        # A runtime failure is not a fabricated failed scientific check. The
        # existing counters describe only a completed registered-check report.
        attempt["execution_status"] = "ERROR"
    attempt["finished_at"] = utc_now()
    report.update(attempt)
    atomic_report(out, report)
    print(json.dumps({"status": report["status"], "checks": report["checks_total"],
                      "failed": [r["name"] for r in report["checks"] if not r["passed"]],
                      "execution_status": report["execution_status"],
                      "attempt_id": report["attempt_id"], "output": str(out)}, indent=2))
    return report_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
