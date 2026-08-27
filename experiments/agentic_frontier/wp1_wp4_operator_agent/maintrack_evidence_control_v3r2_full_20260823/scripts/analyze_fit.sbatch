#!/bin/bash
#SBATCH --job-name=evctrl_v3r2_fit
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12G
#SBATCH --tmp=8G
#SBATCH --output=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2/logs/analyze_fit_%j.out
#SBATCH --error=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2/logs/analyze_fit_%j.err
set -euo pipefail
SOURCE_CLOSURE="$1"
ROOT=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2
REPO="$ROOT/repo"
PY=/cluster/scratch/ooikonomou/verisigs_external_weakpde_v1/venv/bin/python
cd "$REPO"
test "$(find "$ROOT/results/train" -name 'shard_*.json.gz' | wc -l)" -eq 80
test "$(find "$ROOT/results/calibration" -name 'shard_*.json.gz' | wc -l)" -eq 40
PYTHONPATH="$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_evidence_control_v3r2_full_20260823.freeze --verify
PYTHONPATH="$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_evidence_control_v3r2_full_20260823.dominance \
  --shards "$ROOT/results/calibration" --output "$ROOT/results/calibration_dominance.json"
PYTHONPATH="$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_evidence_control_v3r2_full_20260823.learning \
  --train-shards "$ROOT/results/train" --calibration-shards "$ROOT/results/calibration" \
  --dominance "$ROOT/results/calibration_dominance.json" --source-closure "$SOURCE_CLOSURE" \
  --output-dir "$ROOT/results/model_receipt"
PYTHONPATH="$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_evidence_control_v3r2_full_20260823.freeze_confirmation \
  --dominance "$ROOT/results/calibration_dominance.json" \
  --receipt "$ROOT/results/model_receipt/receipt.json" --source-closure "$SOURCE_CLOSURE" \
  --output "$ROOT/results/CONFIRMATION_AUTHORIZATION.json"
