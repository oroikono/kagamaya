#!/bin/bash
#SBATCH --job-name=evctrl_v3r2_eval
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=8G
#SBATCH --output=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2/logs/eval_%j.out
#SBATCH --error=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2/logs/eval_%j.err
set -euo pipefail
ROOT=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2
REPO="$ROOT/repo"
PY=/cluster/scratch/ooikonomou/verisigs_external_weakpde_v1/venv/bin/python
cd "$REPO"
test "$(find "$ROOT/results/confirmation" -name 'shard_*.json.gz' | wc -l)" -eq 200
PYTHONPATH="$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_evidence_control_v3r2_full_20260823.evaluate_confirmation \
  --shards "$ROOT/results/confirmation" \
  --dominance "$ROOT/results/calibration_dominance.json" \
  --receipt "$ROOT/results/model_receipt/receipt.json" \
  --authorization "$ROOT/results/CONFIRMATION_AUTHORIZATION.json" \
  --output "$ROOT/results/CONFIRMATION_RESULT.json"
