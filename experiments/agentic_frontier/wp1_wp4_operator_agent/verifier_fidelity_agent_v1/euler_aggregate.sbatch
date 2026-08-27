#!/bin/bash
#SBATCH --job-name=fidelity-agg
#SBATCH --partition=normal.24h
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/aggregate-%j.out
#SBATCH --error=logs/aggregate-%j.err

set -euo pipefail
: "${FIDELITY_ROOT:?FIDELITY_ROOT must point to staged repository root}"
cd "$FIDELITY_ROOT"
export PYTHONPATH="$FIDELITY_ROOT"

/cluster/scratch/ooikonomou/verisigs-qwen-env/bin/python -m \
  experiments.agentic_frontier.wp1_wp4_operator_agent.verifier_fidelity_agent_v1.aggregate \
  --shard-dir results/shards --output results/confirmation_50seeds.json
