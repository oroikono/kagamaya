#!/bin/bash
#SBATCH --job-name=cons-v2-aggregate
#SBATCH --partition=normal.4h
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/slurm-aggregate-%j.out
#SBATCH --error=logs/slurm-aggregate-%j.err

set -euo pipefail
: "${CONSENSUS_V2_ROOT:?CONSENSUS_V2_ROOT must be set}"
cd "$CONSENSUS_V2_ROOT/repo"
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1

python_bin="/cluster/scratch/ooikonomou/verisigs-qwen-env/bin/python"
module="experiments.agentic_frontier.wp1_wp4_operator_agent.adversarial_verifier_gaming_v1.consensus_mitigation_v2_confirmation"
output="$CONSENSUS_V2_ROOT/results/confirmation_aggregate.json.gz"

test ! -e "$output"
"$python_bin" -m "$module.aggregate" --input-dir "$CONSENSUS_V2_ROOT/results/shards" --output "$output"
"$python_bin" -m "$module.validate" --aggregate "$output"
