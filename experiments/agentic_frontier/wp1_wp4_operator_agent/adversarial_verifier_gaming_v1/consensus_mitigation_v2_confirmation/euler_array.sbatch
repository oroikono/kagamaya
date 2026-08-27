#!/bin/bash
#SBATCH --job-name=cons-v2-confirm
#SBATCH --partition=normal.4h
#SBATCH --time=01:00:00
#SBATCH --array=0-99%20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

set -euo pipefail
: "${CONSENSUS_V2_ROOT:?CONSENSUS_V2_ROOT must be set}"
cd "$CONSENSUS_V2_ROOT/repo"
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [ "$SLURM_ARRAY_TASK_ID" -lt 50 ]; then
  producer="spectral_etdrk4"
  seed=$((10200 + SLURM_ARRAY_TASK_ID))
else
  producer="fd_bdf"
  seed=$((10200 + SLURM_ARRAY_TASK_ID - 50))
fi

tag="${producer}-${seed}"
out="$CONSENSUS_V2_ROOT/results/shards/${tag}.json.gz"
python_bin="/cluster/scratch/ooikonomou/verisigs-qwen-env/bin/python"
module="experiments.agentic_frontier.wp1_wp4_operator_agent.adversarial_verifier_gaming_v1.consensus_mitigation_v2_confirmation"

test ! -e "$out"
"$python_bin" -m "$module.confirmation" --producer "$producer" --seed "$seed" --output "$out"
"$python_bin" -m "$module.validate" "$out"
