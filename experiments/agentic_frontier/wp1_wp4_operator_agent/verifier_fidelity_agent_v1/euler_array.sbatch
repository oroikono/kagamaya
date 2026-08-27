#!/bin/bash
#SBATCH --job-name=fidelity-v1
#SBATCH --partition=normal.24h
#SBATCH --time=05:00:00
#SBATCH --array=0-99%40
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/shard-%A_%a.out
#SBATCH --error=logs/shard-%A_%a.err

set -euo pipefail
: "${FIDELITY_ROOT:?FIDELITY_ROOT must point to staged repository root}"
cd "$FIDELITY_ROOT"
export PYTHONPATH="$FIDELITY_ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [ "$SLURM_ARRAY_TASK_ID" -lt 50 ]; then
  producer="spectral_etdrk4"
  seed=$((9700 + SLURM_ARRAY_TASK_ID))
else
  producer="fd_bdf"
  seed=$((9700 + SLURM_ARRAY_TASK_ID - 50))
fi

/cluster/scratch/ooikonomou/verisigs-qwen-env/bin/python -m \
  experiments.agentic_frontier.wp1_wp4_operator_agent.verifier_fidelity_agent_v1.campaign \
  --producer "$producer" --split confirmation --seeds "$seed" \
  --output "results/shards/${producer}_${seed}.json"

/cluster/scratch/ooikonomou/verisigs-qwen-env/bin/python -m \
  experiments.agentic_frontier.wp1_wp4_operator_agent.verifier_fidelity_agent_v1.validate \
  "results/shards/${producer}_${seed}.json"
