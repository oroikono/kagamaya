#!/bin/bash
#SBATCH --job-name=avg-v1-confirm
#SBATCH --partition=normal.24h
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail
: "${AVG_V1_ROOT:?AVG_V1_ROOT must point to the staged repository root}"

cd "$AVG_V1_ROOT"
export PYTHONPATH="$AVG_V1_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  producer="spectral_etdrk4"
  tag="spectral"
else
  producer="fd_bdf"
  tag="fd"
fi

seeds="$(seq -s, 10000 10049)"
out="experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/results/confirmation"
mkdir -p "$out/logs"
python_bin="/cluster/scratch/ooikonomou/verisigs-qwen-env/bin/python"
module="experiments.agentic_frontier.wp1_wp4_operator_agent.adversarial_verifier_gaming_v1"

"$python_bin" -m "$module.campaign" \
  --producer "$producer" --split confirmation --seeds "$seeds" \
  --public-output "$out/${tag}_public.json.gz" \
  --owner-output "$out/${tag}_owner.json.gz"

"$python_bin" -m "$module.run_non_llm" \
  --public "$out/${tag}_public.json.gz" \
  --output "$out/${tag}_proposals.json.gz"

"$python_bin" -m "$module.score" \
  --public "$out/${tag}_public.json.gz" \
  --owner "$out/${tag}_owner.json.gz" \
  --proposals "$out/${tag}_proposals.json.gz" \
  --output "$out/${tag}_score.json.gz"
