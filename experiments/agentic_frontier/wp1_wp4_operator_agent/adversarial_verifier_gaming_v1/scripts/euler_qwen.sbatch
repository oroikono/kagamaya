#!/bin/bash
#SBATCH --job-name=avg-v1-qwen
#SBATCH --partition=gpupr.4h
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-qwen-%j.out
#SBATCH --error=slurm-qwen-%j.err

set -euo pipefail
: "${AVG_V1_ROOT:?AVG_V1_ROOT must point to the staged repository root}"

cd "$AVG_V1_ROOT"
export PYTHONPATH="$AVG_V1_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python_bin="/cluster/scratch/ooikonomou/verisigs-qwen-minenv/bin/python"
model_source="/cluster/scratch/ooikonomou/models/Qwen2.5-7B-Instruct"
model_local="${TMPDIR}/Qwen2.5-7B-Instruct"
module="experiments.agentic_frontier.wp1_wp4_operator_agent.adversarial_verifier_gaming_v1"
out="experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/results/confirmation"
llm_out="$out/llm/qwen"
mkdir -p "$model_local" "$llm_out/responses"
cp -a "$model_source/." "$model_local/"
cp "$model_source/.verisigs_model.json" "$llm_out/model_identity.json"
nvidia-smi > "$llm_out/nvidia-smi.txt"

"$python_bin" -m "$module.build_llm_campaign" \
  --spectral-public "$out/spectral_public.json.gz" \
  --fd-public "$out/fd_public.json.gz" \
  --output "$out/llm/official_campaign.json.gz"

"$python_bin" -m "$module.llm_runner" preflight \
  --campaign "$out/llm/official_campaign.json.gz" \
  --provider qwen --model "$model_local" \
  --output "$llm_out/preflight.json" \
  --authorization "$llm_out/authorization.json"

"$python_bin" -c \
  "from pathlib import Path; from ${module}.io_utils import read_json,write_json; p=Path('$llm_out/authorization.json'); a=read_json(p); a.update({'billable_calls_authorized':True,'authorization_reference':'explicit-user-bounded-qwen-2026-08-17'}); write_json(p,a)"

"$python_bin" -m "$module.llm_runner" run \
  --campaign "$out/llm/official_campaign.json.gz" \
  --provider qwen --model "$model_local" --policy qwen \
  --preflight "$llm_out/preflight.json" \
  --authorization "$llm_out/authorization.json" \
  --output-dir "$llm_out/responses" \
  --proposal-output "$llm_out/proposals.json.gz"

for tag in spectral fd; do
  "$python_bin" -m "$module.score" \
    --public "$out/${tag}_public.json.gz" \
    --owner "$out/${tag}_owner.json.gz" \
    --proposals "$llm_out/proposals.json.gz" \
    --output "$llm_out/${tag}_score.json.gz"
done
