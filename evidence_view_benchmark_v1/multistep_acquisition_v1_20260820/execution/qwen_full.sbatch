#!/bin/bash
#SBATCH --job-name=evb-multistep-qwen
#SBATCH --partition=gpupr.4h
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/ooikonomou/evb_multistep_20260820/execution/qwen-%j.out
#SBATCH --error=/cluster/scratch/ooikonomou/evb_multistep_20260820/execution/qwen-%j.err

set -euo pipefail

remote_root="/cluster/scratch/ooikonomou/evb_multistep_20260820"
benchmark_root="${remote_root}/evidence_view_benchmark_v1"
lane_root="${benchmark_root}/multistep_acquisition_v1_20260820"
python_bin="/cluster/scratch/ooikonomou/verisigs-qwen-minenv/bin/python"
model_source="/cluster/scratch/ooikonomou/models/Qwen2.5-7B-Instruct"
model_local="${TMPDIR}/Qwen2.5-7B-Instruct"

mkdir -p "${remote_root}/execution" "${lane_root}/results" "$model_local"
cd "$benchmark_root"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"$python_bin" "${lane_root}/execution/verify_remote_source.py" "$benchmark_root" \
  > "${remote_root}/execution/source-preflight.json"
cp -a "${model_source}/." "$model_local/"
nvidia-smi > "${remote_root}/execution/nvidia-smi.txt"

PYTHONPATH="$lane_root" "$python_bin" -m src.qwen_runner \
  --model-path "$model_local" \
  --model-identity "${lane_root}/MODEL_IDENTITY.json" \
  --output "${lane_root}/results/qwen_raw.json"

PYTHONPATH="$lane_root" "$python_bin" -m src.validate_qwen \
  "${lane_root}/results/qwen_raw.json" \
  --output "${lane_root}/results/qwen_validation.json"

find "${lane_root}/results" "${remote_root}/execution" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${remote_root}/execution/FILES.sha256"

