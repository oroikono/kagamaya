#!/bin/bash
set -euo pipefail

LOCAL_REPO=/Users/oroikon/Downloads/verisigs
LANE_REL=experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_spotlight_2d_replication_v1r4_20260824
LANE_MODULE=experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_spotlight_2d_replication_v1r4_20260824
REMOTE_ROOT=/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r4
REMOTE_REPO="$REMOTE_ROOT/repo"
REMOTE_LANE="$REMOTE_REPO/$LANE_REL"
REMOTE_DATA="$REMOTE_ROOT/data/the_well/data/valid"
REVISION=913ef5679d5c4b4ccd6fcd52c055509dfbf190f9
EXPECTED_BYTES=2650800128
IC_SHA=929a69f0619eb5e9f5dc0993706988fb5ee262b3eb11ef1f02474eae627ec166
AUTH_NAME=root_authorization_external_validation_76bb15c4063b0f76.json
SSH=(ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=12 \
  -o ConnectTimeout=15 -o HostName=129.132.93.72 \
  -o HostKeyAlias=euler.ethz.ch euler)

# The same stage authorization is checked before the first network request.
PYTHONPATH="$LOCAL_REPO" "$LOCAL_REPO/.venv/bin/python" -c \
  "from pathlib import Path; from $LANE_MODULE.authorization import validate; validate(Path('$LOCAL_REPO/$LANE_REL/$AUTH_NAME'), stage='external_validation', input_bindings={'internal_confirmation_aggregate_sha256':'$IC_SHA'})"

names=(
  gray_scott_reaction_diffusion_spirals_F_0.018_k_0.051.hdf5
  gray_scott_reaction_diffusion_spots_F_0.03_k_0.062.hdf5
  gray_scott_reaction_diffusion_worms_F_0.058_k_0.065.hdf5
)
hashes=(
  eb83bc9a7f5e2686fba9d4cfc5466adbf8a55c1b0973889c52b0717dca9eeca9
  a64a4f0c1e18d68ef9e59da3191faf72dfd457aefcd49080c53a5ef6823bb016
  f8df51aa04ca360113ad088f2a5038a95b366ffbb049526e889367e6d8a3ddfa
)

for index in 0 1 2; do
  name="${names[$index]}"
  expected_sha="${hashes[$index]}"
  destination="$REMOTE_DATA/$name"
  partial="$destination.part"
  url="https://huggingface.co/datasets/polymathic-ai/gray_scott_reaction_diffusion/resolve/$REVISION/data/valid/$name"

  state=$("${SSH[@]}" "if test -e '$destination'; then echo FINAL; elif test -e '$partial'; then stat -c %s '$partial'; else echo 0; fi")
  if [[ "$state" == FINAL ]]; then
    observed_size=$("${SSH[@]}" "stat -c %s '$destination'")
    observed_sha=$("${SSH[@]}" "sha256sum '$destination' | cut -d' ' -f1")
    [[ "$observed_size" == "$EXPECTED_BYTES" && "$observed_sha" == "$expected_sha" ]]
    printf 'already_valid %s %s\n' "$name" "$observed_sha"
    continue
  fi
  offset="$state"
  [[ "$offset" =~ ^[0-9]+$ ]]
  (( offset <= EXPECTED_BYTES ))
  printf 'resume %s offset=%s remaining=%s\n' "$name" "$offset" "$((EXPECTED_BYTES - offset))"

  if (( offset < EXPECTED_BYTES )); then
    http_code=$(curl --location --silent --show-error --fail \
      --range "$offset-$offset" --output /dev/null --write-out '%{http_code}' \
      "$url")
    [[ "$http_code" == 206 ]]
    curl --location --silent --show-error --fail --range "$offset-" "$url" \
      | "${SSH[@]}" "umask 077; mkdir -p '$REMOTE_DATA'; test ! -e '$destination'; current=\$(if test -e '$partial'; then stat -c %s '$partial'; else echo 0; fi); test \"\$current\" -eq '$offset'; dd of='$partial' bs=8M oflag=append conv=notrunc status=none"
  fi

  observed_size=$("${SSH[@]}" "stat -c %s '$partial'")
  [[ "$observed_size" == "$EXPECTED_BYTES" ]]
  observed_sha=$("${SSH[@]}" "sha256sum '$partial' | cut -d' ' -f1")
  [[ "$observed_sha" == "$expected_sha" ]]
  "${SSH[@]}" "test ! -e '$destination'; ln '$partial' '$destination'; rm '$partial'"
  printf 'published %s %s\n' "$name" "$observed_sha"
done

# Re-enter the frozen downloader after all six destination files exist. It
# rechecks every size/hash and creates the normal immutable download receipt;
# it makes no network request and never opens HDF5 outcomes in this branch.
"${SSH[@]}" "bash -lc 'set -eu; test ! -e \"$REMOTE_ROOT/receipts/external_valid_download.json\"; cd \"$REMOTE_REPO\"; nice -n 10 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=\"$REMOTE_REPO\" /cluster/home/ooikonomou/miniconda3/bin/python -u -m $LANE_MODULE.scripts.download_external_well --split valid --regime all --manifest \"$REMOTE_LANE/EXTERNAL_WELL_MANIFEST.json\" --data-root \"$REMOTE_ROOT/data/the_well\" --authorization \"$REMOTE_LANE/$AUTH_NAME\" --internal-confirmation-aggregate-sha256 $IC_SHA --receipt \"$REMOTE_ROOT/receipts/external_valid_download.json\"'"
