#!/usr/bin/env bash
# Reproduce the reconstruction probe behind paper Table I from committed configs.
# This script intentionally does not compare against or invent reference metrics:
# every result JSON is re-derived from the checkpoint produced in this run.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-25}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid RUN_ID: $RUN_ID" >&2; exit 2; }
[[ "$RUN_ID" != "." && "$RUN_ID" != ".." ]] || { echo "invalid RUN_ID: $RUN_ID" >&2; exit 2; }
RESULT_ROOT="${RESULT_ROOT:-results/reproduction/$RUN_ID}"
MODELS=(fn fcn unet rnn lstm gru transformer gcn)

mkdir -p "$(dirname "$RESULT_ROOT")"
if ! mkdir "$RESULT_ROOT"; then
  echo "refusing to reuse existing RESULT_ROOT: $RESULT_ROOT" >&2
  exit 1
fi
mkdir "$RESULT_ROOT/profile" "$RESULT_ROOT/table1" "$RESULT_ROOT/baselines"

run_train_eval() {
  local config="$1"
  local dataset="$2"
  local bucket_size="$3"
  local experiment_name="$4"
  local artifact="$5"

  "$PYTHON_BIN" -m scripts.train \
    --config "$config" \
    --dataset "$dataset" \
    --bucket_size "$bucket_size" \
    --epochs "$EPOCHS" \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --experiment_name "$experiment_name" \
    --refuse_existing_output
  "$PYTHON_BIN" -m scripts.evaluate \
    --ckpt_path "checkpoints/$experiment_name/best.pth" \
    --out_json "$artifact"
}

run_gi_eval() {
  local dataset="$1"
  local bucket_size="$2"
  local artifact="$3"
  local save_dir="$4"

  "$PYTHON_BIN" -m scripts.GI_rec_eval \
    --config configs/experiments_mnist/GI.yaml \
    --dataset "$dataset" \
    --bucket_size "$bucket_size" \
    --save_dir "$save_dir" \
    --out_json "$artifact"
}

# 0) Prepare exactly the datasets used by Table I. torchvision owns the trusted
# MNIST/Fashion-MNIST checksums. CelebA uses a third-party mirror and therefore
# needs an explicit trusted digest unless train/ and val/ are already prepared.
"$PYTHON_BIN" -m scripts.prepare_datasets --data_root ./data --dataset mnist
"$PYTHON_BIN" -m scripts.prepare_datasets --data_root ./data --dataset fashion_mnist
CELEBA_ARGS=()
if [[ -n "${CELEBA_SHA256:-}" ]]; then
  CELEBA_ARGS+=(--sha256 "celeba.zip=$CELEBA_SHA256")
fi
"$PYTHON_BIN" -m scripts.prepare_datasets \
  --data_root ./data --dataset celeba "${CELEBA_ARGS[@]}"

# 1) Parameter / latency / FLOPs profiles. Use a unique label per sampling rate
# so later runs cannot overwrite earlier JSON files.
for bucket_size in 512 1024 2048; do
  profile_artifact="$RESULT_ROOT/profile/benchmark_mnist_${bucket_size}.json"
  "$PYTHON_BIN" -m scripts.benchmark \
    --dataset "mnist_${bucket_size}" \
    --bucket_size "$bucket_size" \
    --output_dir "$RESULT_ROOT/profile"
  # benchmark.py names the file from --dataset.
  generated="$RESULT_ROOT/profile/benchmark_results_mnist_${bucket_size}.json"
  if [[ "$generated" != "$profile_artifact" ]]; then
    mv "$generated" "$profile_artifact"
  fi
done

# 2) Eight learned methods: MNIST at three rates, then Fashion-MNIST/CelebA at 3.13%.
for model in "${MODELS[@]}"; do
  for bucket_size in 512 1024 2048; do
    name="${model}_mnist_${bucket_size}_seed${SEED}_${RUN_ID}"
    run_train_eval \
      "configs/experiments_mnist/${model}.yaml" mnist "$bucket_size" "$name" \
      "$RESULT_ROOT/table1/${name}.json"
  done

  name="${model}_fmnist_512_seed${SEED}_${RUN_ID}"
  run_train_eval \
    "configs/experiments_mnist/${model}.yaml" fashion_mnist 512 "$name" \
    "$RESULT_ROOT/table1/${name}.json"

  name="${model}_celeba_512_seed${SEED}_${RUN_ID}"
  run_train_eval \
    "configs/experiments_celeba/${model}.yaml" celeba 512 "$name" \
    "$RESULT_ROOT/table1/${name}.json"
done

# 3) Hadamard-adjoint physics row: all five Table-I conditions.
for bucket_size in 512 1024 2048; do
  run_gi_eval \
    mnist "$bucket_size" \
    "$RESULT_ROOT/table1/physics_mnist_${bucket_size}.json" \
    "$RESULT_ROOT/baselines/gi_mnist_${bucket_size}"
done
run_gi_eval \
  fashion_mnist 512 \
  "$RESULT_ROOT/table1/physics_fmnist_512.json" \
  "$RESULT_ROOT/baselines/gi_fmnist_512"
run_gi_eval \
  celeba 512 \
  "$RESULT_ROOT/table1/physics_celeba_512.json" \
  "$RESULT_ROOT/baselines/gi_celeba_512"

# Optional CS diagnostic (not a Table-I row and prohibitively slow at full scale).
# When requested it is still required to emit and validate a JSON artifact.
if [[ "${RUN_CS_BASELINE:-0}" == "1" ]]; then
  cs_artifact="$RESULT_ROOT/baselines/cs_admm_mnist_512.json"
  "$PYTHON_BIN" -m scripts.CS_GI_rec_eval \
    --config configs/experiments_mnist/GI.yaml \
    --dataset mnist \
    --method admm \
    --save_dir "$RESULT_ROOT/baselines/cs_admm_mnist_512" \
    --out_json "$cs_artifact"
fi

# 4) Validate the exact 8x5 learned matrix, five physics conditions, three
# profiles, sample counts, identities, finite metrics, hashes, and JSON inventory.
manifest="$RESULT_ROOT/manifest.json"
VALIDATE_ARGS=(
  --result_root "$RESULT_ROOT"
  --manifest "$manifest"
  --seed "$SEED"
  --epochs "$EPOCHS"
  --run_id "$RUN_ID"
)
if [[ "${RUN_CS_BASELINE:-0}" == "1" ]]; then
  VALIDATE_ARGS+=(--include_cs)
fi
"$PYTHON_BIN" -m scripts.validate_reproduction_artifacts "${VALIDATE_ARGS[@]}"

printf 'Sweep completed successfully. Validated manifest: %s\n' "$manifest"
