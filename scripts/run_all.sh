#!/usr/bin/env bash
# Reproduce the single-pixel reconstruction benchmark (paper Table I) from the
# committed configs. Each training run logs PSNR/SSIM/L1/MSE to TensorBoard under
# experiments/<name>/ and saves the best checkpoint to checkpoints/<name>/best.pth.
#
# NOTE: this is the FULL paper sweep (8 backbones x several datasets/rates ~= 40 runs)
# and is meant as the authoritative reproduction recipe, not a quick smoke test.
# Comment out rows you do not need. Requires a CUDA GPU for reasonable runtime.
cd "$(dirname "$0")/.."   # script lives in scripts/; run from the repo root
export PYTHONPATH="$PWD"

MODELS="fn fcn unet rnn lstm gru transformer gcn"
EPOCHS=25

# 0) Prepare datasets (MNIST/Fashion-MNIST auto-download; CelebA via S3 mirror).
python -m scripts.prepare_datasets --data_root ./data --dataset all || true

# 1) Parameter / latency / FLOPs profile. Reads the same configs, so the printed
#    "Params" column reproduces the Table I parameter column, which lists all three
#    sampling rates per model (3.13 / 6.25 / 12.5%).
python -m scripts.benchmark --dataset mnist
python -m scripts.benchmark --dataset mnist --bucket_size 1024
python -m scripts.benchmark --dataset mnist --bucket_size 2048

# 2) Table I, MNIST columns -- three sampling rates (512=3.13%, 1024=6.25%, 2048=12.5%).
for m in $MODELS; do
  for bs in 512 1024 2048; do
    python -m scripts.train --config configs/experiments_mnist/$m.yaml \
      --dataset mnist --bucket_size $bs --epochs $EPOCHS \
      --experiment_name ${m}_mnist_${bs}
  done
done

# 3) Table I, Fashion-MNIST and CelebA columns -- 3.13% (512 measurements).
for m in $MODELS; do
  python -m scripts.train --config configs/experiments_mnist/$m.yaml \
    --dataset fashion_mnist --bucket_size 512 --epochs $EPOCHS \
    --experiment_name ${m}_fmnist_512
  python -m scripts.train --config configs/experiments_celeba/$m.yaml \
    --dataset celeba --bucket_size 512 --epochs $EPOCHS \
    --experiment_name ${m}_celeba_512
done

# 4) Physics-based (traditional ghost-imaging / CS) baselines.
#    These three GI runs give the 3.13% conditions of the Table I Physics-based row; the
#    MNIST 6.25/12.5% cells need data.bucket_size raised in the YAML, because GI_rec_eval
#    reads the sampling rate from the config rather than from the command line.
python -m scripts.GI_rec_eval --config configs/experiments_mnist/GI.yaml --dataset mnist
python -m scripts.GI_rec_eval --config configs/experiments_mnist/GI.yaml --dataset fashion_mnist
python -m scripts.GI_rec_eval --config configs/experiments_mnist/GI.yaml --dataset celeba
python -m scripts.CS_GI_rec_eval --config configs/experiments_mnist/GI.yaml --dataset mnist --method admm --save_dir ./ADMM_cs_gi_vis

echo "Done. PSNR/SSIM in experiments/<name>/ (TensorBoard); checkpoints in checkpoints/<name>/best.pth."
