# Single-Pixel Imaging Reconstruction — a Deep-Learning Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-2607.22077-b31b1b.svg)](https://arxiv.org/abs/2607.22077)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CI](https://github.com/Hanyuyuan6/Comparison-of-SPI-DLs/actions/workflows/ci.yml/badge.svg)](https://github.com/Hanyuyuan6/Comparison-of-SPI-DLs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Single-pixel imaging (SPI) recovers a 2-D image from a sequence of scalar "bucket" measurements taken
under structured illumination. This repository benchmarks **ten reconstruction methods on one shared,
fixed Hadamard acquisition** under an identical train/eval protocol, so the numbers reflect the
*reconstructor* — not the measurement setup.

It also serves as the reconstruction probe that selects the temporal encoder for **STSF+TPLS**, the
image-free single-pixel segmentation framework at
[Hanyuyuan6/STSF-TPLS](https://github.com/Hanyuyuan6/STSF-TPLS): that repository emits segmentation
metrics only, this one owns the reconstruction PSNR/SSIM comparison.

**Acquisition.** Every method sees the same forward model

```
y = Φ · vec(x),    Φ ∈ {±1}^{M×N}
```

where `x` is a `128×128` image (`N = 16384`) and `Φ` is the first `M` rows of a natural-order
(Sylvester) Hadamard matrix. The sampling rate is `M/N` — `512 → 3.13%`, `1024 → 6.25%`, `2048 → 12.5%`.

## Methods

| Family | config → class |
|---|---|
| Classical | correlation ghost imaging (`GI`); compressed sensing — TV & pixel-domain L1/BPDN via cvxpy (`CS`, `--method tv\|admm`) |
| Fully-connected | `fn` → `FNReconstructionNet` |
| Convolutional | `fcn` → `FCNReconstructionNet`, `unet` → `UNetReconstructionNet` |
| Recurrent | `rnn` / `lstm` / `gru` → `{RNN,LSTM,GRU}ReconstructionNet` |
| Attention | `transformer` → `TransformerReconstructionNet` |
| Graph | `gcn` → `GCNReconstructionNet` |

The eight networks map `y` back to `x` and share decoder-head conventions, so their `state_dict`s and
parameter counts stay directly comparable.

## Install

```bash
git clone https://github.com/Hanyuyuan6/Comparison-of-SPI-DLs.git
cd Comparison-of-SPI-DLs
conda create -n spi-dl python=3.9 -y && conda activate spi-dl

# 1) PyTorch for your CUDA build (see pytorch.org); CUDA 12.1 example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 2) the rest:
pip install -r requirements.txt
```

TensorBoard is the default logger; Weights & Biases is optional (`logging.use_wandb: True`).

## Data

```bash
python -m scripts.prepare_datasets --data_root ./data --dataset all   # or: mnist | fashion_mnist | celeba | div2k | bsd500 | carvana
```

MNIST / Fashion-MNIST download via torchvision. CelebA is pulled from a public mirror (images only) and
split 90/10 into a **disjoint** `train/`–`val/` — a random split, **not** CelebA's official
train/val/test partition. DIV2K / BSD500 auto-download; Carvana needs a manual Kaggle download.

## Train

```bash
python -m scripts.train --config configs/experiments_mnist/gru.yaml \
    --dataset mnist --experiment_name gru_run
```

Each model YAML inherits `configs/base_config.yaml`. Common overrides:

| Flag | Meaning |
|---|---|
| `--dataset {mnist,fashion_mnist,celeba,…}` | dataset (required for the CelebA configs) |
| `--bucket_size {512,1024,2048}` | sampling rate — 3.13 / 6.25 / 12.5 % |
| `--epochs N` · `--batch_size N` · `--experiment_name NAME` | usual overrides |
| `--num_workers 0` | **use on Windows** (see note) |

Best checkpoint (by validation PSNR) → `checkpoints/<name>/best.pth`; curves and sample reconstructions
→ TensorBoard under `experiments/<name>/`.

> **Windows.** `data.preload: true` caches each Hadamard bucket in RAM; with the default `num_workers: 8`,
> Windows *spawns* (not forks) workers and copies the cache into each, which can stall the loader. Pass
> `--num_workers 0` (or set `data.preload: false`). Linux/macOS fork and share the cache.

**Full sweep & profiling**

```bash
bash scripts/run_all.sh                               # all backbones × datasets × sampling rates
python -m scripts.benchmark --dataset mnist   # parameter count / latency / FLOPs
```

## Evaluate & inference

```bash
# metrics on the reported (validation) split (PSNR / SSIM / L1 / MSE)
python -m scripts.evaluate --ckpt_path checkpoints/gru_run/best.pth --dataset mnist

# reconstruct a simulated image, or your own measured bucket (.npy / .txt / .mat)
python -m scripts.predict             --img_path test/GT_0.png       --ckpt_path checkpoints/gru_run/best.pth
python -m scripts.predict_from_bucket --bucket_path your_bucket.mat  --ckpt_path checkpoints/gru_run/best.pth --show

# classical baselines
python -m scripts.GI_rec_eval    --config configs/experiments_mnist/GI.yaml --dataset mnist
python -m scripts.CS_GI_rec_eval --config configs/experiments_mnist/GI.yaml --dataset mnist --method tv
```

Add `--max_batches N` for a quick pass (the CS solver is slow at full resolution). The predict/eval
scripts load `weights_only=True` first and fall back to a full load only for training checkpoints —
load `.pth` files only from a source you trust.

**The classical baselines need no weights.** GI is the closed-form Hadamard adjoint — no training, no
checkpoint, no seed — so it recomputes from a clean clone alone, whereas the eight network rows must be
trained first (this repository ships no weights). `--dataset {mnist,fashion_mnist,celeba}` covers the
3.13% conditions; the higher MNIST rates come from `data.bucket_size` (1024 / 2048) in the YAML, which
`GI_rec_eval.py` reads from the config rather than from the command line.

## Reproducibility & tests

Reproducible by construction: fixed seed (Python / NumPy / PyTorch + cuDNN-deterministic + DataLoader
workers), one YAML per run, a one-command sweep, no magic CLI strings.

**Test suite** — `pytest test/`, CPU-only, run in CI on every push:

| File | Guards |
|---|---|
| `test_models.py` | forward shape, `[0,1]` output range, **exact parameter counts** (pins the reported tables) |
| `test_pipeline.py` | Hadamard operator (`ΦΦᵀ = N·I`), data pipeline, that training actually learns, metric orientation, GI back-projection |
| `test_integration.py` | the **shipped** dataset / Trainer / `evaluate` / baselines end-to-end, with hand-written math twins as differential oracles |

## Evaluation protocol

The protocol is identical for every method, so method-to-method comparison stays fair. Note that the
split used for reporting also selects the checkpoint, so absolute PSNR/SSIM run mildly optimistic.

PSNR and SSIM come from `skimage.metrics` with `data_range=1.0` at scikit-image defaults, computed per
image and averaged. SSIM therefore uses a uniform 7×7 window rather than the 11×11 Gaussian of the
original definition — internally consistent across all methods here, but not directly comparable with
numbers reported under the Gaussian variant.

## Repository layout

```
configs/    base_config.yaml + one YAML per model (experiments_mnist/, experiments_celeba/)
src/        datasets/ · models/ (8 nets + shared base) · losses/ · metrics/ · utils/ · trainer.py
scripts/    train · evaluate · benchmark · predict · predict_from_bucket · GI/CS eval · prepare
test/       CPU test suite + sample images
scripts/run_all.sh  full benchmark sweep
```

## Citation

If this benchmark is useful in your work, please cite the preprint
[arXiv:2607.22077](https://arxiv.org/abs/2607.22077):

```bibtex
@article{han2026liftspectrum,
  title         = {The Lift Spectrum: How Measurement-to-Space Adaptivity Shapes
                   Robustness in Image-Free Single-Pixel Sensing},
  author        = {Han, Yuyuan and Li, Jingwei and Qiu, Long and Wang, Chong and
                   Hao, Wenxuan and Han, Jiangyu and Yao, Xinyu and He, Yuchen and
                   Chen, Hui and Liu, Jianbin and Zheng, Huaibin},
  journal       = {arXiv preprint arXiv:2607.22077},
  year          = {2026},
  eprint        = {2607.22077},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV}
}
```

`CITATION.cff` carries the same entry, so GitHub's *Cite this repository* button exports it directly.
References for the datasets and methods benchmarked here are given in that paper.

## License

Released under the **MIT License** (see [LICENSE](LICENSE)).
