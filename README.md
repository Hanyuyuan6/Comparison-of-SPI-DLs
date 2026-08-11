# Single-Pixel Imaging Reconstruction — a Deep-Learning Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-2607.22077-b31b1b.svg)](https://arxiv.org/abs/2607.22077)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.13](https://img.shields.io/badge/pytorch-2.13-ee4c2c.svg)](https://pytorch.org/)
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

This is a controlled **offline architecture-selection probe**, not the STSF inference pipeline. Its
reconstructed images are used to compare temporal encoders; STSF deploys the selected encoder inside
a direct measurement-to-task lift and never feeds a reconstructed image to the segmentation head.

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

The eight networks map `y` back to `x`. The sequence/graph models (RNN, LSTM, GRU, Transformer and GCN)
share the decoder in `BaseReconstructionNet`; FC, CNN and U-Net retain their native heads. Parameter
budgets are compared, but `state_dict`s are **not** interchangeable across architectures.

## Install

```bash
git clone https://github.com/Hanyuyuan6/Comparison-of-SPI-DLs.git
cd Comparison-of-SPI-DLs
conda create -n spi-dl python=3.10 -y && conda activate spi-dl

# CPU verification environment (for CUDA, use the matching official PyTorch 2.13 index):
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip check
```

TensorBoard is the default logger; Weights & Biases is optional (`logging.use_wandb: True`).

> **Environment provenance.** `requirements.txt` pins the direct dependencies in the reviewed Python
> 3.10 verification environment; transitive packages are resolver-selected rather than hash-locked.
> The historical environment that produced the released probe numbers was
> not archived, so PyTorch 2.13 is **not** claimed to have generated them. Re-running
> the probe under this pinned environment is a new reproduction and should be reported as such.

## Data

```bash
# torchvision verifies MNIST/Fashion-MNIST with its registered checksums:
python -m scripts.prepare_datasets --data_root ./data --dataset mnist

# Other archives require a SHA-256 obtained from a trusted publisher/verified copy:
python -m scripts.prepare_datasets --data_root ./data --dataset celeba \
  --sha256 celeba.zip=<TRUSTED_64_HEX_DIGEST>
python -m scripts.prepare_datasets --data_root ./data --dataset div2k \
  --sha256 DIV2K_train_HR.zip=<TRUSTED_64_HEX_DIGEST> \
  --sha256 DIV2K_valid_HR.zip=<TRUSTED_64_HEX_DIGEST>
```

Downloads and already-present archives are both verified before extraction; a missing or mismatched
digest fails closed. Do not compute a digest from the same untrusted download and treat it as provenance.
CelebA uses a third-party public mirror (images only) and is split 90/10 into a **disjoint** `train/`–`val/`
random split, **not** CelebA's official train/val/test partition. The released protocol requires exactly
202,599 source images and publishes `_prepared_meta.json` for the 182,339/20,260 split; missing, partial,
overlapping, empty, byte-modified, extra-file, or unverified split directories fail closed. The retained
`celeba.zip` is rehashed against the manifest so the prepared-byte inventory stays bound to the trusted
source archive. Carvana remains a manual Kaggle
download and its archives likewise require repeated `--sha256 FILENAME=...` arguments.

## Train

```bash
python -m scripts.train --config configs/experiments_mnist/gru.yaml \
    --dataset mnist --experiment_name gru_run --seed 42
```

Each model YAML inherits `configs/base_config.yaml`. Common overrides:

| Flag | Meaning |
|---|---|
| `--dataset {mnist,fashion_mnist,celeba,…}` | dataset (required for the CelebA configs) |
| `--bucket_size {512,1024,2048}` | sampling rate — 3.13 / 6.25 / 12.5 % |
| `--epochs N` · `--batch_size N` · `--experiment_name NAME` | usual overrides |
| `--seed N` | seed recorded in the resolved checkpoint config (default: 42) |
| `--num_workers 0` | **use on Windows** (see note) |

Best checkpoint (by validation PSNR) → `checkpoints/<name>/best.pth`; curves and sample reconstructions
→ TensorBoard under `experiments/<name>/`.

> **Windows/macOS.** `data.preload: true` caches each Hadamard bucket in RAM. Windows and modern macOS
> Python use `spawn`, which copies the cache into each worker and can stall the loader. Pass
> `--num_workers 0` (or set `data.preload: false`); `run_all.sh` now defaults to zero workers on every
> platform for portability, while Linux users may explicitly raise `NUM_WORKERS` after checking memory.

**Full sweep & profiling**

```bash
# Set CELEBA_SHA256 unless data/celeba/train and val are already prepared and verified.
CELEBA_SHA256=<TRUSTED_64_HEX_DIGEST> SEED=42 RUN_ID=repro01 bash scripts/run_all.sh
python -m scripts.benchmark --dataset mnist   # parameter count / latency / FLOPs
```

`run_all.sh` is fail-closed (`set -euo pipefail`) and atomically reserves a new `RESULT_ROOT` before
writing, so duplicate/concurrent `RUN_ID` values cannot overwrite a prior run. It refuses stale
checkpoint/output directories, evaluates
the exact eight-model by five-condition matrix, runs the Hadamard baseline at all five benchmark conditions,
and profiles all eight models at three rates. Each safe `RUN_ID` gets its own
`results/reproduction/<RUN_ID>/`; the final validator requires exactly 48 identified artifacts (49 with
the optional CS diagnostic), exact sample counts, finite metrics, and no extra JSON files. The validator
rehashes each checkpoint and source config, safely reloads each embedded effective config, and verifies
the reported lineage hashes before writing its SHA-256 manifest. It does **not** compare against or fabricate reference values. Set
`RUN_CS_BASELINE=1` only for the optional, very slow CS diagnostic; it is not a benchmark-matrix row.

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
scripts fail closed if `weights_only=True` cannot load a checkpoint. A legacy checkpoint from a trusted
source may be loaded only with the explicit `--allow_unsafe_pickle` flag; full pickle loads can execute
arbitrary code. For `.mat` bucket files, automatic selection is allowed only when exactly one eligible
numeric vector exists; otherwise select it explicitly with `--mat_key VARIABLE`.

**The classical baselines need no weights.** GI is the closed-form Hadamard adjoint — no training, no
checkpoint, no seed — so it recomputes from a clean clone alone, whereas the eight network rows must be
trained first (this repository ships no weights). `GI_rec_eval.py --bucket_size {512,1024,2048}` exposes
all three MNIST sampling rates without editing a committed YAML.

## Reproducibility & tests

The runner records an explicit seed and enables Python / NumPy / PyTorch, cuDNN, DataLoader-worker and
cuBLAS determinism controls. This makes same-environment reruns inspectable, not statistically replicated:
the released protocol trains one run per cell and therefore has no seed uncertainty estimates.

**Test suite** — `pytest test/`, CPU-only, run in CI on every push:

| File | Guards |
|---|---|
| `test_models.py` | forward shape, `[0,1]` output range, **exact parameter counts** (pins the released configurations) |
| `test_pipeline.py` | Hadamard operator (`ΦΦᵀ = N·I`), data pipeline, that training actually learns, metric orientation, GI back-projection |
| `test_integration.py` | the **shipped** dataset / Trainer / `evaluate` / baselines end-to-end, with hand-written math twins as differential oracles |
| `test_safety.py` | fail-closed checkpoints/buckets, archive links and checksums, complete CelebA manifest, non-finite-loss rejection, cuBLAS seed contract, and exact 48-artifact sweep contract |

## Evaluation protocol

The protocol is identical for every method, so method-to-method comparison stays fair. Checkpoint
selection uses the same split the run reports, so the ranking is internally controlled and the
absolute PSNR/SSIM are not held-out estimates.

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
  author        = {Han, Yuyuan and Li, Jingwei and Zhang, Xiaoxia and Qiu, Long and Wang, Chong and
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
