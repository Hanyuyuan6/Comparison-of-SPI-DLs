"""
Integration tests: exercise the project's REAL code paths.

test_pipeline.py checks pipeline *math* (some of it re-implemented). These tests
instead import and run the shipped objects themselves -- dataset classes, the
Trainer, scripts/evaluate.py, and the classical-baseline functions -- so a change
in project code cannot slip past a re-implemented "twin". Where a math twin is
still used, it serves as a *differential oracle*: the real path and the twin get
the same input and must agree, so any drift in shipped code turns the test red.

CPU-only and download-free by design (synthetic images in tmp dirs), so it runs
in CI. The classical-baseline equivalence tests run at 64x64 to keep the full
NxN Hadamard matrix small on CI runners; the equivalence being tested is
size-independent.
"""
import os
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datasets.dataset_factory import get_dataset
from src.utils.ghost_patterns import get_hadamard_matrix
from src.utils.config_parser import load_config
from src.utils.seed import seed_everything
from src.transforms import ToTensor
from src.losses.recon_losses import CombinedReconstructionLoss
from src.trainer import Trainer
import src.models as models
from scripts.evaluate import evaluate
from scripts.GI_rec_eval import traditional_gi_reconstruction_gpu
from scripts.CS_GI_rec_eval import cs_tv_reconstruction_gpu

IMG, M, N = 128, 512, 128 * 128


@pytest.fixture
def img_dir(tmp_path):
    """Six deterministic synthetic grayscale PNGs (noise + a bright rectangle)."""
    rng = np.random.RandomState(0)
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(6):
        arr = rng.rand(IMG, IMG).astype(np.float32) * 0.3
        arr[20 + 4 * i:60 + 4 * i, 30:90] += 0.6
        arr = np.clip(arr, 0, 1)
        Image.fromarray((arr * 255).astype(np.uint8), mode='L').save(d / f"{i:02d}.png")
    return d


# --------------------------------------------------------------------------- data

def test_dataset_class_matches_math(img_dir):
    """THE twin-killer: the shipped GhostImagingDataset path must agree with the
    independently computed Phi@x + min-max math on the same image."""
    ds = get_dataset('custom', root_dir=str(img_dir), bucket_size=M, img_size=IMG, mode='val')
    s = ds[0]                                   # real shipped code path
    x = ds._load_image(0)                       # the exact image the class loaded
    Phi = get_hadamard_matrix(N, M)
    raw = np.dot(Phi, x.flatten()).astype(np.float32)                       # twin
    norm = ((raw - raw.min()) / (raw.max() - raw.min() + 1e-8)).astype(np.float32)
    assert np.allclose(s['bucket_raw'], raw, rtol=1e-5, atol=1e-3)
    assert np.allclose(s['bucket'], norm, rtol=1e-5, atol=1e-6)
    assert s['image'].shape == (1, IMG, IMG)
    assert s['bucket'].min() >= 0.0 and s['bucket'].max() <= 1.0


def test_preload_cache_matches_lazy(img_dir):
    """preload=True (RAM cache path) must yield the same samples as lazy loading."""
    kw = dict(root_dir=str(img_dir), bucket_size=M, img_size=IMG, mode='val')
    lazy = get_dataset('custom', **kw)
    pre = get_dataset('custom', preload=True, num_workers=2, **kw)
    for i in (0, len(lazy) - 1):
        assert np.allclose(lazy[i]['bucket'], pre[i]['bucket'])
        assert np.allclose(lazy[i]['image'], pre[i]['image'])


def test_totensor_transform(img_dir):
    """The shipped ToTensor transform delivers float32 torch tensors."""
    ds = get_dataset('custom', root_dir=str(img_dir), bucket_size=M, img_size=IMG,
                     transform=ToTensor(), mode='val')
    s = ds[0]
    assert isinstance(s['bucket'], torch.Tensor) and s['bucket'].dtype == torch.float32
    assert isinstance(s['image'], torch.Tensor) and s['image'].shape == (1, IMG, IMG)


# --------------------------------------------------------------------------- config / seed

def test_config_inheritance(tmp_path):
    """load_config: child overrides leaves, inherits the rest (recursive deep merge)."""
    (tmp_path / 'base.yaml').write_text("a: 1\nb:\n  c: 2\n  d: 3\n", encoding='utf-8')
    (tmp_path / 'child.yaml').write_text("base: 'base.yaml'\nb:\n  c: 9\n", encoding='utf-8')
    cfg = load_config(tmp_path / 'child.yaml')
    assert cfg['a'] == 1 and cfg['b']['c'] == 9 and cfg['b']['d'] == 3

    # and the real experiment configs resolve as documented
    real = load_config(os.path.join(ROOT, 'configs', 'experiments_mnist', 'gru.yaml'))
    assert real['model']['name'] == 'GRUReconstructionNet'          # overridden
    assert real['data']['dataset'] == 'mnist'                        # inherited
    assert real['training']['experiment_name'] == 'gru_ghost_imaging_512_f'


def test_seed_determinism():
    """seed_everything makes model init reproducible (training reproducibility)."""
    def build():
        seed_everything(123)
        return models.FNReconstructionNet(bucket_size=M, img_size=IMG, hidden_dims=[64])
    w1 = build().fc[0].weight.detach().clone()
    w2 = build().fc[0].weight.detach().clone()
    assert torch.equal(w1, w2)


# --------------------------------------------------------------------------- full pipeline

def test_train_then_evaluate_full_pipeline(img_dir, tmp_path, monkeypatch, capsys):
    """Integration test of the full pipeline (ML Test Score, Infra-2): the REAL
    Trainer trains a small model for 1 epoch on the custom dataset, writes
    best.pth; the checkpoint round-trips into a fresh model with identical
    outputs; then the REAL scripts/evaluate.py consumes it end to end."""
    monkeypatch.chdir(tmp_path)         # Trainer writes experiments/ and checkpoints/ under cwd
    seed_everything(0)

    kw = dict(root_dir=str(img_dir), bucket_size=M, img_size=IMG, transform=ToTensor())
    train_loader = DataLoader(get_dataset('custom', mode='train', **kw), batch_size=2, shuffle=True)
    val_loader = DataLoader(get_dataset('custom', mode='val', **kw), batch_size=2, shuffle=False)

    model = models.FNReconstructionNet(bucket_size=M, img_size=IMG, hidden_dims=[64])
    config = {
        'training': {'epochs': 1, 'amp': False, 'experiment_name': 'ci_pipeline_smoke',
                     'intra_epoch_val_times': 1, 'batch_size': 2, 'num_workers': 0},
        'model': {'name': 'FNReconstructionNet',
                  'params': {'bucket_size': M, 'img_size': IMG, 'hidden_dims': [64]}},
        'data': {'dataset': 'custom', 'train_dir': str(img_dir), 'val_dir': str(img_dir),
                 'bucket_size': M, 'img_size': IMG},
        'logging': {'use_wandb': False},
    }
    criterion = CombinedReconstructionLoss(l1_weight=0.5, l2_weight=0.3, ssim_weight=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    Trainer(model, optimizer, criterion, train_loader, val_loader,
            scheduler=None, device=torch.device('cpu'), config=config).train()

    best = tmp_path / 'checkpoints' / 'ci_pipeline_smoke' / 'best.pth'
    assert best.exists(), "Trainer did not write best.pth"
    ckpt = torch.load(str(best), map_location='cpu', weights_only=True)
    assert 'model_state_dict' in ckpt and 'config' in ckpt
    assert np.isfinite(ckpt['best_metric']) and ckpt['best_metric'] > 0

    # checkpoint round-trip: fresh model, identical outputs
    m2 = models.FNReconstructionNet(**ckpt['config']['model']['params'])
    m2.load_state_dict(ckpt['model_state_dict'])
    model.eval(); m2.eval()
    xb = torch.rand(2, M)
    with torch.no_grad():
        assert torch.allclose(model(xb), m2(xb), atol=1e-6)

    # the REAL numbers-producing script, end to end on the same checkpoint
    metrics_json = tmp_path / 'evaluation.json'
    evaluate(SimpleNamespace(ckpt_path=str(best), dataset=None, config=None,
                             bucket_size=None, max_batches=None,
                             allow_unsafe_pickle=False, out_json=str(metrics_json)))
    out = capsys.readouterr().out
    assert 'PSNR' in out and 'SSIM' in out
    payload = json.loads(metrics_json.read_text(encoding='utf-8'))
    assert payload['n'] == 6 and payload['checkpoint_sha256']
    assert set(payload['metrics']) == {'l1', 'mse', 'psnr', 'ssim'}


# --------------------------------------------------------------------------- classical baselines

def test_gi_real_function_matches_twin(img_dir):
    """The shipped GI reconstruction (zero-pad + full-H matmul) must equal the
    Phi^T y twin (Hadamard symmetry) -- differential check at 64x64 (CI RAM)."""
    img, m = 64, 128
    n = img * img
    ds = get_dataset('custom', root_dir=str(img_dir), bucket_size=m, img_size=img, mode='val')
    s = ds[0]
    y = torch.from_numpy(s['bucket_raw'])[None]
    Phi = get_hadamard_matrix(n, m)
    real = traditional_gi_reconstruction_gpu(torch.from_numpy(Phi), y, img)[0]

    tw = (Phi.T @ s['bucket_raw']).reshape(img, img)
    tw = tw - tw.mean()
    tw = (tw - tw.min()) / (tw.max() - tw.min() + 1e-8)
    assert np.allclose(real, tw, atol=1e-3), f"max|delta|={np.abs(real - tw).max():.2e}"

    corr = float(np.corrcoef(real.flatten(), ds._load_image(0).flatten())[0, 1])
    assert corr > 0.2, f"GI reconstruction should correlate with GT, corr={corr:.3f}"


def test_cs_tv_real_function_runs(img_dir):
    """The shipped CS-TV solver runs on CPU and returns a sane, GT-correlated
    reconstruction (few iterations -- a smoke of the real function, not a quality test)."""
    img, m = 64, 128
    n = img * img
    ds = get_dataset('custom', root_dir=str(img_dir), bucket_size=m, img_size=img, mode='val')
    s = ds[0]
    Phi = get_hadamard_matrix(n, m).astype(np.float32)
    rec = cs_tv_reconstruction_gpu(Phi, s['bucket_raw'][None], img,
                                   tv_weight=0.05, max_iter=60, lr=0.2)
    assert rec.shape == (1, img, img)
    assert np.isfinite(rec).all() and rec.min() >= 0.0 and rec.max() <= 1.0
    corr = float(np.corrcoef(rec[0].flatten(), ds._load_image(0).flatten())[0, 1])
    assert corr > 0.05, f"CS-TV reconstruction should correlate with GT, corr={corr:.3f}"
