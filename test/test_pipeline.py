"""
Pipeline-level regression tests (complements test_models.py).

test_models.py already pins each network's forward shape, [0,1] range and exact
parameter count. This file guards the parts AROUND the models -- the physics
operator, the data transform, that training actually learns, metric orientation,
and the classical GI baseline math -- so the whole
    data -> model -> loss -> baseline
chain is covered, not just the model forward pass.

CPU-friendly by design: runs in CI (no GPU needed). The heavy full-resolution
overfit smoke test lives in tools/ (local, GPU) and is intentionally NOT in CI.
"""
import os
import sys

import numpy as np
import pytest
import scipy.linalg
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.ghost_patterns import get_hadamard_matrix
from src.metrics.metrics import compute_metrics
from src.losses.recon_losses import CombinedReconstructionLoss
import src.models as models

IMG, M, N = 128, 512, 128 * 128


def _real_image():
    """The repo's sample image, resized to the working resolution and normalized."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GT_0.png')
    return np.array(Image.open(p).convert('L').resize((IMG, IMG), Image.Resampling.LANCZOS),
                    dtype=np.float32) / 255.0


def test_hadamard_operator():
    """Phi = first M rows of a Sylvester Hadamard matrix: +-1 valued, rows orthogonal."""
    Phi = get_hadamard_matrix(N, M)
    assert Phi.shape == (M, N)
    assert set(np.unique(Phi).tolist()) == {-1, 1}
    assert np.allclose(Phi @ Phi.T, N * np.eye(M)), "Hadamard rows must satisfy Phi Phi^T = N*I"


@pytest.mark.parametrize('n,m', [(2, 1), (8, 5), (64, 17), (128, 128)])
def test_memory_efficient_hadamard_matches_scipy(n, m):
    """The O(MN) generator must exactly match SciPy's Sylvester natural order."""
    actual = get_hadamard_matrix(n, m)
    expected = scipy.linalg.hadamard(n)[:m].astype(np.float32)
    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual @ actual.T, n * np.eye(m, dtype=np.float32))


def test_hadamard_rejects_non_power_of_two():
    """A non-power-of-two pixel count would column-truncate the Hadamard into a
    non-orthogonal operator (Phi Phi^T != N*I) -- a physically invalid SPI forward
    model. get_hadamard_matrix must reject it loudly, not return it silently.
    (All shipped configs use 128*128 = 16384 = 2**14, so this guards a real user
    footgun: setting img_size to a non-power-of-two, or feeding a raw N-pixel rig.)"""
    get_hadamard_matrix(N, M)                       # power-of-two (16384): accepted
    for bad_N in (28 * 28, 100 * 100):              # 784, 10000: not powers of two
        with pytest.raises(ValueError):
            get_hadamard_matrix(bad_N, M)
    for bad_m in (0, N + 1):
        with pytest.raises(ValueError):
            get_hadamard_matrix(N, bad_m)


def test_data_pipeline_bucket():
    """The dataset transform Phi @ vec(x) then min-max yields a (M,) bucket in [0,1]."""
    x = _real_image()
    Phi = get_hadamard_matrix(N, M).astype(np.float32)
    raw = Phi @ x.flatten()
    bucket = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    assert bucket.shape == (M,)
    assert bucket.min() >= 0.0 and bucket.max() <= 1.0
    assert x[None].shape == (1, IMG, IMG)


def test_training_step_learns():
    """A single optimizer step must give non-zero grads and change params; a few
    steps must reduce the loss -- i.e. the data->model->loss->backward pipeline
    genuinely learns. CPU-friendly (batch=1, 10 steps)."""
    torch.manual_seed(0)
    x = _real_image()
    Phi = get_hadamard_matrix(N, M).astype(np.float32)
    raw = Phi @ x.flatten()
    bucket = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    bt = torch.from_numpy(bucket)[None]
    tgt = torch.from_numpy(x)[None, None]

    net = models.GRUReconstructionNet(bucket_size=M, img_size=IMG, hidden_size=256, num_layers=3).train()
    crit = CombinedReconstructionLoss(l1_weight=0.5, l2_weight=0.3, ssim_weight=0.2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    w0 = net.decoder[0].weight.detach().clone()
    loss_first = crit(net(bt), tgt)
    opt.zero_grad()
    loss_first.backward()
    assert net.decoder[0].weight.grad is not None
    assert net.decoder[0].weight.grad.norm().item() > 0, "gradient did not flow to the decoder"
    opt.step()
    assert (net.decoder[0].weight.detach() - w0).abs().sum().item() > 0, "params did not change after step()"

    for _ in range(10):
        loss = crit(net(bt), tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < loss_first.item(), "loss did not decrease -> pipeline not learning"


def test_metric_orientation():
    """PSNR / SSIM must increase (and L1 decrease) as prediction approaches ground truth."""
    x = _real_image()
    noisy = np.clip(x + 0.1 * np.random.RandomState(0).randn(IMG, IMG), 0, 1).astype(np.float32)
    same = compute_metrics(x[None], x[None])
    worse = compute_metrics(noisy[None], x[None])
    assert same['ssim'] > worse['ssim']
    assert same['psnr'] > worse['psnr']
    assert same['l1'] < worse['l1']


def test_gi_backprojection_correlates():
    """Classical GI (zero-padded Hadamard inverse == Phi^T y by Hadamard symmetry)
    correlates structurally with the ground truth even at 3.13% sampling."""
    x = _real_image()
    Phi = get_hadamard_matrix(N, M).astype(np.float32)
    raw = Phi @ x.flatten()
    gi = (Phi.T @ raw).reshape(IMG, IMG)
    gi = (gi - gi.min()) / (gi.max() - gi.min() + 1e-8)
    corr = float(np.corrcoef(gi.flatten(), x.flatten())[0, 1])
    assert corr > 0.2, f"GI back-projection should correlate with GT, got corr={corr:.3f}"
