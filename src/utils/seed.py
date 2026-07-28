import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds across Python, NumPy, and PyTorch for reproducibility.

    When ``deterministic`` is True, also enable cuDNN deterministic mode and
    deterministic algorithms so the backbone-benchmark numbers are reproducible
    across invocations (at a small throughput cost). ``warn_only=True`` keeps
    ops without a deterministic kernel from hard-failing.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def seed_worker(worker_id):
    """
    DataLoader ``worker_init_fn``: reseed NumPy and Python ``random`` in each
    worker process from torch's per-worker base seed so augmentation/shuffle
    RNG streams are reproducible (num_workers > 0 otherwise leaves them unseeded).
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
