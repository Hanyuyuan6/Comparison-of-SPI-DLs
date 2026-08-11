from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import logging

from src.utils.ghost_patterns import get_hadamard_matrix

class BaseGhostDataset(Dataset, ABC):
    """Base class for ghost imaging datasets providing common utilities."""
    def __init__(self, root_dir, bucket_size, img_size, transform=None,
                 mode='train', preload=False, num_workers=None, **kwargs):
        self.root_dir = Path(root_dir)
        self.bucket_size = bucket_size
        self.img_size = img_size
        self.transform = transform
        self.mode = mode
        self.preload = preload

        if num_workers is None:
            num_workers = min(mp.cpu_count() // 2, 8)
        self.num_workers = num_workers

        # Load sample list and precompute Hadamard patterns
        self.samples = self._load_samples()
        self.patterns = get_hadamard_matrix(self.img_size ** 2, self.bucket_size)

        if self.preload:
            self._preload_data()
        else:
            self.cached_data = None

    @abstractmethod
    def _load_samples(self):
        """Subclasses must implement: return list of samples or indices."""
        pass

    @abstractmethod
    def _load_image(self, idx):
        """Subclasses must implement: load image by index and return a numpy array."""
        pass

    def _preload_data(self):
        """Preload all samples into memory using a thread pool."""
        logging.info(f"Preloading {len(self.samples)} {self.mode} samples into memory...")
        self.cached_data = {}

        def load_single(idx):
            try:
                return idx, self._pack_sample(self._load_image(idx))
            except Exception as e:
                logging.error(f"Failed to load sample {idx}: {str(e)}")
                return idx, None

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(load_single, i) for i in range(len(self.samples))]

            success_count = 0
            for future in futures:
                idx, data = future.result()
                if data is not None:
                    self.cached_data[idx] = data
                    success_count += 1

        logging.info(f"Preload complete: {success_count}/{len(self.samples)} samples loaded for {self.mode}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Return a sample. If preload is enabled, fetch from cache when available."""
        if self.preload and self.cached_data is not None:
            if idx in self.cached_data:
                sample = self.cached_data[idx].copy()
            else:
                sample = self._load_sample(idx)
        else:
            sample = self._load_sample(idx)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _pack_sample(self, img_np):
        """Compute the Hadamard bucket for one image and pack the sample dict.
        Shared by the lazy and preload paths so the two cannot silently drift."""
        bucket_raw = np.dot(self.patterns, img_np.flatten())
        bucket_norm = (bucket_raw - bucket_raw.min()) / (bucket_raw.max() - bucket_raw.min() + 1e-8)
        return {
            'bucket': bucket_norm.astype(np.float32),
            'bucket_raw': bucket_raw.astype(np.float32),
            'image': img_np[None, ...]  # (1, H, W)
        }

    def _load_sample(self, idx):
        """Load a single sample and compute its bucket signals."""
        return self._pack_sample(self._load_image(idx))