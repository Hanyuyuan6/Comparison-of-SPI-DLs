import torchvision
import numpy as np
from PIL import Image
import cv2
import logging

from .base_dataset import BaseGhostDataset
from .ghost_dataset import GhostImagingDataset, CelebAGhostDataset

class MNISTGhostDataset(BaseGhostDataset):
    """Ghost imaging version of MNIST."""
    def _load_samples(self):
        self.dataset = torchvision.datasets.MNIST(
            root=self.root_dir,
            train=(self.mode == 'train'),
            download=True
        )
        return list(range(len(self.dataset)))

    def _load_image(self, idx):
        img, _ = self.dataset[idx]
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0

class FashionMNISTGhostDataset(BaseGhostDataset):
    """Ghost imaging version of Fashion-MNIST."""
    def _load_samples(self):
        self.dataset = torchvision.datasets.FashionMNIST(
            root=self.root_dir,
            train=(self.mode == 'train'),
            download=True
        )
        return list(range(len(self.dataset)))

    def _load_image(self, idx):
        img, _ = self.dataset[idx]
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0

class DIV2KGhostDataset(BaseGhostDataset):
    """DIV2K high-resolution dataset."""
    def _load_samples(self):
        if self.mode == 'train':
            img_dir = self.root_dir / 'DIV2K_train_HR'
        else:
            img_dir = self.root_dir / 'DIV2K_valid_HR'

        samples = sorted([p for p in img_dir.glob('*.png')])
        if len(samples) == 0:
            raise RuntimeError(f"No PNG files found in {img_dir}")
        return samples

    def _load_image(self, idx):
        img = cv2.imread(str(self.samples[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image: {self.samples[idx]}")
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)
        return img.astype(np.float32) / 255.0

class BSD500GhostDataset(BaseGhostDataset):
    """BSD500 dataset."""
    def _load_samples(self):
        if self.mode == 'train':
            img_dir = self.root_dir / 'images' / 'train'
        else:
            img_dir = self.root_dir / 'images' / 'val'

        samples = sorted([p for p in img_dir.glob('*.jpg')])
        if len(samples) == 0:
            raise RuntimeError(f"No JPG files found in {img_dir}")
        return samples

    def _load_image(self, idx):
        img = Image.open(self.samples[idx]).convert('L')
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0

class CarvanaGhostDataset(BaseGhostDataset):
    """Carvana dataset wrapper."""
    def _load_samples(self):
        if self.mode == 'train':
            img_dir = self.root_dir / 'train'
        elif self.mode == 'val':
            img_dir = self.root_dir / 'val'
        else:
            img_dir = self.root_dir / 'test'

        if not img_dir.exists():
            raise RuntimeError(f"Data directory does not exist: {img_dir}")

        samples = sorted([p for p in img_dir.glob('*.jpg')])
        if len(samples) == 0:
            raise RuntimeError(f"No image files found in {img_dir}")

        logging.info(f"Loaded {len(samples)} {self.mode} images")
        return samples

    def _load_image(self, idx):
        img = cv2.imread(str(self.samples[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image: {self.samples[idx]}")
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)
        return img.astype(np.float32) / 255.0

def get_dataset(dataset_name, **kwargs):
    """Factory function returning dataset instances by name."""
    datasets = {
        'mnist': MNISTGhostDataset,
        'fashion_mnist': FashionMNISTGhostDataset,
        'div2k': DIV2KGhostDataset,
        'bsd500': BSD500GhostDataset,
        'celeba': CelebAGhostDataset,
        'carvana': CarvanaGhostDataset,
        'custom': GhostImagingDataset,
    }

    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(datasets.keys())}")

    return datasets[dataset_name](**kwargs)