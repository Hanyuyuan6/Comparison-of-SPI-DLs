from pathlib import Path
from PIL import Image
import numpy as np
import logging

from .base_dataset import BaseGhostDataset

class GhostImagingDataset(BaseGhostDataset):
    """Generic ghost imaging dataset for image folders."""
    def __init__(self, root_dir, bucket_size, img_size, transform=None, mode='train', **kwargs):
        super().__init__(root_dir, bucket_size, img_size, transform, mode, **kwargs)

    def _load_samples(self):
        """Load image file paths from the root directory."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        img_paths = []
        for ext in image_extensions:
            img_paths.extend(self.root_dir.glob(f'*{ext}'))
            img_paths.extend(self.root_dir.glob(f'*{ext.upper()}'))

        img_paths = sorted(set(img_paths))

        if len(img_paths) == 0:
            raise RuntimeError(f"No image files found in {self.root_dir}")

        logging.info(f"Found {len(img_paths)} images in {self.root_dir}")
        return img_paths

    def _load_image(self, idx):
        """Load and return a grayscale image as a numpy array normalized to [0,1]."""
        img_path = self.samples[idx]

        try:
            img = Image.open(img_path).convert('L')
            img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
            img_np = np.array(img, dtype=np.float32) / 255.0
            return img_np
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {str(e)}")

class CelebAGhostDataset(GhostImagingDataset):
    """CelebA-specific dataset. Selects the train/ or val/ subfolder by ``mode``
    (any other mode falls back to test/). Note: prepare_celeba.py produces only
    train/ and val/ -- a 90/10 random split, with no held-out test/ set."""
    def __init__(self, root_dir, bucket_size, img_size, transform=None, mode='train', **kwargs):
        if mode == 'train':
            img_dir = Path(root_dir) / 'train'
        elif mode == 'val':
            img_dir = Path(root_dir) / 'val'
        else:
            img_dir = Path(root_dir) / 'test'

        super().__init__(img_dir, bucket_size, img_size, transform, mode, **kwargs)

    def _load_samples(self):
        """CelebA-specific sample loading; currently delegates to parent."""
        samples = super()._load_samples()
        # Place for CelebA-specific filtering or handling if needed
        return samples