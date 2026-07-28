import argparse
import hashlib
import logging
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm
import random
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CELEBA_URL = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"
# SHA-256 of celeba.zip. This is a third-party (Udacity) redistribution mirror, not the
# official CelebA host -- set this to a digest from a trusted copy (`sha256sum celeba.zip`)
# to verify integrity/provenance; None skips the check with a warning.
CELEBA_SHA256 = None
OUTPUT_DIR = Path('./data/celeba')


def _verify_checksum(path):
    """True if the file matches CELEBA_SHA256 (or if none is set: warn and pass).
    On mismatch, delete the (possibly tampered) file and return False."""
    if not CELEBA_SHA256:
        logging.warning(
            f"No SHA-256 set for {path.name}; skipping integrity check. Set CELEBA_SHA256 "
            f"(run `sha256sum {path.name}`) to verify this third-party-mirror download."
        )
        return True
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    if h.hexdigest() != CELEBA_SHA256:
        logging.error(f"Checksum mismatch for {path.name}; removing the (possibly tampered) file.")
        path.unlink(missing_ok=True)
        return False
    logging.info(f"Checksum OK: {path.name}")
    return True


def download_file(url, destination):
    logging.info(f"Downloading from {url}...")
    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(destination, 'wb') as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc=destination.name
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        if not _verify_checksum(destination):
            return False
        logging.info(f"Downloaded to: {destination}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Download failed: {e}")
        return False


def prepare_celeba(output_path, val_split=0.1, random_seed=42):
    output_path.mkdir(parents=True, exist_ok=True)
    zip_path = output_path / 'celeba.zip'
    img_dir = output_path / 'img_align_celeba'

    if not zip_path.exists():
        if not download_file(CELEBA_URL, zip_path):
            return
    else:
        logging.info("Archive exists, skipping download.")

    if not img_dir.is_dir() or not any(img_dir.iterdir()):
        logging.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            _root = output_path.resolve()
            for _name in zip_ref.namelist():
                if not (_root / _name).resolve().is_relative_to(_root):
                    raise ValueError(f"Unsafe path in archive (path traversal blocked): {_name}")
            zip_ref.extractall(output_path)
        logging.info("Extraction complete.")
    else:
        logging.info("Image directory exists, skipping extraction.")

    train_dir = output_path / 'train'
    val_dir = output_path / 'val'

    if train_dir.is_dir() and val_dir.is_dir() and any(train_dir.iterdir()):
        logging.info("Train and val directories already prepared.")
        return

    logging.info("Splitting train/val sets...")
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    image_files = sorted(list(img_dir.glob('*.jpg')))
    random.seed(random_seed)
    random.shuffle(image_files)

    split_idx = int(len(image_files) * (1 - val_split))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    logging.info(f"Moving {len(train_files)} files to train...")
    for f in tqdm(train_files, desc="Moving train files"):
        shutil.move(str(f), str(train_dir / f.name))

    logging.info(f"Moving {len(val_files)} files to val...")
    for f in tqdm(val_files, desc="Moving val files"):
        shutil.move(str(f), str(val_dir / f.name))

    try:
        shutil.rmtree(img_dir)
        logging.info(f"Removed directory: {img_dir}")
    except OSError as e:
        logging.error(f"Failed to remove directory: {e}")

    logging.info("CelebA preparation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare CelebA dataset")
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR), help="Output directory for CelebA")
    args = parser.parse_args()
    prepare_celeba(Path(args.output_dir))