import argparse
import hashlib
import logging
from pathlib import Path
import re
import requests
import sys
from tqdm import tqdm
import zipfile
import tarfile
import shutil
import random
import stat

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SHA-256 of each downloadable archive, keyed by filename. After a download the file
# is verified against its digest here; an unset entry fails closed. torchvision
# datasets (MNIST / Fashion-MNIST) carry their own checksums and are intentionally
# not listed.
CHECKSUMS = {}
SHA256_RE = re.compile(r'^[0-9a-fA-F]{64}$')
PREPARE_CELEBA_SCRIPT = Path(__file__).with_name('prepare_celeba.py')


def _validated_sha256(value, filename):
    if not value:
        raise ValueError(
            f"No trusted SHA-256 is configured for {filename}. Refusing to use or "
            f"download an unverified archive. Supply --sha256 {filename}=<64 hex> "
            "using a digest obtained from a trusted publisher or verified copy."
        )
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"Invalid SHA-256 for {filename}: expected exactly 64 hexadecimal characters")
    return value.lower()


def parse_checksum_overrides(items):
    checksums = {}
    for item in items or []:
        if '=' not in item:
            raise ValueError(f"Invalid --sha256 value {item!r}; expected FILENAME=64_HEX")
        filename, digest = item.split('=', 1)
        filename = filename.strip()
        if not filename or Path(filename).name != filename:
            raise ValueError(f"Checksum key must be a plain archive filename, got {filename!r}")
        checksums[filename] = _validated_sha256(digest.strip(), filename)
    return checksums


def _checksum_for(overrides, filename):
    return (overrides or {}).get(filename) or CHECKSUMS.get(filename)


class DatasetDownloader:
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

    def download_file(self, url, dest_path, expected_sha256, allow_redirects=True):
        expected_sha256 = _validated_sha256(expected_sha256, dest_path.name)
        if dest_path.exists():
            self._verify_checksum(dest_path, expected_sha256)
            logging.info(f"Verified existing archive: {dest_path}")
            return True

        partial_path = dest_path.with_name(dest_path.name + '.part')
        try:
            logging.info(f"Starting download: {url}")
            response = requests.get(url, stream=True, allow_redirects=allow_redirects,
                                    timeout=(10, 60))
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(partial_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=partial_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            self._verify_checksum(partial_path, expected_sha256)
            partial_path.replace(dest_path)
            logging.info(f"Downloaded: {dest_path}")
            return True
        except Exception as e:
            if partial_path.exists():
                partial_path.unlink()
            raise RuntimeError(f"Download or verification failed for {url}: {e}") from e

    @staticmethod
    def _verify_checksum(path, expected_sha256):
        """Verify an archive before it is extracted; never trust existence alone."""
        expected = _validated_sha256(expected_sha256, path.name.removesuffix('.part'))
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {path.name}: expected {expected}, got {actual}. "
                "The archive may be corrupted or tampered with; it was not extracted."
            )
        logging.info(f"Checksum OK: {path.name}")

    @staticmethod
    def _assert_no_traversal(names, extract_root):
        """Guard against zip/tar-slip (CVE-2007-4559): reject any member that would
        resolve outside extract_root. Py3.9-compatible (filter= is 3.12+)."""
        root = Path(extract_root).resolve()
        for name in names:
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in archive (path traversal blocked): {name}")

    def extract_archive(self, archive_path, extract_to):
        logging.info(f"Extracting: {archive_path}")
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                members = zip_ref.infolist()
                self._assert_no_traversal([member.filename for member in members], extract_to)
                for member in members:
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        raise ValueError(
                            f"Unsafe symbolic link in ZIP archive: {member.filename}"
                        )
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                members = tar_ref.getmembers()
                self._assert_no_traversal([member.name for member in members], extract_to)
                for member in members:
                    if not (member.isdir() or member.isreg()):
                        raise ValueError(
                            f"Unsafe non-regular TAR member blocked: {member.name}"
                        )
                tar_ref.extractall(extract_to, members=members)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
        logging.info(f"Extraction complete: {extract_to}")


def prepare_mnist_fashion(data_root, checksums=None):
    logging.info("MNIST and Fashion-MNIST will be downloaded automatically via torchvision")
    mnist_dir = data_root / 'mnist'
    fashion_dir = data_root / 'fashion_mnist'
    mnist_dir.mkdir(parents=True, exist_ok=True)
    fashion_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torchvision
        logging.info("torchvision is installed; datasets will download on first use.")
    except ImportError:
        raise ImportError("torchvision is required: install a compatible torch/torchvision pair")


def prepare_div2k(data_root, checksums=None):
    downloader = DatasetDownloader(data_root)
    div2k_dir = data_root / 'div2k'
    div2k_dir.mkdir(exist_ok=True)

    train_url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    valid_url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"

    train_zip = div2k_dir / "DIV2K_train_HR.zip"
    downloader.download_file(train_url, train_zip, _checksum_for(checksums, train_zip.name))
    downloader.extract_archive(train_zip, div2k_dir)

    valid_zip = div2k_dir / "DIV2K_valid_HR.zip"
    downloader.download_file(valid_url, valid_zip, _checksum_for(checksums, valid_zip.name))
    downloader.extract_archive(valid_zip, div2k_dir)


def prepare_bsd500(data_root, checksums=None):
    downloader = DatasetDownloader(data_root)
    bsd_dir = data_root / 'bsd500'
    bsd_dir.mkdir(exist_ok=True)

    urls = [
        "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz",
        "https://github.com/BIDS/BSDS500/releases/download/v1.0/BSR_bsds500.tgz",
    ]

    archive = bsd_dir / "BSR_bsds500.tgz"
    expected_sha256 = _validated_sha256(_checksum_for(checksums, archive.name), archive.name)
    download_success = False
    last_error = None

    for url in urls:
        try:
            downloader.download_file(url, archive, expected_sha256)
            download_success = True
            break
        except Exception as error:
            last_error = error
            logging.warning(f"BSD500 mirror failed: {error}")

    if not download_success:
        raise RuntimeError("BSD500 download failed from every configured mirror") from last_error

    downloader.extract_archive(archive, bsd_dir)

    bsr_dir = bsd_dir / "BSR" / "BSDS500" / "data"
    if bsr_dir.exists():
        images_dir = bsd_dir / "images"
        images_dir.mkdir(exist_ok=True)

        for split in ['train', 'val', 'test']:
            src = bsr_dir / "images" / split
            dst = images_dir / split
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)
                logging.info(f"Copied {split} images to {dst}")


def prepare_celeba(data_root, checksums=None):
    celeba_dir = data_root / 'celeba'
    if not PREPARE_CELEBA_SCRIPT.is_file():
        raise RuntimeError(f"CelebA preparation script not found: {PREPARE_CELEBA_SCRIPT}")

    import subprocess
    logging.info("Preparing CelebA via scripts/prepare_celeba.py")
    command = [sys.executable, str(PREPARE_CELEBA_SCRIPT), '--output_dir', str(celeba_dir)]
    checksum = _checksum_for(checksums, 'celeba.zip')
    if checksum:
        command.extend(['--sha256', checksum])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"CelebA preparation failed: {result.stderr.strip()}")


def prepare_carvana(data_root, checksums=None):
    carvana_dir = data_root / 'carvana'
    carvana_dir.mkdir(exist_ok=True)

    train_imgs_dir = carvana_dir / 'train_imgs'
    test_imgs_dir = carvana_dir / 'test_imgs'

    final_train_dir = carvana_dir / 'train'
    final_val_dir = carvana_dir / 'val'
    final_test_dir = carvana_dir / 'test'
    if all(path.is_dir() and any(path.glob('*.jpg'))
           for path in (final_train_dir, final_val_dir, final_test_dir)):
        logging.info("Carvana dataset already prepared.")
        return

    train_zip = carvana_dir / 'train.zip'
    test_zip = carvana_dir / 'test.zip'
    train_masks_zip = carvana_dir / 'train_masks.zip'

    if not train_zip.exists() or not test_zip.exists():
        raise RuntimeError(
            "Carvana requires manual Kaggle downloads (train.zip, train_masks.zip, test.zip) "
            f"under {carvana_dir}; provide their trusted digests with repeated --sha256 flags."
        )

    downloader = DatasetDownloader(data_root)
    required_archives = [train_zip, test_zip]
    if train_masks_zip.exists():
        required_archives.append(train_masks_zip)
    for archive_path in required_archives:
        downloader._verify_checksum(
            archive_path,
            _checksum_for(checksums, archive_path.name),
        )

    if train_zip.exists():
        logging.info("Extracting training images...")
        train_extract_dir = carvana_dir / 'train_imgs'
        train_extract_dir.mkdir(exist_ok=True)
        downloader.extract_archive(train_zip, train_extract_dir)

        train_subdir = train_extract_dir / 'train'
        if train_subdir.exists():
            for img in train_subdir.glob('*.jpg'):
                shutil.move(str(img), str(train_extract_dir / img.name))
            train_subdir.rmdir()

    if test_zip.exists():
        logging.info("Extracting test images...")
        test_extract_dir = carvana_dir / 'test_imgs'
        test_extract_dir.mkdir(exist_ok=True)
        downloader.extract_archive(test_zip, test_extract_dir)

        test_subdir = test_extract_dir / 'test'
        if test_subdir.exists():
            for img in test_subdir.glob('*.jpg'):
                shutil.move(str(img), str(test_extract_dir / img.name))
            test_subdir.rmdir()

    if train_masks_zip.exists():
        logging.info("Extracting train masks...")
        masks_dir = carvana_dir / 'train_masks'
        masks_dir.mkdir(exist_ok=True)
        downloader.extract_archive(train_masks_zip, masks_dir)

        masks_subdir = masks_dir / 'train_masks'
        if masks_subdir.exists():
            for mask in masks_subdir.glob('*_mask.gif'):
                shutil.move(str(mask), str(masks_dir / mask.name))
            masks_subdir.rmdir()

    if train_imgs_dir.exists():
        train_images = sorted(list(train_imgs_dir.glob('*.jpg')))

        if len(train_images) > 0:
            logging.info(f"Found {len(train_images)} training images")

            final_train_dir.mkdir(exist_ok=True)
            final_val_dir.mkdir(exist_ok=True)

            random.seed(42)
            # Split by vehicle ID: Carvana has 16 near-identical rotated views per
            # car sharing the '<carID>_NN' stem, so a per-image split would leak
            # views of the same car across train and val. Keep whole cars together.
            groups = {}
            for img in train_images:
                groups.setdefault(img.stem.split('_')[0], []).append(img)
            car_ids = sorted(groups.keys())
            random.shuffle(car_ids)

            val_split = 0.1
            target_val = int(len(train_images) * val_split)
            val_images, kept_train = [], []
            for cid in car_ids:
                if len(val_images) < target_val:
                    val_images.extend(groups[cid])
                else:
                    kept_train.extend(groups[cid])
            train_images = kept_train

            logging.info(f"Copying {len(train_images)} images to train...")
            for img in tqdm(train_images, desc="Copy train images"):
                shutil.copy2(img, final_train_dir / img.name)

            logging.info(f"Copying {len(val_images)} images to val...")
            for img in tqdm(val_images, desc="Copy val images"):
                shutil.copy2(img, final_val_dir / img.name)

            if test_imgs_dir.exists():
                test_images = list(test_imgs_dir.glob('*.jpg'))
                if len(test_images) > 0:
                    final_test_dir.mkdir(exist_ok=True)

                    sample_size = min(1000, len(test_images))
                    sampled_test = random.sample(test_images, sample_size)

                    logging.info(f"Sampling {sample_size} test images...")
                    for img in tqdm(sampled_test, desc="Copy test images"):
                        shutil.copy2(img, final_test_dir / img.name)

    logging.info("Carvana dataset preparation complete.")


def main(args):
    data_root = Path(args.data_root)
    checksums = parse_checksum_overrides(getattr(args, 'sha256', []))

    datasets = {
        'mnist': prepare_mnist_fashion,
        'fashion_mnist': prepare_mnist_fashion,
        'div2k': prepare_div2k,
        'bsd500': prepare_bsd500,
        'celeba': prepare_celeba,
        'carvana': prepare_carvana,
    }

    if args.dataset == 'all':
        for name, func in datasets.items():
            logging.info(f"\n{'=' * 50}")
            logging.info(f"Preparing {name.upper()} dataset")
            logging.info(f"{'=' * 50}")
            func(data_root, checksums)
    else:
        if args.dataset in datasets:
            datasets[args.dataset](data_root, checksums)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}; available: {list(datasets.keys())}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare datasets for ghost imaging experiments")
    parser.add_argument('--data_root', type=str, default='./data', help="Root directory for datasets")
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'mnist', 'fashion_mnist', 'div2k', 'bsd500', 'celeba', 'carvana'],
                        help="Dataset to prepare")
    parser.add_argument(
        '--sha256', action='append', default=[], metavar='FILENAME=64_HEX',
        help="Trusted archive digest; repeat for datasets with multiple archives")
    args = parser.parse_args()

    main(args)
