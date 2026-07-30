import argparse
import hashlib
import json
import logging
import re
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm
import random
import shutil
import stat

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CELEBA_URL = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"
# SHA-256 of celeba.zip. This third-party mirror has no publisher-authenticated
# digest in this repository, so callers must supply one from a trusted source.
CELEBA_SHA256 = None
OUTPUT_DIR = Path('./data/celeba')
SHA256_RE = re.compile(r'^[0-9a-fA-F]{64}$')
EXPECTED_TOTAL = 202_599
EXPECTED_TRAIN = 182_339
EXPECTED_VAL = 20_260
PREPARED_META = '_prepared_meta.json'


def _name_digest(paths):
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _content_digest(paths):
    """Bind the manifest to every prepared filename, byte count, and file hash."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Invalid or empty CelebA image: {path}")
        digest.update(path.name.encode('utf-8'))
        digest.update(b'\0')
        digest.update(str(path.stat().st_size).encode('ascii'))
        digest.update(b'\0')
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _validate_prepared(output_path):
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    meta_path = output_path / PREPARED_META
    has_any = any(path.exists() and any(path.iterdir()) for path in (train_dir, val_dir))
    if not has_any and not meta_path.exists():
        return False
    if not (train_dir.is_dir() and val_dir.is_dir() and meta_path.is_file()):
        raise RuntimeError(
            f"Partial or unverified CelebA preparation under {output_path}. Expected non-empty "
            f"train/, val/, and {PREPARED_META}; rebuild in a clean output directory."
        )
    train_files = sorted(train_dir.glob('*.jpg'))
    val_files = sorted(val_dir.glob('*.jpg'))
    unexpected = [
        path for directory in (train_dir, val_dir) for path in directory.iterdir()
        if not (not path.is_symlink() and path.is_file() and path.suffix.lower() == '.jpg')
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected entries in prepared CelebA split: {unexpected[:5]}")
    if len(train_files) != EXPECTED_TRAIN or len(val_files) != EXPECTED_VAL:
        raise RuntimeError(
            f"Incomplete CelebA split: train={len(train_files)} (expected {EXPECTED_TRAIN}), "
            f"val={len(val_files)} (expected {EXPECTED_VAL})"
        )
    train_names = {path.name for path in train_files}
    val_names = {path.name for path in val_files}
    if train_names.intersection(val_names):
        raise RuntimeError("CelebA train/val filename sets overlap")
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid CelebA preparation manifest: {meta_path}") from exc
    expected = {
        'schema_version': 2,
        'source_archive': 'celeba.zip',
        'random_seed': 42,
        'val_split': 0.1,
        'total': EXPECTED_TOTAL,
        'train_count': EXPECTED_TRAIN,
        'val_count': EXPECTED_VAL,
        'train_names_sha256': _name_digest(train_files),
        'val_names_sha256': _name_digest(val_files),
        'train_content_sha256': _content_digest(train_files),
        'val_content_sha256': _content_digest(val_files),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(
                f"CelebA preparation manifest mismatch for {key}: "
                f"{meta.get(key)!r} != {value!r}"
            )
    source_digest = str(meta.get('source_archive_sha256', ''))
    if not SHA256_RE.fullmatch(source_digest):
        raise RuntimeError("CelebA preparation manifest has no valid source archive SHA-256")
    source_archive = output_path / meta['source_archive']
    if not source_archive.is_file() or _file_sha256(source_archive) != source_digest.lower():
        raise RuntimeError("CelebA source archive is missing or does not match the manifest")
    return True


def _validated_sha256(value):
    if not value:
        raise ValueError(
            "No trusted SHA-256 was supplied for celeba.zip. Refusing to download "
            "or extract an unverified third-party archive; pass --sha256 64_HEX."
        )
    if not SHA256_RE.fullmatch(value):
        raise ValueError("--sha256 must contain exactly 64 hexadecimal characters")
    return value.lower()


def _verify_checksum(path, expected_sha256):
    """Verify an archive before extraction; never trust existence alone."""
    expected_sha256 = _validated_sha256(expected_sha256)
    if _file_sha256(path) != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for {path.name}; the archive was not extracted."
        )
    logging.info(f"Checksum OK: {path.name}")
    return True


def download_file(url, destination, expected_sha256):
    expected_sha256 = _validated_sha256(expected_sha256)
    if destination.exists():
        _verify_checksum(destination, expected_sha256)
        logging.info(f"Verified existing archive: {destination}")
        return True

    partial_path = destination.with_name(destination.name + '.part')
    logging.info(f"Downloading from {url}...")
    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(partial_path, 'wb') as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc=partial_path.name
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        _verify_checksum(partial_path, expected_sha256)
        partial_path.replace(destination)
        logging.info(f"Downloaded to: {destination}")
        return True
    except Exception as e:
        if partial_path.exists():
            partial_path.unlink()
        raise RuntimeError(f"CelebA download or verification failed: {e}") from e


def prepare_celeba(output_path, val_split=0.1, random_seed=42, expected_sha256=None):
    output_path.mkdir(parents=True, exist_ok=True)
    zip_path = output_path / 'celeba.zip'
    img_dir = output_path / 'img_align_celeba'
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'

    if _validate_prepared(output_path):
        logging.info("Verified complete CelebA train/val preparation manifest.")
        return

    expected_sha256 = _validated_sha256(expected_sha256 or CELEBA_SHA256)
    download_file(CELEBA_URL, zip_path, expected_sha256)

    if not img_dir.is_dir() or not any(img_dir.iterdir()):
        logging.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            _root = output_path.resolve()
            for _member in zip_ref.infolist():
                _name = _member.filename
                if not (_root / _name).resolve().is_relative_to(_root):
                    raise ValueError(f"Unsafe path in archive (path traversal blocked): {_name}")
                if stat.S_ISLNK(_member.external_attr >> 16):
                    raise ValueError(f"Unsafe symbolic link in ZIP archive: {_name}")
            zip_ref.extractall(output_path)
        logging.info("Extraction complete.")
    else:
        logging.info("Image directory exists, skipping extraction.")

    logging.info("Splitting train/val sets...")
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    image_files = sorted(list(img_dir.glob('*.jpg')))
    if len(image_files) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"CelebA archive produced {len(image_files)} .jpg files under {img_dir}; "
            f"expected {EXPECTED_TOTAL} for the released protocol"
        )
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

    prepared_train_files = sorted(train_dir.glob('*.jpg'))
    prepared_val_files = sorted(val_dir.glob('*.jpg'))
    meta = {
        'schema_version': 2,
        'source_archive': zip_path.name,
        'source_archive_sha256': expected_sha256,
        'random_seed': random_seed,
        'val_split': val_split,
        'total': len(image_files),
        'train_count': len(train_files),
        'val_count': len(val_files),
        'train_names_sha256': _name_digest(prepared_train_files),
        'val_names_sha256': _name_digest(prepared_val_files),
        'train_content_sha256': _content_digest(prepared_train_files),
        'val_content_sha256': _content_digest(prepared_val_files),
    }
    temporary = (output_path / PREPARED_META).with_suffix('.json.tmp')
    temporary.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(output_path / PREPARED_META)
    _validate_prepared(output_path)
    logging.info("CelebA preparation complete and verified.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare CelebA dataset")
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR), help="Output directory for CelebA")
    parser.add_argument('--sha256', type=str, default=None, help="Trusted SHA-256 for celeba.zip")
    args = parser.parse_args()
    prepare_celeba(Path(args.output_dir), expected_sha256=args.sha256)
