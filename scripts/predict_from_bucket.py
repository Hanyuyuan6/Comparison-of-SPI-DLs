import torch
import argparse
import numpy as np
import logging
import os
from PIL import Image

from src.utils.visualization import show_reconstruction
from src.utils.checkpoint import load_checkpoint
import src.models as models


def _mat_bucket_candidates(mat, bucket_size):
    """Return unambiguous one-dimensional, real numeric bucket candidates."""
    candidates = {}
    for name, value in mat.items():
        if name.startswith('__') or not isinstance(value, np.ndarray):
            continue
        squeezed = value.squeeze()
        if (squeezed.ndim == 1 and squeezed.size >= bucket_size
                and np.issubdtype(squeezed.dtype, np.number)
                and np.isrealobj(squeezed)
                and np.isfinite(squeezed[:bucket_size]).all()):
            candidates[name] = squeezed
    return candidates


def load_bucket_signal(bucket_path, bucket_size, mat_key=None):
    """
    Load bucket signal from .npy, .txt or .mat files.
    Support shapes like (N,), (1,N), (N,1). Truncate if longer than bucket_size.
    """
    ext = os.path.splitext(bucket_path)[-1].lower()
    if mat_key is not None and ext != '.mat':
        raise ValueError("--mat_key is valid only for .mat bucket files")
    if ext == '.npy':
        bucket = np.load(bucket_path)
    elif ext == '.txt':
        bucket = np.loadtxt(bucket_path)
    elif ext == '.mat':
        from scipy.io import loadmat
        mat = loadmat(bucket_path)
        candidates = _mat_bucket_candidates(mat, bucket_size)
        if mat_key is not None:
            if mat_key not in mat:
                visible = sorted(k for k in mat if not k.startswith('__'))
                raise ValueError(
                    f"MAT key {mat_key!r} was not found. Available variables: {visible}"
                )
            if mat_key not in candidates:
                value = mat[mat_key]
                raise ValueError(
                    f"MAT key {mat_key!r} is not a finite real numeric vector with at least "
                    f"{bucket_size} values (shape={getattr(value, 'shape', None)})."
                )
            bucket = candidates[mat_key]
        elif len(candidates) == 1:
            bucket = next(iter(candidates.values()))
        elif not candidates:
            raise ValueError(
                "No real numeric vector long enough for the requested bucket size "
                "was found in the .mat file."
            )
        else:
            raise ValueError(
                "Multiple eligible bucket vectors found in .mat file: "
                f"{sorted(candidates)}. Select one explicitly with --mat_key."
            )
    else:
        raise ValueError("Only .npy, .txt, .mat formats are supported")

    bucket = np.asarray(bucket).squeeze()
    if bucket.ndim != 1:
        raise ValueError(f"Bucket signal must be one-dimensional after squeeze; got shape {bucket.shape}")
    if bucket.shape[0] < bucket_size:
        raise ValueError(f"Bucket length is too short: {bucket.shape[0]} < {bucket_size}")
    if bucket.shape[0] > bucket_size:
        logging.warning(f"Bucket length {bucket.shape[0]} > required {bucket_size}; truncating to {bucket_size}")
        bucket = bucket[:bucket_size]
    if not np.issubdtype(bucket.dtype, np.number) or not np.isrealobj(bucket):
        raise ValueError("Bucket signal must contain real numeric values")
    bucket = bucket.astype(np.float32, copy=False)
    if not np.isfinite(bucket).all():
        raise ValueError("Bucket signal contains NaN or infinity")
    if float(np.ptp(bucket)) <= 0.0:
        raise ValueError("Bucket signal has zero dynamic range")
    return bucket


def predict_from_bucket(args):
    logging.info(f"Loading checkpoint: {args.ckpt_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = load_checkpoint(
        args.ckpt_path,
        map_location=device,
        allow_unsafe_pickle=getattr(args, 'allow_unsafe_pickle', False),
    )
    config = checkpoint['config']
    ModelClass = getattr(models, config['model']['name'])
    model = ModelClass(**config['model']['params']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info(f"Model [{config['model']['name']}] loaded and set to eval mode.")

    img_size = config['data']['img_size']
    bucket_size = config['data']['bucket_size']

    bucket = load_bucket_signal(
        args.bucket_path,
        bucket_size,
        mat_key=getattr(args, 'mat_key', None),
    )
    bucket = (bucket - bucket.min()) / (bucket.max() - bucket.min() + 1e-8)

    bucket_tensor = torch.from_numpy(bucket).float().unsqueeze(0).to(device)

    logging.info("Starting reconstruction...")
    with torch.no_grad():
        recon_tensor = model(bucket_tensor)
    recon_np = recon_tensor.cpu().numpy()[0]

    recon_img = recon_np.squeeze()
    if recon_img.shape != (img_size, img_size):
        recon_img = recon_img.reshape((img_size, img_size))
    recon_img = np.clip(recon_img, 0, 1)
    recon_img_uint8 = (recon_img * 255).astype(np.uint8)

    os.makedirs(args.save_dir, exist_ok=True)
    bucket_base = os.path.splitext(os.path.basename(args.bucket_path))[0]
    net_name = config['model']['name']
    save_path = os.path.join(args.save_dir, f"{bucket_base}_{net_name}_recon.png")
    Image.fromarray(recon_img_uint8).save(save_path)
    logging.info(f"Saved reconstruction to: {save_path}")

    if args.show:
        show_reconstruction(None, recon_img)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Reconstruct from bucket signal (.npy/.txt/.mat)")
    parser.add_argument('--bucket_path', type=str, required=True, help="Path to bucket file (.npy/.txt/.mat)")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Model checkpoint (.pth)")
    parser.add_argument('--save_dir', type=str, default='./recon_results', help="Save directory for reconstructions")
    parser.add_argument('--show', action='store_true', help="Display reconstruction")
    parser.add_argument(
        '--mat_key', type=str, default=None,
        help="Variable name for .mat input; required when multiple eligible vectors exist")
    parser.add_argument(
        '--allow_unsafe_pickle', action='store_true',
        help="Allow weights_only=False for a trusted legacy checkpoint (can execute arbitrary code)")
    args = parser.parse_args()

    predict_from_bucket(args)
