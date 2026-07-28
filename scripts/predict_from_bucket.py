import torch
import argparse
import numpy as np
import logging
import os
from PIL import Image

from src.utils.visualization import show_reconstruction
import src.models as models


def load_bucket_signal(bucket_path, bucket_size):
    """
    Load bucket signal from .npy, .txt or .mat files.
    Support shapes like (N,), (1,N), (N,1). Truncate if longer than bucket_size.
    """
    ext = os.path.splitext(bucket_path)[-1].lower()
    if ext == '.npy':
        bucket = np.load(bucket_path)
    elif ext == '.txt':
        bucket = np.loadtxt(bucket_path)
    elif ext == '.mat':
        from scipy.io import loadmat
        mat = loadmat(bucket_path)
        candidate_names = ['bucket', 'd_B', 'b', 'data']
        bucket = None
        for name in candidate_names:
            if name in mat:
                bucket = mat[name].squeeze()
                break
        if bucket is None:
            for k, v in mat.items():
                if k.startswith('__'):
                    continue
                if isinstance(v, np.ndarray):
                    v_squeezed = v.squeeze()
                    if v_squeezed.ndim == 1 and v_squeezed.size > 1:
                        bucket = v_squeezed
                        break
        if bucket is None:
            raise ValueError("No suitable bucket variable found in .mat file.")
    else:
        raise ValueError("Only .npy, .txt, .mat formats are supported")

    bucket = bucket.squeeze()
    if bucket.shape[0] < bucket_size:
        raise ValueError(f"Bucket length is too short: {bucket.shape[0]} < {bucket_size}")
    if bucket.shape[0] > bucket_size:
        logging.warning(f"Bucket length {bucket.shape[0]} > required {bucket_size}; truncating to {bucket_size}")
        bucket = bucket[:bucket_size]
    return bucket


def predict_from_bucket(args):
    logging.info(f"Loading checkpoint: {args.ckpt_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # safe loader first; full training checkpoints need weights_only=False. Only load trusted files.
    try:
        checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=True)
    except Exception:
        logging.warning(
            "weights_only=True failed; falling back to a full (unsafe pickle) load. "
            "Only load checkpoints from a source you trust."
        )
        checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    ModelClass = getattr(models, config['model']['name'])
    model = ModelClass(**config['model']['params']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info(f"Model [{config['model']['name']}] loaded and set to eval mode.")

    img_size = config['data']['img_size']
    bucket_size = config['data']['bucket_size']

    bucket = load_bucket_signal(args.bucket_path, bucket_size)
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
    args = parser.parse_args()

    predict_from_bucket(args)