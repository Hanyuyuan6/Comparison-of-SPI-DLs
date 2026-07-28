import torch
import argparse
from PIL import Image
import numpy as np
import logging
import os

from src.utils.ghost_patterns import get_hadamard_matrix
from src.utils.visualization import show_reconstruction
from src.metrics.metrics import compute_metrics
import src.models as models


def predict(args):
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

    img = Image.open(args.img_path).convert('L').resize((img_size, img_size), Image.Resampling.LANCZOS)
    img_np = np.array(img, dtype=np.float32) / 255.0

    patterns = get_hadamard_matrix(img_size ** 2, bucket_size)
    bucket = np.dot(patterns, img_np.flatten())
    bucket = (bucket - bucket.min()) / (bucket.max() - bucket.min() + 1e-8)
    bucket_tensor = torch.from_numpy(bucket).float().unsqueeze(0).to(device)

    logging.info("Starting reconstruction...")
    with torch.no_grad():
        recon_tensor = model(bucket_tensor)
    recon_np = recon_tensor.cpu().numpy()[0]

    metrics = compute_metrics(recon_np, img_np)
    logging.info(f"Done. Metrics: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")

    show_reconstruction(img_np, recon_np.squeeze())

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    recon_img = recon_np.squeeze()
    if recon_img.shape != (img_size, img_size):
        recon_img = recon_img.reshape((img_size, img_size))
    recon_img = np.clip(recon_img, 0, 1)
    recon_img_uint8 = (recon_img * 255).astype(np.uint8)
    img_basename = os.path.splitext(os.path.basename(args.img_path))[0]
    save_path = os.path.join(save_dir, f"{img_basename}_{config['model']['name']}_recon.png")
    Image.fromarray(recon_img_uint8).save(save_path)
    logging.info(f"Saved reconstruction to: {save_path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Deep-learning ghost imaging prediction script")
    parser.add_argument('--img_path', type=str, required=True, help="Input image path")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Model checkpoint (.pth)")
    parser.add_argument('--save_dir', type=str, default='./recon_results', help="Save directory for reconstructions")
    args = parser.parse_args()

    predict(args)