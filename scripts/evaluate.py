import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config_parser import load_config
from src.datasets.dataset_factory import get_dataset
from src.metrics.metrics import compute_metrics
import src.models as models
import src.transforms as custom_transforms


def evaluate(args):
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

    config = checkpoint.get('config') if isinstance(checkpoint, dict) else None
    if config is None:
        if args.config is None:
            raise ValueError("Checkpoint has no embedded config; pass --config.")
        config = load_config(args.config)
    if args.dataset:
        config['data']['dataset'] = args.dataset
        config['data']['val_dir'] = f'data/{args.dataset}'
    if args.bucket_size:
        config['data']['bucket_size'] = args.bucket_size
        config['model']['params']['bucket_size'] = args.bucket_size

    model = getattr(models, config['model']['name'])(**config['model']['params']).to(device)
    state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    logging.info(f"Loaded {config['model']['name']} from {args.ckpt_path}")

    dataset = get_dataset(
        config['data']['dataset'],
        root_dir=config['data']['val_dir'],
        bucket_size=config['data']['bucket_size'],
        img_size=config['data']['img_size'],
        transform=custom_transforms.ToTensor(),
        mode='val',
    )
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'],
                        shuffle=False, num_workers=config['training']['num_workers'])

    agg = {'psnr': [], 'ssim': [], 'l1': [], 'mse': []}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_batches and i >= args.max_batches:
                break
            pred = model(batch['bucket'].to(device)).cpu().numpy()
            target = batch['image'].numpy()
            for j in range(pred.shape[0]):
                for k, v in compute_metrics(pred[j], target[j]).items():
                    agg[k].append(v)

    print(f"==== {config['model']['name']} on {config['data']['dataset']} "
          f"(bucket_size={config['data']['bucket_size']}, n={len(agg['psnr'])}) ====")
    for k in ['l1', 'mse', 'psnr', 'ssim']:
        print(f"{k.upper()}: {np.mean(agg[k]):.4f}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the validation set (PSNR/SSIM/L1/MSE)")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Checkpoint (.pth)")
    parser.add_argument('--dataset', type=str, default=None, help="Override dataset name")
    parser.add_argument('--config', type=str, default=None, help="Config file (only needed if the checkpoint has none)")
    parser.add_argument('--bucket_size', type=int, default=None, help="Override sampling rate")
    parser.add_argument('--max_batches', type=int, default=None, help="Evaluate only the first N batches (quick check)")
    args = parser.parse_args()
    evaluate(args)
