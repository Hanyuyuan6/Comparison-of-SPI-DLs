import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.config_parser import load_config
from src.datasets.dataset_factory import get_dataset
from src.metrics.metrics import compute_metrics
from src.utils.artifacts import sha256_file, sha256_json, write_json_artifact
from src.utils.ghost_patterns import get_hadamard_matrix


def save_single_images(gt_imgs, gi_imgs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(gi_imgs)):
        gt = gt_imgs[i]
        gi = gi_imgs[i]
        plt.imsave(os.path.join(save_dir, f'GT_{i+1}.png'), gt, cmap='gray', vmin=0, vmax=1)
        plt.imsave(os.path.join(save_dir, f'TradGI_{i+1}.png'), gi, cmap='gray', vmin=0, vmax=1)
    print(f"Saved first {len(gi_imgs)} GT and GI reconstructions to {save_dir}")


def traditional_gi_reconstruction_gpu(patterns, buckets, img_size):
    B, M = buckets.shape
    expected_shape = (M, img_size ** 2)
    if tuple(patterns.shape) != expected_shape:
        raise ValueError(
            f"patterns must have shape {expected_shape} for buckets {tuple(buckets.shape)}, "
            f"got {tuple(patterns.shape)}"
        )
    img_vec = torch.matmul(buckets, patterns)
    img_gi = img_vec.view(B, img_size, img_size)
    # Remove mean and normalize to [0,1]
    img_gi = img_gi - img_gi.mean(dim=[1, 2], keepdim=True)
    img_gi = (img_gi - img_gi.amin(dim=[1, 2], keepdim=True)) / \
             (img_gi.amax(dim=[1, 2], keepdim=True) - img_gi.amin(dim=[1, 2], keepdim=True) + 1e-8)
    return img_gi.cpu().numpy()


def main(args):
    config = load_config(args.config)
    logging.info(f"Loaded config: {args.config}")

    if args.dataset:
        config['data']['dataset'] = args.dataset
        config['data']['val_dir'] = f'data/{args.dataset}'
    if getattr(args, 'bucket_size', None):
        config['data']['bucket_size'] = args.bucket_size

    dataset_name = config['data']['dataset']
    val_dir = config['data']['val_dir']
    img_size = config['data']['img_size']
    bucket_size = config['data']['bucket_size']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    save_dir = args.save_dir

    val_dataset = get_dataset(
        dataset_name,
        root_dir=val_dir,
        bucket_size=bucket_size,
        img_size=img_size,
        transform=None,
        mode='val',
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patterns = get_hadamard_matrix(img_size ** 2, bucket_size)
    patterns = torch.from_numpy(patterns).to(device)

    all_metrics = {}
    gi_images_to_save = []
    gt_images_to_save = []
    saved_images_count = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='[TradGI] Reconstruction')
        for i, batch in enumerate(pbar):
            if args.max_batches and i >= args.max_batches:
                break
            if 'bucket_raw' in batch:
                buckets = batch['bucket_raw']
            else:
                buckets = batch['bucket']
            if not isinstance(buckets, torch.Tensor):
                buckets = torch.from_numpy(buckets)
            buckets = buckets.to(device).float()
            images_gt = batch['image']
            if not isinstance(images_gt, torch.Tensor):
                images_gt = torch.from_numpy(images_gt)
            images_gt = images_gt.float()
            B = buckets.shape[0]
            img_gis = traditional_gi_reconstruction_gpu(patterns, buckets, img_size)
            for b in range(B):
                img_gi = img_gis[b]
                gt = images_gt[b, 0].cpu().numpy()
                if saved_images_count < 6:
                    gi_images_to_save.append(img_gi)
                    gt_images_to_save.append(gt)
                    saved_images_count += 1
                metrics = compute_metrics(img_gi, gt)
                for k, v in metrics.items():
                    all_metrics.setdefault(k, []).append(v)

    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    print("==== Traditional GI Evaluation ====")
    for k, v in avg_metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    save_single_images(gt_images_to_save, gi_images_to_save, save_dir)
    summary = {
        'schema_version': 1,
        'script': 'scripts.GI_rec_eval',
        'method': 'Hadamard adjoint',
        'dataset': dataset_name,
        'bucket_size': int(bucket_size),
        'split': 'val',
        'n': len(next(iter(all_metrics.values()), [])),
        'source_config_sha256': sha256_file(args.config),
        'effective_config_sha256': sha256_json(config),
        'metrics': avg_metrics,
    }
    if getattr(args, 'out_json', None):
        write_json_artifact(args.out_json, summary)
        logging.info(f"Machine-readable metrics written to {args.out_json}")
    return summary


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Traditional GI reconstruction evaluation (GPU-accelerated)")
    parser.add_argument('--config', type=str, required=True, help="Config file path")
    parser.add_argument('--dataset', type=str, default=None, help="Override dataset name")
    parser.add_argument('--bucket_size', type=int, default=None, help="Override measurement count / sampling rate")
    parser.add_argument('--save_dir', type=str, default='./gi_vis', help="Directory to save images")
    parser.add_argument('--max_batches', type=int, default=None, help="Evaluate only the first N batches (quick check)")
    parser.add_argument('--out_json', type=str, default=None, help="Write metrics and config hash as JSON")
    args = parser.parse_args()
    main(args)
