import argparse
import logging
import os

import numpy as np
import torch
import cvxpy as cp
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.config_parser import load_config
from src.datasets.dataset_factory import get_dataset
from src.metrics.metrics import compute_metrics
from src.utils.ghost_patterns import get_hadamard_matrix


def save_single_images(gt_imgs, rec_imgs, save_dir, prefix="CS"):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(rec_imgs)):
        gt = gt_imgs[i]
        rec = rec_imgs[i]
        plt.imsave(os.path.join(save_dir, f'GT_{i+1}.png'), gt, cmap='gray', vmin=0, vmax=1)
        plt.imsave(os.path.join(save_dir, f'{prefix}_{i+1}.png'), rec, cmap='gray', vmin=0, vmax=1)
    print(f"Saved first {len(rec_imgs)} GT and {prefix} reconstructions to {save_dir}")


def cs_tv_reconstruction_gpu(patterns, buckets, img_size, tv_weight=0.1, max_iter=200, lr=0.2):
    """
    TV-based iterative reconstruction using gradient-based optimization on GPU.
    patterns: [M, N], buckets: [B, M]
    Returns: [B, H, W]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patterns_torch = torch.from_numpy(patterns).float().to(device)
    B = buckets.shape[0]
    M, N = patterns.shape
    rec_imgs = []
    for b in range(B):
        y = torch.from_numpy(buckets[b]).float().to(device)
        x = torch.zeros(N, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=lr)
        for _ in range(max_iter):
            optimizer.zero_grad()
            data_loss = torch.mean((patterns_torch @ x - y) ** 2)
            x_img = x.view(img_size, img_size)
            tv_loss = torch.sum(torch.abs(x_img[:, :-1] - x_img[:, 1:])) + \
                      torch.sum(torch.abs(x_img[:-1, :] - x_img[1:, :]))
            loss = data_loss + tv_weight * tv_loss
            loss.backward()
            optimizer.step()
        x_img = x.detach().cpu().numpy().reshape(img_size, img_size)
        x_img = np.clip(x_img, 0, 1)
        rec_imgs.append(x_img)
    return np.stack(rec_imgs, axis=0)


def cs_admm_reconstruction(patterns, buckets, img_size, lmbd=0.01, max_iter=200):
    """
    L1-regularized least squares / basis-pursuit denoising for sparse recovery:
        min_x  0.5 || Phi x - y ||_2^2 + lmbd || x ||_1
    Solved by cvxpy's default convex solver. NOTE: this is NOT a hand-written
    ADMM -- the underlying algorithm is whatever cvxpy dispatches to (e.g. OSQP,
    which is ADMM-based, or an interior-point solver), and the L1 sparsity is in
    the pixel domain (no DCT/wavelet transform). The '--method admm' CLI name is
    kept for backward compatibility; 'l1' would be the more accurate label.
    patterns: [M, N], buckets: [B, M]
    Returns: [B, H, W]
    """
    M, N = patterns.shape
    B = buckets.shape[0]
    rec_imgs = []
    for b in range(B):
        y = buckets[b]
        x = cp.Variable(N)
        objective = cp.Minimize(0.5 * cp.sum_squares(patterns @ x - y) + lmbd * cp.norm1(x))
        prob = cp.Problem(objective)
        prob.solve(verbose=False, max_iter=max_iter)
        x_rec = x.value
        if x_rec is None:
            x_rec = np.zeros(N)
        img = x_rec.reshape(img_size, img_size)
        img = np.clip(img, 0, 1)
        rec_imgs.append(img)
    return np.stack(rec_imgs, axis=0)


def main(args):
    config = load_config(args.config)
    logging.info(f"Loaded config: {args.config}")

    if args.dataset:
        config['data']['dataset'] = args.dataset
        config['data']['val_dir'] = f'data/{args.dataset}'

    dataset_name = config['data']['dataset']
    val_dir = config['data']['val_dir']
    img_size = config['data']['img_size']
    bucket_size = config['data']['bucket_size']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    save_dir = args.save_dir

    rec_method = args.method.lower()
    assert rec_method in ['tv', 'admm'], "Method must be 'tv' or 'admm'"

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

    N = img_size ** 2
    patterns = get_hadamard_matrix(N, bucket_size).astype(np.float32)
    assert patterns.shape == (bucket_size, N), f"Patterns shape should be ({bucket_size}, {N}), got {patterns.shape}"

    all_metrics = {}
    rec_images_to_save = []
    gt_images_to_save = []
    saved_images_count = 0

    pbar = tqdm(val_loader, desc=f'[CS-{rec_method.upper()}] Reconstruction')
    for i, batch in enumerate(pbar):
        if args.max_batches and i >= args.max_batches:
            break
        if 'bucket_raw' in batch:
            buckets = batch['bucket_raw']
        else:
            buckets = batch['bucket']
        if isinstance(buckets, torch.Tensor):
            buckets = buckets.cpu().numpy()
        images_gt = batch['image']
        if isinstance(images_gt, torch.Tensor):
            images_gt = images_gt.cpu().numpy()
        B = buckets.shape[0]
        buckets = buckets.astype(np.float32)

        if rec_method == 'tv':
            rec_imgs = cs_tv_reconstruction_gpu(patterns, buckets, img_size)
        else:
            rec_imgs = cs_admm_reconstruction(patterns, buckets, img_size)

        for b in range(B):
            rec_img = rec_imgs[b]
            gt = images_gt[b, 0]
            if saved_images_count < 6:
                rec_images_to_save.append(rec_img)
                gt_images_to_save.append(gt)
                saved_images_count += 1
            metrics = compute_metrics(rec_img, gt)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)

    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    print(f"==== CS-{rec_method.upper()} Evaluation ====")
    for k, v in avg_metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    save_single_images(gt_images_to_save, rec_images_to_save, save_dir, prefix=f"CS-{rec_method.upper()}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Compressed-sensing GI reconstruction: TV (gradient descent) or "
                    "L1/BPDN via cvxpy (CLI name 'admm', kept for compatibility)")
    parser.add_argument('--config', type=str, required=True, help="Config file path")
    parser.add_argument('--dataset', type=str, default=None, help="Override dataset name")
    parser.add_argument('--save_dir', type=str, default='./cs_gi_vis', help="Save directory")
    parser.add_argument('--method', type=str, required=True, choices=['tv', 'admm'],
                        help="Reconstruction method: 'tv' = TV-regularized gradient descent; "
                             "'admm' = pixel-domain L1/BPDN via cvxpy (name kept for compatibility)")
    parser.add_argument('--max_batches', type=int, default=None, help="Reconstruct only the first N batches (quick check)")
    args = parser.parse_args()
    main(args)