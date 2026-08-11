import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config_parser import load_config
from src.datasets.dataset_factory import get_dataset
from src.metrics.metrics import compute_metrics
from src.utils.artifacts import sha256_file, sha256_json, write_json_artifact
from src.utils.checkpoint import load_checkpoint
import src.models as models
import src.transforms as custom_transforms


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = load_checkpoint(
        args.ckpt_path,
        map_location=device,
        allow_unsafe_pickle=getattr(args, 'allow_unsafe_pickle', False),
    )

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

    metrics = {k: float(np.mean(agg[k])) for k in ['l1', 'mse', 'psnr', 'ssim']}
    summary = {
        'schema_version': 1,
        'script': 'scripts.evaluate',
        'model': config['model']['name'],
        'dataset': config['data']['dataset'],
        'bucket_size': int(config['data']['bucket_size']),
        'split': 'val',
        'experiment_name': config['training']['experiment_name'],
        'seed': int(config['training'].get('seed', 42)),
        'configured_epochs': int(config['training']['epochs']),
        'checkpoint_epoch': int(checkpoint.get('epoch', 0)),
        'n': len(agg['psnr']),
        'checkpoint_sha256': sha256_file(args.ckpt_path),
        'effective_config_sha256': sha256_json(config),
        'metrics': metrics,
    }

    print(f"==== {config['model']['name']} on {config['data']['dataset']} "
          f"(bucket_size={config['data']['bucket_size']}, n={len(agg['psnr'])}) ====")
    for k in ['l1', 'mse', 'psnr', 'ssim']:
        print(f"{k.upper()}: {metrics[k]:.4f}")

    if getattr(args, 'out_json', None):
        write_json_artifact(args.out_json, summary)
        logging.info(f"Machine-readable metrics written to {args.out_json}")
    return summary


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the validation set (PSNR/SSIM/L1/MSE)")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Checkpoint (.pth)")
    parser.add_argument('--dataset', type=str, default=None, help="Override dataset name")
    parser.add_argument('--config', type=str, default=None, help="Config file (only needed if the checkpoint has none)")
    parser.add_argument('--bucket_size', type=int, default=None, help="Override sampling rate")
    parser.add_argument('--max_batches', type=int, default=None, help="Evaluate only the first N batches (quick check)")
    parser.add_argument('--out_json', type=str, default=None, help="Write metrics and checkpoint hash as JSON")
    parser.add_argument(
        '--allow_unsafe_pickle', action='store_true',
        help="Allow weights_only=False for a trusted legacy checkpoint (can execute arbitrary code)")
    args = parser.parse_args()
    evaluate(args)
