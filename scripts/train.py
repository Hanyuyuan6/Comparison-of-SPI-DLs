import torch
import argparse
import logging
import os

# Disable wandb symlink creation on some platforms before importing wandb
os.environ["WANDB_DISABLE_SYMLINKS"] = "true"

from src.utils.config_parser import load_config
from src.utils.seed import seed_everything, seed_worker
from src.datasets.dataset_factory import get_dataset
from torch.utils.data import DataLoader
import src.models as models
import src.losses as losses
import src.transforms as custom_transforms
from src.trainer import Trainer


def main(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    seed_everything(42)
    g = torch.Generator()
    g.manual_seed(42)

    config = load_config(args.config)

    # Override config with CLI arguments if provided
    if args.dataset:
        config['data']['dataset'] = args.dataset
        config['data']['train_dir'] = f'data/{args.dataset}'
        config['data']['val_dir'] = f'data/{args.dataset}'
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.num_workers is not None:
        config['training']['num_workers'] = args.num_workers
    if args.bucket_size:
        config['data']['bucket_size'] = args.bucket_size
        config['model']['params']['bucket_size'] = args.bucket_size
    if args.experiment_name:
        config['training']['experiment_name'] = args.experiment_name

    # Safety: experiments_celeba/* inherit dataset='mnist' from base_config and
    # only become CelebA via --dataset; fail loudly if the flag was forgotten.
    if 'celeba' in args.config.lower() and config['data'].get('dataset') != 'celeba':
        raise ValueError(
            f"Config '{args.config}' looks like a CelebA experiment but the resolved "
            f"dataset is '{config['data'].get('dataset')}'. Pass --dataset celeba (see README)."
        )

    logging.info("Configuration loaded.")
    transform = custom_transforms.ToTensor()

    preload = config['data'].get('preload', False)
    train_dataset = get_dataset(
        config['data']['dataset'],
        root_dir=config['data']['train_dir'],
        bucket_size=config['data']['bucket_size'],
        img_size=config['data']['img_size'],
        transform=transform,
        preload=preload,
        mode='train'
    )
    val_dataset = get_dataset(
        config['data']['dataset'],
        root_dir=config['data']['val_dir'],
        bucket_size=config['data']['bucket_size'],
        img_size=config['data']['img_size'],
        transform=transform,
        preload=preload,
        mode='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    logging.info(f"Dataset: {config['data']['dataset']}")
    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")

    device = torch.device(config['training']['device'])

    # Instantiate model, loss, optimizer and scheduler from config
    ModelClass = getattr(models, config['model']['name'])
    model = ModelClass(**config['model']['params']).to(device)

    LossClass = getattr(losses, config['loss']['name'])
    criterion = LossClass(**config['loss']['params'])

    OptimizerClass = getattr(torch.optim, config['training']['optimizer'])
    optimizer = OptimizerClass(
        model.parameters(),
        lr=config['training']['learning_rate'],
        **config['training']['optimizer_params']
    )

    scheduler_params = config['training']['scheduler_params']
    if 'T_max' in scheduler_params:
        scheduler_params['T_max'] = int(scheduler_params['T_max'])
    if 'eta_min' in scheduler_params:
        scheduler_params['eta_min'] = float(scheduler_params['eta_min'])
    SchedulerClass = getattr(torch.optim.lr_scheduler, config['training']['scheduler'])
    scheduler = SchedulerClass(optimizer, **scheduler_params)

    logging.info(f"Model [{config['model']['name']}] initialized.")

    trainer = Trainer(
        model, optimizer, criterion, train_loader, val_loader,
        scheduler, device, config
    )
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep-learning ghost imaging training script")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--dataset', type=str, default=None, help="Override dataset name")
    parser.add_argument('--epochs', type=int, default=None, help="Override number of epochs")
    parser.add_argument('--batch_size', type=int, default=None, help="Override batch size")
    parser.add_argument('--num_workers', type=int, default=None,
                        help="Override DataLoader worker processes (use 0 on Windows — see README)")
    parser.add_argument('--bucket_size', type=int, default=None,
                        help="Override measurement count / sampling rate (512=3.13%%, 1024=6.25%%, 2048=12.5%%)")
    parser.add_argument('--experiment_name', type=str, default=None,
                        help="Override checkpoint/log dir name (avoids collisions in a multi-run sweep)")
    args = parser.parse_args()
    main(args)