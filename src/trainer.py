import torch
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

from src.metrics.metrics import compute_metrics
from src.utils.wandb_logger import WandbLogger
from src.utils.model_utils import count_parameters

class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, scheduler, device, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device
        self.config = config

        self.epochs = self.config['training']['epochs']
        self.amp = self.config['training']['amp']

        # Paths for logs and checkpoints
        self.exp_path = Path('experiments') / self.config['training']['experiment_name']
        self.checkpoint_path = Path('checkpoints') / self.config['training']['experiment_name']
        self.exp_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Initialize WandB logger if enabled
        self.use_wandb = self.config.get('logging', {}).get('use_wandb', False)
        if self.use_wandb:
            self.wandb_logger = WandbLogger(
                project_name=self.config.get('logging', {}).get('wandb_project', 'default_project'),
                experiment_name=self.config['training']['experiment_name'],
                config=self.config
            )
            # Group epoch-level metrics
            self.wandb_logger.define_metric("train/*", step_metric="epoch")
            self.wandb_logger.define_metric("val/*", step_metric="epoch")
            self.wandb_logger.define_metric("learning_rate", step_metric="epoch")
            # Step-level loss keyed by global_step
            self.wandb_logger.define_metric("train/step_loss", step_metric="global_step")

        self.writer = SummaryWriter(log_dir=self.exp_path)
        self._setup_logging()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        logging.info(f"Model parameters: {count_parameters(self.model):,}")

        self.best_metric = -1.0
        self.start_epoch = 1
        self.global_step = 0

    def _setup_logging(self):
        # force=True so an earlier root-logger config can't suppress INFO logs
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.exp_path / 'training.log'),
                logging.StreamHandler()
            ],
            force=True
        )
        logging.info("Trainer initialized.")
        logging.info(f"Log directory: {self.exp_path}")
        logging.info(f"Checkpoint directory: {self.checkpoint_path}")

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        batch_times = []

        num_batches = len(self.train_loader)
        val_times = min(self.config['training'].get('intra_epoch_val_times', 3), num_batches)
        # Compute intermediate validation points (avoid duplicating the final batch)
        val_points = [round(num_batches * (i + 1) / val_times) for i in range(val_times)]
        if val_points and val_points[-1] == num_batches:
            val_points = val_points[:-1]

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}/{self.epochs}")
        for batch_idx, batch in enumerate(pbar):
            batch_start = time.time()
            bucket = batch['bucket'].to(self.device)
            target_img = batch['image'].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.amp):
                pred_img = self.model(bucket)
                loss = self.criterion(pred_img, target_img)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            total_loss += loss.item()
            self.global_step += 1  # global step counter

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'batch_time': f"{batch_time:.3f}s"})

            # Log step-level loss to TensorBoard/WandB every 10 batches
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Loss/train_step', loss.item(), self.global_step)
                if self.use_wandb:
                    self.wandb_logger.log_metrics({
                        "train/step_loss": loss.item(),
                        "global_step": self.global_step
                    })

            # Mid-epoch validation: logs metrics/images and, for a ReduceLROnPlateau scheduler, steps it
            if (batch_idx + 1) in val_points:
                percent = int((batch_idx + 1) / num_batches * 100)
                logging.info(f"Intermediate validation: Epoch {epoch} ({percent}%)")
                val_metrics = self._validate_epoch(epoch)
                if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['psnr'])
                self.model.train()  # restore training mode after validation

        # End-of-epoch full validation and logging
        avg_loss = total_loss / len(self.train_loader)
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0.0

        self.writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        self.writer.add_scalar('Time/batch_avg', avg_batch_time, epoch)

        if self.use_wandb:
            self.wandb_logger.log_metrics({
                "train/epoch_loss": avg_loss,
                "train/epoch_avg_batch_time": avg_batch_time,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })
        logging.info(f"Epoch {epoch} completed. Avg loss: {avg_loss:.4f}, Avg batch time: {avg_batch_time:.3f}s")

    @torch.no_grad()
    def _validate_epoch(self, epoch):
        self.model.eval()
        all_metrics = {'psnr': [], 'ssim': [], 'l1': [], 'mse': []}
        val_loss = 0
        inference_times = []

        pbar = tqdm(self.val_loader, desc=f"Validate Epoch {int(epoch)}")
        for i, batch in enumerate(pbar):
            bucket = batch['bucket'].to(self.device)
            target_img = batch['image'].to(self.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_start = time.time()
            pred_img = self.model(bucket)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_time = time.time() - infer_start
            inference_times.append(infer_time)

            loss = self.criterion(pred_img, target_img)
            val_loss += loss.item()
            pred_np = pred_img.cpu().numpy()
            target_np = target_img.cpu().numpy()
            for j in range(pred_np.shape[0]):
                metrics = compute_metrics(pred_np[j], target_np[j])
                for k, v in metrics.items():
                    all_metrics[k].append(v)
            # Save images from the first batch for visualization
            if i == 0:
                num_images = min(6, target_img.size(0))
                self.writer.add_images('Images/target', target_img[:num_images], int(epoch))
                self.writer.add_images('Images/reconstruction', torch.clamp(pred_img[:num_images], 0, 1), int(epoch))
                if self.use_wandb:
                    self.wandb_logger.log_images({
                        'val/target': target_img[:num_images],
                        'val/reconstruction': pred_img[:num_images]
                    }, step=int(epoch))

        avg_metrics = {k: (sum(v) / len(v) if len(v) > 0 else 0.0) for k, v in all_metrics.items()}
        avg_val_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        avg_inference_time = (sum(inference_times) / len(inference_times) * 1000) if inference_times else 0.0

        for k, v in avg_metrics.items():
            self.writer.add_scalar(f'Metrics/{k}', v, int(epoch))
        self.writer.add_scalar('Loss/val', avg_val_loss, int(epoch))
        self.writer.add_scalar('Time/inference', avg_inference_time, int(epoch))

        if self.use_wandb:
            metrics_to_log = {f"val/{k}": v for k, v in avg_metrics.items()}
            metrics_to_log['val/loss'] = avg_val_loss
            metrics_to_log['val/inference_time_ms'] = avg_inference_time
            metrics_to_log['epoch'] = int(epoch)
            self.wandb_logger.log_metrics(metrics_to_log)

        logging.info(
            f"Validation completed. PSNR: {avg_metrics.get('psnr', 0.0):.2f}, "
            f"SSIM: {avg_metrics.get('ssim', 0.0):.4f}, Inference time: {avg_inference_time:.2f}ms"
        )
        return avg_metrics

    def _save_checkpoint(self, epoch, is_best):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config
        }
        latest_path = self.checkpoint_path / 'latest.pth'
        torch.save(state, latest_path)
        if is_best:
            best_path = self.checkpoint_path / 'best.pth'
            torch.save(state, best_path)
            logging.info(f"Saved new best model: {best_path}")

    def train(self):
        logging.info("Starting training...")
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)
            # Full validation at epoch end
            val_metrics = self._validate_epoch(epoch)
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('psnr', 0.0))
                else:
                    self.scheduler.step()
            current_metric = val_metrics.get('psnr', 0.0)
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
            self._save_checkpoint(epoch, is_best)
        self.writer.close()
        if self.use_wandb:
            self.wandb_logger.finish()
        logging.info("Training finished.")