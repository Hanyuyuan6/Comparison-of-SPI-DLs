# wandb is optional (only used when logging.use_wandb=True)
try:
    import wandb
except ImportError:
    wandb = None
import torch

class WandbLogger:
    def __init__(self, project_name, experiment_name, config):
        if wandb is None:
            raise ImportError(
                "wandb is not installed but logging.use_wandb=True. "
                "Install it (`pip install wandb`) or set logging.use_wandb=False to use TensorBoard only."
            )
        # Initialize a Weights & Biases run
        self.run = wandb.init(
            project=project_name,
            name=experiment_name,
            config=config
        )
        self.config = config

    def define_metric(self, metric_name, step_metric):
        """
        Define a custom x-axis (step metric) for a given metric name.
        metric_name may contain wildcards (e.g., 'val/*').
        """
        self.run.define_metric(metric_name, step_metric=step_metric)

    def log_metrics(self, metrics_dict):
        """
        Log scalar metrics to W&B.
        If define_metric was used, metrics_dict should include the corresponding x-axis value.
        """
        wandb.log(metrics_dict)

    def log_images(self, images_dict, step):
        """
        Log images to W&B.
        images_dict: dict where values are tensors or numpy arrays.
        step: integer step/epoch to associate with the logged images.
        """
        wandb_images = {}
        for key, images in images_dict.items():
            if isinstance(images, torch.Tensor):
                images = images.cpu().numpy()

            # Expect images in shape (B, C, H, W) for a batch or HxW[xC] for a single image
            if images.ndim == 4:
                images = images[:6]
                converted = []
                for img in images:
                    # If 3-channel, transpose to HWC; if single-channel, squeeze to HxW
                    if img.shape[0] == 3:
                        img_hwc = img.transpose(1, 2, 0)
                        converted.append(wandb.Image(img_hwc))
                    else:
                        converted.append(wandb.Image(img.squeeze()))
                wandb_images[key] = converted
            else:
                wandb_images[key] = [wandb.Image(images)]

        # Log images along with the step (epoch)
        self.log_metrics({**wandb_images, "epoch": step})

    def finish(self):
        wandb.finish()