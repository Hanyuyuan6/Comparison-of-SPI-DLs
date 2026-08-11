import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import ssim_loss

class CombinedReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss that blends L1, L2 and SSIM terms.
    """
    def __init__(self, l1_weight=0.5, l2_weight=0.5, ssim_weight=0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.ssim_weight = ssim_weight

        if self.ssim_weight > 0:
            # SSIM loss function using a sliding window of size 5
            self.ssim_fn = lambda pred, target: ssim_loss(pred, target, window_size=5)

    def forward(self, pred, target):
        loss = 0.0

        if self.l1_weight > 0:
            # Mean absolute error
            loss += self.l1_weight * F.l1_loss(pred, target)

        if self.l2_weight > 0:
            # Mean squared error
            loss += self.l2_weight * F.mse_loss(pred, target)

        if self.ssim_weight > 0:
            # NOTE: kornia's ssim_loss returns DSSIM = (1 - SSIM) / 2, NOT (1 - SSIM).
            # The effective SSIM weight is therefore ssim_weight / 2. Kept as-is so
            # the released runs stay reproducible.
            ssim_val = self.ssim_fn(pred, target).mean()
            loss += self.ssim_weight * ssim_val

        return loss