import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_metrics(pred, gt):
    """
    Compute common quality metrics between prediction and ground truth.
    """
    pred = np.clip(pred.squeeze(), 0, 1)
    gt = np.clip(gt.squeeze(), 0, 1)

    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim = structural_similarity(gt, pred, data_range=1.0)
    l1 = np.mean(np.abs(pred - gt))
    mse = np.mean((pred - gt) ** 2)

    return {'psnr': psnr, 'ssim': ssim, 'l1': l1, 'mse': mse}