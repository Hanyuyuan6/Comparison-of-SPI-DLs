import matplotlib.pyplot as plt
import numpy as np

def show_reconstruction(target, recon, save_path=None):
    # Convert tensors to numpy arrays if needed
    if hasattr(target, 'cpu'):
        target = target.cpu().detach().numpy()
    if hasattr(recon, 'cpu'):
        recon = recon.cpu().detach().numpy()

    if target is None:
        plt.figure(figsize=(5, 5))
        plt.imshow(recon.squeeze(), cmap='gray')
        plt.title('Reconstruction')
        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
        plt.show()
    else:
        # Create a 1x3 subplot: target, reconstruction and absolute difference
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(target.squeeze(), cmap='gray')
        axs[0].set_title('Target')
        axs[1].imshow(recon.squeeze(), cmap='gray')
        axs[1].set_title('Reconstruction')
        axs[2].imshow(np.abs(target - recon).squeeze(), cmap='hot')
        axs[2].set_title('Abs Diff')

        for ax in axs:
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        plt.show()