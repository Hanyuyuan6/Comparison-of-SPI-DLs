import pytest
import torch
import sys
import torch.nn as nn

sys.path.append('.')

import src.models as models
from src.utils.model_utils import count_parameters

# Exact parameter counts at the released setting (bucket_size=512, img_size=128),
# pinning the released model configurations. Comment shows the rounded count in millions.
EXPECTED_PARAMS = {
    'FNReconstructionNet':          136392704,  # FC  136.4
    'FCNReconstructionNet':         141114439,  # CNN 141.1
    'UNetReconstructionNet':        132524161,  # U-Net 132.5
    'RNNReconstructionNet':         135309179,  # RNN 135.3
    'LSTMReconstructionNet':        138464123,  # LSTM 138.5
    'GRUReconstructionNet':         137412475,  # GRU 137.4
    'TransformerReconstructionNet': 146878587,  # Transformer 146.9
    'GCNReconstructionNet':         136372347,  # GCN 136.4
}


class TestModels:
    """Model unit tests."""

    @pytest.fixture
    def test_config(self):
        return {
            'bucket_size': 512,
            'img_size': 128,
            'batch_size': 1
        }

    def test_all_models_consistency(self, test_config):
        model_configs = {
            'FNReconstructionNet': {'hidden_dims': [4096, 4096, 4096, 4096, 4096]},
            'FCNReconstructionNet': {'base_channels': 1024},
            'UNetReconstructionNet': {'bilinear': False},
            'RNNReconstructionNet': {'hidden_size': 256, 'num_layers': 3},
            'LSTMReconstructionNet': {'hidden_size': 256, 'num_layers': 3},
            'GRUReconstructionNet': {'hidden_size': 256, 'num_layers': 3},
            'TransformerReconstructionNet': {'d_model': 512, 'nhead': 16, 'num_layers': 6},
            'GCNReconstructionNet': {'hidden_channels': 512, 'num_layers': 8}
        }

        x = torch.randn(test_config['batch_size'], test_config['bucket_size'])

        for model_name, extra_params in model_configs.items():
            try:
                ModelClass = getattr(models, model_name)

                model = ModelClass(
                    bucket_size=test_config['bucket_size'],
                    img_size=test_config['img_size'],
                    **extra_params
                )

                # set to evaluation mode
                model.eval()

                # forward pass
                with torch.no_grad():
                    y = model(x)

                # verify output shape
                assert y.shape == (test_config['batch_size'], 1,
                                   test_config['img_size'], test_config['img_size'])

                # verify output range
                assert torch.all(y >= 0) and torch.all(y <= 1)

                # verify parameter count matches the released value
                num_params = count_parameters(model)
                print(f"{model_name}: parameters = {num_params:,}")
                assert num_params == EXPECTED_PARAMS[model_name], (
                    f"{model_name} param count {num_params:,} != expected "
                    f"{EXPECTED_PARAMS[model_name]:,} (released configuration). A config or "
                    f"architecture change has altered a released parameter count.")

            except Exception as e:
                print(f"\nError occurred in {model_name}:")
                print(f"Error message: {str(e)}")
                if 'model' in locals():
                    print("\nModel modules (showing BatchNorm modules):")
                    for name, module in model.named_modules():
                        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                            print(f"{name}: {module} (num_features={module.num_features})")
                raise


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
