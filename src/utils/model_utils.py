import torch
import time

def count_parameters(model):
    # Return the number of trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, input_shape, device='cuda', num_runs=100):
    # Set model to eval mode and move to device
    model.eval()
    model = model.to(device)

    # Create dummy input with batch_size=1
    dummy_input = torch.randn(1, *input_shape[1:]).to(device)

    # Warm-up runs to mitigate initial overhead
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    if isinstance(device, torch.device):
        is_cuda = (device.type == 'cuda')
    else:
        is_cuda = str(device).startswith('cuda')

    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)

    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    # Average inference time in milliseconds
    avg_time = (end_time - start_time) / num_runs * 1000
    return avg_time

def calculate_flops(model, input_shape=None, model_params=None):
    """
    Works for:
        - MLP / FC / RNN / LSTM / GRU / Transformer
        - CNN / UNet
    Skips unsupported models like GCN.
    """
    try:
        from thop import profile

        # 1. Choose dummy input by the model's input contract. Every reconstruction
        #    net here consumes a flat measurement vector (1, bucket_size), so prefer
        #    bucket_size; the image-shaped branch is only for genuine image-input CNNs.
        if model_params is not None and "bucket_size" in model_params:
            bucket_size = model_params["bucket_size"]

            # FC / MLP / RNN / LSTM / GRU / Transformer / GCN / UNet recon input shape
            dummy_input = torch.randn(1, bucket_size)
        elif model_params is not None and "img_size" in model_params:
            img_size = model_params["img_size"]

            # True image-input CNN / UNet shape
            dummy_input = torch.randn(1, 1, img_size, img_size)
        else:
            # Fallback: use original input_shape (not recommended)
            dummy_input = torch.randn(1, *input_shape[1:])

        # 2. Run FLOPs calculation
        dummy_input = dummy_input.to(next(model.parameters()).device)

        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9  # convert to GFLOPs

    except Exception as e:
        print(f"[WARN] FLOPs calculation skipped: {e}")
        return None