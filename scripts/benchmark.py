import torch
import argparse
import logging
from pathlib import Path

import src.models as models
from src.utils.artifacts import write_json_artifact
from src.utils.model_utils import count_parameters, measure_inference_time, calculate_flops
from src.utils.config_parser import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def benchmark_model(model_name, model_params, device):
    """Profile a model: parameter count, inference latency, FLOPs. Weight-independent
    (no checkpoints, no PSNR/SSIM). Hyperparameters come from the training YAMLs, so the
    parameter counts match the released configurations."""
    logging.info(f"Benchmarking model: {model_name}")

    ModelClass = getattr(models, model_name)
    model = ModelClass(**model_params).to(device)

    num_params = count_parameters(model)

    input_shape = (1, model_params['bucket_size'])
    inference_time = measure_inference_time(model, input_shape, device)

    flops = calculate_flops(model, input_shape, model_params)

    results = {
        'model_name': model_name,
        'num_params': num_params,
        'inference_time_ms': inference_time,
        'flops_G': flops,
    }

    logging.info(f"Parameters: {num_params:,}")
    logging.info(f"Inference time: {inference_time:.2f} ms")
    logging.info(f"FLOPs: {flops:.2f} G" if flops else "FLOPs: N/A")

    return results


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_dir = Path(args.config_dir)
    # profile each backbone from its training config, in the released benchmark order
    model_keys = ['fn', 'fcn', 'unet', 'gcn', 'transformer', 'rnn', 'lstm', 'gru']

    all_results = []
    for key in model_keys:
        cfg_path = config_dir / f'{key}.yaml'
        if not cfg_path.exists():
            raise FileNotFoundError(f"Required benchmark config not found: {cfg_path}")
        cfg = load_config(str(cfg_path))
        model_name = cfg['model']['name']
        model_params = dict(cfg['model']['params'])
        if args.bucket_size is not None:
            model_params['bucket_size'] = args.bucket_size
        results = benchmark_model(model_name, model_params, device)
        results['config'] = str(cfg_path).replace('\\', '/')
        results['bucket_size'] = model_params['bucket_size']
        all_results.append(results)

    output_file = Path(args.output_dir) / f'benchmark_results_{args.dataset}.json'
    write_json_artifact(output_file, all_results)

    logging.info(f"Benchmark results saved to: {output_file}")

    print("\n" + "=" * 80)
    print(f"{'Model':<30} {'Params':<15} {'Time(ms)':<15} {'FLOPs(G)':<10}")
    print("=" * 80)

    for result in all_results:
        flops = result.get('flops_G')
        flops_str = f"{flops:<10.2f}" if flops else f"{'N/A':<10}"
        print(f"{result['model_name']:<30} "
              f"{result['num_params']:<15,} "
              f"{result['inference_time_ms']:<15.2f} "
              f"{flops_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Model profiler (params / latency / FLOPs). Reads hyperparameters "
                    "from the training YAMLs so the Params column matches the released configurations.")
    parser.add_argument('--config_dir', type=str, default='configs/experiments_mnist',
                        help="Directory holding the 8 per-model YAMLs to profile")
    parser.add_argument('--dataset', type=str, default='mnist', help="Label for the output filename")
    parser.add_argument('--bucket_size', type=int, default=None,
                        help="Override the measurement count (default: use the config's 512 = 3.13%%; "
                             "pass 1024 for 6.25%%, 2048 for 12.5%%)")
    parser.add_argument('--output_dir', type=str, default='./benchmark_results', help="Output directory for results")

    args = parser.parse_args()
    main(args)
