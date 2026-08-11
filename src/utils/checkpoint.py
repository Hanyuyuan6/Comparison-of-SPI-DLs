import logging

import torch


def load_checkpoint(path, map_location, allow_unsafe_pickle=False):
    """Load a checkpoint without executing pickle payloads by default.

    PyTorch's ``weights_only=True`` loader accepts tensors and basic container
    types used by this project's checkpoints.  A full pickle load is available
    only as an explicit opt-in for legacy checkpoints from a trusted source.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as safe_error:
        if not allow_unsafe_pickle:
            raise RuntimeError(
                "Safe checkpoint loading (weights_only=True) failed. Refusing to "
                "fall back to pickle because it can execute arbitrary code. If and "
                "only if this checkpoint comes from a trusted source, re-run with "
                "--allow_unsafe_pickle."
            ) from safe_error

        logging.warning(
            "Safe checkpoint loading failed; --allow_unsafe_pickle was explicitly "
            "set, so a full pickle load will be attempted. Only trusted checkpoints "
            "may be loaded this way."
        )
        return torch.load(path, map_location=map_location, weights_only=False)
