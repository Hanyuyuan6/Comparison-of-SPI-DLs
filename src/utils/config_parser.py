import yaml
from pathlib import Path

def load_config(config_path):
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if 'base' in config:
        base_config_path = config_path.parent / config['base']
        base_config = load_config(base_config_path)
        deep_update(base_config, config)
        config = base_config

    return config

def deep_update(base_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value