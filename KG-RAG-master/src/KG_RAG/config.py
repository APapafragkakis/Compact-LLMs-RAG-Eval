from collections.abc import Mapping

def merge_configs(default, override):
    for key, value in override.items():
        if isinstance(value, Mapping) and key in default:
            default[key] = merge_configs(default.get(key, {}), value)
        else:
            default[key] = value
    return default

class Config:
    def __init__(self, default_config):
        self.config = default_config

    def update(self, updates):
        self.config = merge_configs(self.config, updates)