import argparse
from types import SimpleNamespace
from typing import Dict, List, Optional

import yaml


class ConfigNode(dict):
    def __getattr__(self, name):
        val = self.get(name)
        if isinstance(val, dict):
            return ConfigNode(val)
        return val

    def __setattr__(self, name, value):
        self[name] = value

    def merge_from_list(self, opts):
        for opt in opts:
            if "=" not in opt:
                raise ValueError(f"Option {opt} is not in the format key=val")
            key, val = opt.split("=", 1)
            key_parts = key.split(".")
            self._set_by_path(key_parts, val)

    def _set_by_path(self, keys, val):
        d = self
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        try:
            # Try to convert to int or float
            if val.isdigit():
                val = int(val)
            else:
                val = float(val)
        except ValueError:
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
        d[keys[-1]] = val


def load_config(config_file, opts):
    # Load YAML
    with open(config_file, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Wrap as ConfigNode
    config = ConfigNode(cfg_dict)

    # Merge opts
    if opts:
        config.merge_from_list(opts)

    return config
