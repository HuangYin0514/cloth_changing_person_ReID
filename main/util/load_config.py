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

    def __str__(self):
        """重写字符串输出方法，让打印结果自动换行、格式化"""
        return self._pretty_print(self, indent=0)

    def __repr__(self):
        """保持repr和str输出一致"""
        return self.__str__()

    def _pretty_print(self, obj, indent):
        """递归生成格式化的字符串，实现自动换行"""
        indent_str = "  " * indent  # 每层缩进2个空格
        if isinstance(obj, ConfigNode):
            items = []
            for key, value in obj.items():
                # 递归处理每个值，缩进+1
                items.append(f"{indent_str}{key}: {self._pretty_print(value, indent + 1)}")
            # 每个键值对换行显示
            return "\n".join(items)
        elif isinstance(obj, dict):
            # 兼容普通字典的格式化
            items = []
            for key, value in obj.items():
                items.append(f"{indent_str}{key}: {self._pretty_print(value, indent + 1)}")
            return "\n".join(items)
        else:
            # 基础类型直接返回值
            return str(obj)


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
