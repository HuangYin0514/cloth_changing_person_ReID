import torch
import torch.nn as nn
from loss import CrossEntropyLabelSmooth, TripletLoss


class Build_Criterion:
    def __init__(self, config, *args, **kwargs):
        self.build(config, *args, **kwargs)

    def build(self, config, num_classes):
        self.ce = nn.CrossEntropyLoss()
        self.ce_ls = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1, use_gpu=torch.cuda.is_available())
        self.tri = TripletLoss(margin=0.3)

    def __repr__(self):
        class_name = self.__class__.__name__
        attrs = []
        for attr_name, attr_value in self.__dict__.items():
            if attr_name.startswith("_"):
                continue
            # 智能处理不同类型的属性
            if isinstance(attr_value, (int, float, str, bool, list, dict, tuple)):
                attr_repr = str(attr_value)
            elif hasattr(attr_value, "__repr__"):
                attr_repr = repr(attr_value)
                # 简化torch模块的表示
                if isinstance(attr_value, nn.Module):
                    attr_repr = attr_value.__class__.__name__ + "()"
            else:
                attr_repr = f"<{type(attr_value).__name__} object>"
            attrs.append(f"  {attr_name}: {attr_repr}")
        return f"{class_name}(\n" + ",\n".join(attrs) + f"\n)"
