import torch.nn as nn
from loss.center_triplet_loss import CenterTripletLoss
from loss.hcc import hcc
from loss.ori_triplet_loss import OriTripletLoss


class Build_Criterion:
    def __init__(self, config):
        self.build(config)

    def build(self, config):
        self.id = nn.CrossEntropyLoss()
        self.tri = OriTripletLoss(batch_size=config.DATA.BATCHSIZE, margin=0.3)
        self.hcc = hcc(margin_euc=0.6, margin_kl=6)
        self.ctl = CenterTripletLoss(batch_size=config.DATA.BATCHSIZE, margin=0.3)

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
