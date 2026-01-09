import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# def set_seed_torch(seed):
#     seed = int(seed)
#     random.seed(seed)
#     os.environ["PYTHONASHSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     # 程序浮现 torch.backends.cudnn.benchmark = False / torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True


def set_seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
