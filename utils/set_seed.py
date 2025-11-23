import torch
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    random.seed(seed)
    # PyTorchのCuDNNライブラリ（NVIDIA GPU専用）での動作を制御する設定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False