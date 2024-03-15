import random
import numpy as np
import torch



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def move_to_gpu(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_gpu(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_gpu(value)
                for key, value in maybe_tensor.items()}
        else:
            return maybe_tensor

    return _move_to_gpu(sample)