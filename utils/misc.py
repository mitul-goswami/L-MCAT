import torch
import random
import numpy as np
import yaml

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, filename):
    torch.save(state, filename)
    
def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)
