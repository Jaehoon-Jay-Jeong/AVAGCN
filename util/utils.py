import hashlib
import random
import numpy as np
import torch
import os
import json

import logging
log = logging.getLogger(__name__)
print = log.info

def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ.pop('CUDA_LAUNCH_BLOCKING', None)

def seed_generate(project_name = "JAYHSICAVAGCN2026"):
    hash_object = hashlib.md5(project_name.encode())
    meta_seed = int(hash_object.hexdigest()[:8], 16)

    random.seed(meta_seed)
    seeds = [random.randint(0, 999) for _ in range(10)]
    print(f"Seeds from '{project_name}': {seeds}")

    return seeds

if __name__ == "__main__":
    project_name = input("project name: ")
    print(seed_generate(project_name))