import sys
import os
import torch
from argparse import Namespace
import yaml

# Add the cloned repo to python path so we can import 'models'
sys.path.append(os.path.join(os.getcwd(), 'encoder4editing'))

# Now we can import from the cloned repo
try:
    from models.psp import pSp
except ImportError:
    raise ImportError("Could not import pSp. Did you run 'python setup.py'?")

def load_model(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    ckpt_path = cfg['paths']['ckpt_path']
    print(f"Loading model from {ckpt_path}...")
    
    ckpt = torch.load(ckpt_path, map_location='cuda')
    opts = ckpt['opts']
    opts['checkpoint_path'] = ckpt_path
    opts = Namespace(**opts)

    net = pSp(opts).eval().to('cuda')
    return net