import torch
import argparse
from os import makedirs
from os.path import exists, dirname
from termcolor import cprint

class BatchTensorToVars(object):
    """Move tensors in dict batch to specified device."""

    def __init__(self, use_cuda=True):
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    def __call__(self, batch):
        batch_out = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_out[key] = value.to(self.device)
            else:
                batch_out[key] = value
        return batch_out
    
def save_checkpoint(state, file):
    model_dir = dirname(file)
    if model_dir != '' and not exists(model_dir):
        makedirs(model_dir)
    torch.save(state, file)
        
def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_info(info, _type=None):
    if _type is not None:
        if isinstance(info,str):
            cprint(info, _type[0], attrs=[_type[1]])
        elif isinstance(info,list):
            for i in range(info):
                cprint(i, _type[0], attrs=[_type[1]])
    else:
        print(info)


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """Load checkpoint weights into model.

    Args:
        model: The model instance
        checkpoint_path: Path to checkpoint file
        device: Device to map checkpoint to

    Returns:
        The checkpoint dict (for accessing epoch, args, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint
