import torch.utils.data
import torchvision

from .dataset import build as build_generic

data_path = {
    'shha': r'D:\crowd\data\localization\shanghaitechA',
    'ucf' : r'D:\crowd\data\localization\ucf-qnrf',
    'jhu' : r'D:\crowd\data\localization\jhu-crowd++',
    'nwpu' : r'D:\crowd\data\localization\nwpu-crowd'
}

def build_dataset(purpose, args):
    """
    Builds the dataset using the unified ImageDataset class.

    Args:
        image_set (str): The purpose of the dataset ('train', 'val', 'test').
        args: An object containing configuration arguments, including:
              - args.dataset: The name of the dataset (e.g., 'shha', 'ucf').
              - args.aug_dict (optional): A dictionary or object containing augmentation parameters.
    """
    if args.dataset in data_path:
        args.data_path = data_path[args.dataset]
    else:
        raise ValueError(f"Dataset '{args.dataset}' not recognized in data_path mapping.")
    
    return build_generic(purpose, args)
