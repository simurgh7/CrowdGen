import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import warnings
import random

from models.model import SASNet
from datasets import build_dataset 
import config
from engine import evaluate

warnings.filterwarnings('ignore')

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    # Clean Dataset
    dataset_clean = build_dataset(purpose='val', args=args) 
    test_loader_clean = DataLoader(dataset_clean, batch_size=1, shuffle=False, drop_last=False)
    # --- Setup Adversarial Data Loader ---
    adv_img_list_path = args.adv_npy_path
    adv_img_paths_raw = np.load(adv_img_list_path, allow_pickle=True).tolist()
    
    dataset_adv = build_dataset(purpose='val', args=args) 
    
    new_adv_img_map = {}
    gt_base_path = os.path.join(args.data_root, 'val_data', 'gt-h5_2048') # Assuming this is your GT path structure
    print(gt_base_path)
    for adv_path in adv_img_paths_raw:
        img_filename = os.path.basename(adv_path)
        # Adjust this logic based on your GT filename convention if different
        gt_filename = img_filename.replace('.jpg', '.h5').replace('.png', '.h5')
        gt_full_path = os.path.join(gt_base_path, gt_filename)
        print(adv_path, gt_full_path)
        
        if os.path.exists(adv_path) and os.path.exists(gt_full_path):
            new_adv_img_map[adv_path] = gt_full_path
        else:
            print(f"Warning: Missing adv image {adv_path} or GT {gt_full_path}. Skipping.")

    dataset_adv.img_map = new_adv_img_map
    dataset_adv.img_list = adv_img_paths_raw # Update img_list
    dataset_adv.nSamples = len(dataset_adv.img_list) # Update sample count
    test_loader_adv = DataLoader(dataset_adv, batch_size=1, shuffle=False, drop_last=False)
    
    # Load model - use args.weight for pretrained model
    model = SASNet(args=args).cuda()
    if args.weight and os.path.exists(args.weight):
        model.load_state_dict(torch.load(args.weight, map_location=device))
        print(f"Loaded model from {args.weight}")
    else:
        print(f"Warning: Model weight not found at {args.weight}")
    
    model.to(device)
    model.eval()
    
    print('=' * 50)
    print(f"Testing on dataset: {args.dataset}")
    print("Number of Samples : ", len(test_loader_clean))
    print(f"Using device: {device}")
    #print(f"Test samples: {test_set.get_num_samples()}")
    print('=' * 50)
    clean_mae, clean_mse = evaluate(test_loader_clean, model, device, args)
    print(clean_mae, clean_mse)
    adv_mae, adv_mse = evaluate(test_loader_adv, model, device, args)
    print(adv_mae, adv_mse)

if __name__ == '__main__':
    args = config.args
    main(args)