import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import warnings
import cv2
from PIL import Image
import torchvision.transforms as standard_transforms

from datasets import build_dataset
from models import build_model
import config
import util.misc as utility

import nni
from nni.utils import merge_parameter

warnings.filterwarnings('ignore')

# --- Helper Classes for Metric Accumulation ---
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# --- Data Loading and Processing Utilities ---
def load_and_preprocess_data(img_path, gt_path, transform, device):
    """
    Loads an image and its ground truth points, applies P2PNet's preprocessing,
    and moves them to the specified device.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or could not be read: {img_path}")
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    import h5py
    try:
        with h5py.File(gt_path, 'r') as hf:
            kpoint_map_2d = np.array(hf['kpoint'])
    except Exception as e:
        raise Exception(f"Error loading .h5 kpoint map from {gt_path}: {e}")

    y_coords, x_coords = np.where(kpoint_map_2d == 1)
    gt_points = np.stack((x_coords, y_coords), axis=1).astype(np.float32)
    gt_points_tensor = torch.from_numpy(gt_points).float()

    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    target = {
        'point': gt_points_tensor.to(device),
        'image_id': torch.tensor([int(os.path.basename(img_path).split('.')[0].split('_')[-1])], dtype=torch.long),
        'original_size': torch.tensor([img_pil.height, img_pil.width], dtype=torch.long)
    }

    return img_tensor, target

def get_predicted_points(outputs, threshold=0.5):
    """
    Processes P2PNet raw outputs to extract predicted points based on a confidence threshold.
    """
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]
    
    predicted_points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    return predicted_points

# --- Visualization Function ---
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def visualize_prediction(image_tensor, gt_points, pred_points, save_path, image_name, prefix=""):
    """
    Visualizes ground truth and predicted points on an image and saves it.
    """
    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    pil_to_tensor = standard_transforms.ToTensor()

    sample_pil = restore_transform(image_tensor.squeeze(0).cpu())
    sample_np = (pil_to_tensor(sample_pil.convert('RGB')).numpy() * 255).astype(np.uint8)
    sample_drawn = cv2.cvtColor(sample_np.transpose([1, 2, 0]), cv2.COLOR_RGB2BGR)

    size = 2

    for t in gt_points:
        cv2.circle(sample_drawn, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
    
    for p in pred_points:
        cv2.circle(sample_drawn, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

    cv2.putText(sample_drawn, f"GT: {len(gt_points)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(sample_drawn, f"Pred: {len(pred_points)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    filename = f"{prefix}_{image_name}_gt_{len(gt_points)}_pred_{len(pred_points)}.jpg"
    cv2.imwrite(os.path.join(save_path, filename), sample_drawn)


# --- Core Adversarial Evaluation Function ---
def evaluate_adversarial_performance(model, args, device, transform):
    """
    Computes and compares model performance on clean and adversarial examples.
    """
    model.eval()

    clean_mae_meter = AverageMeter()
    clean_mse_meter = AverageMeter()
    adv_mae_meter = AverageMeter()
    adv_mse_meter = AverageMeter()

    # Define paths for the image lists
    # IMPORTANT: Adjust these paths to your actual data location
    npy_data_dir = args.npy_dir

    if args.dataset == 'shha':
        clean_img_list_path = args.clean_npy_path
        adv_img_list_path = args.adv_npy_path
        gt_base_path = os.path.join(args.data_root, 'val_data', 'gt-h5_2048')
    elif args.dataset == 'ucf':
        clean_img_list_path = args.clean_npy_path
        adv_img_list_path = args.adv_npy_path
        gt_base_path = os.path.join(args.data_root, 'val_data', 'gt-h5_2048')
    else:
        raise ValueError(f"Unsupported dataset for adversarial evaluation: {args.dataset}")

    try:
        clean_img_paths = np.load(clean_img_list_path, allow_pickle=True).tolist()
        adv_img_paths = np.load(adv_img_list_path, allow_pickle=True).tolist()
    except FileNotFoundError as e:
        print(f"Error loading image list NPY file: {e}")
        print("Please ensure your .npy files containing image paths exist and are correct.")
        return [], [], [], [] # Return empty lists to prevent further errors

    if len(clean_img_paths) != len(adv_img_paths):
        if utility.is_main_process():
            print(f"Warning: Clean image list ({len(clean_img_paths)}) and adversarial image list ({len(adv_img_paths)}) have different lengths. Proceeding with min length.")
        num_samples_to_process = min(len(clean_img_paths), len(adv_img_paths))
    else:
        num_samples_to_process = len(clean_img_paths)

    if utility.is_main_process():
        print(f"\n--- Starting Adversarial Performance Evaluation for {args.dataset} ---")
        print(f"Total samples to process: {num_samples_to_process}")

    start_time = time.time()
    
    for i in range(num_samples_to_process):
        clean_img_path = clean_img_paths[i]
        adv_img_path = adv_img_paths[i]
        
        img_filename = os.path.basename(clean_img_path)
        gt_filename = img_filename.replace('.jpg', '.h5').replace('.png', '.h5')
        gt_path = os.path.join(gt_base_path, gt_filename)

        if not os.path.exists(adv_img_path):
            if utility.is_main_process():
                print(f"Warning: Adversarial image not found: {adv_img_path}. Skipping this sample.")
            continue
        if not os.path.exists(gt_path):
            if utility.is_main_process():
                print(f"Warning: Ground truth not found for {clean_img_path} at {gt_path}. Skipping this sample.")
            continue

        if utility.is_main_process() and i % 100 == 0:
            print(f"Processing sample {i+1}/{num_samples_to_process}")

        try:
            clean_img_tensor, clean_target = load_and_preprocess_data(clean_img_path, gt_path, transform, device)
            gt_count = clean_target['point'].shape[0]

            with torch.no_grad():
                clean_outputs = model(clean_img_tensor)
            clean_pred_points = get_predicted_points(clean_outputs)
            clean_pred_count = len(clean_pred_points)

            clean_mae = abs(clean_pred_count - gt_count)
            clean_mse = (clean_pred_count - gt_count) ** 2
            clean_mae_meter.update(clean_mae)
            clean_mse_meter.update(clean_mse)

            adv_img_tensor, adv_target = load_and_preprocess_data(adv_img_path, gt_path, transform, device)
            
            with torch.no_grad():
                adv_outputs = model(adv_img_tensor)
            adv_pred_points = get_predicted_points(adv_outputs)
            adv_pred_count = len(adv_pred_points)

            adv_mae = abs(adv_pred_count - gt_count)
            adv_mse = (adv_pred_count - gt_count) ** 2
            adv_mae_meter.update(adv_mae)
            adv_mse_meter.update(adv_mse)

            # if args.visualization_dir and utility.is_main_process():
            #     vis_image_name = os.path.basename(clean_img_path).split('.')[0]
            #     visualize_prediction(clean_img_tensor, clean_target['point'].cpu().numpy().tolist(), clean_pred_points, args.visualization_dir, vis_image_name, prefix="clean")
            #     visualize_prediction(adv_img_tensor, adv_target['point'].cpu().numpy().tolist(), adv_pred_points, args.visualization_dir, vis_image_name, prefix="adv")

        except Exception as e:
            if utility.is_main_process():
                print(f"Error processing {clean_img_path} / {adv_img_path}: {e}. Skipping this pair.")
            continue

    total_eval_time = time.time() - start_time
    total_eval_time_str = str(datetime.timedelta(seconds=int(total_eval_time)))

    if utility.is_main_process():
        print(f"\n--- Evaluation Complete for {args.dataset} ---")
        print(f"Total processing time: {total_eval_time_str}")
        print("\nClean Performance:")
        print(f"  MAE: {clean_mae_meter.avg:.4f}")
        print(f"  RMSE: {np.sqrt(clean_mse_meter.avg):.4f}") # Report RMSE for MSE
        print("\nAdversarial Performance:")
        print(f"  MAE: {adv_mae_meter.avg:.4f}")
        print(f"  RMSE: {np.sqrt(adv_mse_meter.avg):.4f}") # Report RMSE for MSE
        print("---------------------------------------------")

    return clean_mae_meter.avg, np.sqrt(clean_mse_meter.avg), adv_mae_meter.avg, np.sqrt(adv_mse_meter.avg)


def main(args):
    utility.init_distributed_mode(args)

    if not args.distributed:
        os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    
    if utility.is_main_process():
        print("Arguments received by main function:", args)

    device = torch.device(args.device)
    if utility.is_main_process():
        print(f"Using device: {device} (Rank {utility.get_rank()})")

    seed = args.seed + utility.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_build_args = argparse.Namespace(**vars(args))

    if args.weight: 
        if utility.is_main_process():
            print(f"Attempting to load model weights from: {args.weight}")
        try:
            if args.weight.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.weight, map_location=device, check_hash=True)
            else:
                checkpoint = torch.load(args.weight, map_location=device)
            
            if 'args' in checkpoint:
                trained_model_args = checkpoint['args']
                if utility.is_main_process():
                    print("Found 'args' in checkpoint. Overriding model-specific arguments for consistency.")
                
                for arg_name in ['backbone', 'row', 'line', 'set_cost_class', 
                                 'set_cost_point', 'point_loss_coef', 'eos_coef', 
                                 'hidden_dim', 'dec_layers', 'nheads', 'dropout']:
                    if hasattr(trained_model_args, arg_name):
                        setattr(model_build_args, arg_name, getattr(trained_model_args, arg_name))
            else:
                if utility.is_main_process():
                    print("Warning: No 'args' found in checkpoint. Building model with current config.py arguments. "
                          "This might lead to mismatches if the training config was different.")
            
            model = build_model(model_build_args, training=False) 
            model.to(device)

            model_without_ddp = model 
            if args.distributed: 
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
                model_without_ddp = model.module

            if 'model' in checkpoint:
                model_without_ddp.load_state_dict(checkpoint['model']) 
            else:
                model_without_ddp.load_state_dict(checkpoint) 
            
            if utility.is_main_process():
                print("Model weights loaded successfully.")
                cur_epoch = checkpoint.get('epoch', 'N/A')
                print(f"Model trained for epoch: {cur_epoch}")

        except Exception as e:
            if utility.is_main_process(): 
                print(f"Error loading model weights from {args.weight}: {e}") 
                print("Please ensure the path is correct and the file is a valid PyTorch model checkpoint.")
            if args.distributed:
                dist.destroy_process_group() 
            return 
    else: 
        if utility.is_main_process():
            print("Warning: No model weights path (--weight) provided. Building model with untrained weights based on current config.py arguments.")
        model = build_model(model_build_args, training=False) 
        model.to(device)
        model_without_ddp = model 
        if args.distributed: 
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

    if model is None: 
        if utility.is_main_process():
            print("Error: Model could not be built or loaded. Exiting.")
        if args.distributed:
            dist.destroy_process_group()
        return

    if utility.is_main_process():
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {n_parameters/1e6:.2f}M')

    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.visualization_dir and utility.is_main_process():
        Path(args.visualization_dir).mkdir(parents=True, exist_ok=True)
        print(f"Visualization results will be saved to: {args.visualization_dir}")

    clean_mae, clean_rmse, adv_mae, adv_rmse = evaluate_adversarial_performance(model, args, device, transform)

    if utility.is_main_process():
        nni.report_final_result(clean_mae)
    
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    base_args = config.args 
    tuner_params = nni.get_next_parameter()
    args_for_main = merge_parameter(base_args, tuner_params)
    main(args_for_main)
