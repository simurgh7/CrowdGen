import datetime
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np 

import torchvision.transforms as standard_transforms 
import os
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from datasets import build_dataset
from engine import evaluate_crowd_no_overlap 
from models import build_model
import config 
import util.misc as utility 


import os
import warnings
warnings.filterwarnings('ignore')

import argparse

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Wrapper 
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs['pred_points']  # or any tensor you want Grad-CAM to explain

# Custom Grad-CAM target that converts the tensor output into a scalar by summing
class PointCountTarget:
    def __call__(self, model_output):
        # model_output is tensor of shape [batch_size, num_points, ...]
        return model_output.sum()  # scalar value for backward()
        #return model_output.new_tensor(model_output.shape[1], requires_grad=True)

# ========== VISUALIZATION ==========
def save_gradcam_overlay(original_tensor, mask, out_path):
    # De-normalize
    inv_transform = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    img_tensor = inv_transform(original_tensor).clamp(0, 1)
    rgb_image = img_tensor.permute(1, 2, 0).cpu().numpy()

    cam_overlay = show_cam_on_image(rgb_image, mask, use_rgb=True)
    cv2.imwrite(out_path, cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))

def main(args):
    os.makedirs(args.explanation_dir, exist_ok=True)
    # Load model
    model = build_model(args, training=False)
    checkpoint = torch.load(args.weight, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval().to(args.device)

    # Load dataset
    clean_dataset = build_dataset(image_set='val', args=args)
    clean_dataloader = torch.utils.data.DataLoader(clean_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Advex dataloader
    adv_img_list_path = args.adv_npy_path
    adv_img_paths_raw = np.load(adv_img_list_path, allow_pickle=True).tolist()
    adv_dataset = build_dataset(image_set='val', args=args) 
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

    adv_dataset.img_map = new_adv_img_map
    adv_dataset.img_list = adv_img_paths_raw # Update img_list
    adv_dataset.nSamples = len(adv_dataset.img_list) # Update sample count
    adv_dataloader = torch.utils.data.DataLoader(adv_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=1)
    # Grad-CAM target layer
    # backbone_layer = model.backbone[-1] 
    # target_layer = dict([*model.named_modules()])[backbone_layer]
    #target_layer = model.backbone.body4[5]
    target_layer = model.backbone.body4[9]
    wrapped_model = WrappedModel(model)
    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
    #cam = GradCAM(model=model, target_layers=[target_layer])

    #for idx, (images, targets) in enumerate(tqdm(clean_dataloader, desc="Generating Grad-CAM for Clean Images")):
    for idx, (images, targets) in enumerate(tqdm(adv_dataloader, desc="Generating Grad-CAM for Adv Images")):
        # if idx >= 10:
        #     break

        images = images.to(args.device)
        image_id = targets["name"][0]
        cam_input_tensor = images.clone()

        grayscale_cam = cam(input_tensor=cam_input_tensor, targets=[PointCountTarget()])[0]
        out_path = os.path.join(args.explanation_dir, f"{image_id}.jpg")
        save_gradcam_overlay(images[0], grayscale_cam, out_path)

    print(f"Grad-CAM visualizations saved in {args.explanation_dir}")


if __name__ == '__main__':
    base_args = config.args 
    main(base_args)
