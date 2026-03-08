from models.model import SASNet
from models import build_unet
from datasets import build_dataset 
import config
from engine import evaluate
########################################
from camgen import CAM
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as tff
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import gc
from enum import Enum
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs
class DensityMapTarget:
    def __init__(self, density_map):
        self.density_map = density_map
        
    def __call__(self, model_output):
        # For SASNet, we want to maximize the correlation with the original density
        # or use it to guide where to focus the attack
        return self.density_map.sum()
def counting(input):
    input_max = torch.max(input).item()
    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input
    input[input < 100.0 / 255.0 * torch.max(input)] = 0
    input[input > 0] = 1
    '''negative sample'''
    if input_max<0.1:
        input = input * 0
    count = int(torch.sum(input).item())
    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()
    return count, kpoint

### Loss Functions ###
# Option 1: Density Suppression Loss
def suppress_potential_peaks(output_density, threshold_ratio):
    """
    Fallback: suppress highest values that could become peaks
    """
    # Find top k% values that could potentially become peaks
    density_flat = output_density.view(output_density.size(0), -1)
    k = max(1, int(threshold_ratio * density_flat.size(1)))
    
    topk_values, _ = torch.topk(density_flat, k, dim=1)
    threshold = topk_values[:, -1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    
    # Penalize high values
    high_value_mask = (output_density >= threshold).float()
    loss = (output_density * high_value_mask).mean()
    return loss

def density_suppression_loss(output_density, threshold_ratio=0.1):
    """
    Suppress regions that are likely to be detected as peaks by your counting function
    """
    # First, apply the same local maxima detection as your counting function
    keep = nn.functional.max_pool2d(output_density, (3, 3), stride=1, padding=1)
    is_local_max = (keep == output_density).float()
    local_maxima = output_density * is_local_max
    
    # Get the maximum value (used for thresholding in your counting function)
    input_max = torch.max(output_density)
    
    # If it's a negative sample (max < 0.1), return zero loss
    if input_max < 0.1:
        return torch.tensor(0.0, device=output_density.device)
    
    # Calculate the adaptive threshold used in your counting function
    counting_threshold = 100.0 / 255.0 * input_max
    
    # Focus on peaks that would pass the counting threshold
    significant_peaks = local_maxima * (local_maxima >= counting_threshold).float()
    
    if significant_peaks.sum() == 0:
        # No significant peaks found, try alternative suppression
        return suppress_potential_peaks(output_density, threshold_ratio)
    
    # Option 1: Direct suppression of significant peaks
    peak_loss = significant_peaks.mean()
    
    # Option 2: Also suppress values near the threshold to prevent new peaks from forming
    near_threshold = output_density * (
        (output_density >= counting_threshold * 0.8) & 
        (output_density < counting_threshold)
    ).float()
    threshold_loss = near_threshold.mean()
    
    # Combined loss
    loss = peak_loss + threshold_loss
    return loss

def multi_scale_density_loss(output_density, scales=[0.5, 1.0, 2.0]):
    """Fixed version using mean instead of sum"""
    loss = 0
    for scale in scales:
        if scale == 1.0:
            scaled_density = output_density
        else:
            size = (int(output_density.size(2) * scale), int(output_density.size(3) * scale))
            scaled_density = F.interpolate(output_density, size=size, mode='bilinear')
            scaled_density = scaled_density / (scale * scale)
        
        # Use MEAN instead of SUM to avoid huge numbers
        loss += torch.mean(scaled_density)
    
    return loss / len(scales)

# Option 2: Peak Suppression Loss
def peak_suppression_loss(output_density, threshold_ratio=0.1):
    """
    Main loss: Targets the exact peak detection mechanism in your counting function
    """
    # Step 1: Local maxima detection (same as your counting function)
    keep = nn.functional.max_pool2d(output_density, (3, 3), stride=1, padding=1)
    is_local_max = (keep == output_density).float()
    local_maxima = output_density * is_local_max
    
    input_max = torch.max(output_density)
    
    # Handle negative samples
    if input_max < 0.1:
        return torch.tensor(0.0, device=output_density.device)
    
    # Step 2: Adaptive thresholding (same as your counting function)
    counting_threshold = 100.0 / 255.0 * input_max
    significant_peaks = local_maxima * (local_maxima >= counting_threshold).float()
    
    # Loss component 1: Suppress peaks that would be counted
    peak_loss = significant_peaks.mean()
    
    # Loss component 2: Reduce peak prominence
    neighborhood_mean = nn.functional.avg_pool2d(output_density, (3, 3), stride=1, padding=1)
    peak_prominence = (output_density - neighborhood_mean) * is_local_max
    prominence_loss = peak_prominence.mean()
    
    # Combined loss
    loss = peak_loss + prominence_loss * 0.3
    
    return loss

def multi_scale_peak_loss(output_density, scales=[0.5, 1.0, 2.0]):
    """
    Multi-scale version for better attack robustness
    """
    total_loss = 0
    
    for scale in scales:
        if scale == 1.0:
            scaled_density = output_density
        else:
            size = (int(output_density.size(2) * scale), int(output_density.size(3) * scale))
            scaled_density = F.interpolate(output_density, size=size, mode='bilinear')
            scaled_density = scaled_density / (scale * scale)  # Normalize
        
        total_loss += peak_suppression_loss(scaled_density)
    
    return total_loss / len(scales)

# Perturbation Specific Loss
def total_variation_loss_sum(pert):
    return (torch.sum(torch.abs(pert[:,:,:,:-1] - pert[:,:,:,1:])) + 
            torch.sum(torch.abs(pert[:,:,:-1,:] - pert[:,:,1:,:])))

def total_variation_loss_norm(pert):
    # Normalized by image dimensions
    h, w = pert.shape[-2:]
    return (torch.sum(torch.abs(pert[...,:-1] - pert[...,1:])) / (h*w) + 
           torch.sum(torch.abs(pert[...,:-1,:] - pert[...,1:,:])) / (h*w))

### Training ###
def get_density_cam(target_model, img):
    """Simple CAM using density map directly - more reliable for SASNet"""
    with torch.no_grad():
        density_map = target_model(img)  # [batch, 1, H, W]
        # Use density map as attention/importance map
        cam_map = density_map  # Already [batch, 1, H, W]
        # Normalize to [0, 1]
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-6)
        return cam_map

def adversarial_training_density_cam(unet, target_model, train_loader, optimizer, args, epoch):
    device = args.device
    unet.train()
    total_loss = 0
    
    # Learning rate scheduling
    if epoch >= 4:
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * (epoch-4) / (args.epochs-4)))
    else:
        lr = args.lr * (epoch / 4)
    for g in optimizer.param_groups:
        g['lr'] = lr
        
    for idx, (imgs, targets) in enumerate(train_loader):
        img = imgs.to(device)
        gt_count = targets['points'].shape[1]
        
        # Temporary downscaled
        img = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)
        
        # Get density-based CAM (simple approach)
        with torch.no_grad():
            cam_map = get_density_cam(target_model, img)
            cam_map = F.interpolate(cam_map, size=img.shape[-2:], mode='bilinear', align_corners=False)
        
        # Generate perturbation from U-Net
        perturbation = unet(img)
        
        # Apply CAM weighting (stronger perturbation in hot regions)
        perturbation = perturbation * (1 + args.cam_weight * cam_map)
        
        # Clamp perturbation
        perturbation = torch.clamp(perturbation, -args.epsilon, args.epsilon)
        perturbation = F.interpolate(perturbation, size=img.shape[-2:], mode='bilinear', align_corners=False)
        
        adv_img = torch.clamp(img + perturbation, 0, 1)
        
        # Prediction
        with torch.no_grad():
            pred_adv = target_model(adv_img)
        
        # Use density loss
        density_loss = density_suppression_loss(pred_adv)
        # density_loss = multi_scale_density_loss(pred_adv)
        
        # Compute adaptive weight
        alpha = min(1.0, gt_count / 100.)
        
        # Regularization losses
        hinge_loss = perturbation.pow(2).mean()
        tv_loss = total_variation_loss_norm(perturbation)
        pert_fft = torch.fft.rfft2(perturbation)
        freq_loss = torch.mean(torch.abs(pert_fft[..., 1:]))
        
        # CAM loss
        cam_loss = F.l1_loss(perturbation.abs().mean(1, keepdim=True), cam_map)
        
        # Combined loss
        loss = (density_loss * alpha 
                + hinge_loss * args.beta
                #+ tv_loss * args.gamma
                #+ freq_loss * args.zeta
                #+ cam_loss * args.kappa
                )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()

        gc.collect()
        torch.cuda.empty_cache()
        total_loss += loss.item()
        
        # Get actual count for monitoring
        with torch.no_grad():
            pred_adv_np = pred_adv.data.cpu().numpy()
            actual_count = np.sum(pred_adv_np) / args.log_para if hasattr(args, 'log_para') else np.sum(pred_adv_np)
        
        if idx % 50 == 0:
            print(f"Epoch [{epoch}] Batch [{idx}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Density: {density_loss.item():.4f} | "
                  f"Hinge: {hinge_loss.item():.4f} | "
                  f"TV: {tv_loss.item():.3f} | "
                  f"Freq: {freq_loss.item():.3f} | "
                  f"Cam: {cam_loss.item():.3f}")
            print(f"Count - Pred: {actual_count:.1f}, GT: {gt_count} | "
                  f"Perturbation Max: {perturbation.abs().max().item():.3f} | "
                  f"CAM Mean: {cam_map.mean().item():.3f}")
    
    total_loss /= len(train_loader)
    print(f"[Epoch {epoch}] Total Loss: {total_loss:.4f}")

def adversarial_training_peak_cam(unet, target_model, train_loader, optimizer, args, epoch):
    device = args.device
    unet.train()
    total_loss = 0
    
    # Learning rate scheduling
    if epoch >= 4:
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * (epoch-4) / (args.epochs-4)))
    else:
        lr = args.lr * (epoch / 4)
    for g in optimizer.param_groups:
        g['lr'] = lr
        
    for idx, (imgs, targets) in enumerate(train_loader):
        img = imgs.to(device)
        gt_count = targets['points'].shape[1]
        
        # Temporary downscaled
        img = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)
        
        # Get density-based CAM (simple approach)
        with torch.no_grad():
            cam_map = get_density_cam(target_model, img)
            cam_map = F.interpolate(cam_map, size=img.shape[-2:], mode='bilinear', align_corners=False)
        
        # Generate perturbation from U-Net
        perturbation = unet(img)
        
        # Apply CAM weighting (stronger perturbation in hot regions)
        perturbation = perturbation * (1 + args.cam_weight * cam_map)
        
        # Clamp perturbation
        perturbation = torch.clamp(perturbation, -args.epsilon, args.epsilon)
        perturbation = F.interpolate(perturbation, size=img.shape[-2:], mode='bilinear', align_corners=False)
        
        adv_img = torch.clamp(img + perturbation, 0, 1)
        
        # Prediction
        with torch.no_grad():
            pred_adv = target_model(adv_img)
        
        # Use density loss
        #density_loss = peak_suppression_loss(pred_adv)
        density_loss = multi_scale_peak_loss(pred_adv)
        
        # Compute adaptive weight
        alpha = min(1.0, gt_count / 100.)
        
        # Regularization losses
        hinge_loss = perturbation.pow(2).mean()
        tv_loss = total_variation_loss_norm(perturbation)
        pert_fft = torch.fft.rfft2(perturbation)
        freq_loss = torch.mean(torch.abs(pert_fft[..., 1:]))
        
        # CAM loss
        cam_loss = F.l1_loss(perturbation.abs().mean(1, keepdim=True), cam_map)
        
        # Combined loss
        loss = (density_loss * alpha 
                + hinge_loss * args.beta
                #+ tv_loss * args.gamma
                #+ freq_loss * args.zeta
                #+ cam_loss * args.kappa
                )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()

        gc.collect()
        torch.cuda.empty_cache()
        total_loss += loss.item()
        
        # Get actual count for monitoring
        with torch.no_grad():
            pred_adv_np = pred_adv.data.cpu().numpy()
            actual_count = np.sum(pred_adv_np) / args.log_para if hasattr(args, 'log_para') else np.sum(pred_adv_np)
        
        if idx % 50 == 0:
            print(f"Epoch [{epoch}] Batch [{idx}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Density: {density_loss.item():.4f} | "
                  f"Hinge: {hinge_loss.item():.4f} | "
                  f"TV: {tv_loss.item():.3f} | "
                  f"Freq: {freq_loss.item():.3f} | "
                  f"Cam: {cam_loss.item():.3f}")
            print(f"Count - Pred: {actual_count:.1f}, GT: {gt_count} | "
                  f"Perturbation Max: {perturbation.abs().max().item():.3f} | "
                  f"CAM Mean: {cam_map.mean().item():.3f}")
    
    total_loss /= len(train_loader)
    print(f"[Epoch {epoch}] Total Loss: {total_loss:.4f}")

### Validation for Density Map Models ###
def adversarial_validation(unet, target_model, val_loader, args, epoch):
    device = args.device
    unet.eval()
    total_count_clean = 0
    total_count_adv = 0
    total_gt_count = 0
    mae_clean = 0
    mae_adv = 0
    
    with torch.no_grad():
        for idx, (imgs, targets) in enumerate(val_loader):
            # img = imgs.to(device)
            img = imgs[0].unsqueeze(0).to(device)
            gt_count = targets['points'].shape[1]
            total_gt_count += gt_count
            # temporary downscaled
            img = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)
            # Generate adversarial image
            perturbation = unet(img)
            perturbation = torch.clamp(perturbation, -args.epsilon, args.epsilon)
            perturbation = F.interpolate(perturbation, size=img.shape[-2:], mode='bilinear', align_corners=False)
            adv_img = torch.clamp(img + perturbation, 0, 1)
            
            # Get predictions
            pred_clean = target_model(img)
            pred_adv = target_model(adv_img)
            
            pred_adv = pred_adv.data.cpu().numpy()
            adv_count = np.sum(pred_adv) / args.log_para
            pred_clean = pred_clean.data.cpu().numpy()
            clean_count = np.sum(pred_clean) / args.log_para
            
            total_count_clean += clean_count
            total_count_adv += adv_count
            
            # Calculate MAE
            mae_clean += abs(clean_count - gt_count)
            mae_adv += abs(adv_count - gt_count)
            
            # Print batch results
            if idx % args.log_interval == 0:
                reduction_pct = ((clean_count - adv_count) / max(clean_count, 1)) * 100
                print(f"Batch [{idx}] GT: {gt_count}, Clean: {clean_count}, Adv: {adv_count}, "
                      f"Reduction: {reduction_pct:.1f}%")
            
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    # Calculate metrics
    num_samples = len(val_loader)
    count_difference = total_count_clean - total_count_adv
    count_reduction = ((total_count_clean - total_count_adv) / max(total_count_clean, 1) * 100)
    
    print(f"\n[Epoch {epoch}] Validation Results:")
    print(f"  Total GT Count: {total_gt_count}")
    print(f"  Total Clean Count: {total_count_clean}")
    print(f"  Total Adv Count: {total_count_adv}")
    print(f"  Count Difference: {count_difference}")
    print(f"  Count Reduction: {count_reduction:.2f}%")
    print(f"  MAE Clean: {mae_clean/num_samples:.2f}, MAE Adv: {mae_adv/num_samples:.2f}")
    print(f"  Attack Success Rate: {count_reduction :.1f}%")
    
    return total_count_clean, total_count_adv, count_reduction

# Save the Images #
def save_adversarial_image(unet, image_path, weight_path, weight_name, device, data_loader, args):
    """
    Save adversarial images and perturbations - only reverse normalization, no rescaling
    """
    os.makedirs(image_path, exist_ok=True)
    # perturb_path = os.path.join(image_path, "perturbations")
    # os.makedirs(perturb_path, exist_ok=True)
    dataset = "shha"#args.dataset
    # Load weights
    ckpt = torch.load(os.path.join(weight_path, dataset, weight_name), map_location=device)
    unet.load_state_dict(ckpt['model_state_dict'])
    unet.eval()

    # Reverse normalization parameters (from your transform)
    reverse_mean = torch.tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225]).view(1, 3, 1, 1).to(device)
    reverse_std = torch.tensor([1/0.229, 1/0.224, 1/0.225]).view(1, 3, 1, 1).to(device)

    with torch.no_grad():
        for idx, (img, target) in enumerate(data_loader):
            img = img.to(device)  # [1, C, H, W]

            # Generate perturbation
            perturbation = unet(img)
            perturbation = torch.clamp(perturbation, -args.epsilon, args.epsilon)
            perturbation = F.interpolate(perturbation, size=img.shape[-2:], mode='bilinear', align_corners=False)

            # Generate adversarial image
            adv_img = torch.clamp(img + perturbation, 0, 1)

            # Reverse normalization for saving
            adv_img_denorm = adv_img * reverse_std + reverse_mean
            adv_img_denorm = torch.clamp(adv_img_denorm, 0, 1)

            # Convert to PIL
            adv_img_pil = transforms.ToPILImage()(adv_img_denorm.squeeze(0).cpu())

            # Save adversarial image
            img_name = target['name'][0]
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_name += '.jpg'
            adv_img_pil.save(os.path.join(image_path, img_name))
            print(f"Saved adversarial image: {img_name}")

            # ---- Save perturbation ---- #
            # Normalize to [0,1] for visualization
            # pert_norm = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)
            # pert_pil = transforms.ToPILImage()(pert_norm.squeeze(0).cpu())
            # pert_pil.save(os.path.join(perturb_path, img_name))
            # print(f"Saved perturbation: {img_name}")

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

### Main ###
def main(args):
    device = args.device
    image_path = args.advex_dir
    weight_path = args.weight_dir

    dataset_train = build_dataset('train', args)
    dataset_val = build_dataset('val', args)

    train_loader = DataLoader(dataset_train, batch_size=1, sampler=RandomSampler(dataset_train), num_workers=args.num_workers)
    val_loader = DataLoader(dataset_val, batch_size=1, sampler=SequentialSampler(dataset_val),num_workers=args.num_workers)
    # Load the target model
    checkpoint_path = args.weight
    target_model = SASNet(args=args).cuda()
    if checkpoint_path and os.path.exists(checkpoint_path):
        target_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Warning: Model weight not found at {checkpoint_path}")
    
    target_model.to(device)
    target_model.eval()

    # Initialize U-Net for perturbations
    unet = build_unet(args, type='vanilla')
    # unet = build_diffusion(args, type='vanilla')
    unet.to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    best_gap = -float('inf')  # Best (clean - adv) gap
    weight_name = r'sasnet_shha_unet_peak.pth'
    # for epoch in range(1, args.epochs + 1):
    #     #adversarial_training_density_cam(unet, target_model, train_loader, optimizer, args, epoch)
    #     adversarial_training_peak_cam(unet, target_model, train_loader, optimizer, args, epoch)
    #     clean_count, adv_count, gap = adversarial_validation(unet, target_model, val_loader, args, epoch)
    #     if gap > best_gap:
    #         best_gap = gap
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': unet.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #         }, os.path.join(args.weight_dir, weight_name))
    #         print(f"Saved best model (epoch {epoch}) with count gap {gap:.2f}")

    
    save_adversarial_image(unet, image_path, weight_path, weight_name, device, val_loader, args)

if __name__ == '__main__':
    args = config.args
    main(args)