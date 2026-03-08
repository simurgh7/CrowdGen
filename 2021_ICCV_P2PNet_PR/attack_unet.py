from models import build_model, build_unet, build_diffusion
from datasets import build_dataset
import config
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
import util.misc as utils
import gc
import os
from enum import Enum
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# Wrapper 
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs['pred_points']
class PointCountTarget:
    def __call__(self, model_output):
        # model_output is tensor of shape [batch_size, num_points, ...]
        return model_output.sum()  # scalar value for backward()
        #return model_output.new_tensor(model_output.shape[1], requires_grad=True)

### Utility ###
def visualize_cam_vs_perturb(image, cam, perturb_mag):
    img_np = image.squeeze().permute(1,2,0).cpu().numpy()
    cam_np = cam.cpu().numpy()
    pert_np = perturb_mag.cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_np)
    axs[0].set_title("Image")
    axs[1].imshow(cam_np, cmap='jet')
    axs[1].set_title("Grad-CAM")
    axs[2].imshow(pert_np, cmap='gray')
    axs[2].set_title("Perturbation Focus")
    for ax in axs: ax.axis('off')
    plt.show()

def plot_perturbation(pert):
    pert_np = pert.squeeze().cpu().numpy()
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(pert_np.mean(0), cmap='RdBu', vmin=-args.epsilon, vmax=args.epsilon)
    plt.colorbar()
    plt.subplot(132)
    plt.plot(pert_np.mean((1,2)))
    plt.title('Channel-wise Mean')
    plt.subplot(133)
    plt.hist(pert_np.flatten(), bins=50)
    plt.show()

def counting_points(output, threshold = 0.5):
    pred_scores = torch.nn.functional.softmax(output['pred_logits'], -1)[:, :, 1]
    pred_points = output['pred_points']
    points = pred_points[pred_scores > threshold].detach().cpu().numpy().tolist()
    cnt = int((pred_scores > threshold).sum())
    return points, cnt
    
def save_adversarial_image(unet, image_path, weight_path, weight_name, device, data_loader, args):
    """
    Save adversarial images - only reverse normalization, no rescaling
    """
    os.makedirs(image_path, exist_ok=True)
    # Load weights
    dataset = "shha_new"#args.dataset
    ckpt = torch.load(os.path.join(weight_path, dataset , weight_name), map_location=device)
    unet.load_state_dict(ckpt['model_state_dict'])
    unet.eval()
    # Reverse normalization parameters (from your transform)
    reverse_mean = torch.tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225]).view(1, 3, 1, 1).to(device)
    reverse_std = torch.tensor([1/0.229, 1/0.224, 1/0.225]).view(1, 3, 1, 1).to(device)
    # filename = os.path.join(save_path, f"{prefix}_epoch{epoch}_img{idx}.png")
    # vutils.save_image(img_tensor, filename, normalize=True)
    with torch.no_grad():
        for idx, (imgs, targets) in enumerate(data_loader):
            img = imgs[0].unsqueeze(0).to(device)
            target = targets[0]
            #perturbation = unet(img) * args.epsilon
            perturbation = unet(img)
            perturbation = torch.clamp(perturbation, -args.epsilon, args.epsilon)
            perturbation = F.interpolate(perturbation, size=img.shape[-2:], mode='bilinear', align_corners=False)
            adv_img = torch.clamp(img + perturbation, 0, 1)
            # adv_img = torch.mul(img , perturbation) + img
            # adv_img = img + perturbation * 255
            adv_img_denorm = adv_img * reverse_std + reverse_mean
            # adv_img_denorm = adv_img * std + mean
            adv_img_denorm = torch.clamp(adv_img, 0, 1)  # Ensure valid pixel range
            
            # Convert to PIL and save
            adv_img_pil = transforms.ToPILImage()(adv_img_denorm.squeeze(0).cpu())
            img_name = target['name'][0] if isinstance(target['name'], list) else target['name']
            adv_img_pil.save(os.path.join(image_path, img_name))
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print(f"Saved: {img_name}")

### Loss Functions ###
def logit_undercount_loss_conf(output, threshold=0.5):
    """More stable version with threshold control"""
    # Get human probabilities and mask
    p_human = torch.softmax(output['pred_logits'], -1)[:, :, 1]
    y_human_mask = (p_human > threshold)
    
    # Focus only on confidently detected humans
    if not y_human_mask.any():
        return torch.tensor(0., device=output['pred_logits'].device)
    
    # Direct logit minimization + probability suppression
    human_logits = output['pred_logits'][:, :, 1][y_human_mask]
    loss = human_logits.mean() - torch.log(1 - p_human[y_human_mask] + 1e-6).mean()
    return loss

def logit_undercount_loss_varth(output, threshold=0.5, epoch=0, max_epochs=12):
    # Linear decay from 0.5 to 0.3 over training
    adaptive_threshold = max(0.3, threshold - 0.02*(epoch/max_epochs))  
    p_human = torch.softmax(output['pred_logits'], -1)[:, :, 1]
    y_human_mask = (p_human > adaptive_threshold)
    
    if not y_human_mask.any():
        return torch.tensor(0., device=output['pred_logits'].device)
    
    human_logits = output['pred_logits'][:, :, 1][y_human_mask]
    # Focused suppression for borderline cases
    weights = torch.sigmoid(5*(p_human[y_human_mask] - adaptive_threshold))
    loss = (human_logits * weights).mean() - torch.log(1 - p_human[y_human_mask] + 1e-6).mean()
    
    return y_human_mask, loss

def logit_undercount_loss_sparsity(output, gt_count, epoch, max_epochs=12):
    """Enhanced loss with density adaptation"""
    # Dynamic threshold (0.5 → 0.3 over training)
    thresh = max(0.3, 0.5 - 0.2*(epoch/max_epochs))  
    
    # Density-adaptive adjustment
    if gt_count < 100:  # Sparse scenes
        thresh = min(0.4, thresh + 0.1)  # Easier threshold
        
    p_human = torch.softmax(output['pred_logits'], -1)[:,:,1]
    y_human_mask = (p_human > thresh)
    
    if not y_human_mask.any():
        return y_human_mask, torch.tensor(0., device=output['pred_logits'].device)
    
    human_logits = output['pred_logits'][:,:,1][y_human_mask]
    
    # Sparse scene handling
    if gt_count < 100:
        loss = -torch.log(1 - p_human[y_human_mask] + 1e-6).mean()  # Force undercount
    else:
        weights = (p_human[y_human_mask] - thresh).abs()  # Focus on borderline cases
        loss = (human_logits * weights).mean() / (weights.mean() + 1e-6)
    
    return y_human_mask, loss

def total_variation_loss_sum(pert):
    return (torch.sum(torch.abs(pert[:,:,:,:-1] - pert[:,:,:,1:])) + 
            torch.sum(torch.abs(pert[:,:,:-1,:] - pert[:,:,1:,:])))

def total_variation_loss_norm(pert):
    # Normalized by image dimensions
    h, w = pert.shape[-2:]
    return (torch.sum(torch.abs(pert[...,:-1] - pert[...,1:])) / (h*w) + 
           torch.sum(torch.abs(pert[...,:-1,:] - pert[...,1:,:])) / (h*w))

### Training ###
def adversarial_training_cam(unet, target_model, train_loader, optimizer, args, epoch):
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
    threshold = 0.5
    adaptive_threshold = max(0.3, threshold - 0.02*(epoch/args.epochs))  
    # Choose correct target layer for CAM
    target_layer = target_model.backbone.body4[-1]  # last conv block
    wrapped_model = WrappedModel(target_model)
    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
    for idx, (imgs, targets) in enumerate(train_loader):
        img = imgs[0].unsqueeze(0).to(device)  # [1,C,H,W]
        gt_count = targets[0]['points'].shape[0]
        # Temporary downscaled
        img = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)
        # Generate Grad-CAM
        wrapped_model.zero_grad()
        cam_map = cam(input_tensor=img, targets=[PointCountTarget()])[0]  # [H, W]
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-6)
        cam_map = torch.tensor(cam_map).to(device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        #Generate perturbation from U-Net
        perturbation = unet(img)  # [1,C,H,W]
        perturbation = F.interpolate(perturbation, size=img.shape[-2:], mode='bilinear', align_corners=False)
        # Apply CAM weighting (stronger perturbation in hot regions)
        perturbation = perturbation * (1 + args.cam_weight * cam_map)
        # Clamp perturbation
        perturbation = torch.clamp(perturbation, -args.epsilon, args.epsilon)
        adv_img = (img + perturbation).clamp(0, 1)
        # Forward pass with adversarial image
        pred_adv = target_model(adv_img)
        #Compute losses
        y_human_mask, logit_loss = logit_undercount_loss_sparsity(pred_adv, gt_count, epoch, args.epochs)
        clean_probs = torch.softmax(target_model(img)['pred_logits'], -1)[:,:,1]
        confidence = clean_probs[y_human_mask].mean() if y_human_mask.any() else 0.
        alpha = min(1.0, gt_count / 100.)

        hinge_loss = perturbation.pow(2).mean()
        tv_loss = total_variation_loss_norm(perturbation)
        pert_fft = torch.fft.rfft2(perturbation)
        freq_loss = torch.mean(torch.abs(pert_fft[..., 1:]))
        cam_loss = F.l1_loss(perturbation.abs().mean(1, keepdim=True), cam_map)

        #Fix loss equation
        loss = logit_loss * alpha \
            + hinge_loss * args.beta \
            + cam_loss * args.kappa
            + freq_loss * args.zeta \
            + tv_loss * args.gamma \


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()

        gc.collect()
        torch.cuda.empty_cache()
        total_loss += loss.item()
        if idx % 50 == 0:
            print(f"Epoch [{epoch}] Batch [{idx}] Loss: {loss.item():.4f}")
            # print(f"Current Threshold: {adaptive_threshold:.3f}")
            # print(f"Perturbation Stats - Max: {perturbation.abs().max().item():.3f}, "f"TV: {tv_loss.item():.3f}, Freq: {freq_loss.item():.3f}, Cam: {cam_loss.item():.3f}")
    total_loss /= len(train_loader)
    print(f"[Epoch {epoch}] Total Loss: {total_loss:.4f}")

### Validation ###
def adversarial_validation(unet, target_model, val_loader, args, epoch):
    device = args.device
    unet.eval()
    total_count_clean = 0
    total_count_adv = 0

    with torch.no_grad():
        for idx, (imgs, targets) in enumerate(val_loader):
            img = imgs[0].unsqueeze(0).to(device)
            gt_count = targets[0]['points'].shape[0]
            # temporary downsize to avoid memory error
            img = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)
            perturbation = unet(img)
            perturbation = torch.clamp(perturbation, -args.epsilon, args.epsilon)
            perturbation = F.interpolate(perturbation, size=img.shape[-2:], mode='bilinear', align_corners=False)
            adv_img = torch.clamp(img + perturbation, 0, 1)

            #pred_clean = target_model(img)
            with torch.no_grad():
                pred_adv = target_model(adv_img)

            #clean_points, clean_count = counting_points(pred_clean)
            adv_points, adv_count = counting_points(pred_adv)

            #total_count_clean += clean_count
            total_count_clean += gt_count
            total_count_adv += adv_count

            gc.collect()
            torch.cuda.empty_cache()

    gap = total_count_clean - total_count_adv
    print(f"[Epoch {epoch}] Validation Clean Count: {total_count_clean:.2f}, Adv Count: {total_count_adv:.2f}, Gap: {gap:.2f}")
    return total_count_clean, total_count_adv, gap

def calculate_psnr(clean_img, adv_img):
    """
    Calculate Peak Signal-to-Noise Ratio between clean and adversarial images
    """
    try:
        # Convert tensors to numpy arrays
        clean_np = clean_img.squeeze().cpu().numpy()
        adv_np = adv_img.squeeze().cpu().numpy()
        
        # Handle different channel dimensions
        if len(clean_np.shape) == 3:
            # Move channel to last dimension
            clean_np = np.transpose(clean_np, (1, 2, 0))
            adv_np = np.transpose(adv_np, (1, 2, 0))
        
        # Resize images to have the same dimensions
        if clean_np.shape != adv_np.shape:
            # Resize to the minimum dimensions
            min_height = min(clean_np.shape[0], adv_np.shape[0])
            min_width = min(clean_np.shape[1], adv_np.shape[1])
            
            # Use scikit-image resize
            from skimage.transform import resize
            clean_np = resize(clean_np, (min_height, min_width), preserve_range=True, anti_aliasing=True)
            adv_np = resize(adv_np, (min_height, min_width), preserve_range=True, anti_aliasing=True)
        
        # Ensure images are in [0, 1] range
        clean_np = np.clip(clean_np, 0, 1)
        adv_np = np.clip(adv_np, 0, 1)
        
        # Calculate MSE (Mean Squared Error)
        mse = np.mean((clean_np - adv_np) ** 2)
        
        # Avoid division by zero
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        max_pixel = 1.0  # Since images are in [0, 1] range
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        
        return psnr
    
    except Exception as e:
        print(f"PSNR calculation failed: {e}")
        print(f"Clean shape: {clean_img.shape}, Adv shape: {adv_img.shape}")
        return 0.0  # Return 0 on error

def validate_attack(target_model, unet, loader, weight_path, weight_name, device, args):
    target_model.eval()
    unet.eval()
    gap_stats = []
    
    # Load UNet weights
    ckpt = torch.load(os.path.join(weight_path, weight_name), map_location=device)
    if 'unet' in ckpt:
        unet.load_state_dict(ckpt['unet'])
    
    total_psnr = 0
    total_gap = 0
    num_images = 0
    
    with torch.no_grad():
        for img, targets in loader:
            img = img.to(device)  # [1, C, H, W]
            
            # Generate perturbation
            pert = unet(img).clamp(-args.epsilon, args.epsilon)
            pert = F.interpolate(pert, size=img.shape[-2:], mode='bilinear', align_corners=False)
            
            # Create adversarial image
            adv_img = (img + pert).clamp(0, 1)
            
            # --- Calculate PSNR (single image) ---
            psnr = calculate_psnr(img, adv_img)
            total_psnr += psnr
            # Model inference
            clean_out = target_model(img)
            adv_out = target_model(adv_img)
            
            clean_cnt = counting_points(clean_out)[1]
            adv_cnt = counting_points(adv_out)[1]
            
            gap = clean_cnt - adv_cnt
            total_gap += gap
            # Record stats
            gap_stats.append({
                'image_idx': num_images,
                'clean': float(clean_cnt),
                'adv': float(adv_cnt),
                'gap': float(gap),
                'pert_magnitude': pert.abs().mean().item(),
                'psnr': psnr
            })
            
            print(f'Image {num_images}: Gap={gap:.1f}, PSNR={psnr:.2f} dB')
            num_images += 1
    
    # Calculate averages
    avg_psnr = total_psnr / num_images if num_images > 0 else 0
    avg_gap = total_gap / num_images if num_images > 0 else 0
    print(f'\n=== Summary ===')
    print(f'Images processed: {num_images}')
    print(f'Average PSNR: {avg_psnr:.2f} dB')
    print(f'Average Count Gap: {avg_gap:.2f}')
    # Create DataFrame
    df = pd.DataFrame(gap_stats)
    
    return df

### Main ###
def main(args):
    device = args.device
    image_path = args.advex_dir
    weight_path = args.weight_dir

    dataset_train = build_dataset('train', args)
    dataset_val = build_dataset('val', args)

    train_loader = DataLoader(dataset_train, batch_size=1, sampler=RandomSampler(dataset_train),
                              collate_fn=utils.collate_fn, num_workers=0)
    val_loader = DataLoader(dataset_val, batch_size=1, sampler=SequentialSampler(dataset_val),
                            collate_fn=utils.collate_fn, num_workers=0)

    model = build_model(args, training=False)
    model.to(device)
    checkpoint = torch.load(args.weight, map_location=device)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    model.eval()

    # Initialize U-Net for perturbations
    unet = build_unet(args, type='vanilla')
    # unet = build_diffusion(args, type='vanilla')
    unet.to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    best_gap = -float('inf')  # Best (clean - adv) gap
    weight_name = r'shha_unet_cam.pth'
    for epoch in range(1, args.epochs + 1):
        adversarial_training_cam(unet, model, train_loader, optimizer, args, epoch)
        clean_count, adv_count, gap = adversarial_validation(unet, model, val_loader, args, epoch)
        if gap > best_gap:
            best_gap = gap
            torch.save({
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.weight_dir, weight_name))
            print(f"Saved best model (epoch {epoch}) with count gap {gap:.2f}")
    validate_attack(model, unet, val_loader, weight_path, weight_name, device, args)
    save_adversarial_image(unet, image_path, weight_path, weight_name, device, val_loader, args)

if __name__ == '__main__':
    args = config.args
    main(args)