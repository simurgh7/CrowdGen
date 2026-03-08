import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from tqdm import tqdm

from models.model import SASNet
from datasets import build_dataset 
import config

def true_gradcam_sasnet():
    """Real Grad-CAM using the last backbone layer"""
    args = config.args
    os.makedirs(args.explanation_dir, exist_ok=True)
    
    # 1. Load model
    model = SASNet(args=args).to(args.device)
    if args.weight and os.path.exists(args.weight):
        model.load_state_dict(torch.load(args.weight, map_location=args.device))
    
    model.eval()
    
    # 2. Get the last Conv2d in features5 (VGG16 last block)
    target_layer = None
    
    # Search backwards through features5
    for i in range(len(model.features5) - 1, -1, -1):
        if isinstance(model.features5[i], torch.nn.Conv2d):
            target_layer = model.features5[i]
            print(f"Using last backbone Conv2d: features5[{i}] = {target_layer}")
            break
    
    if target_layer is None:
        print("ERROR: No Conv2d found in features5!")
        return
    
    # 3. Load dataset
    clean_dataset = build_dataset(purpose='val', args=args)
    clean_dataloader = torch.utils.data.DataLoader(
        clean_dataset, batch_size=1, shuffle=False, num_workers=1
    )
    adv_img_list_path = args.adv_npy_path
    adv_img_paths_raw = np.load(adv_img_list_path, allow_pickle=True).tolist()
    
    adv_dataset = build_dataset(purpose='val', args=args)
    # Update paths
    gt_base_path = os.path.join(args.data_root, 'val_data', 'gt-h5_2048')
    new_adv_img_map = {}
    
    for adv_path in adv_img_paths_raw:
        img_filename = os.path.basename(adv_path)
        gt_filename = img_filename.replace('.jpg', '.h5').replace('.png', '.h5')
        gt_full_path = os.path.join(gt_base_path, gt_filename)
        
        if os.path.exists(adv_path) and os.path.exists(gt_full_path):
            new_adv_img_map[adv_path] = gt_full_path
    
    adv_dataset.img_map = new_adv_img_map
    adv_dataset.img_list = adv_img_paths_raw
    adv_dataset.nSamples = len(adv_dataset.img_list)
    
    adv_dataloader = torch.utils.data.DataLoader(
        adv_dataset, batch_size=1, shuffle=False, num_workers=1
    )
    
    # 4. REAL Grad-CAM Implementation
    def compute_gradcam(model, input_tensor, target_layer):
        """Compute Grad-CAM for a single image"""
        # Hook to capture activations and gradients
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        try:
            # Forward pass
            output = model(input_tensor)
            
            # Target: sum of density map
            target = output.sum()
            
            # Backward pass
            model.zero_grad()
            target.backward(retain_graph=True)
            
            # Compute Grad-CAM
            if activations and gradients:
                activation = activations[0]
                gradient = gradients[0]
                
                # Global average pooling of gradients
                weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
                
                # Weighted combination of activations
                cam = torch.sum(weights * activation, dim=1, keepdim=True)
                cam = F.relu(cam)  # ReLU to keep positive contributions
                
                # Normalize
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-8)
                cam = cam.squeeze().cpu().numpy()
                
                return cam
            else:
                return None
                
        finally:
            # Remove hooks
            forward_handle.remove()
            backward_handle.remove()
    
    # 5. Process images
    for idx, (images, targets) in enumerate(tqdm(adv_dataloader)):
    #for idx, (images, targets) in enumerate(tqdm(clean_dataloader)):
        # if idx >= 5:  # Process only 5 images
        #     break
        
        images = images.to(args.device)
        image_id = targets["name"][0].split('.')[0]
        
        # Compute REAL Grad-CAM
        cam = compute_gradcam(model, images, target_layer)
        
        if cam is None:
            print(f"Failed to compute Grad-CAM for {image_id}")
            continue
        
        # Denormalize image
        img_np = images[0].cpu().detach()
        for i in range(3):
            if i == 0:
                img_np[i] = img_np[i] * 0.229 + 0.485
            elif i == 1:
                img_np[i] = img_np[i] * 0.224 + 0.456
            else:
                img_np[i] = img_np[i] * 0.225 + 0.406
        
        img_np = torch.clamp(img_np, 0, 1)
        rgb_img = img_np.permute(1, 2, 0).numpy()
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]))
        
        # Create overlay
        overlay = cv2.addWeighted(np.uint8(255 * rgb_img), 0.6, heatmap, 0.4, 0)
        
        # Save
        out_path = os.path.join(args.explanation_dir, f"{image_id}_sasnet.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        print(f"Saved: {out_path}")
    
    print(f"Done! Check {args.explanation_dir} for results.")

if __name__ == '__main__':
    true_gradcam_sasnet()