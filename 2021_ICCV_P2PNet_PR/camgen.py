import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import os
import sys

from torchvision import datasets, transforms
import glob

class GradCam():
    """
    Implements the core Grad-CAM algorithm.
    It hooks into a specified convolutional layer of a model to capture activations
    and gradients, then computes a heatmap indicating important regions.
    """
    # Class-level variables to store activations and gradients
    # These are reset for each call to ensure no stale data.
    _activations = None
    _gradients = None
    # List to keep track of hook handles for proper removal
    _hook_handles = []
    def __init__(self, model, target_layer, use_cuda=True):
        """
        Initializes the GradCam instance.
        Args:
            model (torch.nn.Module): The neural network model. It will be set to eval mode.
            target_layer (torch.nn.Module): The specific convolutional layer module
                                            within the model to hook into for Grad-CAM.
                                            This should be a direct reference to the layer,
                                            e.g., `model.backbone.layer4[-1]`.
            use_cuda (bool): Whether to use CUDA for computations.
        """
        self.model = model.eval() # Ensure model is in evaluation mode
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

        if not isinstance(target_layer, torch.nn.Module):
            raise TypeError("`target_layer` must be a `torch.nn.Module` instance from your model.")

        # Clear any existing hooks from previous GradCam instances
        self.clear_hooks() 
        # Highly Model Dependent - P2PNet #
        # Register forward hook to capture activations
        self._hook_handles.append(target_layer.register_forward_hook(self._save_activation))
        # Register backward hook to capture gradients
        self._hook_handles.append(target_layer.register_backward_hook(self._save_gradient))
        
        print(f"GradCam: Hooks registered on layer: {target_layer.__class__.__name__}")
        # Highly Model Dependent - P2PNet #
        
    def _save_activation(self, module, input, output):
        """Forward hook: Stores the output (activations) of the target layer."""
        GradCam._activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: Stores the gradients with respect to the output of the target layer."""
        # grad_output is a tuple, we need the first element which is the gradient w.r.t. the output
        GradCam._gradients = grad_output[0] 
    
    def clear_hooks(self):
        """Removes all registered hooks to prevent memory leaks."""
        for handle in GradCam._hook_handles:
            handle.remove()
        GradCam._hook_handles = []
        # Also clear stored activations and gradients
        GradCam._activations = None
        GradCam._gradients = None

    def _backprop_from_scalar_score(self, scalar_score_tensor):
        """
        Performs a backward pass from a scalar score to compute gradients
        needed for Grad-CAM.
        
        Args:
            scalar_score_tensor (torch.Tensor): A scalar tensor representing the score
                                                (e.g., predicted count) to backpropagate from.
        """
        self.model.zero_grad() # Clear gradients for all model parameters
        # Backpropagate from the scalar score. retain_graph=True is often used
        # if you need to call .backward() multiple times on the same graph,
        # but for a single CAM computation, it might not be strictly necessary.
        scalar_score_tensor.backward(retain_graph=True) 
    
    def _get_alpha_weights(self, scalar_score_tensor):
        """
        Calculates the neuron importance weights (alpha_k) for each channel
        of the target layer's feature maps.
        
        Args:
            scalar_score_tensor (torch.Tensor): The scalar score tensor to backpropagate from.
            
        Returns:
            torch.Tensor: A 1D tensor of alpha weights, one for each channel.
        """
        self._backprop_from_scalar_score(scalar_score_tensor) # Compute gradients first
        
        if GradCam._gradients is None:
            raise RuntimeError("Gradients are None. Ensure backward pass was successful "
                               "and the target layer received gradients.")
        
        # Global average pooling of gradients across spatial dimensions (H, W)
        # Squeeze batch dimension (assuming batch size 1)
        # Resulting shape: (Channels,)
        alpha_weights = GradCam._gradients.squeeze(0).mean(dim=(1, 2))
        return alpha_weights
    
    def __call__(self, input_tensor):
        """
        Generates the Grad-CAM heatmap for the given input tensor.
        
        Args:
            input_tensor (torch.Tensor): The preprocessed (e.g., normalized) input image tensor.
                                         Should have `requires_grad=True`.
                                         Shape: (N, C, H, W), typically N=1 for single image.
                                         
        Returns:
            tuple: A tuple containing:
                - heatmap_tensor (torch.Tensor): The computed Grad-CAM heatmap as a tensor,
                                                 resized to the input image's spatial dimensions.
                                                 This is suitable for use in loss functions.
                - heatmap_numpy (numpy.ndarray): The computed Grad-CAM heatmap as a normalized
                                                 NumPy array (0-1 range), suitable for visualization.
        """
        self.model.eval() # Ensure model is in eval mode before forward pass
        
        # Forward pass through the model to get its output
        # For P2PNet, this `outputs` will be a dictionary.
        outputs = self.model(input_tensor)
        
        # --- CRITICAL: Extract a scalar score for backpropagation from P2PNet's output ---
        # This part needs to be adapted based on P2PNet's exact output structure.
        # Assuming P2PNet returns a dictionary with 'pred_logits' (scores for queries)
        scalar_score_for_cam = torch.tensor(0.0, device=input_tensor.device) # Default to 0.0
        
        if isinstance(outputs, dict) and 'pred_logits' in outputs:
            # Apply sigmoid to get probabilities for each query
            prob = outputs['pred_logits'].sigmoid()
            # Assuming class 0 is the 'person' class, sum its probabilities across all queries.
            # This provides a differentiable proxy for the total predicted count.
            scalar_score_for_cam = prob[:, :, 0].sum() 
        else:
            print("Warning: P2PNet output structure not recognized for score calculation. "
                  "Expected 'pred_logits' in output dictionary. Using default 0.0 score.")
            # If your P2PNet model returns a direct count or a density map, adjust this:
            # e.g., `scalar_score_for_cam = outputs.sum()` if `outputs` is a density map tensor.
            
        # Get the channel importance weights (alpha_k)
        alpha_weights = self._get_alpha_weights(scalar_score_for_cam)
        
        if GradCam._activations is None:
            raise RuntimeError("Activations are None. Ensure forward pass was successful "
                               "and the target layer produced activations.")

        # Compute the raw Grad-CAM heatmap
        # Linear combination of activations and alpha weights
        # `_activations` shape: (N, C, H', W'), `alpha_weights` shape: (C,)
        # Unsqueeze alpha_weights to (C, 1, 1) for broadcasting
        cam_raw = (alpha_weights.unsqueeze(-1).unsqueeze(-1) * GradCam._activations.squeeze(0)).sum(dim=0)
        cam_raw = F.relu(cam_raw) # Apply ReLU to focus on positive contributions
        
        # Normalize the heatmap tensor to [0, 1] for consistency and potential use in loss
        # This normalization is differentiable.
        heatmap_tensor = cam_raw - cam_raw.min()
        if heatmap_tensor.max() > 0:
            heatmap_tensor = heatmap_tensor / heatmap_tensor.max()
        else:
            heatmap_tensor = torch.zeros_like(heatmap_tensor) # Handle case where heatmap is all zeros

        # Resize the heatmap tensor to the original input image's spatial dimensions
        # This is important if the regularization loss term needs to be image-sized.
        heatmap_tensor_resized = F.interpolate(heatmap_tensor.unsqueeze(0).unsqueeze(0), 
                                               size=(input_tensor.shape[2], input_tensor.shape[3]), 
                                               mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        # Convert the resized heatmap tensor to NumPy for visualization (non-differentiable)
        heatmap_numpy = heatmap_tensor_resized.data.cpu().numpy()
        
        return heatmap_tensor_resized, heatmap_numpy

class CAM:
    """
    A wrapper class for GradCam, providing a simplified interface for
    generating and optionally visualizing CAMs for crowd counting models.
    """
    
    def __init__(self, model, target_layer):
        """
        Initializes the CAM instance.
        
        Args:
            model (torch.nn.Module): The crowd counting model.
            target_layer (torch.nn.Module): The specific layer within the model
                                            for Grad-CAM computation.
        """
        self.grad_cam = GradCam(model=model, target_layer=target_layer, use_cuda=True)
        
    def __call__(self, img_tensor):
        """
        Generates the Grad-CAM heatmap for an input image tensor.
        
        Args:
            img_tensor (torch.Tensor): The preprocessed (e.g., normalized) input image tensor.
                                       This tensor should have `requires_grad=True` if used
                                       in an adversarial attack context.
                                       
        Returns:
            tuple: A tuple containing:
                - heatmap_tensor (torch.Tensor): The computed Grad-CAM heatmap as a tensor,
                                                 resized to the input image's spatial dimensions.
                                                 This is the `s_map` used in the PAP loss.
                - heatmap_numpy (numpy.ndarray): The computed Grad-CAM heatmap as a normalized
                                                 NumPy array (0-1 range), suitable for visualization.
        """
        # The `img_tensor` passed here is expected to be already preprocessed (normalized)
        # by the calling script (e.g., `pap_attack_p2pnet.py`).
        heatmap_tensor, heatmap_numpy = self.grad_cam(img_tensor)
        return heatmap_tensor, heatmap_numpy
        
    def show_cam_on_image(self, original_img_pil, heatmap_numpy, save_path, filename="cam_overlay.jpg"):
        """
        Overlays the Grad-CAM heatmap onto the original image and saves it.
        
        Args:
            original_img_pil (PIL.Image.Image): The original, unnormalized PIL image.
            heatmap_numpy (numpy.ndarray): The normalized (0-1) Grad-CAM heatmap NumPy array.
            save_path (str): Directory where the image will be saved.
            filename (str): Name of the file to save.
        """
        # Convert PIL image to NumPy array (H, W, C) and scale to 0-255
        img_np = np.array(original_img_pil.convert('RGB'))
        
        # Resize heatmap to original image dimensions
        h, w = img_np.shape[0], img_np.shape[1]
        heatmap_resized = cv2.resize(heatmap_numpy, (w, h))
        
        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = np.float32(heatmap_colored) / 255 # Scale to 0-1
        
        # Convert original image to float32 and scale to 0-1
        img_float = np.float32(img_np) / 255
        
        # Overlay heatmap on image
        # You can adjust the alpha (transparency) here if needed, e.g., 0.7 * heatmap_colored
        cam_overlay = img_float + heatmap_colored 
        cam_overlay = cam_overlay / np.max(cam_overlay) # Normalize combined image to 0-1
        
        # Convert back to uint8 for saving
        final_image = Image.fromarray(np.uint8(255 * cam_overlay))
        
        os.makedirs(save_path, exist_ok=True)
        final_image.save(os.path.join(save_path, filename))
        print(f"CAM overlay saved to: {os.path.join(save_path, filename)}")