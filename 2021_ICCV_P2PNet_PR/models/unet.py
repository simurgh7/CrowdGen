import math
import torch
import torch.nn as nn
import torch.nn.functional as F
####################################
# CBAM modules
####################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))
class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        ca = self.ca(x)
        sa = self.sa(ca * x)
        self.last_channel_att = self.ca.sigmoid  # You can store attention weights here
        self.last_spatial_att = self.sa.sigmoid
        return sa * ca * x  # or however you want to combine
    # def forward(self, x):
    #     x = x * self.ca(x)
    #     x = x * self.sa(x)
    #     return x
####################################
# Attention-guided U-Net with CBAM
####################################
class UNetCBAMPerturbationGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        self.enc1 = self._block(in_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self._block(base_channels * 4, base_channels * 8)

        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.cbam3 = CBAM(base_channels * 4)
        self.dec3 = self._block(base_channels * 8, base_channels * 4)

        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.cbam2 = CBAM(base_channels * 2)
        self.dec2 = self._block(base_channels * 4, base_channels * 2)

        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.cbam1 = CBAM(base_channels)
        self.dec1 = self._block(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, out_channels, 1)
        self.tanh = nn.Tanh()

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, density_map=None):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        bottleneck = self.bottleneck(self.pool(enc3))

        dec3 = self.upconv3(bottleneck)
        enc3 = self.cbam3(enc3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.cbam2(enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = self.cbam1(enc1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.tanh(self.out(dec1))
####################################
# Vanilla UNet
####################################
class UNetPerturbationGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        # Encoder (Downsampling)
        self.enc1 = self._block(in_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._block(base_channels * 4, base_channels * 8)
        
        # Decoder (Upsampling)
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = self._block(base_channels * 8, base_channels * 4)
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = self._block(base_channels * 4, base_channels * 2)
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = self._block(base_channels * 2, base_channels)
        
        # Output layer (tanh to constrain perturbations to [-ε, ε])
        self.out = nn.Conv2d(base_channels, out_channels, 1)
        self.tanh = nn.Tanh()

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))

        # Decoder with feature alignment via interpolation
        dec3 = self.upconv3(bottleneck)
        enc3 = _resize_to_match(enc3, dec3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = _resize_to_match(enc2, dec2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = _resize_to_match(enc1, dec1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.tanh(self.out(dec1))
####################################
# Diffusion-Aware CBAM UNet
####################################
class UNetCBAMDiffusion(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()
        # --- time embedding (for diffusion) ---
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, base_channels * 4),
            nn.ReLU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        # --- encoder ---
        self.enc1 = self._block(in_channels + base_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        # --- bottleneck ---
        self.bottleneck = self._block(base_channels * 4, base_channels * 8)
        # --- decoder with CBAM ---
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.cbam3 = CBAM(base_channels * 4)
        self.dec3 = self._block(base_channels * 8, base_channels * 4)

        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.cbam2 = CBAM(base_channels * 2)
        self.dec2 = self._block(base_channels * 4, base_channels * 2)

        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.cbam1 = CBAM(base_channels)
        self.dec1 = self._block(base_channels * 2, base_channels)

        # --- output ---
        self.out = nn.Conv2d(base_channels, out_channels, 1)
        self.tanh = nn.Tanh()

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, t=None):
        # --- Encoder ---
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # --- Bottleneck ---
        b = self.bottleneck(self.pool(e3))

        # --- Time embedding injection (for diffusion) ---
        if t is not None:
            t_emb = t_emb = _timestep_embedding(t, self.time_emb_dim, x.device)  # Pass device
            t_emb = self.time_mlp(t_emb)  # [B, base_channels*4]
            b = b + t_emb[:, :, None, None]  # broadcast into feature map

        # --- Decoder with CBAM attention on skip connections ---
        d3 = self.upconv3(b)
        e3 = self.cbam3(e3)       # apply CBAM
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        e2 = self.cbam2(e2)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        e1 = self.cbam1(e1)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        return self.tanh(self.out(d1))
#######################################
# Diffusion-Ready U-Net (with time)
#######################################
class UNetDiffusion(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_emb_dim=256, cond_channels=3):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        # Time Embedding Projection
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(time_emb_dim, time_emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(time_emb_dim, time_emb_dim)
        )
        # Condition projection (processes the conditioning image)
        self.cond_conv = torch.nn.Sequential(
            torch.nn.Conv2d(cond_channels, base_channels, 3, padding=1),
            torch.nn.ReLU()
        )
        # Encoder Downsamples
        self.enc1 = self._block(in_channels + base_channels, base_channels) # First block takes only the input noise
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.pool = torch.nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = self._block(base_channels * 4, base_channels * 8)
        # Decoder Upsamples
        self.upconv3 = torch.nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self._block((base_channels * 4) * 2, base_channels * 4) # x2 for skip connection
        self.upconv2 = torch.nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self._block((base_channels * 2) * 2, base_channels * 2) # x2 for skip connection
        self.upconv1 = torch.nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self._block(base_channels * 2, base_channels) # x2 for skip connection (base_channels from enc1 + base_channels from upconv1)
        # Final output layer
        self.out = torch.nn.Conv2d(base_channels, out_channels, kernel_size=1)
        # Time projection layers for each stage
        # This will project the time embedding to the channel size of each stage
        self.time_proj_enc1 = torch.nn.Linear(time_emb_dim, base_channels)
        self.time_proj_enc2 = torch.nn.Linear(time_emb_dim, base_channels * 2)
        self.time_proj_enc3 = torch.nn.Linear(time_emb_dim, base_channels * 4)
        self.time_proj_bottleneck = torch.nn.Linear(time_emb_dim, base_channels * 8)
        self.time_proj_dec3 = torch.nn.Linear(time_emb_dim, base_channels * 4)
        self.time_proj_dec2 = torch.nn.Linear(time_emb_dim, base_channels * 2)
        self.time_proj_dec1 = torch.nn.Linear(time_emb_dim, base_channels)

    def _block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
    
    def _add_time_embedding(self, x, t_emb, proj_layer):
        """Adds the time embedding to feature map x."""
        # Project time embedding to have the same number of channels as x
        time_proj = proj_layer(t_emb) # [B, Channels]
        time_proj = time_proj[:, :, None, None] # [B, Channels, 1, 1]
        # Add it to each spatial position in x
        return x + time_proj

    def forward(self, x, t, cond_img):
        """
        x: Noisy perturbation [B, C, H, W] (the thing we are denoising)
        t: Timestep [B]
        cond_img: Conditioning image [B, C, H, W] (the clean image)
        """
        # 1. Process time embedding
        t_emb = _timestep_embedding(t, self.time_emb_dim, x.device)  # Pass device
        t_emb = self.time_mlp(t_emb) # [B, time_emb_dim]
        # 2. Process condition image
        cond_feat = self.cond_conv(cond_img) # [B, base_channels, H, W]
        # 3. Encoder Path
        # Concatenate condition at the INPUT of the first block
        x = torch.cat([x, cond_feat], dim=1) # [B, in_channels + base_channels, H, W]
        e1 = self.enc1(x) # [B, base_channels, H, W]
        e1 = self._add_time_embedding(e1, t_emb, self.time_proj_enc1)
        e2 = self.enc2(self.pool(e1)) # [B, base_channels*2, H/2, W/2]
        e2 = self._add_time_embedding(e2, t_emb, self.time_proj_enc2)
        e3 = self.enc3(self.pool(e2)) # [B, base_channels*4, H/4, W/4]
        e3 = self._add_time_embedding(e3, t_emb, self.time_proj_enc3)
        # 4. Bottleneck
        b = self.bottleneck(self.pool(e3)) # [B, base_channels*8, H/8, W/8]
        b = self._add_time_embedding(b, t_emb, self.time_proj_bottleneck)
        # 5. Decoder Path
        d3 = self.upconv3(b) # [B, base_channels*4, H/4, W/4]
        # Ensure spatial size matches for skip connection
        e3 = _crop_to_match(e3, d3)
        d3 = torch.cat([d3, e3], dim=1) # [B, (base_channels*4)*2, H/4, W/4]
        d3 = self.dec3(d3)
        d3 = self._add_time_embedding(d3, t_emb, self.time_proj_dec3)
        d2 = self.upconv2(d3) # [B, base_channels*2, H/2, W/2]
        e2 = _crop_to_match(e2, d2)
        d2 = torch.cat([d2, e2], dim=1) # [B, (base_channels*2)*2, H/2, W/2]
        d2 = self.dec2(d2)
        d2 = self._add_time_embedding(d2, t_emb, self.time_proj_dec2)
        d1 = self.upconv1(d2) # [B, base_channels, H, W]
        e1 = _crop_to_match(e1, d1)
        d1 = torch.cat([d1, e1], dim=1) # [B, base_channels*2, H, W]
        d1 = self.dec1(d1)
        d1 = self._add_time_embedding(d1, t_emb, self.time_proj_dec1)
        # 6. Final output
        return self.out(d1) # Predicts the noise [B, out_channels, H, W]
######################################
### Utility Functions ###
######################################
def _crop_to_match(enc_feat, dec_feat):
    _, _, h_dec, w_dec = dec_feat.shape
    _, _, h_enc, w_enc = enc_feat.shape
    # Compute crop margins
    crop_top = (h_enc - h_dec) // 2
    crop_left = (w_enc - w_dec) // 2
    return enc_feat[:, :, crop_top:crop_top + h_dec, crop_left:crop_left + w_dec]
def _resize_to_match(src, target):
    return F.interpolate(src, size=target.shape[2:], mode='bilinear', align_corners=False)
def _timestep_embedding(timesteps, dim, device=None):
    half = dim // 2
    # Create freqs on the same device as timesteps
    if device is None:
        device = timesteps.device
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=device) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb