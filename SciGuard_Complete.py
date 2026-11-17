import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import time
import timm
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

# ============================================================
# 1. FORENSIC LAYERS (M2 Components)
# ============================================================

class BayarConv2d(nn.Module):
    """
    Constrained convolution layer for noise extraction.
    Ensures that the kernel learns residuals (high-pass filters).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)

    def _constrain(self):
        with torch.no_grad():
            # Force center of kernel to 0 for initial sum calculation
            mid = self.kernel_size // 2
            k = self.kernel.clone()
            k[:, :, mid, mid] = 0.0
            
            # Normalize so each kernel sums to 1 (excluding center)
            s = k.sum(dim=(-1, -2), keepdim=True)
            k = k / (s + 1e-6)
            
            # Set center to -1, making the whole kernel sum to 0 (High-pass)
            k[:, :, mid, mid] = -1.0
            self.kernel.data.copy_(k)

    def forward(self, x):
        self._constrain()
        return F.conv2d(x, self.kernel, padding=self.kernel_size//2)

class SRMFilterBank(nn.Module):
    """
    Static Filter Bank: Implements standard Forensic SRM kernels.
    Used for detecting basic digital inconsistencies.
    """
    def __init__(self):
        super().__init__()
        # Define 3 standard SRM kernels
        f1 = np.array([[ 0,  0,  0,  0,  0], [ 0, -1,  2, -1,  0], [ 0,  2, -4,  2,  0], [ 0, -1,  2, -1,  0], [ 0,  0,  0,  0,  0]], dtype=np.float32) / 4.0
        f2 = np.array([[-1,  2, -2,  2, -1], [ 2, -6,  8, -6,  2], [-2,  8,-12,  8, -2], [ 2, -6,  8, -6,  2], [-1,  2, -2,  2, -1]], dtype=np.float32) / 12.0
        f3 = np.array([[ 0,  0,  0,  0,  0], [ 0,  0,  0,  0,  0], [ 0,  0, -1,  0,  0], [ 0,  0,  2,  0,  0], [ 0,  0, -1,  0,  0]], dtype=np.float32) / 2.0
        
        # Stack into a weight buffer [Out, In, H, W]
        kernels = torch.from_numpy(np.stack([f1, f2, f3], axis=0)[:, np.newaxis]).repeat(3, 1, 1, 1)
        self.register_buffer("weight", kernels)

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=2, groups=3)

# ============================================================
# 2. M1/M2 STREAM COMPONENTS
# ============================================================

class ForensicStream(nn.Module):
    """
    M2: Forensic Stream. Processes noise maps via a parallel encoder path.
    """
    def __init__(self, channels=[64, 128, 256, 512]):
        super().__init__()
        self.bayar = BayarConv2d(3, 3)
        self.srm = SRMFilterBank()
        
        # Input to fusion is 12 channels (3 Bayar + 9 SRM)
        self.init_conv = nn.Sequential(
            nn.Conv2d(12, channels[0], 3, padding=1),
            nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True)
        )
        
        # Encoder sequence to match Swin's 4 stages (plus initial patch-4 scale)
        self.stages = nn.ModuleList()
        # Initial scale in Swin is 1/4 (e.g. 64x64). Forensic starts at 1/1 (256x256).
        # We need 2 stages to reach 1/4, then match the rest.
        
        # Stage 0: 256 -> 128 -> 64 (Match beginning of Swin Stage 1)
        self.prepare = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], channels[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True)
        )
        
        curr_ch = channels[0]
        for ch in channels[1:]:
            self.stages.append(nn.Sequential(
                nn.Conv2d(curr_ch, ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True)
            ))
            curr_ch = ch

    def forward(self, x):
        noise = torch.cat([self.bayar(x), self.srm(x)], dim=1)
        f = self.init_conv(noise)
        f = self.prepare(f) # Reaches 64x64
        features = [f]
        for stage in self.stages:
            f = stage(f)
            features.append(f)
        return features # [64x64, 32x32, 16x16, 8x8] — Matches Swin!

# ============================================================
# 3. M3: CROSS-ATTENTION FUSION
# ============================================================

class CrossAttentionGate(nn.Module):
    """
    M3 Fusion: Gated Multi-Head Cross Attention.
    Cross-references Spatial (M1) and Forensic (M2) features.
    """
    def __init__(self, in_ch, embed_dim=256, num_heads=8):
        super().__init__()
        self.proj_s = nn.Conv2d(in_ch, embed_dim, 1)
        self.proj_f = nn.Conv2d(in_ch, embed_dim, 1)
        
        self.attn_s2f = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_f2s = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        self.out_proj = nn.Conv2d(embed_dim, in_ch, 1)

    def forward(self, feat_s, feat_f):
        B, C, H, W = feat_s.shape
        S = self.proj_s(feat_s).flatten(2).transpose(1, 2)
        F_feat = self.proj_f(feat_f).flatten(2).transpose(1, 2)
        
        # Cross-Attend
        s_enh, _ = self.attn_s2f(query=S, key=F_feat, value=F_feat)
        f_enh, _ = self.attn_f2s(query=F_feat, key=S, value=S)
        
        # Gate
        g = self.gate(torch.cat([s_enh, f_enh], dim=-1))
        fused = g * s_enh + (1 - g) * f_enh
        
        fused = fused.transpose(1, 2).reshape(B, -1, H, W)
        return self.out_proj(fused), g

# ============================================================
# 4. DECODER & FINAL MODEL
# ============================================================

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear')
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SciGuard(nn.Module):
    """
    THE COMPLETE ARCHITECTURE:
    M1 (Swin V2) + M2 (Forensic) fused via M3 (Cross-Attention)
    with a hierarchical U-Net decoder.
    """
    def __init__(self):
        super().__init__()
        # M1: Swin V2 Encoder (Extracts Hierarchical Features)
        self.m1 = timm.create_model('swinv2_tiny_window8_256', 
                                    pretrained=True, features_only=True)
        m1_channels = self.m1.feature_info.channels() # e.g. [96, 192, 384, 768]
        
        # M2: Forensic Path (SRM/Bayar based)
        self.m2 = ForensicStream(channels=m1_channels)
        
        # M3: Fusion Gate at the bottleneck
        self.fusion = CrossAttentionGate(in_ch=m1_channels[-1])
        
        # Decoder Blocks
        self.up4 = DecoderBlock(m1_channels[-1], m1_channels[-2] * 2, 256) # *2 for combined skip
        self.up3 = DecoderBlock(256, m1_channels[-3] * 2, 128)
        self.up2 = DecoderBlock(128, m1_channels[-4] * 2, 64)
        
        self.final_mask = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'), # Patch-4 to pixel
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        # 1. Feature Extraction
        m1_raw = self.m1(x)
        m2_feats = self.m2(x)
        
        # Swin V2 returns [B, H, W, C]. Transform to [B, C, H, W] for the rest of the pipeline
        m1_feats = [f.permute(0, 3, 1, 2) if len(f.shape) == 4 and f.shape[-1] > f.shape[1] else f for f in m1_raw]
        
        # 2. Fusion at Bottleneck
        fused_bottleneck, gate_weights = self.fusion(m1_feats[-1], m2_feats[-1])
        
        # 3. Decoding with Combined Skips (M1 + M2 logic)
        s3 = torch.cat([m1_feats[-2], m2_feats[-2]], dim=1)
        x = self.up4(fused_bottleneck, s3)
        
        s2 = torch.cat([m1_feats[-3], m2_feats[-3]], dim=1)
        x = self.up3(x, s2)
        
        s1 = torch.cat([m1_feats[-4], m2_feats[-4]], dim=1)
        x = self.up2(x, s1)
        
        mask = self.final_mask(x)
        return mask, gate_weights

# ============================================================
# 5. DATASET & LOSS
# ============================================================

class SciGuardDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.transform = transform
        self.samples = []
        
        # Paths for Forged and Authentic
        base_img_dir = os.path.join(base_path, "recodai-luc-scientific-image-forgery-detection", "train_images")
        mask_dir = os.path.join(base_path, "recodai-luc-scientific-image-forgery-detection", "train_masks")

        # 1. Load Forged Samples
        forged_dir = os.path.join(base_img_dir, "forged")
        for f in sorted(os.listdir(forged_dir)):
            if f.endswith('.png'):
                self.samples.append({
                    'img': os.path.join(forged_dir, f),
                    'mask': os.path.join(mask_dir, f.replace('.png', '.npy')),
                    'is_forged': True
                })

        # 2. Load Authentic Samples
        authentic_dir = os.path.join(base_img_dir, "authentic")
        if os.path.exists(authentic_dir):
            for f in sorted(os.listdir(authentic_dir)):
                if f.endswith('.png'):
                    self.samples.append({
                        'img': os.path.join(authentic_dir, f),
                        'mask': None, # Will generate zero-mask
                        'is_forged': False
                    })

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.cvtColor(cv2.imread(sample['img']), cv2.COLOR_BGR2RGB)
        
        if sample['is_forged']:
            mask = np.load(sample['mask']).astype(np.float32)
        else:
            # Generate zero mask for authentic images
            mask = np.zeros(image.shape[:2], dtype=np.float32)

        if len(mask.shape) > 2: mask = mask[:, :, 0]
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']
            
        return image, mask.unsqueeze(0)

def combo_loss(pred, target):
    # Forensic Weighting: Since only ~1% of pixels are forged, we weight the 
    # '1' pixels much higher to force the model to learn the forgery features.
    pos_weight = torch.tensor([10.0]).to(pred.device)  # Balanced — was 25 (too aggressive)
    bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
    
    pred_sig = torch.sigmoid(pred).view(-1)
    target_f = target.view(-1)
    
    # Dice Loss helps with boundary shape
    dice = 1 - (2. * (pred_sig * target_f).sum() + 1e-6) / (pred_sig.sum() + target_f.sum() + 1e-6)
    
    return bce + dice

# ============================================================
# 6. EXECUTION SCRIPT
# ============================================================

def train_engine(model, train_loader, val_loader, epochs=30):
    # 🌍 DEVICE PICKER — MPS (Apple Silicon GPU) is NOT a CPU, never cool it
    is_gpu = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()
    COOLING_MODE = not (is_gpu or is_mps)  # Only throttle on pure CPU

    if is_gpu:
        device = torch.device("cuda")
    elif is_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device} | Cooling: {'ON (CPU only)' if COOLING_MODE else 'OFF ✅'}")
    model.to(device)

    # Tuned optimizer: slightly higher LR + cosine decay for smooth convergence
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        # ── Training ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Train]", leave=False)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds, _ = model(images)
            loss = combo_loss(preds, masks)
            loss.backward()
            # Gradient clipping — prevents exploding gradients with Swin
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if COOLING_MODE:
                time.sleep(0.02)  # Reduced: 20ms vs old 50ms

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Val]  ", leave=False)
            for images, masks in vbar:
                images, masks = images.to(device), masks.to(device)
                preds, _ = model(images)
                val_loss += combo_loss(preds, masks).item()
                # Dice coefficient — real quality indicator (higher = better)
                pred_bin = (torch.sigmoid(preds) > 0.5).float()
                dice = (2*(pred_bin*masks).sum() + 1e-6) / (pred_bin.sum() + masks.sum() + 1e-6)
                val_dice += dice.item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        avg_dice  = val_dice   / len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Dice: {avg_dice:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        torch.save(model.state_dict(), "sciguard_final.pth")
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_sciguard.pth")
            print(f"  ✅ Best saved! Val Loss: {best_val_loss:.4f} | Dice: {avg_dice:.4f}")

if __name__ == "__main__":
    # 1. Dynamic path — works on any machine
    # Auto-detects Colab (/content/sifd) vs local (./sifd) — no manual edit needed
    if os.path.exists("/content/sifd"):
        base_path = "/content/sifd"
    else:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sifd")

    # 2. Separate transforms: augmented for train, clean for val
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])
    val_transform = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])

    try:
        # Load dataset once (no transform) just to count and get indices
        full_ds = SciGuardDataset(base_path, transform=None)
        forged_count = sum(1 for s in full_ds.samples if s['is_forged'])
        auth_count   = len(full_ds) - forged_count
        print(f"Dataset Stats: Found {len(full_ds)} images.")
        print(f"   - Forged:    {forged_count}")
        print(f"   - Authentic: {auth_count}")

        if len(full_ds) == 0:
            print("WARNING: No images found. Check path structure.")
        else:
            # Split indices
            indices = list(range(len(full_ds)))
            train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

            # Two dataset instances with their own transforms
            train_ds = SciGuardDataset(base_path, transform=train_transform)
            val_ds   = SciGuardDataset(base_path, transform=val_transform)

            # num_workers=0 required for MPS/Mac stability
            train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=8, shuffle=True,  num_workers=0, pin_memory=False)
            val_loader   = DataLoader(Subset(val_ds,   val_idx),   batch_size=8, shuffle=False, num_workers=0, pin_memory=False)

            print(f"   Train: {len(train_idx)} | Val: {len(val_idx)} | Batch size: 8")

            # Model init + sanity check
            model = SciGuard()
            print("Sci-Guard: Swin-V2 + Forensic Stream + Cross-Attention Fusion ✅")
            sample_img, _ = val_ds[0]
            model.eval()
            with torch.no_grad():
                out, _ = model(sample_img.unsqueeze(0))
            print(f"Sanity check PASSED — Input: {sample_img.shape} → Mask: {out.shape}")

            # Start training
            print("\n🚀 Starting Training...\n")
            train_engine(model, train_loader, val_loader, epochs=30)

    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
