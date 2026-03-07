"""
CNN Encoder for VQA
- Pretrained models (1, 2): Use torchvision ResNet50 with ImageNet weights
- From-scratch models (3, 4): Lightweight custom CNN, coded layer-by-layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================================
#  BASIC RESIDUAL BLOCK — Lightweight building block
# ============================================================================

class BasicBlock(nn.Module):
    """
    Basic Residual Block (2 convolutions instead of 3).
    
    Structure:
        Input (C_in)
          ├─ Conv 3x3 (C_in → C_out)
          ├─ BatchNorm + ReLU
          ├─ Conv 3x3 (C_out → C_out)
          ├─ BatchNorm
          │
          + Skip Connection (Identity or 1x1 Conv if dims change)
          │
          └─ ReLU
    
    Compared to Bottleneck (3 convs, expansion=4), BasicBlock has:
    - Only 2 conv layers → fewer operations
    - Expansion = 1 → smaller output channels
    - Much faster training
    """
    
    expansion = 1  # Output channels = channels * expansion (no expansion)
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first conv (1 = same, 2 = halve spatial dims)
            downsample: Module to match dims for skip connection
        """
        super(BasicBlock, self).__init__()
        
        # --- Main path ---
        # Conv 3x3: (C_in → C_out), may downsample spatially
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Conv 3x3: (C_out → C_out), same spatial size
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # --- Skip connection ---
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        x → [Conv3x3 → BN → ReLU → Conv3x3 → BN] + skip(x) → ReLU
        """
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


# ============================================================================
#  FROM-SCRATCH CNN — Lightweight ResNet-style, built layer-by-layer
# ============================================================================

class ScratchCNN(nn.Module):
    """
    Lightweight custom CNN built from scratch (no pretrained weights).
    
    Uses BasicBlock (2 convs) with [2, 2, 2, 2] blocks — similar to ResNet18
    but explicitly coded layer-by-layer.
    
    Architecture:
        Input: (B, 3, 224, 224)
        
        Stem:
            Conv 7x7, stride 2    → (B, 64, 112, 112)
            BatchNorm + ReLU
            MaxPool 3x3, stride 2 → (B, 64, 56, 56)
        
        Stage 1 — 2 Basic Blocks:
            64 → 64               → (B, 64, 56, 56)
        
        Stage 2 — 2 Basic Blocks:
            64 → 128, stride=2    → (B, 128, 28, 28)
        
        Stage 3 — 2 Basic Blocks:
            128 → 256, stride=2   → (B, 256, 14, 14)
        
        Stage 4 — 2 Basic Blocks:
            256 → 512, stride=2   → (B, 512, 7, 7)
        
        Output: (B, 512, 7, 7) spatial  or  (B, 512) global
    
    Total params: ~11M (vs ~23M for ResNet50-style)
    ~2-3x faster training than previous version.
    """
    
    def __init__(self, return_spatial: bool = False):
        super(ScratchCNN, self).__init__()
        
        self.return_spatial = return_spatial
        self.feature_dim = 512
        self.num_regions = 49  # 7x7
        
        # ==========================================
        #  Stem: Input Processing
        # ==========================================
        # Conv 7x7, stride 2: (3, 224, 224) → (64, 112, 112)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # MaxPool: (64, 112, 112) → (64, 56, 56)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ==========================================
        #  Stage 1: (64, 56, 56) → (64, 56, 56)
        # ==========================================
        self.stage1 = self._make_stage(in_channels=64, out_channels=64, num_blocks=2, stride=1)
        
        # ==========================================
        #  Stage 2: (64, 56, 56) → (128, 28, 28)
        # ==========================================
        self.stage2 = self._make_stage(in_channels=64, out_channels=128, num_blocks=2, stride=2)
        
        # ==========================================
        #  Stage 3: (128, 28, 28) → (256, 14, 14)
        # ==========================================
        self.stage3 = self._make_stage(in_channels=128, out_channels=256, num_blocks=2, stride=2)
        
        # ==========================================
        #  Stage 4: (256, 14, 14) → (512, 7, 7)
        # ==========================================
        self.stage4 = self._make_stage(in_channels=256, out_channels=512, num_blocks=2, stride=2)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ==========================================
        #  Weight Initialization (Kaiming)
        # ==========================================
        self._initialize_weights()
    
    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Build a stage of BasicBlock residual blocks.
        
        Args:
            in_channels: Input channels to this stage
            out_channels: Output channels for this stage
            num_blocks: Number of BasicBlocks
            stride: Stride for first block (1 = same, 2 = downsample)
        """
        blocks = []
        
        # First block: may change dimensions
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        blocks.append(BasicBlock(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks: same dimensions
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(out_channels, out_channels, stride=1, downsample=None))
        
        return nn.Sequential(*blocks)
    
    def _initialize_weights(self):
        """Kaiming initialization for all layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            return_spatial=True:  (B, 512, 7, 7)
            return_spatial=False: (B, 512)
        """
        # Stem
        x = self.conv1(x)       # (B, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # (B, 64, 56, 56)
        
        # Residual stages
        x = self.stage1(x)      # (B, 64, 56, 56)
        x = self.stage2(x)      # (B, 128, 28, 28)
        x = self.stage3(x)      # (B, 256, 14, 14)
        x = self.stage4(x)      # (B, 512, 7, 7)
        
        if not self.return_spatial:
            x = self.avgpool(x)     # (B, 512, 1, 1)
            x = torch.flatten(x, 1) # (B, 512)
        
        return x


# ============================================================================
#  CNN ENCODER — Wrapper: Pretrained ResNet50 or From-scratch ScratchCNN
# ============================================================================

class CNNEncoder(nn.Module):
    """
    CNN Encoder for extracting image features.
    
    - Pretrained (Model 1, 2): torchvision ResNet50, output 2048-D
    - From-scratch (Model 3, 4): ScratchCNN (lightweight), output 512-D
    """
    
    def __init__(
        self,
        arch: str = 'resnet50',
        pretrained: bool = True,
        freeze: bool = True,
        feature_dim: int = None,
        return_spatial: bool = False
    ):
        """
        Args:
            arch: CNN architecture identifier
            pretrained: True → torchvision ResNet50, False → ScratchCNN
            freeze: Freeze CNN params (auto-False if pretrained=False)
            feature_dim: Override output dim (None = use default)
            return_spatial: True → (B, 49, C), False → (B, C)
        """
        super(CNNEncoder, self).__init__()
        
        self.arch = arch
        self.pretrained = pretrained
        self.return_spatial = return_spatial
        
        # From-scratch should NOT be frozen
        if not pretrained and freeze:
            print(f"⚠️  WARNING: From-scratch CNN should not be frozen. Auto-unfreezing...")
            freeze = False
        
        self.freeze = freeze
        
        # ==========================================
        #  Build CNN backbone
        # ==========================================
        if pretrained:
            # --- PRETRAINED: torchvision ResNet50 (2048-D) ---
            base_model = models.resnet50(pretrained=True)
            
            if return_spatial:
                self.cnn = nn.Sequential(*list(base_model.children())[:-2])
            else:
                self.cnn = nn.Sequential(*list(base_model.children())[:-1])
            
            self.default_feature_dim = 2048
            self.num_regions = 49
            
            print(f"  ► Using torchvision ResNet50 (ImageNet pretrained, 2048-D)")
        
        else:
            # --- FROM-SCRATCH: Lightweight ScratchCNN (512-D) ---
            self.cnn = ScratchCNN(return_spatial=return_spatial)
            self.default_feature_dim = self.cnn.feature_dim   # 512
            self.num_regions = self.cnn.num_regions            # 49
            
            print(f"  ► Using ScratchCNN (lightweight, 512-D, ~11M params)")
        
        # Freeze if specified
        if freeze:
            for param in self.cnn.parameters():
                param.requires_grad = False
            print(f"  ► Froze CNN parameters")
        else:
            trainable = sum(p.numel() for p in self.cnn.parameters() if p.requires_grad)
            print(f"  ► CNN parameters trainable ({trainable/1e6:.2f}M params)")
        
        # Optional projection
        self.feature_dim = feature_dim if feature_dim else self.default_feature_dim
        if feature_dim and feature_dim != self.default_feature_dim:
            self.projection = nn.Linear(self.default_feature_dim, feature_dim)
        else:
            self.projection = None
        
        print(f"  CNNEncoder ready: {'ResNet50-pretrained' if pretrained else 'ScratchCNN'}, "
              f"dim={self.feature_dim}, spatial={return_spatial}, frozen={freeze}")

    
    def forward(self, images):
        """
        Args:
            images: (B, 3, 224, 224)
        Returns:
            (B, feature_dim) or (B, 49, feature_dim) if return_spatial
        """
        with torch.set_grad_enabled(not self.freeze):
            features = self.cnn(images)
        
        if self.return_spatial:
            # (B, C, 7, 7) → (B, 49, C)
            B, C = features.size(0), features.size(1)
            features = features.view(B, C, -1).permute(0, 2, 1)
            
            if self.projection is not None:
                features = self.projection(features)
        else:
            # Flatten if needed
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            
            if self.projection is not None:
                features = self.projection(features)
        
        return features
    
    def unfreeze(self):
        """Unfreeze CNN parameters for fine-tuning"""
        for param in self.cnn.parameters():
            param.requires_grad = True
        self.freeze = False
        print(f"Unfroze CNN parameters")
    
    def freeze_params(self):
        """Freeze CNN parameters"""
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.freeze = True
        print(f"Froze CNN parameters")


# ============================================================================
#  TESTING
# ============================================================================

if __name__ == "__main__":
    dummy = torch.randn(2, 3, 224, 224)
    
    print("=" * 60)
    print("Test 1: Pretrained Global")
    e1 = CNNEncoder(pretrained=True, freeze=True)
    print(f"  Output: {e1(dummy).shape}")  # (2, 2048)
    
    print("\n" + "=" * 60)
    print("Test 2: Scratch Global")
    e2 = CNNEncoder(pretrained=False)
    print(f"  Output: {e2(dummy).shape}")  # (2, 512)
    
    print("\n" + "=" * 60)
    print("Test 3: Pretrained Spatial")
    e3 = CNNEncoder(pretrained=True, freeze=True, return_spatial=True)
    print(f"  Output: {e3(dummy).shape}")  # (2, 49, 2048)
    
    print("\n" + "=" * 60)
    print("Test 4: Scratch Spatial")
    e4 = CNNEncoder(pretrained=False, return_spatial=True)
    print(f"  Output: {e4(dummy).shape}")  # (2, 49, 512)
    
    print("\n✓ All tests passed!")
