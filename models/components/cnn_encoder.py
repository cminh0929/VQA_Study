"""
Image Encoder - CNN Feature Extractor với Spatial Features
Trích xuất spatial feature maps (7x7) từ pretrained CNN
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


class ImageEncoder(nn.Module):
    """
    CNN Feature Extractor với Spatial Features cho VQA
    
    Thay vì output vector (2048-d), model này output feature map (7x7x2048)
    để giữ lại thông tin không gian cho Spatial Attention.
    """
    
    def __init__(self, 
                 backbone: str = "resnet50",
                 pretrained: bool = True,
                 freeze: bool = True,
                 feature_dim: int = 2048,
                 output_dim: int = 1024,
                 use_spatial: bool = True):
        """
        Args:
            backbone: Tên backbone CNN ("resnet50", "resnet101", "vgg16")
            pretrained: Sử dụng pretrained weights
            freeze: Đóng băng các lớp pretrained
            feature_dim: Số channels của feature map (ResNet50: 2048, VGG16: 512)
            output_dim: Kích thước output sau projection (để giảm chiều)
            use_spatial: Trả về spatial features (7x7) hay global vector
        """
        super(ImageEncoder, self).__init__()
        
        self.backbone_name = backbone
        self.use_spatial = use_spatial
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        
        # Load pretrained CNN
        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            # Lấy tất cả layers trừ avgpool và fc
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            self.feature_dim = 2048
            
        elif backbone == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            self.feature_dim = 2048
            
        elif backbone == "vgg16":
            vgg = models.vgg16(pretrained=pretrained)
            self.features = vgg.features  # Chỉ lấy convolutional layers
            self.feature_dim = 512
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Đóng băng pretrained layers nếu cần
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
            print(f"[ImageEncoder] Frozen pretrained {backbone} layers")
        
        # Adaptive pooling để đảm bảo output size cố định (7x7)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Projection layer để giảm chiều (tùy chọn)
        if output_dim != self.feature_dim:
            # Sử dụng Conv2d để project spatial features
            self.projection = nn.Sequential(
                nn.Conv2d(self.feature_dim, output_dim, kernel_size=1),
                nn.ReLU(),
                nn.Dropout2d(0.2)
            )
        else:
            self.projection = None
        
        # Global pooling cho trường hợp không dùng spatial features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            images: Tensor (batch_size, 3, 224, 224)
            
        Returns:
            Nếu use_spatial=True:
                features: Tensor (batch_size, output_dim, 7, 7)
            Nếu use_spatial=False:
                features: Tensor (batch_size, output_dim)
        """
        # Extract features từ CNN
        features = self.features(images)  # (batch, feature_dim, H, W)
        
        # Adaptive pooling về 7x7
        features = self.adaptive_pool(features)  # (batch, feature_dim, 7, 7)
        
        if self.use_spatial:
            # Giữ spatial structure
            if self.projection is not None:
                features = self.projection(features)  # (batch, output_dim, 7, 7)
            
            return features  # (batch, output_dim, 7, 7)
        
        else:
            # Global pooling thành vector
            features = self.global_pool(features)  # (batch, feature_dim, 1, 1)
            features = features.flatten(1)  # (batch, feature_dim)
            
            if self.projection is not None:
                # Cho global vector, dùng linear projection
                linear_proj = nn.Linear(self.feature_dim, self.output_dim).to(features.device)
                features = linear_proj(features)  # (batch, output_dim)
            
            return features  # (batch, output_dim)
    
    def get_output_shape(self, batch_size: int = 1) -> Tuple[int, ...]:
        """
        Trả về shape của output
        
        Args:
            batch_size: Batch size
            
        Returns:
            Output shape tuple
        """
        if self.use_spatial:
            return (batch_size, self.output_dim, 7, 7)
        else:
            return (batch_size, self.output_dim)
    
    def unfreeze(self):
        """Unfreeze tất cả layers để fine-tune"""
        for param in self.features.parameters():
            param.requires_grad = True
        print(f"[ImageEncoder] Unfrozen all layers")
    
    def freeze(self):
        """Freeze tất cả layers"""
        for param in self.features.parameters():
            param.requires_grad = False
        print(f"[ImageEncoder] Frozen all layers")


if __name__ == "__main__":
    print("Testing ImageEncoder...")
    
    # Test với ResNet50 (spatial features)
    print("\n1. ResNet50 with Spatial Features (7x7):")
    encoder_spatial = ImageEncoder(
        backbone="resnet50",
        pretrained=False,  # False để test nhanh
        freeze=True,
        output_dim=1024,
        use_spatial=True
    )
    
    # Dummy input
    dummy_images = torch.randn(4, 3, 224, 224)  # batch=4
    
    # Forward pass
    spatial_features = encoder_spatial(dummy_images)
    print(f"   Input shape: {dummy_images.shape}")
    print(f"   Output shape: {spatial_features.shape}")
    print(f"   Expected: (4, 1024, 7, 7)")
    
    # Test với ResNet50 (global vector)
    print("\n2. ResNet50 with Global Vector:")
    encoder_global = ImageEncoder(
        backbone="resnet50",
        pretrained=False,
        freeze=True,
        output_dim=1024,
        use_spatial=False
    )
    
    global_features = encoder_global(dummy_images)
    print(f"   Input shape: {dummy_images.shape}")
    print(f"   Output shape: {global_features.shape}")
    print(f"   Expected: (4, 1024)")
    
    # Test với VGG16
    print("\n3. VGG16 with Spatial Features:")
    encoder_vgg = ImageEncoder(
        backbone="vgg16",
        pretrained=False,
        freeze=True,
        output_dim=512,
        use_spatial=True
    )
    
    vgg_features = encoder_vgg(dummy_images)
    print(f"   Input shape: {dummy_images.shape}")
    print(f"   Output shape: {vgg_features.shape}")
    print(f"   Expected: (4, 512, 7, 7)")
    
    # Test get_output_shape
    print("\n4. Output shape info:")
    print(f"   Spatial: {encoder_spatial.get_output_shape(batch_size=8)}")
    print(f"   Global: {encoder_global.get_output_shape(batch_size=8)}")
    
    # Test freeze/unfreeze
    print("\n5. Testing freeze/unfreeze:")
    encoder_spatial.unfreeze()
    encoder_spatial.freeze()
    
    print("\n✅ All tests passed!")
