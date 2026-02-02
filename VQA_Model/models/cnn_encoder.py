"""
CNN Encoder for VQA
Extracts visual features from images using ResNet50 or VGG16
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    """CNN Encoder for extracting image features"""
    
    def __init__(
        self,
        arch: str = 'resnet50',
        pretrained: bool = True,
        freeze: bool = True,
        feature_dim: int = None
    ):
        """
        Args:
            arch: CNN architecture ('resnet50' or 'vgg16')
            pretrained: Whether to use pretrained weights
            freeze: Whether to freeze CNN parameters
            feature_dim: Output feature dimension (None = use default)
        """
        super(CNNEncoder, self).__init__()
        
        self.arch = arch
        self.pretrained = pretrained
        self.freeze = freeze
        
        # Load pretrained model
        if arch == 'resnet50':
            self.cnn = models.resnet50(pretrained=pretrained)
            # Remove final FC layer
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
            self.default_feature_dim = 2048
        
        elif arch == 'vgg16':
            self.cnn = models.vgg16(pretrained=pretrained)
            # Remove classifier, keep features + avgpool
            self.cnn = nn.Sequential(
                self.cnn.features,
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                *list(self.cnn.classifier.children())[:-1]  # Remove last FC
            )
            self.default_feature_dim = 4096
        
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # Freeze CNN parameters if specified
        if freeze:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        # Optional projection layer
        self.feature_dim = feature_dim if feature_dim else self.default_feature_dim
        if feature_dim and feature_dim != self.default_feature_dim:
            self.projection = nn.Linear(self.default_feature_dim, feature_dim)
        else:
            self.projection = None
        
        print(f"Created CNNEncoder:")
        print(f"  Architecture: {arch}")
        print(f"  Pretrained: {pretrained}")
        print(f"  Frozen: {freeze}")
        print(f"  Output dim: {self.feature_dim}")
    
    def forward(self, images):
        """
        Extract features from images
        
        Args:
            images: Tensor (B, 3, 224, 224)
        
        Returns:
            features: Tensor (B, feature_dim)
        """
        # Extract features
        with torch.set_grad_enabled(not self.freeze):
            features = self.cnn(images)
        
        # Flatten if needed (ResNet50 outputs (B, 2048, 1, 1))
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        # Project if needed
        if self.projection is not None:
            features = self.projection(features)
        
        return features
    
    def unfreeze(self):
        """Unfreeze CNN parameters for fine-tuning"""
        for param in self.cnn.parameters():
            param.requires_grad = True
        self.freeze = False
        print(f"Unfroze {self.arch} parameters")
    
    def freeze_params(self):
        """Freeze CNN parameters"""
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.freeze = True
        print(f"Froze {self.arch} parameters")


# Example usage
if __name__ == "__main__":
    # Test ResNet50
    print("="*60)
    print("Testing ResNet50 Encoder")
    print("="*60)
    
    encoder_resnet = CNNEncoder(arch='resnet50', pretrained=True, freeze=True)
    
    # Dummy input
    dummy_images = torch.randn(4, 3, 224, 224)
    features = encoder_resnet(dummy_images)
    
    print(f"\nInput shape: {dummy_images.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output range: [{features.min():.3f}, {features.max():.3f}]")
    
    # Test VGG16
    print("\n" + "="*60)
    print("Testing VGG16 Encoder")
    print("="*60)
    
    encoder_vgg = CNNEncoder(arch='vgg16', pretrained=True, freeze=True)
    features_vgg = encoder_vgg(dummy_images)
    
    print(f"\nInput shape: {dummy_images.shape}")
    print(f"Output shape: {features_vgg.shape}")
    print(f"Output range: [{features_vgg.min():.3f}, {features_vgg.max():.3f}]")
    
    # Test projection
    print("\n" + "="*60)
    print("Testing with Projection")
    print("="*60)
    
    encoder_proj = CNNEncoder(arch='resnet50', pretrained=True, freeze=True, feature_dim=512)
    features_proj = encoder_proj(dummy_images)
    
    print(f"\nInput shape: {dummy_images.shape}")
    print(f"Output shape: {features_proj.shape}")
    
    print("\n✓ CNN Encoder working correctly!")
