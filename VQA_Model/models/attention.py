"""
Attention Mechanism for VQA
Computes attention weights between image and question features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(nn.Module):
    """Additive (Bahdanau-style) Attention"""
    
    def __init__(
        self,
        img_dim: int,
        q_dim: int,
        attn_dim: int = 512
    ):
        """
        Args:
            img_dim: Image feature dimension
            q_dim: Question feature dimension
            attn_dim: Attention hidden dimension
        """
        super(AttentionModule, self).__init__()
        
        self.img_dim = img_dim
        self.q_dim = q_dim
        self.attn_dim = attn_dim
        
        # Linear layers for attention
        self.img_proj = nn.Linear(img_dim, attn_dim)
        self.q_proj = nn.Linear(q_dim, attn_dim)
        self.attn_linear = nn.Linear(attn_dim, 1)
        
        print(f"Created AttentionModule:")
        print(f"  Image dim: {img_dim}")
        print(f"  Question dim: {q_dim}")
        print(f"  Attention dim: {attn_dim}")
    
    def forward(self, img_features, q_features):
        """
        Compute attention-weighted image features
        
        Args:
            img_features: Tensor (B, img_dim)
            q_features: Tensor (B, q_dim)
        
        Returns:
            attended_features: Tensor (B, img_dim)
            attention_weights: Tensor (B, 1)
        """
        batch_size = img_features.size(0)
        
        # Project image and question features
        img_proj = self.img_proj(img_features)  # (B, attn_dim)
        q_proj = self.q_proj(q_features)  # (B, attn_dim)
        
        # Combine features
        combined = torch.tanh(img_proj + q_proj)  # (B, attn_dim)
        
        # Compute attention scores
        attn_scores = self.attn_linear(combined)  # (B, 1)
        
        # Attention weights (softmax over spatial dimension)
        # For single feature vector, weights are just sigmoid
        attention_weights = torch.sigmoid(attn_scores)  # (B, 1)
        
        # Weighted image features
        attended_features = attention_weights * img_features  # (B, img_dim)
        
        return attended_features, attention_weights


class SpatialAttention(nn.Module):
    """Spatial Attention for feature maps (if using spatial features)"""
    
    def __init__(
        self,
        img_dim: int,
        q_dim: int,
        attn_dim: int = 512,
        num_regions: int = 49  # 7x7 for VGG16
    ):
        """
        Args:
            img_dim: Image feature dimension per region
            q_dim: Question feature dimension
            attn_dim: Attention hidden dimension
            num_regions: Number of spatial regions
        """
        super(SpatialAttention, self).__init__()
        
        self.img_dim = img_dim
        self.q_dim = q_dim
        self.attn_dim = attn_dim
        self.num_regions = num_regions
        
        # Linear layers
        self.img_proj = nn.Linear(img_dim, attn_dim)
        self.q_proj = nn.Linear(q_dim, attn_dim)
        self.attn_linear = nn.Linear(attn_dim, 1)
        
        print(f"Created SpatialAttention:")
        print(f"  Image dim: {img_dim}")
        print(f"  Question dim: {q_dim}")
        print(f"  Attention dim: {attn_dim}")
        print(f"  Num regions: {num_regions}")
    
    def forward(self, img_features, q_features):
        """
        Compute spatial attention
        
        Args:
            img_features: Tensor (B, num_regions, img_dim)
            q_features: Tensor (B, q_dim)
        
        Returns:
            attended_features: Tensor (B, img_dim)
            attention_weights: Tensor (B, num_regions)
        """
        batch_size = img_features.size(0)
        num_regions = img_features.size(1)
        
        # Project image features
        img_proj = self.img_proj(img_features)  # (B, num_regions, attn_dim)
        
        # Project question features and expand
        q_proj = self.q_proj(q_features).unsqueeze(1)  # (B, 1, attn_dim)
        q_proj = q_proj.expand(-1, num_regions, -1)  # (B, num_regions, attn_dim)
        
        # Combine features
        combined = torch.tanh(img_proj + q_proj)  # (B, num_regions, attn_dim)
        
        # Compute attention scores
        attn_scores = self.attn_linear(combined).squeeze(-1)  # (B, num_regions)
        
        # Attention weights
        attention_weights = F.softmax(attn_scores, dim=1)  # (B, num_regions)
        
        # Weighted sum of image features
        attended_features = torch.bmm(
            attention_weights.unsqueeze(1),  # (B, 1, num_regions)
            img_features  # (B, num_regions, img_dim)
        ).squeeze(1)  # (B, img_dim)
        
        return attended_features, attention_weights


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing Attention Module")
    print("="*60)
    
    # Create attention module
    attn = AttentionModule(img_dim=2048, q_dim=512, attn_dim=512)
    
    # Dummy input
    batch_size = 4
    img_features = torch.randn(batch_size, 2048)
    q_features = torch.randn(batch_size, 512)
    
    print(f"\nImage features shape: {img_features.shape}")
    print(f"Question features shape: {q_features.shape}")
    
    # Forward pass
    attended, weights = attn(img_features, q_features)
    
    print(f"\nAttended features shape: {attended.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention weights: {weights.squeeze().tolist()}")
    
    # Test Spatial Attention
    print("\n" + "="*60)
    print("Testing Spatial Attention")
    print("="*60)
    
    spatial_attn = SpatialAttention(
        img_dim=512,
        q_dim=512,
        attn_dim=512,
        num_regions=49
    )
    
    # Dummy spatial features
    img_spatial = torch.randn(batch_size, 49, 512)
    
    print(f"\nSpatial image features shape: {img_spatial.shape}")
    print(f"Question features shape: {q_features.shape}")
    
    attended_spatial, weights_spatial = spatial_attn(img_spatial, q_features)
    
    print(f"\nAttended features shape: {attended_spatial.shape}")
    print(f"Attention weights shape: {weights_spatial.shape}")
    print(f"Attention weights sum: {weights_spatial.sum(dim=1).tolist()}")
    
    print("\n✓ Attention modules working correctly!")
