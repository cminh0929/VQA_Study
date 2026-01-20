"""
Attention Mechanisms cho VQA
Bao gồm: Spatial Attention, Multi-Head Attention, Stacked Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpatialAttention(nn.Module):
    """
    Spatial Attention Mechanism cho VQA
    
    Tính attention weights cho từng vùng trong spatial feature map (7x7)
    dựa trên câu hỏi, sau đó tạo weighted sum của visual features.
    """
    
    def __init__(self,
                 visual_dim: int = 1024,
                 question_dim: int = 512,
                 hidden_dim: int = 512,
                 dropout: float = 0.2):
        """
        Args:
            visual_dim: Kích thước visual features (mỗi spatial location)
            question_dim: Kích thước question features
            hidden_dim: Kích thước hidden layer
            dropout: Dropout rate
        """
        super(SpatialAttention, self).__init__()
        
        self.visual_dim = visual_dim
        self.question_dim = question_dim
        self.hidden_dim = hidden_dim
        
        # Project visual features
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        
        # Project question features
        self.question_proj = nn.Linear(question_dim, hidden_dim)
        
        # Attention scoring
        self.attention_proj = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        
    def forward(self, 
                visual_features: torch.Tensor,
                question_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            visual_features: (batch, visual_dim, 7, 7) - spatial feature map
            question_features: (batch, question_dim) - question vector
            
        Returns:
            attended_features: (batch, visual_dim) - weighted visual features
            attention_weights: (batch, 49) - attention weights cho visualization
        """
        batch_size = visual_features.size(0)
        
        # Reshape visual features: (batch, visual_dim, 7, 7) -> (batch, 49, visual_dim)
        visual_features = visual_features.view(batch_size, self.visual_dim, -1)  # (batch, visual_dim, 49)
        visual_features = visual_features.permute(0, 2, 1)  # (batch, 49, visual_dim)
        
        num_regions = visual_features.size(1)  # 49
        
        # Project visual features
        v_proj = self.visual_proj(visual_features)  # (batch, 49, hidden_dim)
        
        # Project question features
        q_proj = self.question_proj(question_features)  # (batch, hidden_dim)
        q_proj = q_proj.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Combine visual and question
        # Broadcasting: (batch, 49, hidden_dim) + (batch, 1, hidden_dim) = (batch, 49, hidden_dim)
        combined = self.tanh(v_proj + q_proj)  # (batch, 49, hidden_dim)
        combined = self.dropout(combined)
        
        # Compute attention scores
        attention_scores = self.attention_proj(combined)  # (batch, 49, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch, 49)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, 49)
        
        # Weighted sum of visual features
        # (batch, 1, 49) @ (batch, 49, visual_dim) = (batch, 1, visual_dim)
        attended_features = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, 49)
            visual_features  # (batch, 49, visual_dim)
        )
        attended_features = attended_features.squeeze(1)  # (batch, visual_dim)
        
        return attended_features, attention_weights


class MultiHeadSpatialAttention(nn.Module):
    """
    Multi-Head Spatial Attention
    Nhiều attention heads để học các khía cạnh khác nhau
    """
    
    def __init__(self,
                 visual_dim: int = 1024,
                 question_dim: int = 512,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0.2):
        """
        Args:
            visual_dim: Kích thước visual features
            question_dim: Kích thước question features
            hidden_dim: Kích thước hidden (phải chia hết cho num_heads)
            num_heads: Số attention heads
            dropout: Dropout rate
        """
        super(MultiHeadSpatialAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.visual_dim = visual_dim
        self.question_dim = question_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Projections
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.question_proj = nn.Linear(question_dim, hidden_dim)
        
        # Multi-head attention
        self.attention_proj = nn.Linear(self.head_dim, 1)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, visual_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        
    def forward(self, visual_features, question_features):
        """
        Forward pass
        
        Args:
            visual_features: (batch, visual_dim, 7, 7)
            question_features: (batch, question_dim)
            
        Returns:
            attended_features: (batch, visual_dim)
            attention_weights: (batch, num_heads, 49)
        """
        batch_size = visual_features.size(0)
        
        # Reshape visual: (batch, visual_dim, 7, 7) -> (batch, 49, visual_dim)
        visual_features = visual_features.view(batch_size, self.visual_dim, -1).permute(0, 2, 1)
        num_regions = visual_features.size(1)
        
        # Project
        v_proj = self.visual_proj(visual_features)  # (batch, 49, hidden_dim)
        q_proj = self.question_proj(question_features)  # (batch, hidden_dim)
        
        # Reshape for multi-head: (batch, 49, num_heads, head_dim)
        v_proj = v_proj.view(batch_size, num_regions, self.num_heads, self.head_dim)
        v_proj = v_proj.permute(0, 2, 1, 3)  # (batch, num_heads, 49, head_dim)
        
        q_proj = q_proj.view(batch_size, self.num_heads, self.head_dim)
        q_proj = q_proj.unsqueeze(2)  # (batch, num_heads, 1, head_dim)
        
        # Combine
        combined = self.tanh(v_proj + q_proj)  # (batch, num_heads, 49, head_dim)
        
        # Attention scores
        attention_scores = self.attention_proj(combined).squeeze(-1)  # (batch, num_heads, 49)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, num_heads, 49)
        
        # Weighted sum per head
        # (batch, num_heads, 1, 49) @ (batch, num_heads, 49, head_dim) = (batch, num_heads, 1, head_dim)
        attended = torch.matmul(
            attention_weights.unsqueeze(2),
            v_proj
        ).squeeze(2)  # (batch, num_heads, head_dim)
        
        # Concatenate heads
        attended = attended.permute(0, 1, 2).contiguous()  # (batch, num_heads, head_dim)
        attended = attended.view(batch_size, self.hidden_dim)  # (batch, hidden_dim)
        
        # Output projection
        attended_features = self.output_proj(attended)  # (batch, visual_dim)
        attended_features = self.dropout(attended_features)
        
        return attended_features, attention_weights


class StackedAttention(nn.Module):
    """
    Stacked Attention Networks (SAN)
    Nhiều lớp attention chồng lên nhau để tinh chỉnh dần
    """
    
    def __init__(self,
                 visual_dim: int = 1024,
                 question_dim: int = 512,
                 hidden_dim: int = 512,
                 num_stacks: int = 2,
                 dropout: float = 0.2):
        """
        Args:
            visual_dim: Kích thước visual features
            question_dim: Kích thước question features
            hidden_dim: Kích thước hidden
            num_stacks: Số lớp attention
            dropout: Dropout rate
        """
        super(StackedAttention, self).__init__()
        
        self.num_stacks = num_stacks
        
        # Tạo nhiều attention layers
        self.attention_layers = nn.ModuleList([
            SpatialAttention(visual_dim, question_dim, hidden_dim, dropout)
            for _ in range(num_stacks)
        ])
        
    def forward(self, visual_features, question_features):
        """
        Forward pass
        
        Args:
            visual_features: (batch, visual_dim, 7, 7)
            question_features: (batch, question_dim)
            
        Returns:
            attended_features: (batch, visual_dim)
            all_attention_weights: List of (batch, 49) - weights từ mỗi stack
        """
        batch_size = visual_features.size(0)
        all_attention_weights = []
        
        # Reshape visual features
        visual_flat = visual_features.view(batch_size, visual_features.size(1), -1)
        visual_flat = visual_flat.permute(0, 2, 1)  # (batch, 49, visual_dim)
        
        # Apply stacked attention
        current_features = visual_features
        
        for i, attention_layer in enumerate(self.attention_layers):
            attended, weights = attention_layer(current_features, question_features)
            all_attention_weights.append(weights)
            
            # Reshape attended features back to spatial
            # Để làm input cho layer tiếp theo
            current_features = attended.unsqueeze(-1).unsqueeze(-1)  # (batch, visual_dim, 1, 1)
            current_features = current_features.expand(-1, -1, 7, 7)  # (batch, visual_dim, 7, 7)
        
        # Final attended features
        attended_features = attended
        
        return attended_features, all_attention_weights


if __name__ == "__main__":
    print("Testing Attention Mechanisms...")
    
    batch_size = 4
    visual_dim = 1024
    question_dim = 512
    
    # Dummy inputs
    visual_features = torch.randn(batch_size, visual_dim, 7, 7)
    question_features = torch.randn(batch_size, question_dim)
    
    # Test 1: Spatial Attention
    print("\n1. Spatial Attention:")
    spatial_attn = SpatialAttention(visual_dim, question_dim, hidden_dim=512)
    attended, weights = spatial_attn(visual_features, question_features)
    print(f"   Visual input: {visual_features.shape}")
    print(f"   Question input: {question_features.shape}")
    print(f"   Attended output: {attended.shape}")
    print(f"   Attention weights: {weights.shape}")
    print(f"   Weights sum: {weights.sum(dim=1)}")  # Should be ~1.0
    
    # Test 2: Multi-Head Attention
    print("\n2. Multi-Head Spatial Attention:")
    multihead_attn = MultiHeadSpatialAttention(visual_dim, question_dim, hidden_dim=512, num_heads=8)
    attended_mh, weights_mh = multihead_attn(visual_features, question_features)
    print(f"   Attended output: {attended_mh.shape}")
    print(f"   Attention weights: {weights_mh.shape}")
    
    # Test 3: Stacked Attention
    print("\n3. Stacked Attention:")
    stacked_attn = StackedAttention(visual_dim, question_dim, hidden_dim=512, num_stacks=2)
    attended_st, weights_st = stacked_attn(visual_features, question_features)
    print(f"   Attended output: {attended_st.shape}")
    print(f"   Number of attention layers: {len(weights_st)}")
    print(f"   Each layer weights shape: {weights_st[0].shape}")
    
    print("\n✅ All tests passed!")
