"""
VQA Model - Complete Visual Question Answering Model
Kết hợp Image Encoder, Question Encoder, Attention, và Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from models.image_encoder import ImageEncoder
from models.question_encoder import QuestionEncoder, QuestionEncoderWithAttention
from models.attention import SpatialAttention, MultiHeadSpatialAttention, StackedAttention


class VQAModel(nn.Module):
    """
    Complete VQA Model với Spatial Attention
    
    Architecture:
        Image → CNN → Spatial Features (7x7x2048)
                                ↓
        Question → LSTM → Question Vector (512-d)
                                ↓
                    Spatial Attention
                                ↓
                    Attended Visual Features
                                ↓
                Fusion (Concat + FC layers)
                                ↓
                    Softmax → Answer
    """
    
    def __init__(self,
                 # Vocabulary
                 question_vocab_size: int,
                 answer_vocab_size: int,
                 # Image Encoder
                 cnn_backbone: str = "resnet50",
                 cnn_pretrained: bool = True,
                 freeze_cnn: bool = True,
                 cnn_feature_dim: int = 2048,
                 image_feature_dim: int = 1024,
                 # Question Encoder
                 embedding_dim: int = 300,
                 lstm_hidden_size: int = 512,
                 lstm_num_layers: int = 2,
                 lstm_dropout: float = 0.3,
                 bidirectional: bool = False,
                 # Attention
                 use_attention: bool = True,
                 attention_type: str = "spatial",
                 attention_hidden_dim: int = 512,
                 num_attention_heads: int = 8,
                 num_attention_stacks: int = 2,
                 attention_dropout: float = 0.2,
                 # Classifier
                 fusion_dim: int = 1024,
                 hidden_dims: list = [1024, 512],
                 classifier_dropout: float = 0.5):
        """
        Args:
            question_vocab_size: Kích thước question vocabulary
            answer_vocab_size: Số lượng câu trả lời (classes)
            ... (xem docstring của từng component)
        """
        super(VQAModel, self).__init__()
        
        self.use_attention = use_attention
        self.attention_type = attention_type
        
        # ==================== Image Encoder ====================
        self.image_encoder = ImageEncoder(
            backbone=cnn_backbone,
            pretrained=cnn_pretrained,
            freeze=freeze_cnn,
            feature_dim=cnn_feature_dim,
            output_dim=image_feature_dim,
            use_spatial=use_attention  # Spatial nếu dùng attention
        )
        
        # ==================== Question Encoder ====================
        self.question_encoder = QuestionEncoder(
            vocab_size=question_vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )
        
        question_feature_dim = self.question_encoder.get_output_dim()
        
        # ==================== Attention ====================
        if use_attention:
            if attention_type == "spatial":
                self.attention = SpatialAttention(
                    visual_dim=image_feature_dim,
                    question_dim=question_feature_dim,
                    hidden_dim=attention_hidden_dim,
                    dropout=attention_dropout
                )
            elif attention_type == "multi_head":
                self.attention = MultiHeadSpatialAttention(
                    visual_dim=image_feature_dim,
                    question_dim=question_feature_dim,
                    hidden_dim=attention_hidden_dim,
                    num_heads=num_attention_heads,
                    dropout=attention_dropout
                )
            elif attention_type == "stacked":
                self.attention = StackedAttention(
                    visual_dim=image_feature_dim,
                    question_dim=question_feature_dim,
                    hidden_dim=attention_hidden_dim,
                    num_stacks=num_attention_stacks,
                    dropout=attention_dropout
                )
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")
        
        # ==================== Fusion & Classifier ====================
        # Fusion dimension
        fusion_input_dim = image_feature_dim + question_feature_dim
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout)
        )
        
        # Classifier layers
        classifier_layers = []
        prev_dim = fusion_dim
        
        for hidden_dim in hidden_dims:
            classifier_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(classifier_dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        classifier_layers.append(nn.Linear(prev_dim, answer_vocab_size))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
    def forward(self, 
                images: torch.Tensor,
                questions: torch.Tensor,
                question_lengths: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: (batch, 3, 224, 224)
            questions: (batch, max_length)
            question_lengths: (batch,) - optional
            
        Returns:
            Dict containing:
                - logits: (batch, answer_vocab_size)
                - attention_weights: (batch, 49) if use_attention else None
        """
        # Extract image features
        image_features = self.image_encoder(images)
        # (batch, image_feature_dim, 7, 7) nếu use_attention
        # (batch, image_feature_dim) nếu không
        
        # Extract question features
        question_features = self.question_encoder(questions, question_lengths)
        # (batch, question_feature_dim)
        
        # Apply attention
        attention_weights = None
        if self.use_attention:
            attended_features, attention_weights = self.attention(
                image_features, question_features
            )
            # attended_features: (batch, image_feature_dim)
            # attention_weights: (batch, 49) hoặc (batch, num_heads, 49)
            visual_features = attended_features
        else:
            visual_features = image_features
        
        # Fusion
        fused_features = torch.cat([visual_features, question_features], dim=1)
        # (batch, image_feature_dim + question_feature_dim)
        
        fused_features = self.fusion(fused_features)
        # (batch, fusion_dim)
        
        # Classification
        logits = self.classifier(fused_features)
        # (batch, answer_vocab_size)
        
        return {
            'logits': logits,
            'attention_weights': attention_weights
        }
    
    def predict(self, images: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        """
        Predict answers
        
        Args:
            images: (batch, 3, 224, 224)
            questions: (batch, max_length)
            
        Returns:
            predicted_answers: (batch,) - answer indices
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, questions)
            logits = outputs['logits']
            predicted = torch.argmax(logits, dim=1)
        return predicted


if __name__ == "__main__":
    print("Testing VQAModel...")
    
    # Parameters
    question_vocab_size = 1000
    answer_vocab_size = 500
    batch_size = 4
    max_length = 20
    
    # Test 1: VQA với Spatial Attention
    print("\n1. VQA Model with Spatial Attention:")
    model = VQAModel(
        question_vocab_size=question_vocab_size,
        answer_vocab_size=answer_vocab_size,
        cnn_backbone="resnet50",
        cnn_pretrained=False,  # False để test nhanh
        freeze_cnn=True,
        use_attention=True,
        attention_type="spatial"
    )
    
    # Dummy inputs
    images = torch.randn(batch_size, 3, 224, 224)
    questions = torch.randint(0, question_vocab_size, (batch_size, max_length))
    question_lengths = torch.tensor([15, 20, 10, 18])
    
    # Forward pass
    outputs = model(images, questions, question_lengths)
    logits = outputs['logits']
    attention_weights = outputs['attention_weights']
    
    print(f"   Images shape: {images.shape}")
    print(f"   Questions shape: {questions.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Attention weights shape: {attention_weights.shape if attention_weights is not None else None}")
    
    # Test prediction
    predicted = model.predict(images, questions)
    print(f"   Predicted answers: {predicted.shape}")
    
    # Test 2: VQA với Multi-Head Attention
    print("\n2. VQA Model with Multi-Head Attention:")
    model_mh = VQAModel(
        question_vocab_size=question_vocab_size,
        answer_vocab_size=answer_vocab_size,
        cnn_backbone="resnet50",
        cnn_pretrained=False,
        use_attention=True,
        attention_type="multi_head",
        num_attention_heads=8
    )
    
    outputs_mh = model_mh(images, questions)
    print(f"   Logits shape: {outputs_mh['logits'].shape}")
    print(f"   Attention weights shape: {outputs_mh['attention_weights'].shape}")
    
    # Test 3: VQA không có Attention (baseline)
    print("\n3. VQA Model without Attention (Baseline):")
    model_baseline = VQAModel(
        question_vocab_size=question_vocab_size,
        answer_vocab_size=answer_vocab_size,
        cnn_backbone="resnet50",
        cnn_pretrained=False,
        use_attention=False
    )
    
    outputs_baseline = model_baseline(images, questions)
    print(f"   Logits shape: {outputs_baseline['logits'].shape}")
    print(f"   Attention weights: {outputs_baseline['attention_weights']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n4. Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\n✅ All tests passed!")
