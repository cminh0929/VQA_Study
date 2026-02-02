"""
Complete VQA Model
Combines CNN Encoder, LSTM Encoder, Attention, and LSTM Decoder
Supports 8 model variants
"""

import torch
import torch.nn as nn

from .cnn_encoder import CNNEncoder
from .lstm_encoder import LSTMEncoder
from .attention import AttentionModule
from .lstm_decoder import LSTMDecoder


class VQAModel(nn.Module):
    """Complete VQA Model"""
    
    def __init__(
        self,
        # Vocabulary sizes
        question_vocab_size: int,
        answer_vocab_size: int,
        
        # CNN settings
        cnn_arch: str = 'resnet50',  # 'resnet50' or 'vgg16'
        cnn_pretrained: bool = True,
        cnn_freeze: bool = True,
        
        # Attention
        use_attention: bool = False,
        
        # Dimensions
        embed_dim: int = 300,
        lstm_hidden_dim: int = 512,
        attn_dim: int = 512,
        
        # LSTM settings
        num_lstm_layers: int = 2,
        dropout: float = 0.5
    ):
        """
        Args:
            question_vocab_size: Question vocabulary size
            answer_vocab_size: Answer vocabulary size
            cnn_arch: CNN architecture ('resnet50' or 'vgg16')
            cnn_pretrained: Use pretrained CNN
            cnn_freeze: Freeze CNN parameters
            use_attention: Use attention mechanism
            embed_dim: Word embedding dimension
            lstm_hidden_dim: LSTM hidden dimension
            attn_dim: Attention dimension
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(VQAModel, self).__init__()
        
        self.use_attention = use_attention
        self.cnn_arch = cnn_arch
        self.cnn_pretrained = cnn_pretrained
        
        # CNN Encoder
        self.cnn_encoder = CNNEncoder(
            arch=cnn_arch,
            pretrained=cnn_pretrained,
            freeze=cnn_freeze
        )
        img_dim = self.cnn_encoder.feature_dim
        
        # LSTM Question Encoder
        self.question_encoder = LSTMEncoder(
            vocab_size=question_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout,
            bidirectional=False
        )
        q_dim = self.question_encoder.output_dim
        
        # Attention (optional)
        if use_attention:
            self.attention = AttentionModule(
                img_dim=img_dim,
                q_dim=q_dim,
                attn_dim=attn_dim
            )
            # Fused dimension: attended_img + question
            fused_dim = img_dim + q_dim
        else:
            self.attention = None
            # Fused dimension: img + question
            fused_dim = img_dim + q_dim
        
        # LSTM Answer Decoder
        self.answer_decoder = LSTMDecoder(
            vocab_size=answer_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=lstm_hidden_dim,
            input_dim=fused_dim,
            num_layers=num_lstm_layers,
            dropout=dropout
        )
        
        print(f"\n{'='*60}")
        print(f"Created VQAModel:")
        print(f"  CNN: {cnn_arch} ({'pretrained' if cnn_pretrained else 'from-scratch'})")
        print(f"  Attention: {'Yes' if use_attention else 'No'}")
        print(f"  Image dim: {img_dim}")
        print(f"  Question dim: {q_dim}")
        print(f"  Fused dim: {fused_dim}")
        print(f"{'='*60}\n")
    
    def forward(self, images, questions, question_lengths=None, target_answers=None, teacher_forcing_ratio=0.5):
        """
        Forward pass
        
        Args:
            images: Tensor (B, 3, 224, 224)
            questions: Tensor (B, max_q_len)
            question_lengths: Tensor (B,) - optional
            target_answers: Tensor (B, max_a_len) - for training
            teacher_forcing_ratio: Teacher forcing probability
        
        Returns:
            outputs: Tensor (B, max_a_len, answer_vocab_size)
            attention_weights: Tensor (B, 1) or None
        """
        # Encode image
        img_features = self.cnn_encoder(images)  # (B, img_dim)
        
        # Encode question
        q_features = self.question_encoder(questions, question_lengths)  # (B, q_dim)
        
        # Attention (optional)
        attention_weights = None
        if self.use_attention:
            img_features_attended, attention_weights = self.attention(img_features, q_features)
            # Fuse attended image + question
            fused_features = torch.cat([img_features_attended, q_features], dim=1)
        else:
            # Fuse image + question directly
            fused_features = torch.cat([img_features, q_features], dim=1)
        
        # Decode answer
        outputs = self.answer_decoder(
            fused_features,
            target_answers=target_answers,
            max_len=10,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        return outputs, attention_weights
    
    def generate_answer(self, images, questions, question_lengths=None, max_len=10):
        """
        Generate answer (inference mode)
        
        Args:
            images: Tensor (B, 3, 224, 224)
            questions: Tensor (B, max_q_len)
            question_lengths: Tensor (B,)
            max_len: Maximum answer length
        
        Returns:
            generated: Tensor (B, max_len) - word indices
            attention_weights: Tensor (B, 1) or None
        """
        # Encode
        img_features = self.cnn_encoder(images)
        q_features = self.question_encoder(questions, question_lengths)
        
        # Attention
        attention_weights = None
        if self.use_attention:
            img_features_attended, attention_weights = self.attention(img_features, q_features)
            fused_features = torch.cat([img_features_attended, q_features], dim=1)
        else:
            fused_features = torch.cat([img_features, q_features], dim=1)
        
        # Generate
        generated = self.answer_decoder.generate(fused_features, max_len=max_len)
        
        return generated, attention_weights
    
    def get_model_name(self):
        """Get descriptive model name"""
        cnn_name = self.cnn_arch.upper()
        pretrained = "Pretrained" if self.cnn_pretrained else "FromScratch"
        attention = "WithAttn" if self.use_attention else "NoAttn"
        return f"{cnn_name}_{pretrained}_{attention}"


def create_model_variant(model_id: int, question_vocab_size: int, answer_vocab_size: int):
    """
    Create one of the 8 model variants
    
    Model variants:
        1: ResNet50 + Pretrained + No Attention
        2: ResNet50 + Pretrained + Attention
        3: ResNet50 + From-scratch + No Attention
        4: ResNet50 + From-scratch + Attention
        5: VGG16 + Pretrained + No Attention
        6: VGG16 + Pretrained + Attention
        7: VGG16 + From-scratch + No Attention
        8: VGG16 + From-scratch + Attention
    
    Args:
        model_id: Model ID (1-8)
        question_vocab_size: Question vocabulary size
        answer_vocab_size: Answer vocabulary size
    
    Returns:
        model: VQAModel instance
    """
    configs = {
        1: {'cnn_arch': 'resnet50', 'cnn_pretrained': True, 'use_attention': False},
        2: {'cnn_arch': 'resnet50', 'cnn_pretrained': True, 'use_attention': True},
        3: {'cnn_arch': 'resnet50', 'cnn_pretrained': False, 'use_attention': False},
        4: {'cnn_arch': 'resnet50', 'cnn_pretrained': False, 'use_attention': True},
        5: {'cnn_arch': 'vgg16', 'cnn_pretrained': True, 'use_attention': False},
        6: {'cnn_arch': 'vgg16', 'cnn_pretrained': True, 'use_attention': True},
        7: {'cnn_arch': 'vgg16', 'cnn_pretrained': False, 'use_attention': False},
        8: {'cnn_arch': 'vgg16', 'cnn_pretrained': False, 'use_attention': True},
    }
    
    if model_id not in configs:
        raise ValueError(f"Invalid model_id: {model_id}. Must be 1-8.")
    
    config = configs[model_id]
    
    model = VQAModel(
        question_vocab_size=question_vocab_size,
        answer_vocab_size=answer_vocab_size,
        **config
    )
    
    return model


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing VQA Model")
    print("="*60)
    
    # Create Model 1 (ResNet50 + Pretrained + No Attention)
    model = create_model_variant(
        model_id=1,
        question_vocab_size=47,
        answer_vocab_size=29
    )
    
    # Dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    questions = torch.randint(0, 47, (batch_size, 20))
    question_lengths = torch.tensor([15, 12])
    target_answers = torch.randint(0, 29, (batch_size, 10))
    
    print(f"\nInput shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Questions: {questions.shape}")
    print(f"  Target answers: {target_answers.shape}")
    
    # Forward pass (training)
    outputs, attn_weights = model(images, questions, question_lengths, target_answers, teacher_forcing_ratio=0.5)
    print(f"\nTraining output shape: {outputs.shape}")
    print(f"Attention weights: {attn_weights}")
    
    # Generate (inference)
    generated, attn_weights = model.generate_answer(images, questions, question_lengths, max_len=10)
    print(f"\nGenerated shape: {generated.shape}")
    print(f"Generated sample: {generated[0].tolist()}")
    
    # Test Model 2 (with attention)
    print("\n" + "="*60)
    print("Testing Model 2 (With Attention)")
    print("="*60)
    
    model2 = create_model_variant(2, 47, 29)
    outputs2, attn_weights2 = model2(images, questions, question_lengths, target_answers)
    print(f"\nOutput shape: {outputs2.shape}")
    print(f"Attention weights shape: {attn_weights2.shape if attn_weights2 is not None else None}")
    
    print("\n✓ VQA Model working correctly!")
    print(f"\nModel name: {model2.get_model_name()}")
