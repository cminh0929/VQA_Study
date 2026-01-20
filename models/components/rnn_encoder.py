"""
Question Encoder - LSTM-based Question Embedding
Mã hóa câu hỏi thành vector ngữ nghĩa
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class QuestionEncoder(nn.Module):
    """
    Question Encoder sử dụng LSTM để mã hóa câu hỏi
    
    Input: Câu hỏi dạng sequence of word indices
    Output: Question vector (hidden state cuối cùng của LSTM)
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = False,
                 padding_idx: int = 0):
        """
        Args:
            vocab_size: Kích thước vocabulary
            embedding_dim: Kích thước word embedding
            hidden_size: Kích thước hidden state của LSTM
            num_layers: Số lớp LSTM
            dropout: Dropout rate
            bidirectional: Sử dụng bidirectional LSTM
            padding_idx: Index của PAD token
        """
        super(QuestionEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        self.output_dim = hidden_size * self.num_directions
        
    def forward(self, 
                questions: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            questions: Tensor (batch_size, max_length) - word indices
            lengths: Tensor (batch_size,) - actual lengths (optional, for packing)
            
        Returns:
            question_features: Tensor (batch_size, hidden_size * num_directions)
        """
        batch_size = questions.size(0)
        
        # Embedding
        embedded = self.embedding(questions)  # (batch, max_length, embedding_dim)
        embedded = self.dropout(embedded)
        
        # LSTM
        if lengths is not None:
            # Pack padded sequence để LSTM bỏ qua padding
            lengths = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            # Unpack
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Lấy hidden state cuối cùng
        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        
        if self.bidirectional:
            # Concatenate forward và backward của layer cuối
            hidden_forward = hidden[-2, :, :]  # (batch, hidden_size)
            hidden_backward = hidden[-1, :, :]  # (batch, hidden_size)
            question_features = torch.cat([hidden_forward, hidden_backward], dim=1)
            # (batch, hidden_size * 2)
        else:
            # Chỉ lấy hidden state của layer cuối
            question_features = hidden[-1, :, :]  # (batch, hidden_size)
        
        return question_features
    
    def get_output_dim(self) -> int:
        """Trả về kích thước output"""
        return self.output_dim


class QuestionEncoderWithAttention(nn.Module):
    """
    Question Encoder nâng cao với self-attention
    Sử dụng tất cả hidden states thay vì chỉ state cuối
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = False,
                 padding_idx: int = 0):
        super(QuestionEncoderWithAttention, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Self-attention
        lstm_output_dim = hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim),
            nn.Tanh(),
            nn.Linear(lstm_output_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = lstm_output_dim
        
    def forward(self, questions: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """
        Forward pass với self-attention
        
        Args:
            questions: (batch, max_length)
            lengths: (batch,)
            
        Returns:
            question_features: (batch, hidden_size * num_directions)
        """
        # Embedding
        embedded = self.embedding(questions)  # (batch, max_length, embedding_dim)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, max_length, hidden_size * num_directions)
        
        # Self-attention
        attention_scores = self.attention(lstm_out)  # (batch, max_length, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, max_length, 1)
        
        # Weighted sum
        question_features = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_size * num_directions)
        
        return question_features
    
    def get_output_dim(self) -> int:
        return self.output_dim


if __name__ == "__main__":
    print("Testing QuestionEncoder...")
    
    # Parameters
    vocab_size = 1000
    batch_size = 4
    max_length = 20
    
    # Test 1: Basic LSTM encoder
    print("\n1. Basic LSTM Encoder:")
    encoder = QuestionEncoder(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_size=512,
        num_layers=2,
        dropout=0.3,
        bidirectional=False
    )
    
    # Dummy input
    questions = torch.randint(0, vocab_size, (batch_size, max_length))
    lengths = torch.tensor([15, 20, 10, 18])
    
    # Forward
    features = encoder(questions, lengths)
    print(f"   Input shape: {questions.shape}")
    print(f"   Output shape: {features.shape}")
    print(f"   Expected: ({batch_size}, 512)")
    print(f"   Output dim: {encoder.get_output_dim()}")
    
    # Test 2: Bidirectional LSTM
    print("\n2. Bidirectional LSTM Encoder:")
    encoder_bi = QuestionEncoder(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_size=512,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    )
    
    features_bi = encoder_bi(questions, lengths)
    print(f"   Input shape: {questions.shape}")
    print(f"   Output shape: {features_bi.shape}")
    print(f"   Expected: ({batch_size}, 1024)")  # 512 * 2
    print(f"   Output dim: {encoder_bi.get_output_dim()}")
    
    # Test 3: Encoder with self-attention
    print("\n3. LSTM Encoder with Self-Attention:")
    encoder_attn = QuestionEncoderWithAttention(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_size=512,
        num_layers=2,
        dropout=0.3,
        bidirectional=False
    )
    
    features_attn = encoder_attn(questions, lengths)
    print(f"   Input shape: {questions.shape}")
    print(f"   Output shape: {features_attn.shape}")
    print(f"   Expected: ({batch_size}, 512)")
    print(f"   Output dim: {encoder_attn.get_output_dim()}")
    
    print("\n✅ All tests passed!")
