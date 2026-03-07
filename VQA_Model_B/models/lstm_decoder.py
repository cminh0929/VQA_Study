"""
LSTM Decoder for VQA
Generates answer sequences word-by-word
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class LSTMDecoder(nn.Module):
    """LSTM Decoder for generating answers"""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 512,
        input_dim: int = 2560,  # 2048 (img) + 512 (q)
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        """
        Args:
            vocab_size: Answer vocabulary size
            embed_dim: Word embedding dimension
            hidden_dim: LSTM hidden dimension
            input_dim: Input feature dimension (fused features)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(LSTMDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Input projection (fused features → hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # CRITICAL FIX: Cell state projection for better memory initialization
        self.cell_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,  # word + context
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        print(f"Created LSTMDecoder:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Input dim: {input_dim}")
        print(f"  Num layers: {num_layers}")

    
    def forward(self, fused_features, target_answers=None, max_len=10, teacher_forcing_ratio=0.5):
        """
        Generate answer sequence
        
        Args:
            fused_features: Tensor (B, input_dim) - combined img+question features
            target_answers: Tensor (B, max_len) - ground truth (for training)
            max_len: Maximum answer length
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: Tensor (B, max_len, vocab_size) - word probabilities
        """
        batch_size = fused_features.size(0)
        
        # Project fused features to hidden and cell states
        context = self.input_proj(fused_features)  # (B, hidden_dim)
        
        # CRITICAL FIX: Initialize BOTH hidden and cell states from context
        hidden = context.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, B, hidden_dim)
        cell_init = self.cell_proj(fused_features)  # (B, hidden_dim)
        cell = cell_init.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, B, hidden_dim)
        
        # Start token (<SOS> = 2)
        input_word = torch.full((batch_size,), 2, dtype=torch.long, device=fused_features.device)
        
        outputs = []
        
        for t in range(max_len):
            # Embed current word
            embedded = self.embedding(input_word)  # (B, embed_dim)
            
            # Concatenate with context
            lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)  # (B, 1, embed_dim+hidden_dim)
            
            # LSTM step
            lstm_out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            lstm_out = lstm_out.squeeze(1)  # (B, hidden_dim)
            
            # Predict word
            output = self.fc_out(self.dropout(lstm_out))  # (B, vocab_size)
            outputs.append(output)
            
            # Teacher forcing
            if target_answers is not None and random.random() < teacher_forcing_ratio:
                input_word = target_answers[:, t]
            else:
                input_word = output.argmax(dim=1)
        
        outputs = torch.stack(outputs, dim=1)  # (B, max_len, vocab_size)
        return outputs
    
    def generate(self, fused_features, max_len=10, sos_idx=2, eos_idx=3):
        """
        Generate answer (inference mode)
        
        Args:
            fused_features: Tensor (B, input_dim)
            max_len: Maximum length
            sos_idx: Start-of-sequence index
            eos_idx: End-of-sequence index
        
        Returns:
            generated: Tensor (B, max_len) - generated word indices
        """
        batch_size = fused_features.size(0)
        device = fused_features.device
        
        # Project features to hidden and cell states
        context = self.input_proj(fused_features)
        
        # Initialize BOTH hidden and cell states
        hidden = context.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell_init = self.cell_proj(fused_features)
        cell = cell_init.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # Start token
        input_word = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
        
        generated = []
        
        for t in range(max_len):
            # Embed
            embedded = self.embedding(input_word)
            
            # LSTM input
            lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
            
            # LSTM step
            lstm_out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            lstm_out = lstm_out.squeeze(1)
            
            # Predict
            output = self.fc_out(lstm_out)
            predicted = output.argmax(dim=1)
            
            generated.append(predicted)
            input_word = predicted
        
        generated = torch.stack(generated, dim=1)  # (B, max_len)
        return generated


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing LSTM Decoder")
    print("="*60)
    
    # Create decoder
    decoder = LSTMDecoder(
        vocab_size=29,  # Answer vocab size
        embed_dim=300,
        hidden_dim=512,
        input_dim=2560,  # 2048 + 512
        num_layers=2
    )
    
    # Dummy input
    batch_size = 4
    fused_features = torch.randn(batch_size, 2560)
    target_answers = torch.randint(0, 29, (batch_size, 10))
    
    print(f"\nFused features shape: {fused_features.shape}")
    print(f"Target answers shape: {target_answers.shape}")
    
    # Training mode
    outputs = decoder(fused_features, target_answers, max_len=10, teacher_forcing_ratio=0.5)
    print(f"\nOutput shape: {outputs.shape}")
    print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    
    # Inference mode
    generated = decoder.generate(fused_features, max_len=10)
    print(f"\nGenerated shape: {generated.shape}")
    print(f"Generated sample: {generated[0].tolist()}")
    
    print("\n✓ LSTM Decoder working correctly!")
