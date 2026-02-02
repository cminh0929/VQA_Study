"""
LSTM Encoder for VQA
Encodes questions into fixed-size vectors
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):
    """LSTM Encoder for encoding questions"""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = False
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        print(f"Created LSTMEncoder:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Bidirectional: {bidirectional}")
        print(f"  Output dim: {self.output_dim}")
    
    def forward(self, questions, lengths=None):
        """
        Encode questions
        
        Args:
            questions: Tensor (B, max_len) - word indices
            lengths: Tensor (B,) - actual lengths (optional)
        
        Returns:
            encoded: Tensor (B, output_dim) - encoded question
        """
        batch_size = questions.size(0)
        
        # Embed words
        embedded = self.embedding(questions)  # (B, max_len, embed_dim)
        
        # Pack sequence if lengths provided
        if lengths is not None:
            # Sort by length (required for pack_padded_sequence)
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            embedded_sorted = embedded[sorted_idx]
            
            # Pack
            packed = pack_padded_sequence(
                embedded_sorted,
                lengths_sorted.cpu(),
                batch_first=True,
                enforce_sorted=True
            )
            
            # LSTM
            packed_output, (hidden, cell) = self.lstm(packed)
            
            # Unpack
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            
            # Unsort
            _, unsorted_idx = sorted_idx.sort()
            output = output[unsorted_idx]
            hidden = hidden[:, unsorted_idx, :]
            cell = cell[:, unsorted_idx, :]
        else:
            # LSTM without packing
            output, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden_forward = hidden[-2, :, :]  # (B, hidden_dim)
            hidden_backward = hidden[-1, :, :]  # (B, hidden_dim)
            encoded = torch.cat([hidden_forward, hidden_backward], dim=1)  # (B, 2*hidden_dim)
        else:
            encoded = hidden[-1, :, :]  # (B, hidden_dim)
        
        return encoded
    
    def init_embeddings(self, pretrained_embeddings):
        """
        Initialize embeddings with pretrained vectors (e.g., GloVe)
        
        Args:
            pretrained_embeddings: Tensor (vocab_size, embed_dim)
        """
        self.embedding.weight.data.copy_(pretrained_embeddings)
        print(f"Initialized embeddings with pretrained vectors")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing LSTM Encoder")
    print("="*60)
    
    # Create encoder
    vocab_size = 47  # From question_vocab
    encoder = LSTMEncoder(
        vocab_size=vocab_size,
        embed_dim=300,
        hidden_dim=512,
        num_layers=2,
        bidirectional=False
    )
    
    # Dummy input
    batch_size = 4
    max_len = 20
    questions = torch.randint(0, vocab_size, (batch_size, max_len))
    lengths = torch.tensor([15, 12, 18, 10])
    
    print(f"\nInput questions shape: {questions.shape}")
    print(f"Lengths: {lengths.tolist()}")
    
    # Forward pass
    encoded = encoder(questions, lengths)
    
    print(f"\nOutput shape: {encoded.shape}")
    print(f"Output range: [{encoded.min():.3f}, {encoded.max():.3f}]")
    
    # Test bidirectional
    print("\n" + "="*60)
    print("Testing Bidirectional LSTM")
    print("="*60)
    
    encoder_bi = LSTMEncoder(
        vocab_size=vocab_size,
        embed_dim=300,
        hidden_dim=512,
        num_layers=2,
        bidirectional=True
    )
    
    encoded_bi = encoder_bi(questions, lengths)
    print(f"\nOutput shape (bidirectional): {encoded_bi.shape}")
    
    print("\n✓ LSTM Encoder working correctly!")
