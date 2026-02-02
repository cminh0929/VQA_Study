"""
Test script for Phase 2: Model Architecture
"""

import sys
sys.path.append('.')

import torch
from models import create_model_variant

print("="*60)
print("TESTING PHASE 2: MODEL ARCHITECTURE")
print("="*60)

# Test all 8 model variants
for model_id in range(1, 9):
    print(f"\n{'='*60}")
    print(f"Testing Model {model_id}")
    print(f"{'='*60}")
    
    # Create model
    model = create_model_variant(
        model_id=model_id,
        question_vocab_size=47,
        answer_vocab_size=29
    )
    
    # Dummy input
    images = torch.randn(2, 3, 224, 224)
    questions = torch.randint(0, 47, (2, 20))
    question_lengths = torch.tensor([15, 12])
    target_answers = torch.randint(0, 29, (2, 10))
    
    # Forward pass
    outputs, attn_weights = model(images, questions, question_lengths, target_answers)
    
    print(f"  Output shape: {outputs.shape}")
    print(f"  Attention: {attn_weights.shape if attn_weights is not None else 'None'}")
    print(f"  Model name: {model.get_model_name()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

print("\n" + "="*60)
print("✅ PHASE 2 COMPLETE: ALL 8 MODELS WORKING!")
print("="*60)
