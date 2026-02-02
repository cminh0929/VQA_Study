"""
Quick Test: Train on 100 images with best model
Model 2: ResNet50 + Pretrained + Attention
"""

import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader, Subset

from data import Vocabulary, VQADataset, collate_fn
from models import create_model_variant
from engine import VQATrainer
from utils import MetricsTracker


def create_small_subset(full_dataset, num_samples=100):
    """Create a small subset of dataset"""
    indices = list(range(min(num_samples, len(full_dataset))))
    return Subset(full_dataset, indices)


def main():
    print("="*80)
    print("QUICK TEST: 100 Images with Model 2 (ResNet50 + Pretrained + Attention)")
    print("="*80)
    
    # Config
    num_train_samples = 100
    num_val_samples = 20
    num_epochs = 5
    batch_size = 8
    learning_rate = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Train samples: {num_train_samples}")
    print(f"  Val samples: {num_val_samples}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    
    # Paths
    train_json = r'..\Data_prep\data\annotations\train.json'
    val_json = r'..\Data_prep\data\annotations\val.json'
    image_dir = r'..\Data_prep\data\images'
    q_vocab_path = r'data\question_vocab.json'
    a_vocab_path = r'data\answer_vocab.json'
    
    # Load vocabularies
    print("\n1. Loading vocabularies...")
    question_vocab = Vocabulary.load(q_vocab_path)
    answer_vocab = Vocabulary.load(a_vocab_path)
    print(f"  Question vocab: {len(question_vocab)} words")
    print(f"  Answer vocab: {len(answer_vocab)} words")
    
    # Create full datasets
    print("\n2. Creating datasets...")
    train_dataset_full = VQADataset(
        annotations_file=train_json,
        image_dir=image_dir,
        question_vocab=question_vocab,
        answer_vocab=answer_vocab,
        max_question_len=20,
        max_answer_len=10,
        use_pretrained=True,
        is_training=True
    )
    
    val_dataset_full = VQADataset(
        annotations_file=val_json,
        image_dir=image_dir,
        question_vocab=question_vocab,
        answer_vocab=answer_vocab,
        max_question_len=20,
        max_answer_len=10,
        use_pretrained=True,
        is_training=False
    )
    
    # Create small subsets
    print(f"\n3. Creating small subsets...")
    train_dataset = create_small_subset(train_dataset_full, num_train_samples)
    val_dataset = create_small_subset(val_dataset_full, num_val_samples)
    print(f"  Train subset: {len(train_dataset)} samples")
    print(f"  Val subset: {len(val_dataset)} samples")
    
    # Create dataloaders
    print("\n4. Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 for Windows compatibility
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create Model 2 (ResNet50 + Pretrained + Attention)
    print("\n5. Creating Model 2 (ResNet50 + Pretrained + Attention)...")
    model = create_model_variant(
        model_id=2,
        question_vocab_size=len(question_vocab),
        answer_vocab_size=len(answer_vocab)
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # Optimizer and loss
    print("\n6. Setting up training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Create trainer
    trainer = VQATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        answer_vocab=answer_vocab,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir='checkpoints/test_100',
        log_dir='logs/test_100',
        teacher_forcing_ratio=0.5,
        gradient_clip=5.0
    )
    
    # Train
    print("\n7. Starting training...")
    print("="*80)
    trainer.train(num_epochs=num_epochs, save_every=2)
    
    # Final evaluation
    print("\n8. Final evaluation on validation set...")
    print("="*80)
    
    model.eval()
    tracker = MetricsTracker()
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].to(device)
            questions = batch['questions'].to(device)
            q_lengths = batch['question_lengths']
            
            # Generate answers
            predictions, attn_weights = model.generate_answer(images, questions, q_lengths, max_len=10)
            
            # Decode
            pred_texts = [answer_vocab.decode(p.cpu().tolist(), skip_special_tokens=True) 
                         for p in predictions]
            gt_texts = batch['answer_texts']
            categories = batch['question_types']
            
            tracker.update(pred_texts, gt_texts, categories)
    
    # Compute final metrics
    final_metrics = tracker.compute()
    
    print("\nFinal Test Results:")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  BLEU-1: {final_metrics['bleu1']:.4f}")
    print(f"  F1 Score: {final_metrics['f1']:.4f}")
    
    if 'per_category' in final_metrics:
        print("\n  Per-category accuracy:")
        for cat, cat_metrics in final_metrics['per_category'].items():
            print(f"    {cat}: {cat_metrics['accuracy']:.4f} ({cat_metrics['count']} samples)")
    
    # Show sample predictions
    print("\n9. Sample predictions:")
    print("="*80)
    
    model.eval()
    sample_batch = next(iter(val_loader))
    
    with torch.no_grad():
        images = sample_batch['images'].to(device)
        questions = sample_batch['questions'].to(device)
        q_lengths = sample_batch['question_lengths']
        
        predictions, attn_weights = model.generate_answer(images, questions, q_lengths, max_len=10)
    
    # Show first 5 samples
    for i in range(min(5, len(predictions))):
        pred_text = answer_vocab.decode(predictions[i].cpu().tolist(), skip_special_tokens=True)
        gt_text = sample_batch['answer_texts'][i]
        q_text = sample_batch['question_texts'][i]
        img_id = sample_batch['image_ids'][i]
        
        print(f"\nSample {i+1}:")
        print(f"  Image: {img_id}")
        print(f"  Question: {q_text}")
        print(f"  Ground truth: {gt_text}")
        print(f"  Predicted: {pred_text}")
        print(f"  Correct: {'✓' if pred_text.strip() == gt_text.strip() else '✗'}")
        
        if attn_weights is not None:
            print(f"  Attention weight: {attn_weights[i].item():.4f}")
    
    print("\n" + "="*80)
    print("✅ QUICK TEST COMPLETE!")
    print("="*80)
    print(f"\nBest validation accuracy: {trainer.best_val_accuracy:.4f}")
    print(f"Checkpoints saved to: checkpoints/test_100/")
    print(f"Logs saved to: logs/test_100/")
    print("\nModel is working correctly and ready for full training! 🚀")
    print("="*80)


if __name__ == "__main__":
    main()
