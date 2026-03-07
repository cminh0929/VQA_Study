"""
Main Training Script for VQA Models
Train all 4 model variants
"""

import torch
import torch.nn as nn
import argparse
import os

from data import Vocabulary, create_dataloaders
from models import create_model_variant
from engine import VQATrainer


def train_model(
    model_id: int,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    dataset: str = 'small'
):
    """
    Train a single model variant
    
    Args:
        model_id: Model ID (1-4)
        num_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
    """
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL {model_id}")
    print(f"{'='*80}\n")
    
    # Paths - support small/full dataset
    ann_dir = rf'..\Data_prep\data\annotations\{dataset}'
    train_json = os.path.join(ann_dir, 'train.json')
    val_json = os.path.join(ann_dir, 'val.json')
    test_json = os.path.join(ann_dir, 'test.json')
    image_dir = r'..\Data_prep\data\images'
    q_vocab_path = r'data\question_vocab.json'
    a_vocab_path = r'data\answer_vocab.json'
    
    print(f"Dataset: {dataset.upper()}")
    print(f"Annotations: {ann_dir}")
    
    # Load vocabularies
    print("Loading vocabularies...")
    question_vocab = Vocabulary.load(q_vocab_path)
    answer_vocab = Vocabulary.load(a_vocab_path)
    
    # Create model first to get its configuration
    print(f"Creating Model {model_id}...")
    model = create_model_variant(
        model_id=model_id,
        question_vocab_size=len(question_vocab),
        answer_vocab_size=len(answer_vocab)
    )
    
    # NEW: Automatically unfreeze CNN if it's a from-scratch model
    if not model.cnn_pretrained:
        print(f"  (!) From-scratch model detected. Unfreezing CNN weights for training...")
        model.cnn_encoder.unfreeze()
    
    # Create dataloaders with the correct normalization flag
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_json=train_json,
        val_json=val_json,
        test_json=test_json,
        image_dir=image_dir,
        question_vocab=question_vocab,
        answer_vocab=answer_vocab,
        batch_size=batch_size,
        num_workers=0,  # 0 for Windows compatibility
        use_pretrained=model.cnn_pretrained  # Correctly synced with model
    )
    
    # Optimizer and loss
    # If using from-scratch, we might need a different LR or weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Create trainer
    checkpoint_dir = f'checkpoints/model_{model_id}'
    log_dir = f'logs/model_{model_id}'
    
    trainer = VQATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        answer_vocab=answer_vocab,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        teacher_forcing_ratio=0.5,
        gradient_clip=5.0
    )
    
    # Train
    trainer.train(num_epochs=num_epochs, save_every=5)
    
    print(f"\n✓ Model {model_id} training complete!")
    print(f"  Best validation accuracy: {trainer.best_val_accuracy:.4f}")
    print(f"  Checkpoints saved to: {checkpoint_dir}")
    print(f"  Logs saved to: {log_dir}\n")


def main():
    parser = argparse.ArgumentParser(description='Train VQA Models')
    parser.add_argument('--model_id', type=int, default=None,
                       help='Model ID to train (1-4). If None, train all models.')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    parser.add_argument('--dataset', type=str, default='small', choices=['small', 'full'],
                       help='Dataset to use: small (dog+cat) or full (10 animals)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("VQA MODEL TRAINING")
    print(f"{'='*80}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*80}\n")
    
    if args.model_id is not None:
        # Train single model
        if args.model_id < 1 or args.model_id > 4:
            print(f"Error: model_id must be between 1 and 4")
            return
        
        train_model(
            model_id=args.model_id,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
            dataset=args.dataset
        )
    else:
        # Train all 8 models
        print("Training all 4 model variants...")
        for model_id in range(1, 5):
            train_model(
                model_id=model_id,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=args.device,
                dataset=args.dataset
            )
        
        print(f"\n{'='*80}")
        print("ALL MODELS TRAINED SUCCESSFULLY!")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
