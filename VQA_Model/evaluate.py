"""
Evaluation Script for VQA Models
Evaluate trained models on test set
"""

import torch
import argparse
import os

from data import Vocabulary, create_dataloaders
from models import create_model_variant
from engine import VQAEvaluator


def evaluate_model(
    model_id: int,
    checkpoint_path: str = None,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_predictions: bool = True
):
    """
    Evaluate a single model variant
    
    Args:
        model_id: Model ID (1-8)
        checkpoint_path: Path to model checkpoint
        batch_size: Batch size
        device: Device to evaluate on
        save_predictions: Whether to save predictions
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING MODEL {model_id}")
    print(f"{'='*80}\n")
    
    # Paths
    train_json = r'..\Data_prep\data\annotations\train.json'
    val_json = r'..\Data_prep\data\annotations\val.json'
    test_json = r'..\Data_prep\data\annotations\test.json'
    image_dir = r'..\Data_prep\data\images'
    q_vocab_path = r'data\question_vocab.json'
    a_vocab_path = r'data\answer_vocab.json'
    
    # Load vocabularies
    print("Loading vocabularies...")
    question_vocab = Vocabulary.load(q_vocab_path)
    answer_vocab = Vocabulary.load(a_vocab_path)
    
    # Create dataloaders
    print("Creating dataloaders...")
    _, _, test_loader = create_dataloaders(
        train_json=train_json,
        val_json=val_json,
        test_json=test_json,
        image_dir=image_dir,
        question_vocab=question_vocab,
        answer_vocab=answer_vocab,
        batch_size=batch_size,
        num_workers=4,
        use_pretrained=True
    )
    
    # Create model
    print(f"Creating Model {model_id}...")
    model = create_model_variant(
        model_id=model_id,
        question_vocab_size=len(question_vocab),
        answer_vocab_size=len(answer_vocab)
    )
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = f'checkpoints/model_{model_id}/best_model.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            print(f"  Val accuracy: {checkpoint['metrics']['accuracy']:.4f}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Evaluating with random weights...")
    
    # Create evaluator
    evaluator = VQAEvaluator(
        model=model,
        test_loader=test_loader,
        answer_vocab=answer_vocab,
        device=device
    )
    
    # Evaluate
    output_file = f'results/model_{model_id}_predictions.json' if save_predictions else None
    if save_predictions:
        os.makedirs('results', exist_ok=True)
    
    metrics = evaluator.evaluate(
        save_predictions=save_predictions,
        output_file=output_file
    )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate VQA Models')
    parser.add_argument('--model_id', type=int, default=None,
                       help='Model ID to evaluate (1-8). If None, evaluate all models.')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to evaluate on')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("VQA MODEL EVALUATION")
    print(f"{'='*80}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save predictions: {args.save_predictions}")
    print(f"{'='*80}\n")
    
    if args.model_id is not None:
        # Evaluate single model
        if args.model_id < 1 or args.model_id > 8:
            print(f"Error: model_id must be between 1 and 8")
            return
        
        evaluate_model(
            model_id=args.model_id,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
            save_predictions=args.save_predictions
        )
    else:
        # Evaluate all 8 models
        print("Evaluating all 8 model variants...")
        all_results = {}
        
        for model_id in range(1, 9):
            metrics = evaluate_model(
                model_id=model_id,
                checkpoint_path=None,  # Use default path
                batch_size=args.batch_size,
                device=args.device,
                save_predictions=args.save_predictions
            )
            all_results[f'model_{model_id}'] = metrics
        
        # Print comparison
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}\n")
        
        print(f"{'Model':<8} {'Accuracy':<12} {'BLEU-1':<12} {'BLEU-4':<12} {'F1':<12}")
        print("-" * 80)
        
        for model_id in range(1, 9):
            metrics = all_results[f'model_{model_id}']
            print(f"Model {model_id:<2} {metrics['accuracy']:<12.4f} "
                  f"{metrics['bleu1']:<12.4f} {metrics['bleu4']:<12.4f} "
                  f"{metrics['f1']:<12.4f}")
        
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
