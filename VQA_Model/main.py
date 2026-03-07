"""
==========================================================================
 VQA Main Script — Train, Evaluate, Compare models from one file
==========================================================================
 Usage:
   py -3.10 main.py train        # Train model
   py -3.10 main.py evaluate     # Evaluate on test set
   py -3.10 main.py both         # Train + Evaluate
   py -3.10 main.py compare      # Compare all trained models
   py -3.10 main.py build_vocab  # Rebuild vocabularies
==========================================================================
"""

import os
import sys
import json
import torch
import torch.nn as nn

# ============================================================================
#  CONFIG — Edit all settings here
# ============================================================================

CONFIG = {
    # --- Model ---
    'model_id': 2,           # 1-4 (see model table below)
    
    # --- Dataset ---
    'dataset': 'full',      # 'small' (dog+cat) or 'full' (10 animals)
    
    # --- Training ---
    'epochs': 5,
    'batch_size': 32,
    'learning_rate': 0.001,
    'teacher_forcing': 0.5,
    'gradient_clip': 5.0,
    
    # --- Hardware ---
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,        # 0 for Windows, 4 for Linux
    
    # --- Save ---
    'save_every': 5,         # Save checkpoint every N epochs
    'save_predictions': True,
}

# ============================================================================
#  MODEL VARIANTS
# ============================================================================
#  ID | CNN       | Pretrained | Attention | Description
#  ---|-----------|------------|-----------|---------------------------
#  1  | ResNet50  | Yes        | No        | Baseline pretrained
#  2  | ResNet50  | Yes        | Yes       | Pretrained + Attention
#  3  | ResNet50  | No         | No        | From-scratch baseline
#  4  | ResNet50  | No         | Yes       | From-scratch + Attention
# ============================================================================


# ============================================================================
#  PATHS (auto-generated, usually no need to modify)
# ============================================================================

def get_paths(dataset='small'):
    """Get all file paths based on dataset choice."""
    ann_dir = os.path.join('..', 'Data_prep', 'data', 'annotations', dataset)
    return {
        'train_json': os.path.join(ann_dir, 'train.json'),
        'val_json': os.path.join(ann_dir, 'val.json'),
        'test_json': os.path.join(ann_dir, 'test.json'),
        'image_dir': os.path.join('..', 'Data_prep', 'data', 'images'),
        'q_vocab': os.path.join('data', 'question_vocab.json'),
        'a_vocab': os.path.join('data', 'answer_vocab.json'),
    }


# ============================================================================
#  BUILD VOCAB
# ============================================================================

def cmd_build_vocab():
    """Build vocabularies from training data."""
    from data import build_vqa_vocabularies
    
    paths = get_paths(CONFIG['dataset'])
    print(f"\nBuilding vocab from: {paths['train_json']}")
    
    build_vqa_vocabularies(
        train_json_path=paths['train_json'],
        question_vocab_path=paths['q_vocab'],
        answer_vocab_path=paths['a_vocab'],
        min_word_freq=1
    )


# ============================================================================
#  TRAIN
# ============================================================================

def cmd_train():
    """Train model based on CONFIG."""
    from data import Vocabulary, create_dataloaders
    from models import create_model_variant
    from engine import VQATrainer
    
    model_id = CONFIG['model_id']
    paths = get_paths(CONFIG['dataset'])
    
    print(f"\n{'='*60}")
    print(f"  TRAINING MODEL {model_id} on {CONFIG['dataset'].upper()} dataset")
    print(f"{'='*60}\n")
    
    # Load vocabularies
    print("Loading vocabularies...")
    q_vocab = Vocabulary.load(paths['q_vocab'])
    a_vocab = Vocabulary.load(paths['a_vocab'])
    
    # Create model
    print(f"Creating Model {model_id}...")
    model = create_model_variant(
        model_id=model_id,
        question_vocab_size=len(q_vocab),
        answer_vocab_size=len(a_vocab)
    )
    
    # Auto-unfreeze from-scratch models
    if not model.cnn_pretrained:
        print("  From-scratch model: unfreezing CNN weights")
        model.cnn_encoder.unfreeze()
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        train_json=paths['train_json'],
        val_json=paths['val_json'],
        test_json=paths['test_json'],
        image_dir=paths['image_dir'],
        question_vocab=q_vocab,
        answer_vocab=a_vocab,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        use_pretrained=model.cnn_pretrained
    )
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Trainer
    checkpoint_dir = f'checkpoints/model_{model_id}'
    log_dir = f'logs/model_{model_id}'
    
    trainer = VQATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        answer_vocab=a_vocab,
        optimizer=optimizer,
        criterion=criterion,
        device=CONFIG['device'],
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        teacher_forcing_ratio=CONFIG['teacher_forcing'],
        gradient_clip=CONFIG['gradient_clip']
    )
    
    # Train!
    trainer.train(num_epochs=CONFIG['epochs'], save_every=CONFIG['save_every'])
    
    print(f"\nTraining complete.")
    print(f"  Best val accuracy : {trainer.best_val_accuracy:.4f}")
    print(f"  Checkpoint        : {checkpoint_dir}/best_model.pth")
    print(f"  Logs              : {log_dir}/training_history.json")


# ============================================================================
#  EVALUATE
# ============================================================================

def cmd_evaluate():
    """Evaluate model on test set."""
    from data import Vocabulary, create_dataloaders
    from models import create_model_variant
    from engine import VQAEvaluator
    
    model_id = CONFIG['model_id']
    paths = get_paths(CONFIG['dataset'])
    
    print(f"\n{'='*60}")
    print(f"  EVALUATING MODEL {model_id} on {CONFIG['dataset'].upper()} test set")
    print(f"{'='*60}\n")
    
    # Load vocabularies
    q_vocab = Vocabulary.load(paths['q_vocab'])
    a_vocab = Vocabulary.load(paths['a_vocab'])
    
    # Create model
    model = create_model_variant(
        model_id=model_id,
        question_vocab_size=len(q_vocab),
        answer_vocab_size=len(a_vocab)
    )
    
    # Load checkpoint
    checkpoint_path = f'checkpoints/model_{model_id}/best_model.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  From epoch {checkpoint['epoch']}")
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}")
        print("Evaluating with random weights...")
    
    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        train_json=paths['train_json'],
        val_json=paths['val_json'],
        test_json=paths['test_json'],
        image_dir=paths['image_dir'],
        question_vocab=q_vocab,
        answer_vocab=a_vocab,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        use_pretrained=model.cnn_pretrained
    )
    
    # Evaluate
    evaluator = VQAEvaluator(
        model=model,
        test_loader=test_loader,
        answer_vocab=a_vocab,
        device=CONFIG['device']
    )
    
    output_file = None
    if CONFIG['save_predictions']:
        os.makedirs('results', exist_ok=True)
        output_file = f'results/model_{model_id}_predictions.json'
    
    metrics = evaluator.evaluate(
        save_predictions=CONFIG['save_predictions'],
        output_file=output_file
    )
    
    return metrics


# ============================================================================
#  COMPARE — Compare all trained models
# ============================================================================

def cmd_compare():
    """Compare all trained models."""
    print(f"\n{'='*60}")
    print("  MODEL COMPARISON")
    print(f"{'='*60}\n")
    
    results = {}
    for i in range(1, 5):
        result_file = f'results/model_{i}_predictions.json'
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
            results[i] = data['metrics']
    
    if not results:
        print("No results found. Train and evaluate models first!")
        return
    
    # Model configs for display
    model_info = {
        1: ('ResNet50', 'Pretrained', 'No'),
        2: ('ResNet50', 'Pretrained', 'Yes'),
        3: ('ResNet50', 'Scratch',    'No'),
        4: ('ResNet50', 'Scratch',    'Yes'),
    }
    
    # Print table
    header = f"{'ID':>3} {'CNN':>9} {'Weights':>11} {'Attn':>5} | {'Accuracy':>9} {'BLEU-1':>8} {'F1':>8}"
    print(header)
    print("-" * len(header))
    
    for mid, m in sorted(results.items()):
        cnn, wt, attn = model_info[mid]
        print(f"{mid:>3} {cnn:>9} {wt:>11} {attn:>5} | "
              f"{m['accuracy']:>9.4f} {m['bleu1']:>8.4f} {m['f1']:>8.4f}")
    
    # Per-category for each model
    print(f"\n{'='*60}")
    print("  PER-CATEGORY ACCURACY")
    print(f"{'='*60}\n")
    
    categories = ['animal_recognition', 'color_recognition', 'yes_no', 'counting_simple']
    cat_header = f"{'ID':>3} | " + " | ".join(f"{c[:12]:>12}" for c in categories)
    print(cat_header)
    print("-" * len(cat_header))
    
    for mid, m in sorted(results.items()):
        if 'per_category' in m:
            vals = []
            for c in categories:
                acc = m['per_category'].get(c, {}).get('accuracy', 0)
                vals.append(f"{acc:>12.4f}")
            print(f"{mid:>3} | " + " | ".join(vals))
    
    # Best model
    best_id = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\nBest model: Model {best_id} (Accuracy: {results[best_id]['accuracy']:.4f})")


# ============================================================================
#  MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Current CONFIG:")
        for k, v in CONFIG.items():
            print(f"  {k:20s} = {v}")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'train':
        cmd_train()
    elif command == 'evaluate' or command == 'eval':
        cmd_evaluate()
    elif command == 'both':
        cmd_train()
        cmd_evaluate()
    elif command == 'compare':
        cmd_compare()
    elif command == 'build_vocab' or command == 'vocab':
        cmd_build_vocab()
    else:
        print(f"Unknown command: {command}")
        print("Available: train, evaluate, both, compare, build_vocab")


if __name__ == "__main__":
    main()
