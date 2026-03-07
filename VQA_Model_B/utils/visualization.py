"""
Visualization utilities for VQA
Includes attention visualization, training curves, and sample predictions
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import json
import os


def plot_training_history(history_file: str, save_path: str = None):
    """
    Plot training history (loss, accuracy, etc.)
    
    Args:
        history_file: Path to training_history.json
        save_path: Path to save plot
    """
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['val_accuracy'], 'g-', label='Val Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # BLEU-1
    axes[1, 0].plot(epochs, history['val_bleu1'], 'm-', label='Val BLEU-1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('BLEU-1')
    axes[1, 0].set_title('Validation BLEU-1')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Score
    axes[1, 1].plot(epochs, history['val_f1'], 'c-', label='Val F1')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Validation F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_attention_weights(image, question, answer, attention_weights, save_path: str = None):
    """
    Visualize attention weights on image
    
    Args:
        image: Image tensor or numpy array (H, W, 3)
        question: Question text
        answer: Answer text
        attention_weights: Attention weights
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Display image
    if isinstance(image, np.ndarray):
        ax.imshow(image)
    else:
        # Convert tensor to numpy
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        # Denormalize if needed
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        ax.imshow(img_np)
    
    ax.axis('off')
    
    # Add text
    title = f"Q: {question}\nA: {answer}\nAttention: {attention_weights:.4f}"
    ax.set_title(title, fontsize=12, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_model_comparison(results: Dict[str, Dict], save_path: str = None):
    """
    Plot comparison of multiple models
    
    Args:
        results: Dict mapping model_name to metrics dict
        save_path: Path to save plot
    """
    models = list(results.keys())
    metrics = ['accuracy', 'bleu1', 'bleu4', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in models]
        
        axes[idx].bar(range(len(models)), values, color='skyblue', edgecolor='navy')
        axes[idx].set_xticks(range(len(models)))
        axes[idx].set_xticklabels(models, rotation=45, ha='right')
        axes[idx].set_ylabel(metric.upper())
        axes[idx].set_title(f'{metric.upper()} Comparison')
        axes[idx].grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_category_metrics(metrics: Dict[str, Dict], save_path: str = None):
    """
    Plot per-category metrics
    
    Args:
        metrics: Dict mapping category to metrics
        save_path: Path to save plot
    """
    categories = list(metrics.keys())
    accuracies = [metrics[cat]['accuracy'] for cat in categories]
    bleu_scores = [metrics[cat]['bleu1'] for cat in categories]
    f1_scores = [metrics[cat]['f1'] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, accuracies, width, label='Accuracy', color='skyblue')
    ax.bar(x, bleu_scores, width, label='BLEU-1', color='lightcoral')
    ax.bar(x + width, f1_scores, width, label='F1', color='lightgreen')
    
    ax.set_xlabel('Question Category')
    ax.set_ylabel('Score')
    ax.set_title('Per-Category Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-category metrics to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_results_summary(results_dir: str, output_dir: str = 'visualizations'):
    """
    Create comprehensive visualization summary from results
    
    Args:
        results_dir: Directory containing results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating visualizations in {output_dir}...")
    
    # Find all model results
    model_results = {}
    for i in range(1, 9):
        result_file = os.path.join(results_dir, f'model_{i}_predictions.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
                model_results[f'Model {i}'] = data['metrics']
    
    if model_results:
        # Plot model comparison
        plot_model_comparison(
            model_results,
            save_path=os.path.join(output_dir, 'model_comparison.png')
        )
        print("✓ Created model comparison")
    
    # Plot training histories
    for i in range(1, 9):
        history_file = f'logs/model_{i}/training_history.json'
        if os.path.exists(history_file):
            plot_training_history(
                history_file,
                save_path=os.path.join(output_dir, f'model_{i}_training.png')
            )
            print(f"✓ Created training curves for Model {i}")
    
    print(f"\n✓ All visualizations saved to {output_dir}/")


# Example usage
if __name__ == "__main__":
    print("Visualization utilities for VQA")
    print("Use in notebooks or scripts to visualize results")
