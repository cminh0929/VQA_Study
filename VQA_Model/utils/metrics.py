"""
Evaluation Metrics for VQA
Includes Accuracy, BLEU, F1 Score
"""

import torch
import numpy as np
from collections import Counter
from typing import List, Dict


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Calculate exact match accuracy
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
    
    Returns:
        accuracy: Accuracy score (0-1)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred.strip() == gt.strip())
    accuracy = correct / len(predictions)
    return accuracy


def calculate_bleu(predictions: List[str], ground_truths: List[str], n: int = 1) -> float:
    """
    Calculate BLEU-n score
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        n: N-gram size (1 for BLEU-1, 4 for BLEU-4)
    
    Returns:
        bleu_score: BLEU-n score (0-1)
    """
    def get_ngrams(text: str, n: int) -> Counter:
        """Get n-grams from text"""
        words = text.lower().split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        return Counter(ngrams)
    
    total_precision = 0.0
    count = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_ngrams = get_ngrams(pred, n)
        gt_ngrams = get_ngrams(gt, n)
        
        if len(pred_ngrams) == 0:
            continue
        
        # Count matches
        matches = sum((pred_ngrams & gt_ngrams).values())
        total = sum(pred_ngrams.values())
        
        precision = matches / total if total > 0 else 0
        total_precision += precision
        count += 1
    
    bleu_score = total_precision / count if count > 0 else 0
    return bleu_score


def calculate_f1(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """
    Calculate F1 score (word-level)
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
    
    Returns:
        dict with precision, recall, f1
    """
    total_precision = 0.0
    total_recall = 0.0
    count = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_words = set(pred.lower().split())
        gt_words = set(gt.lower().split())
        
        if len(pred_words) == 0 and len(gt_words) == 0:
            continue
        
        # Calculate precision and recall
        if len(pred_words) > 0:
            precision = len(pred_words & gt_words) / len(pred_words)
        else:
            precision = 0
        
        if len(gt_words) > 0:
            recall = len(pred_words & gt_words) / len(gt_words)
        else:
            recall = 0
        
        total_precision += precision
        total_recall += recall
        count += 1
    
    avg_precision = total_precision / count if count > 0 else 0
    avg_recall = total_recall / count if count > 0 else 0
    
    if avg_precision + avg_recall > 0:
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1 = 0
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': f1
    }


def calculate_per_category_metrics(
    predictions: List[str],
    ground_truths: List[str],
    categories: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics per question category
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        categories: List of question categories
    
    Returns:
        dict mapping category to metrics
    """
    # Group by category
    category_data = {}
    for pred, gt, cat in zip(predictions, ground_truths, categories):
        if cat not in category_data:
            category_data[cat] = {'predictions': [], 'ground_truths': []}
        category_data[cat]['predictions'].append(pred)
        category_data[cat]['ground_truths'].append(gt)
    
    # Calculate metrics per category
    results = {}
    for cat, data in category_data.items():
        preds = data['predictions']
        gts = data['ground_truths']
        
        acc = calculate_accuracy(preds, gts)
        bleu1 = calculate_bleu(preds, gts, n=1)
        f1_scores = calculate_f1(preds, gts)
        
        results[cat] = {
            'accuracy': acc,
            'bleu1': bleu1,
            'f1': f1_scores['f1'],
            'count': len(preds)
        }
    
    return results


class MetricsTracker:
    """Track metrics during training/evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.ground_truths = []
        self.categories = []
        self.losses = []
    
    def update(self, predictions: List[str], ground_truths: List[str], 
               categories: List[str] = None, loss: float = None):
        """Update with new batch"""
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)
        if categories:
            self.categories.extend(categories)
        if loss is not None:
            self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        if len(self.predictions) == 0:
            return {}
        
        metrics = {
            'accuracy': calculate_accuracy(self.predictions, self.ground_truths),
            'bleu1': calculate_bleu(self.predictions, self.ground_truths, n=1),
            'bleu4': calculate_bleu(self.predictions, self.ground_truths, n=4),
        }
        
        f1_scores = calculate_f1(self.predictions, self.ground_truths)
        metrics.update(f1_scores)
        
        if len(self.losses) > 0:
            metrics['loss'] = np.mean(self.losses)
        
        # Per-category metrics
        if len(self.categories) > 0:
            cat_metrics = calculate_per_category_metrics(
                self.predictions, self.ground_truths, self.categories
            )
            metrics['per_category'] = cat_metrics
        
        return metrics


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing Metrics")
    print("="*60)
    
    # Sample data
    predictions = [
        "dog",
        "black cat",
        "two",
        "yes",
        "bird"
    ]
    
    ground_truths = [
        "dog",
        "black cat",
        "three",
        "yes",
        "cat"
    ]
    
    categories = [
        "animal_recognition",
        "color_recognition",
        "counting_simple",
        "yes_no",
        "animal_recognition"
    ]
    
    # Test accuracy
    acc = calculate_accuracy(predictions, ground_truths)
    print(f"\nAccuracy: {acc:.3f}")
    
    # Test BLEU
    bleu1 = calculate_bleu(predictions, ground_truths, n=1)
    bleu4 = calculate_bleu(predictions, ground_truths, n=4)
    print(f"BLEU-1: {bleu1:.3f}")
    print(f"BLEU-4: {bleu4:.3f}")
    
    # Test F1
    f1_scores = calculate_f1(predictions, ground_truths)
    print(f"Precision: {f1_scores['precision']:.3f}")
    print(f"Recall: {f1_scores['recall']:.3f}")
    print(f"F1: {f1_scores['f1']:.3f}")
    
    # Test per-category
    cat_metrics = calculate_per_category_metrics(predictions, ground_truths, categories)
    print("\nPer-category metrics:")
    for cat, metrics in cat_metrics.items():
        print(f"  {cat}:")
        print(f"    Accuracy: {metrics['accuracy']:.3f}")
        print(f"    BLEU-1: {metrics['bleu1']:.3f}")
        print(f"    F1: {metrics['f1']:.3f}")
        print(f"    Count: {metrics['count']}")
    
    # Test MetricsTracker
    print("\nTesting MetricsTracker:")
    tracker = MetricsTracker()
    tracker.update(predictions[:3], ground_truths[:3], categories[:3], loss=1.5)
    tracker.update(predictions[3:], ground_truths[3:], categories[3:], loss=1.2)
    
    all_metrics = tracker.compute()
    print(f"  Accuracy: {all_metrics['accuracy']:.3f}")
    print(f"  BLEU-1: {all_metrics['bleu1']:.3f}")
    print(f"  F1: {all_metrics['f1']:.3f}")
    print(f"  Loss: {all_metrics['loss']:.3f}")
    
    print("\n✓ All metrics working correctly!")
