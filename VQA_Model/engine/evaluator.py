"""
Evaluator for VQA Models
Comprehensive evaluation on test set
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from typing import Dict, List

from utils import MetricsTracker


class VQAEvaluator:
    """Evaluator for VQA models"""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        answer_vocab,
        device: str = 'cuda'
    ):
        """
        Args:
            model: VQA model
            test_loader: Test DataLoader
            answer_vocab: Answer vocabulary
            device: Device to evaluate on
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.answer_vocab = answer_vocab
        self.device = device
    
    def evaluate(self, save_predictions: bool = False, output_file: str = None) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            save_predictions: Whether to save predictions
            output_file: File to save predictions
        
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.model.eval()
        tracker = MetricsTracker()
        all_predictions = []
        
        print("Evaluating model on test set")
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move to device
                images = batch['images'].to(self.device)
                questions = batch['questions'].to(self.device)
                q_lengths = batch['question_lengths']
                
                # Generate answers
                predictions, attn_weights = self.model.generate_answer(
                    images, questions, q_lengths, max_len=10
                )
                
                # Decode predictions
                pred_texts = [self.answer_vocab.decode(p.cpu().tolist(), skip_special_tokens=True) 
                             for p in predictions]
                gt_texts = batch['answer_texts']
                categories = batch['question_types']
                
                # Update tracker
                tracker.update(pred_texts, gt_texts, categories)
                
                # Save predictions if requested
                if save_predictions:
                    for i in range(len(pred_texts)):
                        all_predictions.append({
                            'image_id': batch['image_ids'][i],
                            'question': batch['question_texts'][i],
                            'predicted_answer': pred_texts[i],
                            'ground_truth': gt_texts[i],
                            'question_type': categories[i],
                            'tier': batch['tiers'][i]
                        })
        
        # Compute metrics
        metrics = tracker.compute()
        
        # Print results
        self.print_results(metrics)
        
        # Save predictions
        if save_predictions and output_file:
            self.save_predictions(all_predictions, metrics, output_file)
        
        return metrics
    
    def print_results(self, metrics: Dict):
        """Print evaluation results"""
        print("EVALUATION RESULTS\n")
        
        print(f"Overall Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  BLEU-1: {metrics['bleu1']:.4f}")
        print(f"  BLEU-4: {metrics['bleu4']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        
        if 'per_category' in metrics:
            print(f"\nPer-Category Metrics:")
            print(f"{'Category':<25} {'Accuracy':<12} {'BLEU-1':<12} {'F1':<12} {'Count':<8}")
            print("-" * 80)
            
            for cat, cat_metrics in sorted(metrics['per_category'].items()):
                print(f"{cat:<25} {cat_metrics['accuracy']:<12.4f} "
                      f"{cat_metrics['bleu1']:<12.4f} {cat_metrics['f1']:<12.4f} "
                      f"{cat_metrics['count']:<8}")
        

    
    def save_predictions(self, predictions: List[Dict], metrics: Dict, output_file: str):
        """Save predictions to JSON file"""
        output = {
            'metrics': metrics,
            'predictions': predictions
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Saved predictions to {output_file}")


