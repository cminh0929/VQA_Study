"""
Analyze and compare results from all trained models
"""

import json
import os
import argparse
from utils import plot_model_comparison, plot_per_category_metrics, create_results_summary


def load_model_results(results_dir='results'):
    """Load results from all models"""
    model_results = {}
    
    for i in range(1, 9):
        result_file = os.path.join(results_dir, f'model_{i}_predictions.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
                model_results[f'Model {i}'] = data['metrics']
                print(f"✓ Loaded Model {i} results")
        else:
            print(f"⚠ Model {i} results not found")
    
    return model_results


def print_comparison_table(results):
    """Print comparison table"""
    print("\n" + "="*100)
    print("MODEL COMPARISON TABLE")
    print("="*100)
    
    print(f"\n{'Model':<12} {'Accuracy':<12} {'BLEU-1':<12} {'BLEU-4':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-"*100)
    
    for model_name, metrics in sorted(results.items()):
        print(f"{model_name:<12} "
              f"{metrics.get('accuracy', 0):<12.4f} "
              f"{metrics.get('bleu1', 0):<12.4f} "
              f"{metrics.get('bleu4', 0):<12.4f} "
              f"{metrics.get('f1', 0):<12.4f} "
              f"{metrics.get('precision', 0):<12.4f} "
              f"{metrics.get('recall', 0):<12.4f}")
    
    print("="*100)


def analyze_by_architecture(results):
    """Analyze results by CNN architecture"""
    print("\n" + "="*80)
    print("ANALYSIS BY CNN ARCHITECTURE")
    print("="*80)
    
    resnet_models = {k: v for k, v in results.items() if '1' in k or '2' in k or '3' in k or '4' in k}
    vgg_models = {k: v for k, v in results.items() if '5' in k or '6' in k or '7' in k or '8' in k}
    
    if resnet_models:
        avg_resnet_acc = sum(m['accuracy'] for m in resnet_models.values()) / len(resnet_models)
        print(f"\nResNet50 Models (1-4):")
        print(f"  Average Accuracy: {avg_resnet_acc:.4f}")
        print(f"  Models: {len(resnet_models)}")
    
    if vgg_models:
        avg_vgg_acc = sum(m['accuracy'] for m in vgg_models.values()) / len(vgg_models)
        print(f"\nVGG16 Models (5-8):")
        print(f"  Average Accuracy: {avg_vgg_acc:.4f}")
        print(f"  Models: {len(vgg_models)}")


def analyze_by_attention(results):
    """Analyze impact of attention"""
    print("\n" + "="*80)
    print("ANALYSIS BY ATTENTION MECHANISM")
    print("="*80)
    
    # Models with attention: 2, 4, 6, 8
    with_attn = {k: v for k, v in results.items() if any(x in k for x in ['2', '4', '6', '8'])}
    without_attn = {k: v for k, v in results.items() if any(x in k for x in ['1', '3', '5', '7'])}
    
    if with_attn:
        avg_with = sum(m['accuracy'] for m in with_attn.values()) / len(with_attn)
        print(f"\nWith Attention (2, 4, 6, 8):")
        print(f"  Average Accuracy: {avg_with:.4f}")
    
    if without_attn:
        avg_without = sum(m['accuracy'] for m in without_attn.values()) / len(without_attn)
        print(f"\nWithout Attention (1, 3, 5, 7):")
        print(f"  Average Accuracy: {avg_without:.4f}")
    
    if with_attn and without_attn:
        improvement = avg_with - avg_without
        print(f"\nAttention Improvement: {improvement:+.4f} ({improvement/avg_without*100:+.2f}%)")


def find_best_model(results):
    """Find best performing model"""
    print("\n" + "="*80)
    print("BEST MODEL")
    print("="*80)
    
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"\nBest Model: {best_model[0]}")
    print(f"  Accuracy: {best_model[1]['accuracy']:.4f}")
    print(f"  BLEU-1: {best_model[1]['bleu1']:.4f}")
    print(f"  F1: {best_model[1]['f1']:.4f}")
    
    if 'per_category' in best_model[1]:
        print(f"\n  Per-category accuracy:")
        for cat, metrics in best_model[1]['per_category'].items():
            print(f"    {cat}: {metrics['accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze VQA Results')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing results')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    print("="*80)
    print("VQA RESULTS ANALYSIS")
    print("="*80)
    
    # Load results
    results = load_model_results(args.results_dir)
    
    if not results:
        print("\n⚠ No results found. Train models first!")
        return
    
    # Print comparison table
    print_comparison_table(results)
    
    # Analyze by architecture
    analyze_by_architecture(results)
    
    # Analyze by attention
    analyze_by_attention(results)
    
    # Find best model
    find_best_model(results)
    
    # Create visualizations
    if args.visualize:
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)
        create_results_summary(args.results_dir, args.output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
