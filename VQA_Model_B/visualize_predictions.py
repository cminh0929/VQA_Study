import os
import json
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import Vocabulary, get_val_transforms
from models import create_model_variant

def load_models_and_vocabs(device='cuda'):
    print("Loading vocabularies...")
    q_vocab_path = 'data/question_vocab.json'
    a_vocab_path = 'data/answer_vocab.json'
    
    if not os.path.exists(q_vocab_path) or not os.path.exists(a_vocab_path):
        print(f"Error: Vocabulary files not found at {q_vocab_path} or {a_vocab_path}")
        return None, None, None

    q_vocab = Vocabulary.load(q_vocab_path)
    a_vocab = Vocabulary.load(a_vocab_path)
    
    models = {}
    for i in range(1, 5):
        checkpoint_path = f'checkpoints/model_{i}/best_model.pth'
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found for Model {i} at {checkpoint_path}")
            models[i] = None
            continue
            
        print(f"Loading Model {i}...")
        model = create_model_variant(i, len(q_vocab), len(a_vocab))
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        models[i] = model
        print(f"✓ Model {i} loaded (Epoch {checkpoint.get('epoch', 'N/A')})")
        
    return models, q_vocab, a_vocab

def get_sample_data(annotation_file, num_images=10, questions_per_image=4):
    print(f"Loading test annotations from {annotation_file}...")
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Group by image_id
    grouped = {}
    for item in data:
        img_id = item['image_id']
        if img_id not in grouped:
            grouped[img_id] = []
        grouped[img_id].append(item)
        
    # Filter images that have enough questions
    valid_images = {k: v for k, v in grouped.items() if len(v) >= questions_per_image}
    
    if len(valid_images) == 0:
        print(f"No images with at least {questions_per_image} questions. Falling back to any valid images.")
        valid_images = {k: v for k, v in grouped.items() if len(v) >= 1}
        
    # Sample images
    image_ids = list(valid_images.keys())
    sampled_ids = random.sample(image_ids, min(num_images, len(image_ids)))
    
    samples = []
    for img_id in sampled_ids:
        qs = valid_images[img_id]
        # Pick requested number of questions or max available
        sampled_qs = random.sample(qs, min(questions_per_image, len(qs)))
        samples.append({
            'image_id': img_id,
            'questions': sampled_qs
        })
        
    return samples

def visualize_results(models, q_vocab, a_vocab, samples, image_dir, device='cuda'):
    print("\nGenerating visualizations...")
    os.makedirs('results/visualizations', exist_ok=True)
    
    transform_pretrained = get_val_transforms(use_pretrained=True)
    transform_scratch = get_val_transforms(use_pretrained=False)
    
    for idx, sample in enumerate(samples):
        img_id = sample['image_id']
        img_path = os.path.join(image_dir, img_id)
        
        try:
            pil_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
            
        tensor_pre = transform_pretrained(pil_image).unsqueeze(0).to(device)
        tensor_scr = transform_scratch(pil_image).unsqueeze(0).to(device)
        
        # Create plot
        fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(15, max(6, 1.8 * len(sample['questions']))), gridspec_kw={'width_ratios': [1, 1.2]})
        
        # Left side: Image
        ax_img.imshow(pil_image)
        ax_img.axis('off')
        ax_img.set_title(f"Image: {img_id}", fontsize=14, fontweight='bold')
        
        # Right side: Questions and Answers
        ax_text.axis('off')
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
        
        y_pos = 0.95
        total_qs = len(sample['questions'])
        block_height = 0.9 / max(total_qs, 1)
        line_height = block_height / 5.0
        
        for q_idx, qa in enumerate(sample['questions']):
            question_text = qa['question']
            ground_truth = qa['answer']
            q_type = qa.get('question_type', 'unknown')
            
            # Predict for each model
            q_tokens = q_vocab.encode(question_text)
            q_tensor = torch.tensor([q_tokens], dtype=torch.long).to(device)
            q_length = torch.tensor([len(q_tokens)])
            
            predictions = {}
            for m_id, model in models.items():
                if model is None:
                    predictions[f"M{m_id}"] = "N/A"
                    continue
                    
                input_tensor = tensor_pre if m_id in [1, 2] else tensor_scr
                
                with torch.no_grad():
                    pred_out, _ = model.generate_answer(
                        input_tensor, q_tensor, q_length, max_len=10
                    )
                
                pred_tokens = pred_out[0].cpu().tolist()
                pred_text = a_vocab.decode(pred_tokens, skip_special_tokens=True)
                predictions[f"M{m_id}"] = pred_text
                
            # Render text on the right side
            
            # Q & A GT
            ax_text.text(0.01, y_pos, f"Q{q_idx+1}: {question_text}", fontsize=13, fontweight='bold', color='darkblue')
            y_pos -= line_height
            ax_text.text(0.01, y_pos, f"GT: {ground_truth} ({q_type})", fontsize=12, fontweight='bold', color='darkgreen')
            y_pos -= line_height * 1.2
            
            # Models predictions
            model_texts = []
            for i in range(1, 5):
                m_pred = predictions.get(f"M{i}", "N/A")
                color = 'green' if m_pred == ground_truth else 'red'
                model_texts.append((f"M{i}: {m_pred}", color))
                
            ax_text.text(0.05, y_pos, f"{model_texts[0][0]}", fontsize=11, color=model_texts[0][1])
            ax_text.text(0.4, y_pos, f"{model_texts[1][0]}", fontsize=11, color=model_texts[1][1])
            y_pos -= line_height
            
            ax_text.text(0.05, y_pos, f"{model_texts[2][0]}", fontsize=11, color=model_texts[2][1])
            ax_text.text(0.4, y_pos, f"{model_texts[3][0]}", fontsize=11, color=model_texts[3][1])
            y_pos -= line_height * 0.5
            
            ax_text.plot([0.01, 0.9], [y_pos, y_pos], color='gray', linewidth=0.5, linestyle=':')
            y_pos -= line_height * 1.3

        plt.tight_layout()
        save_path = f'results/visualizations/pred_img_{idx+1}_{img_id}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved visualization {idx+1}/{len(samples)} -> {save_path}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check project structure to find correct dataset
    dataset_type = 'vqa2_full'
    
    test_json = f'../Data_prep/data/annotations/{dataset_type}/test.json'
    if not os.path.exists(test_json):
        # Fallback 1: check full dataset
        test_json = '../Data_prep/data/annotations/full/test.json'
        
    if not os.path.exists(test_json):
        print(f"Error: Could not find dataset JSON at {test_json}")
        return

    img_dir = '../Data_prep/data/images'

    models, q_vocab, a_vocab = load_models_and_vocabs(device)
    if not models:
        return
        
    # Get 10 random images with up to 4 questions each
    samples = get_sample_data(test_json, num_images=10, questions_per_image=4)
    
    # Generate charts
    visualize_results(models, q_vocab, a_vocab, samples, img_dir, device)
    print("\nAll done! Check VQA_Workspace/VQA_Model/results/visualizations/ folder.")

if __name__ == '__main__':
    main()
