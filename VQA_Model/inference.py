"""
Inference script for single image VQA
Load trained model and answer questions about an image
"""

import torch
import argparse
from PIL import Image

from data import Vocabulary, get_val_transforms
from models import create_model_variant


def load_model(model_id: int, checkpoint_path: str, device: str = 'cuda'):
    """Load trained model"""
    # Load vocabularies
    q_vocab = Vocabulary.load('data/question_vocab.json')
    a_vocab = Vocabulary.load('data/answer_vocab.json')
    
    # Create model
    model = create_model_variant(
        model_id=model_id,
        question_vocab_size=len(q_vocab),
        answer_vocab_size=len(a_vocab)
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded Model {model_id} from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"  Val Accuracy: {checkpoint['metrics']['accuracy']:.4f}")
    
    return model, q_vocab, a_vocab


def answer_question(model, image_path: str, question: str, q_vocab, a_vocab, device: str = 'cuda'):
    """Answer a question about an image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_val_transforms(use_pretrained=True)
    image_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    
    # Encode question
    question_tokens = q_vocab.encode(question)
    question_tensor = torch.tensor([question_tokens], dtype=torch.long).to(device)  # (1, len)
    question_length = torch.tensor([len(question_tokens)])
    
    # Generate answer
    with torch.no_grad():
        predictions, attention_weights = model.generate_answer(
            image_tensor,
            question_tensor,
            question_length,
            max_len=10
        )
    
    # Decode answer
    answer_tokens = predictions[0].cpu().tolist()
    answer = a_vocab.decode(answer_tokens, skip_special_tokens=True)
    
    return answer, attention_weights[0].item() if attention_weights is not None else None


def interactive_mode(model, q_vocab, a_vocab, device: str = 'cuda'):
    """Interactive question answering"""
    print("\n" + "="*80)
    print("INTERACTIVE VQA MODE")
    print("="*80)
    print("Commands:")
    print("  'image <path>' - Load new image")
    print("  'quit' - Exit")
    print("  Or just type a question")
    print("="*80)
    
    current_image = None
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.lower().startswith('image '):
            image_path = user_input[6:].strip()
            try:
                Image.open(image_path)
                current_image = image_path
                print(f"✓ Loaded image: {image_path}")
            except Exception as e:
                print(f"✗ Error loading image: {e}")
            continue
        
        if current_image is None:
            print("⚠ Please load an image first using 'image <path>'")
            continue
        
        # Answer question
        try:
            answer, attn = answer_question(model, current_image, user_input, q_vocab, a_vocab, device)
            print(f"\nQuestion: {user_input}")
            print(f"Answer: {answer}")
            if attn is not None:
                print(f"Attention: {attn:.4f}")
        except Exception as e:
            print(f"✗ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='VQA Inference')
    parser.add_argument('--model_id', type=int, required=True,
                       help='Model ID (1-8)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (default: checkpoints/model_{id}/best_model.pth)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image')
    parser.add_argument('--question', type=str, default=None,
                       help='Question to ask')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Default checkpoint path
    if args.checkpoint is None:
        args.checkpoint = f'checkpoints/model_{args.model_id}/best_model.pth'
    
    # Load model
    model, q_vocab, a_vocab = load_model(args.model_id, args.checkpoint, args.device)
    
    if args.interactive:
        # Interactive mode
        interactive_mode(model, q_vocab, a_vocab, args.device)
    elif args.image and args.question:
        # Single question mode
        answer, attn = answer_question(model, args.image, args.question, q_vocab, a_vocab, args.device)
        
        print("\n" + "="*80)
        print(f"Image: {args.image}")
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
        if attn is not None:
            print(f"Attention: {attn:.4f}")
        print("="*80)
    else:
        print("Error: Provide --image and --question, or use --interactive mode")


if __name__ == "__main__":
    main()
