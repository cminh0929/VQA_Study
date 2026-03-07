"""
PyTorch Dataset for VQA
Loads images, questions, and answers for training/evaluation
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Tuple

from .vocab import Vocabulary
from .transforms import ImageTransform


class VQADataset(Dataset):
    """VQA Dataset class"""
    
    def __init__(
        self,
        annotations_file: str,
        image_dir: str,
        question_vocab: Vocabulary,
        answer_vocab: Vocabulary,
        max_question_len: int = 20,
        max_answer_len: int = 10,
        image_size: int = 224,
        use_pretrained: bool = True,
        is_training: bool = True
    ):
        """
        Args:
            annotations_file: Path to JSON file (train.json/val.json/test.json)
            image_dir: Directory containing images
            question_vocab: Question vocabulary
            answer_vocab: Answer vocabulary
            max_question_len: Maximum question length
            max_answer_len: Maximum answer length
            image_size: Image size for CNN input
            use_pretrained: Whether using pretrained CNN
            is_training: Whether in training mode (affects augmentation)
        """
        self.image_dir = image_dir
        self.question_vocab = question_vocab
        self.answer_vocab = answer_vocab
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.is_training = is_training
        
        # Load annotations
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # Image transforms
        self.transform = ImageTransform(image_size=image_size, use_pretrained=use_pretrained)
        

    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            Dictionary containing:
                - image: Tensor (3, H, W)
                - question: Tensor (max_question_len,)
                - answer: Tensor (max_answer_len,)
                - question_length: int
                - answer_length: int
                - image_id: str
                - question_text: str
                - answer_text: str
        """
        ann = self.annotations[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, ann['image_id'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image, is_training=self.is_training)
        
        # Encode question
        question_text = ann['question']
        question_indices = self.question_vocab.encode(
            question_text,
            max_length=self.max_question_len,
            add_sos_eos=False
        )
        question_length = min(len(question_text.split()), self.max_question_len)
        
        # Encode answer (with SOS/EOS for decoder)
        answer_text = ann['answer']
        answer_indices = self.answer_vocab.encode(
            answer_text,
            max_length=self.max_answer_len,
            add_sos_eos=True
        )
        answer_length = min(len(answer_text.split()) + 2, self.max_answer_len)  # +2 for SOS/EOS
        
        return {
            'image': image,
            'question': torch.tensor(question_indices, dtype=torch.long),
            'answer': torch.tensor(answer_indices, dtype=torch.long),
            'question_length': question_length,
            'answer_length': answer_length,
            'image_id': ann['image_id'],
            'question_text': question_text,
            'answer_text': answer_text,
            'question_type': ann.get('question_type', 'unknown'),
            'tier': ann.get('tier', 1)
        }


def collate_fn(batch):
    """
    Custom collate function for DataLoader
    Handles variable-length sequences
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Batched dictionary
    """
    # Stack images
    images = torch.stack([item['image'] for item in batch])
    
    # Stack questions and answers
    questions = torch.stack([item['question'] for item in batch])
    answers = torch.stack([item['answer'] for item in batch])
    
    # Get lengths
    question_lengths = torch.tensor([item['question_length'] for item in batch])
    answer_lengths = torch.tensor([item['answer_length'] for item in batch])
    
    # Get metadata
    image_ids = [item['image_id'] for item in batch]
    question_texts = [item['question_text'] for item in batch]
    answer_texts = [item['answer_text'] for item in batch]
    question_types = [item['question_type'] for item in batch]
    tiers = [item['tier'] for item in batch]
    
    return {
        'images': images,
        'questions': questions,
        'answers': answers,
        'question_lengths': question_lengths,
        'answer_lengths': answer_lengths,
        'image_ids': image_ids,
        'question_texts': question_texts,
        'answer_texts': answer_texts,
        'question_types': question_types,
        'tiers': tiers
    }


def create_dataloaders(
    train_json: str,
    val_json: str,
    test_json: str,
    image_dir: str,
    question_vocab: Vocabulary,
    answer_vocab: Vocabulary,
    batch_size: int = 32,
    num_workers: int = 4,
    max_question_len: int = 20,
    max_answer_len: int = 10,
    image_size: int = 224,
    use_pretrained: bool = True
):
    """
    Create train/val/test dataloaders
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = VQADataset(
        train_json, image_dir, question_vocab, answer_vocab,
        max_question_len, max_answer_len, image_size, use_pretrained, is_training=True
    )
    
    val_dataset = VQADataset(
        val_json, image_dir, question_vocab, answer_vocab,
        max_question_len, max_answer_len, image_size, use_pretrained, is_training=False
    )
    
    test_dataset = VQADataset(
        test_json, image_dir, question_vocab, answer_vocab,
        max_question_len, max_answer_len, image_size, use_pretrained, is_training=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"\nCreated DataLoaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Paths
    train_json = r"c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\Data_prep\data\annotations\train.json"
    image_dir = r"c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\Data_prep\data\images"
    q_vocab_path = r"c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\VQA_Model\data\question_vocab.json"
    a_vocab_path = r"c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\VQA_Model\data\answer_vocab.json"
    
    # Load vocabularies
    print("Loading vocabularies...")
    question_vocab = Vocabulary.load(q_vocab_path)
    answer_vocab = Vocabulary.load(a_vocab_path)
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = VQADataset(
        train_json, image_dir, question_vocab, answer_vocab,
        max_question_len=20, max_answer_len=10, use_pretrained=True, is_training=True
    )
    
    # Test single sample
    print("\nTesting single sample...")
    sample = dataset[0]
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Question shape: {sample['question'].shape}")
    print(f"  Answer shape: {sample['answer'].shape}")
    print(f"  Question: {sample['question_text']}")
    print(f"  Answer: {sample['answer_text']}")
    print(f"  Question encoded: {sample['question'][:10].tolist()}...")
    print(f"  Answer encoded: {sample['answer'].tolist()}")
    
    # Test dataloader
    print("\nTesting dataloader...")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"  Batch images shape: {batch['images'].shape}")
    print(f"  Batch questions shape: {batch['questions'].shape}")
    print(f"  Batch answers shape: {batch['answers'].shape}")
    
    print("\n✓ Dataset working correctly!")
