"""
Test script for Phase 1: Data Pipeline
"""

import sys
sys.path.append('.')

from data.vocab import Vocabulary
from data.dataset import VQADataset, collate_fn
from torch.utils.data import DataLoader

# Paths
train_json = r"c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\Data_prep\data\annotations\train.json"
image_dir = r"c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\Data_prep\data\images"
q_vocab_path = r"c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\VQA_Model\data\question_vocab.json"
a_vocab_path = r"c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\VQA_Model\data\answer_vocab.json"

print("="*60)
print("TESTING VQA DATASET")
print("="*60)

# Load vocabularies
print("\n1. Loading vocabularies...")
question_vocab = Vocabulary.load(q_vocab_path)
answer_vocab = Vocabulary.load(a_vocab_path)

# Create dataset
print("\n2. Creating dataset...")
dataset = VQADataset(
    train_json, image_dir, question_vocab, answer_vocab,
    max_question_len=20, max_answer_len=10, use_pretrained=True, is_training=True
)

# Test single sample
print("\n3. Testing single sample...")
sample = dataset[0]
print(f"  Image shape: {sample['image'].shape}")
print(f"  Question shape: {sample['question'].shape}")
print(f"  Answer shape: {sample['answer'].shape}")
print(f"  Question: '{sample['question_text']}'")
print(f"  Answer: '{sample['answer_text']}'")
print(f"  Question type: {sample['question_type']}")
print(f"  Tier: {sample['tier']}")

# Test dataloader
print("\n4. Testing dataloader...")
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)
batch = next(iter(loader))
print(f"  Batch images shape: {batch['images'].shape}")
print(f"  Batch questions shape: {batch['questions'].shape}")
print(f"  Batch answers shape: {batch['answers'].shape}")
print(f"  Batch size: {len(batch['image_ids'])}")

print("\n" + "="*60)
print("✅ PHASE 1 COMPLETE: DATA PIPELINE WORKING!")
print("="*60)
