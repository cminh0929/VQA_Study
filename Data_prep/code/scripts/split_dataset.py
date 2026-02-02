"""
Split annotations_complete.json into train/val/test sets

Strategy: Split by image_id (not Q&A pairs) to prevent data leakage
Ratio: 70% train, 15% val, 15% test
Random seed: 42 (for reproducibility)
"""

import json
import random
from collections import defaultdict

# ===== CONFIGURATION =====
INPUT_FILE = r'c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\Data_prep\data\annotations\annotations_complete.json'
OUTPUT_DIR = r'c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\Data_prep\data\annotations'

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

def main():
    print("="*60)
    print("SPLIT DATASET: Train/Val/Test")
    print("="*60)
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Load annotations
    print("\n1. Loading annotations...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_annotations = json.load(f)
    
    print(f"   Total Q&A pairs: {len(all_annotations):,}")
    
    # Group by image_id
    print("\n2. Grouping by image_id...")
    image_to_qa = defaultdict(list)
    for qa in all_annotations:
        image_to_qa[qa['image_id']].append(qa)
    
    unique_images = list(image_to_qa.keys())
    print(f"   Unique images: {len(unique_images):,}")
    print(f"   Avg Q&A per image: {len(all_annotations)/len(unique_images):.1f}")
    
    # Shuffle images
    print("\n3. Shuffling images (seed=42)...")
    random.shuffle(unique_images)
    
    # Calculate split sizes
    total_images = len(unique_images)
    train_size = int(total_images * TRAIN_RATIO)
    val_size = int(total_images * VAL_RATIO)
    test_size = total_images - train_size - val_size  # Remaining
    
    print(f"\n4. Splitting images:")
    print(f"   Train: {train_size:,} images ({train_size/total_images*100:.1f}%)")
    print(f"   Val:   {val_size:,} images ({val_size/total_images*100:.1f}%)")
    print(f"   Test:  {test_size:,} images ({test_size/total_images*100:.1f}%)")
    
    # Split images
    train_images = set(unique_images[:train_size])
    val_images = set(unique_images[train_size:train_size+val_size])
    test_images = set(unique_images[train_size+val_size:])
    
    # Validate no overlap
    assert len(train_images & val_images) == 0, "Train and Val overlap!"
    assert len(train_images & test_images) == 0, "Train and Test overlap!"
    assert len(val_images & test_images) == 0, "Val and Test overlap!"
    assert len(train_images) + len(val_images) + len(test_images) == total_images
    print("   ✓ Validation passed: No overlap between splits")
    
    # Extract Q&A for each split
    print("\n5. Extracting Q&A pairs for each split...")
    train_qa = []
    val_qa = []
    test_qa = []
    
    for qa in all_annotations:
        img_id = qa['image_id']
        if img_id in train_images:
            train_qa.append(qa)
        elif img_id in val_images:
            val_qa.append(qa)
        elif img_id in test_images:
            test_qa.append(qa)
    
    print(f"   Train Q&A: {len(train_qa):,}")
    print(f"   Val Q&A:   {len(val_qa):,}")
    print(f"   Test Q&A:  {len(test_qa):,}")
    
    # Validate total
    assert len(train_qa) + len(val_qa) + len(test_qa) == len(all_annotations)
    print("   ✓ Validation passed: All Q&A pairs accounted for")
    
    # Save splits
    print("\n6. Saving split files...")
    
    train_file = f"{OUTPUT_DIR}/train.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_qa, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved: {train_file}")
    
    val_file = f"{OUTPUT_DIR}/val.json"
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_qa, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved: {val_file}")
    
    test_file = f"{OUTPUT_DIR}/test.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_qa, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved: {test_file}")
    
    # Summary
    print("\n" + "="*60)
    print("SPLIT COMPLETE")
    print("="*60)
    print("\nSummary:")
    print(f"  Train: {len(train_images):,} images, {len(train_qa):,} Q&A pairs")
    print(f"  Val:   {len(val_images):,} images, {len(val_qa):,} Q&A pairs")
    print(f"  Test:  {len(test_images):,} images, {len(test_qa):,} Q&A pairs")
    print(f"  Total: {total_images:,} images, {len(all_annotations):,} Q&A pairs")
    
    # Statistics by question type
    from collections import Counter
    
    print("\nTrain set statistics:")
    train_types = Counter(qa['question_type'] for qa in train_qa)
    for qtype, count in sorted(train_types.items()):
        print(f"  {qtype}: {count:,}")
    
    print("\nVal set statistics:")
    val_types = Counter(qa['question_type'] for qa in val_qa)
    for qtype, count in sorted(val_types.items()):
        print(f"  {qtype}: {count:,}")
    
    print("\nTest set statistics:")
    test_types = Counter(qa['question_type'] for qa in test_qa)
    for qtype, count in sorted(test_types.items()):
        print(f"  {qtype}: {count:,}")
    
    print("\n" + "="*60)
    print("✅ Dataset split successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
