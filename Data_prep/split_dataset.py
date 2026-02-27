"""
Dataset Splitting Script
========================
Creates TWO dataset versions:
1. FULL: All 10 animal types (70/15/15 split)
2. SMALL: Only dog + cat (70/15/15 split) - for quick experiments

Usage:
    python split_dataset.py
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_ANNOTATIONS = Path("data/annotations/annotations_complete_v2.json")

# Output directories
FULL_OUTPUT_DIR = Path("data/annotations/full")
SMALL_OUTPUT_DIR = Path("data/annotations/small")

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Small dataset animals
SMALL_ANIMALS = ['dog', 'cat']

# All animals
ALL_ANIMALS = ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_annotations(path: Path):
    """Load annotations from JSON file."""
    print(f"📂 Loading annotations from: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} Q&A pairs")
    return data


def group_by_image(annotations):
    """Group Q&A pairs by image_id."""
    image_groups = defaultdict(list)
    for qa in annotations:
        image_groups[qa['image_id']].append(qa)
    return image_groups


def get_animal_type(qa_list):
    """Get animal type from Q&A list."""
    for qa in qa_list:
        if qa['question_type'] == 'animal_recognition':
            return qa['answer']
    return None


def split_images(image_ids, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Split image IDs into train/val/test sets.
    
    Args:
        image_ids: List of image IDs
        train_ratio: Training set ratio (e.g., 0.70)
        val_ratio: Validation set ratio (e.g., 0.15)
        test_ratio: Test set ratio (e.g., 0.15)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    random.seed(seed)
    
    # Shuffle
    shuffled_ids = list(image_ids)
    random.shuffle(shuffled_ids)
    
    # Calculate split points
    n_total = len(shuffled_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split
    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]
    
    return train_ids, val_ids, test_ids


def create_split_files(image_groups, train_ids, val_ids, test_ids, output_dir):
    """
    Create train.json, val.json, test.json files.
    
    Args:
        image_groups: Dictionary mapping image_id to list of Q&A pairs
        train_ids: List of training image IDs
        val_ids: List of validation image IDs
        test_ids: List of test image IDs
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect Q&A pairs for each split
    train_data = []
    val_data = []
    test_data = []
    
    for img_id in train_ids:
        train_data.extend(image_groups[img_id])
    
    for img_id in val_ids:
        val_data.extend(image_groups[img_id])
    
    for img_id in test_ids:
        test_data.extend(image_groups[img_id])
    
    # Save to files
    splits = {
        'train.json': train_data,
        'val.json': val_data,
        'test.json': test_data
    }
    
    for filename, data in splits.items():
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✅ {filename}: {len(data)} Q&A pairs ({len(data)//5} images)")
    
    return len(train_data), len(val_data), len(test_data)


# ============================================================================
# MAIN SPLITTING LOGIC
# ============================================================================

def split_dataset():
    """Main function to split dataset into FULL and SMALL versions."""
    
    print("=" * 80)
    print("DATASET SPLITTING")
    print("=" * 80)
    print()
    
    # Load annotations
    annotations = load_annotations(INPUT_ANNOTATIONS)
    
    # Group by image
    print("\n📊 Grouping by image...")
    image_groups = group_by_image(annotations)
    print(f"✅ Found {len(image_groups)} unique images")
    
    # Categorize images by animal type
    print("\n🔍 Categorizing by animal type...")
    animal_images = defaultdict(list)
    
    for img_id, qa_list in image_groups.items():
        animal = get_animal_type(qa_list)
        if animal:
            animal_images[animal].append(img_id)
    
    print("\nAnimal distribution:")
    for animal in sorted(animal_images.keys()):
        count = len(animal_images[animal])
        print(f"  - {animal:10s}: {count:5d} images")
    
    # ========================================================================
    # SPLIT 1: FULL DATASET (All animals)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SPLIT 1: FULL DATASET (All 10 animals)")
    print("=" * 80)
    
    all_image_ids = list(image_groups.keys())
    print(f"\n📊 Total images: {len(all_image_ids)}")
    print(f"   Split ratio: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    
    train_ids_full, val_ids_full, test_ids_full = split_images(
        all_image_ids, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    
    print(f"\n💾 Saving to: {FULL_OUTPUT_DIR}")
    n_train, n_val, n_test = create_split_files(
        image_groups, train_ids_full, val_ids_full, test_ids_full, FULL_OUTPUT_DIR
    )
    
    print(f"\n✅ FULL dataset created:")
    print(f"   Train: {n_train:6d} Q&A ({len(train_ids_full):5d} images)")
    print(f"   Val:   {n_val:6d} Q&A ({len(val_ids_full):5d} images)")
    print(f"   Test:  {n_test:6d} Q&A ({len(test_ids_full):5d} images)")
    print(f"   Total: {n_train + n_val + n_test:6d} Q&A ({len(all_image_ids):5d} images)")
    
    # ========================================================================
    # SPLIT 2: SMALL DATASET (Dog + Cat only)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(f"SPLIT 2: SMALL DATASET (Only {' + '.join(SMALL_ANIMALS)})")
    print("=" * 80)
    
    # Filter only dog and cat images
    small_image_ids = []
    for animal in SMALL_ANIMALS:
        small_image_ids.extend(animal_images[animal])
    
    print(f"\n📊 Total images: {len(small_image_ids)}")
    for animal in SMALL_ANIMALS:
        count = len(animal_images[animal])
        print(f"   - {animal}: {count} images")
    print(f"   Split ratio: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    
    train_ids_small, val_ids_small, test_ids_small = split_images(
        small_image_ids, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    
    print(f"\n💾 Saving to: {SMALL_OUTPUT_DIR}")
    n_train_s, n_val_s, n_test_s = create_split_files(
        image_groups, train_ids_small, val_ids_small, test_ids_small, SMALL_OUTPUT_DIR
    )
    
    print(f"\n✅ SMALL dataset created:")
    print(f"   Train: {n_train_s:6d} Q&A ({len(train_ids_small):5d} images)")
    print(f"   Val:   {n_val_s:6d} Q&A ({len(val_ids_small):5d} images)")
    print(f"   Test:  {n_test_s:6d} Q&A ({len(test_ids_small):5d} images)")
    print(f"   Total: {n_train_s + n_val_s + n_test_s:6d} Q&A ({len(small_image_ids):5d} images)")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print()
    print("FULL Dataset (10 animals):")
    print(f"  Location: {FULL_OUTPUT_DIR}")
    print(f"  Images:   {len(all_image_ids):,}")
    print(f"  Q&A:      {n_train + n_val + n_test:,}")
    print()
    print("SMALL Dataset (dog + cat):")
    print(f"  Location: {SMALL_OUTPUT_DIR}")
    print(f"  Images:   {len(small_image_ids):,}")
    print(f"  Q&A:      {n_train_s + n_val_s + n_test_s:,}")
    print(f"  Speedup:  ~{len(all_image_ids) / len(small_image_ids):.1f}x faster training")
    print()
    print("=" * 80)
    print("✅ DATASET SPLITTING COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Use SMALL dataset for quick experiments and debugging")
    print("2. Use FULL dataset for final model training")
    print("3. Rebuild vocabularies for each dataset")
    print()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    split_dataset()
