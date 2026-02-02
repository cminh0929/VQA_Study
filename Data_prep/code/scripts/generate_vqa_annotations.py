"""
Generate VQA Annotations from COCO Dataset

Tạo Q&A pairs cho 4 loại câu hỏi:
1. Animal Recognition (MỤC TIÊU CHÍNH)
2. Color Recognition (MỤC TIÊU CHÍNH)
3. Yes/No Questions (MỤC TIÊU PHỤ)
4. Simple Counting (MỤC TIÊU THỬ NGHIỆM - Pretrained only)
"""

import json
import os
import random
from collections import defaultdict
from tqdm import tqdm

# ===== CẤU HÌNH =====
COCO_ANNOTATIONS_PATH = r'C:\Users\cminh\Downloads\annotations_extracted\annotations\instances_train2017.json'
IMAGE_DIR = r'c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\Data_prep\data\images'
OUTPUT_DIR = r'c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\Data_prep\data\annotations'

# Animal categories (COCO IDs)
ANIMAL_IDS = {
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe"
}

# Predefined color mapping (simplified)
ANIMAL_COLORS = {
    "bird": ["blue", "red", "yellow", "brown", "white", "black"],
    "cat": ["black", "white", "orange", "gray", "brown"],
    "dog": ["brown", "black", "white", "golden", "gray"],
    "horse": ["brown", "black", "white"],
    "sheep": ["white", "black"],
    "cow": ["black", "white", "brown"],
    "elephant": ["gray"],
    "bear": ["brown", "black", "white"],
    "zebra": ["black and white"],
    "giraffe": ["yellow and brown"]
}

# Other animals for negative Yes/No questions
ALL_ANIMALS = list(ANIMAL_IDS.values())

def load_coco_data():
    """Load COCO annotations"""
    print("Loading COCO annotations...")
    with open(COCO_ANNOTATIONS_PATH, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def build_image_annotations(coco_data):
    """Build mapping: image_id -> list of annotations"""
    print("Building image annotations mapping...")
    image_annotations = defaultdict(list)
    
    for ann in coco_data['annotations']:
        image_annotations[ann['image_id']].append(ann)
    
    return image_annotations

def get_image_filename_mapping(coco_data):
    """Build mapping: image_id -> filename"""
    return {img['id']: img['file_name'] for img in coco_data['images']}

def generate_qa_for_image(filename, animals_count):
    """
    Generate Q&A pairs for one image
    
    Args:
        filename: Image filename
        animals_count: Dict {animal_name: count}
    
    Returns:
        List of Q&A pairs
    """
    qa_pairs = []
    
    # Skip if no animals
    if not animals_count:
        return qa_pairs
    
    # Get primary animal (most frequent)
    primary_animal = max(animals_count.items(), key=lambda x: x[1])[0]
    primary_count = animals_count[primary_animal]
    
    # ===== 1. ANIMAL RECOGNITION (MỤC TIÊU CHÍNH) =====
    qa_pairs.append({
        "image_id": filename,
        "question": "What animal is in the image?",
        "answer": primary_animal,
        "question_type": "animal_recognition",
        "tier": 1  # All models
    })
    
    # ===== 2. COLOR RECOGNITION (MỤC TIÊU CHÍNH) =====
    # Random color from predefined list
    possible_colors = ANIMAL_COLORS.get(primary_animal, ["brown"])
    color = random.choice(possible_colors)
    
    qa_pairs.append({
        "image_id": filename,
        "question": f"What color is the {primary_animal}?",
        "answer": color,
        "question_type": "color_recognition",
        "tier": 1  # All models
    })
    
    # ===== 3. YES/NO QUESTIONS (MỤC TIÊU PHỤ) =====
    # Positive: Animal present
    qa_pairs.append({
        "image_id": filename,
        "question": f"Is there a {primary_animal} in the image?",
        "answer": "yes",
        "question_type": "yes_no",
        "tier": 1  # All models
    })
    
    # Negative: Random animal NOT present
    absent_animals = [a for a in ALL_ANIMALS if a not in animals_count]
    if absent_animals:
        random_absent = random.choice(absent_animals)
        qa_pairs.append({
            "image_id": filename,
            "question": f"Is there a {random_absent} in the image?",
            "answer": "no",
            "question_type": "yes_no",
            "tier": 1  # All models
        })
    
    # ===== 4. SIMPLE COUNTING (MỤC TIÊU THỬ NGHIỆM) =====
    # Only for counts 1-3 (pretrained models only)
    if 1 <= primary_count <= 3:
        qa_pairs.append({
            "image_id": filename,
            "question": f"How many {primary_animal}s are there?",
            "answer": str(primary_count),
            "question_type": "counting_simple",
            "tier": 2  # Pretrained only
        })
    
    return qa_pairs

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load COCO data
    coco_data = load_coco_data()
    image_annotations = build_image_annotations(coco_data)
    filename_mapping = get_image_filename_mapping(coco_data)
    
    # Get list of images in data/images
    print("Scanning images directory...")
    image_files = set(os.listdir(IMAGE_DIR))
    print(f"Found {len(image_files)} images in data/images/")
    
    # Generate Q&A pairs
    all_annotations = []
    processed_count = 0
    skipped_count = 0
    
    print("Generating Q&A pairs...")
    for img_info in tqdm(coco_data['images'], desc="Processing"):
        img_id = img_info['id']
        filename = img_info['file_name']
        
        # Skip if image not in our filtered set
        if filename not in image_files:
            skipped_count += 1
            continue
        
        # Count animals in this image
        animals_count = defaultdict(int)
        for ann in image_annotations[img_id]:
            cat_id = ann['category_id']
            if cat_id in ANIMAL_IDS:
                animal_name = ANIMAL_IDS[cat_id]
                animals_count[animal_name] += 1
        
        # Generate Q&A pairs
        qa_pairs = generate_qa_for_image(filename, animals_count)
        all_annotations.extend(qa_pairs)
        
        if qa_pairs:
            processed_count += 1
    
    # Save annotations
    output_path = os.path.join(OUTPUT_DIR, 'annotations.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, indent=2, ensure_ascii=False)
    
    # Statistics
    print("\n" + "="*60)
    print("HOÀN THÀNH TẠO ANNOTATIONS")
    print("="*60)
    print(f"Tổng số ảnh xử lý: {processed_count}")
    print(f"Tổng số ảnh bỏ qua: {skipped_count}")
    print(f"Tổng số Q&A pairs: {len(all_annotations)}")
    print(f"\nPhân loại theo question type:")
    
    type_counts = defaultdict(int)
    tier_counts = defaultdict(int)
    for qa in all_annotations:
        type_counts[qa['question_type']] += 1
        tier_counts[qa['tier']] += 1
    
    for qtype, count in sorted(type_counts.items()):
        print(f"  - {qtype}: {count}")
    
    print(f"\nPhân loại theo tier:")
    print(f"  - Tier 1 (All models): {tier_counts[1]}")
    print(f"  - Tier 2 (Pretrained only): {tier_counts[2]}")
    
    print(f"\nFile saved: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
