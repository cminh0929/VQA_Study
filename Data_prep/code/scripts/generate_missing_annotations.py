"""
Generate annotations for images that are missing from annotations.json
Uses YOLO to detect animals and create Q&A pairs
"""

import json
import os
import random
from collections import defaultdict
from ultralytics import YOLO
from tqdm import tqdm

# ===== CẤU HÌNH =====
IMAGE_DIR = r'c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\Data_prep\data\images'
ANNOTATIONS_FILE = r'c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\Data_prep\data\annotations\annotations.json'
OUTPUT_FILE = r'c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\Data_prep\data\annotations\annotations_complete.json'

# Animal categories (COCO IDs)
ANIMAL_IDS = {
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe"
}

# Predefined color mapping
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

ALL_ANIMALS = list(ANIMAL_IDS.values())

def generate_qa_for_image(filename, animals_count):
    """Generate Q&A pairs for one image"""
    qa_pairs = []
    
    if not animals_count:
        return qa_pairs
    
    # Primary animal
    primary_animal = max(animals_count.items(), key=lambda x: x[1])[0]
    primary_count = animals_count[primary_animal]
    
    # 1. Animal Recognition
    qa_pairs.append({
        "image_id": filename,
        "question": "What animal is in the image?",
        "answer": primary_animal,
        "question_type": "animal_recognition",
        "tier": 1
    })
    
    # 2. Color Recognition
    possible_colors = ANIMAL_COLORS.get(primary_animal, ["brown"])
    color = random.choice(possible_colors)
    
    qa_pairs.append({
        "image_id": filename,
        "question": f"What color is the {primary_animal}?",
        "answer": color,
        "question_type": "color_recognition",
        "tier": 1
    })
    
    # 3. Yes/No - Positive
    qa_pairs.append({
        "image_id": filename,
        "question": f"Is there a {primary_animal} in the image?",
        "answer": "yes",
        "question_type": "yes_no",
        "tier": 1
    })
    
    # 4. Yes/No - Negative
    absent_animals = [a for a in ALL_ANIMALS if a not in animals_count]
    if absent_animals:
        random_absent = random.choice(absent_animals)
        qa_pairs.append({
            "image_id": filename,
            "question": f"Is there a {random_absent} in the image?",
            "answer": "no",
            "question_type": "yes_no",
            "tier": 1
        })
    
    # 5. Simple Counting (1-3 only)
    if 1 <= primary_count <= 3:
        qa_pairs.append({
            "image_id": filename,
            "question": f"How many {primary_animal}s are there?",
            "answer": str(primary_count),
            "question_type": "counting_simple",
            "tier": 2
        })
    
    return qa_pairs

def main():
    print("Loading existing annotations...")
    with open(ANNOTATIONS_FILE, 'r') as f:
        existing_annotations = json.load(f)
    
    # Get images that already have annotations
    images_with_annotations = set(qa['image_id'] for qa in existing_annotations)
    print(f"Images with annotations: {len(images_with_annotations)}")
    
    # Get all images in folder
    all_images = set(os.listdir(IMAGE_DIR))
    print(f"Total images in folder: {len(all_images)}")
    
    # Find missing images
    missing_images = all_images - images_with_annotations
    print(f"Images without annotations: {len(missing_images)}")
    
    if len(missing_images) == 0:
        print("✅ All images already have annotations!")
        return
    
    # Load YOLO model
    print("\nLoading YOLO model...")
    model = YOLO("yolov8n.pt")
    
    # Generate annotations for missing images
    new_annotations = []
    processed_count = 0
    
    print(f"\nGenerating annotations for {len(missing_images)} images...")
    for filename in tqdm(missing_images, desc="Processing"):
        img_path = os.path.join(IMAGE_DIR, filename)
        
        # Run YOLO
        results = model(img_path, verbose=False)[0]
        
        # Count animals
        animals_count = defaultdict(int)
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if class_id in ANIMAL_IDS and conf >= 0.5:
                animal_name = ANIMAL_IDS[class_id]
                animals_count[animal_name] += 1
        
        # Generate Q&A pairs
        qa_pairs = generate_qa_for_image(filename, animals_count)
        new_annotations.extend(qa_pairs)
        
        if qa_pairs:
            processed_count += 1
    
    # Combine with existing annotations
    all_annotations = existing_annotations + new_annotations
    
    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, indent=2, ensure_ascii=False)
    
    # Statistics
    print("\n" + "="*60)
    print("HOÀN THÀNH")
    print("="*60)
    print(f"Existing annotations: {len(existing_annotations)}")
    print(f"New annotations generated: {len(new_annotations)}")
    print(f"Total annotations: {len(all_annotations)}")
    print(f"\nImages processed: {processed_count}/{len(missing_images)}")
    
    type_counts = defaultdict(int)
    tier_counts = defaultdict(int)
    for qa in new_annotations:
        type_counts[qa['question_type']] += 1
        tier_counts[qa['tier']] += 1
    
    print(f"\nNew annotations by type:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  - {qtype}: {count}")
    
    print(f"\nNew annotations by tier:")
    print(f"  - Tier 1 (All models): {tier_counts[1]}")
    print(f"  - Tier 2 (Pretrained only): {tier_counts[2]}")
    
    print(f"\nSaved to: {OUTPUT_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()
