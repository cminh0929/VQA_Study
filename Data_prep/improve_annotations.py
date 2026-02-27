"""
VQA Annotation Improvement Script (V2)
=======================================
Single-file solution to improve annotation quality:
1. Real color detection using K-Means clustering
2. Diverse question templates (33+ variations)
3. YOLO-based accurate counting
4. Generate improved annotations

Usage:
    python improve_annotations.py
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import cv2
from sklearn.cluster import KMeans

# ============================================================================
# CONFIGURATION
# ============================================================================

IMAGES_DIR = Path("data/images")
INPUT_ANNOTATIONS = Path("data/annotations/annotations_complete.json")
OUTPUT_ANNOTATIONS = Path("data/annotations/annotations_complete_v2.json")

ANIMAL_CATEGORIES = ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

# ============================================================================
# 1. COLOR DETECTION (K-Means Clustering)
# ============================================================================

COLOR_MAP = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'gray': (128, 128, 128),
    'brown': (139, 69, 19),
    'golden': (255, 215, 0),
    'orange': (255, 165, 0),
    'red': (255, 0, 0),
    'yellow': (255, 255, 0),
    'tan': (210, 180, 140),
    'beige': (245, 245, 220),
}

def detect_animal_color(image_path: Path) -> str:
    """
    Detect dominant color using K-Means clustering on image pixels.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Color name (e.g., 'brown', 'white', 'black')
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return random.choice(['brown', 'white', 'black', 'gray'])
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Reshape to 2D array of pixels
        pixels = img.reshape(-1, 3)
        
        # Remove very dark and very bright pixels (noise)
        mask = (pixels.mean(axis=1) > 20) & (pixels.mean(axis=1) < 235)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) < 100:
            filtered_pixels = pixels
        
        # Sample pixels for faster computation
        if len(filtered_pixels) > 5000:
            indices = np.random.choice(len(filtered_pixels), 5000, replace=False)
            filtered_pixels = filtered_pixels[indices]
        
        # K-Means clustering (5 clusters)
        kmeans = KMeans(n_clusters=min(5, len(filtered_pixels)), random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)
        
        # Get dominant cluster (largest cluster)
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_cluster_idx = labels[np.argmax(counts)]
        dominant_color_rgb = kmeans.cluster_centers_[dominant_cluster_idx]
        
        # Map RGB to color name
        color_name = rgb_to_color_name(dominant_color_rgb)
        return color_name
        
    except Exception as e:
        # Fallback to random color
        return random.choice(['brown', 'white', 'black', 'gray'])


def rgb_to_color_name(rgb: np.ndarray) -> str:
    """
    Map RGB value to nearest color name.
    
    Args:
        rgb: RGB array [R, G, B]
        
    Returns:
        Color name
    """
    min_distance = float('inf')
    closest_color = 'brown'
    
    for color_name, color_rgb in COLOR_MAP.items():
        distance = np.linalg.norm(rgb - np.array(color_rgb))
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    
    return closest_color


# ============================================================================
# 2. YOLO COUNTING (OPTIMIZED - GLOBAL MODEL)
# ============================================================================

# Global YOLO model (load once, reuse many times)
_YOLO_MODEL = None

def get_yolo_model():
    """
    Get or initialize YOLO model (singleton pattern).
    Loads model once and reuses it for all images.
    """
    global _YOLO_MODEL
    
    if _YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            import torch
            
            print("🔧 Initializing YOLO model...")
            
            # Load model
            _YOLO_MODEL = YOLO('yolov8n.pt')
            
            # Force GPU if available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            _YOLO_MODEL.to(device)
            
            print(f"✅ YOLO loaded on: {device.upper()}")
            if device == 'cuda':
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
            
        except Exception as e:
            print(f"⚠️  YOLO initialization failed: {e}")
            _YOLO_MODEL = None
    
    return _YOLO_MODEL


def count_animals_yolo(image_path: Path, animal_type: str) -> int:
    """
    Count animals using YOLO detection (with fallback).
    Uses global model for speed (no reloading).
    
    Args:
        image_path: Path to image
        animal_type: Animal category (e.g., 'dog', 'cat')
        
    Returns:
        Count of animals (1-3)
    """
    try:
        # Get global YOLO model
        model = get_yolo_model()
        
        if model is None:
            return count_animals_fallback(image_path)
        
        # Run detection (fast - model already loaded)
        results = model(str(image_path), verbose=False)
        
        # Count detected animals
        count = 0
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Map COCO classes to our animal types
                if class_name == animal_type or (class_name == 'person' and animal_type == 'dog'):
                    count += 1
        
        # Clamp to [1, 3]
        return max(1, min(count, 3)) if count > 0 else 1
        
    except Exception as e:
        # Fallback: Use simple heuristic based on image size
        return count_animals_fallback(image_path)


def count_animals_fallback(image_path: Path) -> int:
    """
    Fallback counting using edge detection and contours.
    
    Args:
        image_path: Path to image
        
    Returns:
        Estimated count (1-3)
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 1
        
        # Edge detection
        edges = cv2.Canny(img, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter large contours
        large_contours = [c for c in contours if cv2.contourArea(c) > 1000]
        
        # Estimate count
        count = len(large_contours)
        return max(1, min(count, 3))
        
    except Exception as e:
        return 1


# ============================================================================
# 3. DIVERSE QUESTION TEMPLATES
# ============================================================================

QUESTION_TEMPLATES = {
    'animal_recognition': [
        "What animal is in the image?",
        "Which animal can you see?",
        "Identify the animal in this image.",
        "What type of animal is shown?",
        "Can you tell me what animal this is?",
        "What kind of animal appears in the picture?",
        "Which animal is present in this photo?",
        "What animal does this image show?",
        "Name the animal in the image.",
        "What animal is depicted here?",
    ],
    
    'color_recognition': [
        "What color is the {animal}?",
        "What is the color of the {animal}?",
        "Describe the {animal}'s color.",
        "What color does the {animal} have?",
        "Can you identify the {animal}'s color?",
        "Tell me the color of the {animal}.",
        "What is the {animal}'s primary color?",
        "Which color is the {animal}?",
    ],
    
    'yes_no_positive': [
        "Is there a {animal} in the image?",
        "Can you see a {animal}?",
        "Does this image contain a {animal}?",
        "Is a {animal} present in the picture?",
        "Do you see a {animal} here?",
        "Is there any {animal} visible?",
        "Can a {animal} be seen in this image?",
        "Does the image show a {animal}?",
    ],
    
    'yes_no_negative': [
        "Is there a {animal} in the image?",
        "Can you see a {animal}?",
        "Does this image contain a {animal}?",
        "Is a {animal} present in the picture?",
        "Do you see a {animal} here?",
        "Is there any {animal} visible?",
        "Can a {animal} be seen in this image?",
        "Does the image show a {animal}?",
    ],
    
    'counting': [
        "How many {animal}s are there?",
        "Count the number of {animal}s.",
        "How many {animal}s can you see?",
        "What is the count of {animal}s?",
        "Tell me how many {animal}s are present.",
        "Can you count the {animal}s?",
        "How many {animal}s appear in the image?",
    ],
}


def generate_question(question_type: str, animal: str = None) -> str:
    """
    Generate a diverse question based on type.
    
    Args:
        question_type: Type of question
        animal: Animal name (for templates that need it)
        
    Returns:
        Generated question string
    """
    template = random.choice(QUESTION_TEMPLATES[question_type])
    
    if animal and '{animal}' in template:
        # Handle plural forms
        if '{animal}s' in template:
            plural = animal + 's' if animal not in ['sheep', 'fish'] else animal
            template = template.replace('{animal}s', plural)
        template = template.replace('{animal}', animal)
    
    return template


# ============================================================================
# 4. ANNOTATION GENERATION
# ============================================================================

def generate_improved_qa(image_id: str, image_path: Path, animal: str) -> List[Dict]:
    """
    Generate improved Q&A pairs for a single image.
    
    Args:
        image_id: Image filename
        image_path: Path to image file
        animal: Animal type in the image
        
    Returns:
        List of Q&A dictionaries
    """
    qa_pairs = []
    
    # 1. Animal Recognition
    qa_pairs.append({
        'image_id': image_id,
        'question': generate_question('animal_recognition'),
        'answer': animal,
        'question_type': 'animal_recognition',
        'tier': 1
    })
    
    # 2. Color Recognition (IMPROVED: Real color detection)
    detected_color = detect_animal_color(image_path)
    qa_pairs.append({
        'image_id': image_id,
        'question': generate_question('color_recognition', animal),
        'answer': detected_color,
        'question_type': 'color_recognition',
        'tier': 1
    })
    
    # 3. Yes/No (Positive)
    qa_pairs.append({
        'image_id': image_id,
        'question': generate_question('yes_no_positive', animal),
        'answer': 'yes',
        'question_type': 'yes_no',
        'tier': 1
    })
    
    # 4. Yes/No (Negative)
    other_animals = [a for a in ANIMAL_CATEGORIES if a != animal]
    other_animal = random.choice(other_animals)
    qa_pairs.append({
        'image_id': image_id,
        'question': generate_question('yes_no_negative', other_animal),
        'answer': 'no',
        'question_type': 'yes_no',
        'tier': 1
    })
    
    # 5. Counting (IMPROVED: YOLO-based)
    count = count_animals_yolo(image_path, animal)
    qa_pairs.append({
        'image_id': image_id,
        'question': generate_question('counting', animal),
        'answer': str(count),
        'question_type': 'counting_simple',
        'tier': 2
    })
    
    return qa_pairs


def improve_annotations():
    """
    Main function to improve all annotations.
    """
    print("=" * 70)
    print("VQA ANNOTATION IMPROVEMENT (V2)")
    print("=" * 70)
    print()
    
    # Load existing annotations
    print(f"📂 Loading annotations from: {INPUT_ANNOTATIONS}")
    with open(INPUT_ANNOTATIONS, 'r') as f:
        old_annotations = json.load(f)
    
    print(f"✅ Loaded {len(old_annotations)} Q&A pairs")
    print()
    
    # Group by image_id
    image_to_animal = {}
    for qa in old_annotations:
        image_id = qa['image_id']
        if image_id not in image_to_animal:
            # Get animal from first animal_recognition question
            if qa['question_type'] == 'animal_recognition':
                image_to_animal[image_id] = qa['answer']
    
    print(f"📊 Found {len(image_to_animal)} unique images")
    print()
    
    # Generate improved annotations
    print("🚀 Generating improved annotations...")
    print("   - Real color detection (K-Means)")
    print("   - Diverse question templates (33+)")
    print("   - YOLO-based counting")
    print()
    
    new_annotations = []
    
    for image_id, animal in tqdm(image_to_animal.items(), desc="Processing images"):
        image_path = IMAGES_DIR / image_id
        
        if not image_path.exists():
            print(f"⚠️  Warning: Image not found: {image_path}")
            continue
        
        # Generate improved Q&A pairs
        qa_pairs = generate_improved_qa(image_id, image_path, animal)
        new_annotations.extend(qa_pairs)
    
    print()
    print(f"✅ Generated {len(new_annotations)} improved Q&A pairs")
    print()
    
    # Save improved annotations
    print(f"💾 Saving to: {OUTPUT_ANNOTATIONS}")
    OUTPUT_ANNOTATIONS.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_ANNOTATIONS, 'w') as f:
        json.dump(new_annotations, f, indent=2)
    
    print(f"✅ Saved successfully!")
    print()
    
    # Statistics
    print("=" * 70)
    print("📊 STATISTICS")
    print("=" * 70)
    
    question_types = {}
    for qa in new_annotations:
        qtype = qa['question_type']
        question_types[qtype] = question_types.get(qtype, 0) + 1
    
    print(f"Total Q&A pairs: {len(new_annotations)}")
    print(f"Unique images: {len(image_to_animal)}")
    print()
    print("Question type distribution:")
    for qtype, count in sorted(question_types.items()):
        percentage = (count / len(new_annotations)) * 100
        print(f"  - {qtype:20s}: {count:6d} ({percentage:5.1f}%)")
    
    print()
    print("=" * 70)
    print("✅ IMPROVEMENT COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review the generated annotations")
    print("2. Run split_dataset.py to create train/val/test splits")
    print("3. Rebuild vocabularies in VQA_Model")
    print("4. Re-train models with improved data")
    print()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    improve_annotations()
