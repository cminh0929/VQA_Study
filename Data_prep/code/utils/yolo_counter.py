"""
YOLO-based Object Counting for VQA
Accurate counting using object detection instead of CNN global features
"""

import torch
import cv2
import numpy as np
from pathlib import Path


class YOLOCounter:
    """YOLO-based object counter for accurate counting"""
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.25):
        """
        Initialize YOLO counter
        
        Args:
            model_path: Path to YOLO weights
            confidence_threshold: Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.available = True
            print(f"✓ YOLO model loaded: {model_path}")
        except Exception as e:
            print(f"⚠ YOLO not available: {e}")
            print("  Falling back to heuristic counting")
            self.available = False
    
    def count_objects(self, image_path, target_class=None):
        """
        Count objects in image
        
        Args:
            image_path: Path to image
            target_class: Specific class to count (e.g., 'dog', 'cat')
                         If None, counts all detected objects
        
        Returns:
            count: Number of objects detected
            detections: List of detected objects with confidence
        """
        if not self.available:
            return self._heuristic_count(image_path)
        
        try:
            # Run YOLO detection
            results = self.model(image_path, conf=self.confidence_threshold, verbose=False)
            
            if len(results) == 0:
                return 1, []  # Default to 1 if no detection
            
            # Get detections
            boxes = results[0].boxes
            
            if boxes is None or len(boxes) == 0:
                return 1, []
            
            # Extract class names and confidences
            detections = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = results[0].names[cls_id]
                
                detections.append({
                    'class': cls_name,
                    'confidence': conf
                })
            
            # Filter by target class if specified
            if target_class:
                filtered = [d for d in detections if target_class.lower() in d['class'].lower()]
                count = len(filtered)
            else:
                count = len(detections)
            
            # Ensure count is in range [1, 3] for our dataset
            count = max(1, min(3, count))
            
            return count, detections
            
        except Exception as e:
            print(f"Error in YOLO counting: {e}")
            return self._heuristic_count(image_path)
    
    def _heuristic_count(self, image_path):
        """
        Fallback heuristic counting based on image analysis
        (Used when YOLO is not available)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return 1, []
            
            # Simple heuristic: Use connected components on edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by area (remove noise)
            min_area = (image.shape[0] * image.shape[1]) * 0.01  # 1% of image
            significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            count = len(significant_contours)
            count = max(1, min(3, count))  # Clamp to [1, 3]
            
            return count, []
            
        except Exception as e:
            print(f"Error in heuristic counting: {e}")
            return 1, []


def count_animals_in_image(image_path, animal_type=None):
    """
    Convenience function to count animals in image
    
    Args:
        image_path: Path to image
        animal_type: Type of animal to count (optional)
    
    Returns:
        count: Number of animals (1-3)
    """
    counter = YOLOCounter()
    count, _ = counter.count_objects(image_path, target_class=animal_type)
    return count


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python yolo_counter.py <image_path> [animal_type]")
        print("\nExample:")
        print("  python yolo_counter.py ../data/images/000000000001.jpg dog")
        sys.exit(1)
    
    image_path = sys.argv[1]
    animal_type = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"\nCounting objects in: {image_path}")
    if animal_type:
        print(f"Target: {animal_type}")
    print("="*60)
    
    counter = YOLOCounter()
    count, detections = counter.count_objects(image_path, animal_type)
    
    print(f"\nCount: {count}")
    
    if detections:
        print(f"\nDetections:")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det['class']} (confidence: {det['confidence']:.2f})")
    
    print("="*60)
