"""
Color Detection Utility for VQA Data Generation
Extract REAL dominant color from images using K-Means clustering
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import os


def rgb_to_color_name(rgb):
    """
    Convert RGB to human-readable color name
    
    Args:
        rgb: Tuple (R, G, B)
    
    Returns:
        color_name: String color name
    """
    r, g, b = rgb
    
    # Define color ranges (expanded for animals)
    colors = {
        'black': ([0, 0, 0], [60, 60, 60]),
        'white': ([200, 200, 200], [255, 255, 255]),
        'gray': ([60, 60, 60], [180, 180, 180]),
        'brown': ([80, 50, 20], [150, 100, 60]),
        'orange': ([200, 100, 0], [255, 165, 50]),
        'yellow': ([200, 200, 0], [255, 255, 120]),
        'red': ([150, 0, 0], [255, 80, 80]),
        'pink': ([200, 100, 150], [255, 200, 255]),
        'blue': ([0, 0, 150], [100, 100, 255]),
        'green': ([0, 100, 0], [100, 200, 100]),
        'tan': ([180, 140, 100], [220, 180, 140]),
        'golden': ([200, 150, 50], [255, 215, 100]),
    }
    
    # Find closest color
    min_dist = float('inf')
    closest_color = 'brown'  # Default for animals
    
    for color_name, (lower, upper) in colors.items():
        # Calculate distance to center of range
        center = [(lower[i] + upper[i]) / 2 for i in range(3)]
        dist = sum((rgb[i] - center[i])**2 for i in range(3)) ** 0.5
        
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name
    
    return closest_color


def get_dominant_color(image_path, n_colors=5):
    """
    Extract dominant color from image using K-Means
    
    Args:
        image_path: Path to image file
        n_colors: Number of clusters for K-Means
    
    Returns:
        color_name: Dominant color name
        confidence: Confidence score (0-1)
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return 'brown', 0.5
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize for faster processing
        h, w = image.shape[:2]
        if max(h, w) > 500:
            scale = 500 / max(h, w)
            image = cv2.resize(image, (int(w*scale), int(h*scale)))
        
        # Reshape to list of pixels
        pixels = image.reshape(-1, 3)
        
        # Remove very dark (shadows) and very bright (highlights) pixels
        brightness = pixels.sum(axis=1)
        mask = (brightness > 50) & (brightness < 700)
        pixels = pixels[mask]
        
        if len(pixels) < 10:
            return 'brown', 0.5
        
        # K-Means clustering
        n_clusters = min(n_colors, len(pixels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Count pixels in each cluster
        labels = kmeans.labels_
        counts = Counter(labels)
        
        # Get most common cluster
        dominant_cluster = counts.most_common(1)[0][0]
        dominant_rgb = tuple(colors[dominant_cluster])
        
        # Calculate confidence (proportion of pixels in dominant cluster)
        confidence = counts[dominant_cluster] / len(labels)
        
        # Convert to color name
        color_name = rgb_to_color_name(dominant_rgb)
        
        return color_name, confidence
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 'brown', 0.5


def detect_animal_color(image_path):
    """
    Detect animal color from image (simplified wrapper)
    
    Args:
        image_path: Path to image
    
    Returns:
        color: Color name (str)
    """
    color, confidence = get_dominant_color(image_path, n_colors=5)
    
    # If confidence is very low, default to common animal colors
    if confidence < 0.15:
        # Return most common animal color
        return 'brown'
    
    return color


# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python color_detection.py <image_path>")
        print("\nExample:")
        print("  python color_detection.py ../data/images/000000000001.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"\nAnalyzing: {image_path}")
    print("="*60)
    
    color, confidence = get_dominant_color(image_path)
    
    print(f"Dominant color: {color}")
    print(f"Confidence: {confidence:.2%}")
    print("="*60)
