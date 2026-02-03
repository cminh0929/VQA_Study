"""
Data Preparation Utilities
Tools for improving VQA data quality
"""

from .color_detection import detect_animal_color, get_dominant_color
from .question_templates import (
    get_animal_question,
    get_color_question,
    get_yes_no_question,
    get_counting_question,
    get_animal_variation
)
from .yolo_counter import count_animals_in_image, YOLOCounter

__all__ = [
    # Color detection
    'detect_animal_color',
    'get_dominant_color',
    
    # Question templates
    'get_animal_question',
    'get_color_question',
    'get_yes_no_question',
    'get_counting_question',
    'get_animal_variation',
    
    # Counting
    'count_animals_in_image',
    'YOLOCounter',
]
