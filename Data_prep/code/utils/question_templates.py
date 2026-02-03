"""
Question Templates for VQA Data Generation
Provides diverse question phrasings to avoid overfitting to sentence structure
"""

import random


# Animal Recognition Templates
ANIMAL_RECOGNITION_TEMPLATES = [
    "What animal is in the image?",
    "What animal is this?",
    "Which animal can you see?",
    "What type of animal is shown?",
    "What kind of animal is in the picture?",
    "Identify the animal in this image.",
    "What animal appears in this photo?",
    "Can you identify the animal?",
    "What species is shown?",
    "Which animal is present?",
]

# Color Recognition Templates
COLOR_RECOGNITION_TEMPLATES = [
    "What color is the {animal}?",
    "What is the color of the {animal}?",
    "Describe the color of the {animal}.",
    "What color does the {animal} have?",
    "Tell me the {animal}'s color.",
    "What is the {animal}'s color?",
    "How would you describe the {animal}'s color?",
    "What color fur/skin does the {animal} have?",
]

# Yes/No Templates - Positive
YES_NO_POSITIVE_TEMPLATES = [
    "Is there a {animal} in the image?",
    "Is there a {animal} in this picture?",
    "Can you see a {animal}?",
    "Is a {animal} present?",
    "Does this image contain a {animal}?",
    "Is a {animal} visible?",
    "Do you see a {animal}?",
    "Is this a {animal}?",
]

# Yes/No Templates - Negative  
YES_NO_NEGATIVE_TEMPLATES = [
    "Is there a {animal} in the image?",
    "Is there a {animal} in this picture?",
    "Can you see a {animal}?",
    "Is a {animal} present?",
    "Does this image contain a {animal}?",
    "Is a {animal} visible?",
    "Do you see a {animal}?",
]

# Counting Templates
COUNTING_TEMPLATES = [
    "How many {animal}s are there?",
    "How many {animal}s can you see?",
    "Count the {animal}s in the image.",
    "How many {animal}s are in the picture?",
    "What is the number of {animal}s?",
    "How many {animal}s are present?",
    "Can you count the {animal}s?",
]


def get_animal_question(animal_name=None):
    """Get random animal recognition question"""
    return random.choice(ANIMAL_RECOGNITION_TEMPLATES)


def get_color_question(animal_name):
    """Get random color recognition question"""
    template = random.choice(COLOR_RECOGNITION_TEMPLATES)
    return template.format(animal=animal_name)


def get_yes_no_question(animal_name, is_positive=True):
    """Get random yes/no question"""
    if is_positive:
        template = random.choice(YES_NO_POSITIVE_TEMPLATES)
    else:
        template = random.choice(YES_NO_NEGATIVE_TEMPLATES)
    return template.format(animal=animal_name)


def get_counting_question(animal_name):
    """Get random counting question"""
    template = random.choice(COUNTING_TEMPLATES)
    return template.format(animal=animal_name)


# Animal name variations for diversity
ANIMAL_VARIATIONS = {
    'dog': ['dog', 'canine', 'pup', 'hound'],
    'cat': ['cat', 'feline', 'kitty', 'kitten'],
    'bird': ['bird', 'avian'],
    'horse': ['horse', 'equine', 'mare', 'stallion'],
    'sheep': ['sheep', 'lamb'],
    'cow': ['cow', 'cattle', 'bovine'],
    'elephant': ['elephant', 'pachyderm'],
    'bear': ['bear'],
    'zebra': ['zebra'],
    'giraffe': ['giraffe'],
}


def get_animal_variation(animal_name):
    """Get random variation of animal name"""
    if animal_name in ANIMAL_VARIATIONS:
        return random.choice(ANIMAL_VARIATIONS[animal_name])
    return animal_name


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("QUESTION TEMPLATE EXAMPLES")
    print("="*60)
    
    animal = "dog"
    
    print(f"\n1. Animal Recognition (10 variations):")
    for i in range(5):
        print(f"   - {get_animal_question()}")
    
    print(f"\n2. Color Recognition (8 variations):")
    for i in range(4):
        print(f"   - {get_color_question(animal)}")
    
    print(f"\n3. Yes/No Questions (8 variations):")
    for i in range(4):
        print(f"   - {get_yes_no_question(animal, is_positive=True)}")
    
    print(f"\n4. Counting Questions (7 variations):")
    for i in range(4):
        print(f"   - {get_counting_question(animal)}")
    
    print(f"\n5. Animal Name Variations:")
    for _ in range(5):
        print(f"   - {get_animal_variation('dog')}")
    
    print("\n" + "="*60)
    print("✓ Total unique templates: 33+")
    print("="*60)
