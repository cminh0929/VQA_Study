# DATA QUALITY IMPROVEMENTS - Phase 2A

## 🎯 OBJECTIVE
Fix 3 critical data quality issues identified in the weakness analysis.

---

## ✅ IMPROVEMENTS IMPLEMENTED

### 1. Real Color Detection (K-Means Clustering)
**File**: `code/utils/color_detection.py`

**Problem (Before)**:
```python
# OLD: Random color assignment
color = random.choice(['black', 'white', 'brown', 'gray', 'orange'])
```

**Solution (After)**:
```python
# NEW: K-Means clustering on actual image pixels
def detect_animal_color(image_path):
    # 1. Read image
    # 2. Remove noise (very dark/bright pixels)
    # 3. K-Means clustering (5 clusters)
    # 4. Get dominant cluster
    # 5. Map RGB to color name
    return color_name
```

**Impact**:
- ✅ Colors now reflect actual image content
- ✅ Model learns to look at image, not memorize patterns
- ✅ Color recognition becomes meaningful metric

---

### 2. Diverse Question Templates
**File**: `code/utils/question_templates.py`

**Problem (Before)**:
```python
# OLD: Fixed templates
question = "What animal is in the image?"
```

**Solution (After)**:
```python
# NEW: 33+ diverse templates
ANIMAL_RECOGNITION_TEMPLATES = [
    "What animal is in the image?",
    "Which animal can you see?",
    "Identify the animal in this image.",
    "What type of animal is shown?",
    # ... 10 total variations
]

question = random.choice(ANIMAL_RECOGNITION_TEMPLATES)
```

**Templates by Category**:
- Animal Recognition: 10 variations
- Color Recognition: 8 variations
- Yes/No Questions: 8 variations
- Counting: 7 variations
- Animal Name Variations: 4+ per animal

**Impact**:
- ✅ Prevents overfitting to sentence structure
- ✅ Model learns semantics, not patterns
- ✅ Better generalization to new phrasings

---

### 3. YOLO-Based Accurate Counting
**File**: `code/utils/yolo_counter.py`

**Problem (Before)**:
```python
# OLD: Random or heuristic counting
count = random.choice([1, 2, 3])
```

**Solution (After)**:
```python
# NEW: YOLO object detection
def count_animals_in_image(image_path, animal_type):
    # 1. Run YOLOv8 detection
    # 2. Filter by animal type
    # 3. Count detected objects
    # 4. Fallback to heuristic if YOLO unavailable
    return count
```

**Fallback Strategy**:
- If YOLO available: Use detection count
- If YOLO unavailable: Use edge detection + contours
- Always clamp to [1, 3] range

**Impact**:
- ✅ Counting becomes accurate (not random)
- ✅ Model can actually learn to count
- ✅ Meaningful evaluation metric

---

## 📁 NEW FILES CREATED

```
Data_prep/code/utils/
├── __init__.py
├── color_detection.py       # K-Means color detection
├── question_templates.py    # 33+ diverse templates
└── yolo_counter.py          # YOLO-based counting

Data_prep/code/scripts/
└── generate_vqa_annotations_v2.py  # Improved generator
```

---

## 🚀 USAGE

### Generate Improved Annotations:
```bash
cd Data_prep/code/scripts
python generate_vqa_annotations_v2.py
```

### Test Individual Components:
```bash
# Test color detection
cd Data_prep/code/utils
python color_detection.py ../../data/images/000000000001.jpg

# Test question templates
python question_templates.py

# Test YOLO counting
python yolo_counter.py ../../data/images/000000000001.jpg dog
```

---

## 📊 EXPECTED IMPROVEMENTS

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| **Color Accuracy** | Random (20%) | Real (60-80%) |
| **Question Diversity** | 1 template | 33+ templates |
| **Counting Accuracy** | Random (33%) | YOLO (70-90%) |
| **Overall Quality** | Low | High |

---

## ⚠️ REQUIREMENTS

### Python Packages:
```bash
pip install opencv-python scikit-learn ultralytics
```

### YOLO Weights:
- YOLOv8n will auto-download on first use
- Or manually download: `yolov8n.pt`

---

## 🔄 MIGRATION STEPS

### Option A: Full Re-generation (Recommended)
1. Run `generate_vqa_annotations_v2.py`
2. Creates `annotations_complete_v2.json`
3. Run `split_dataset.py` on v2 annotations
4. Rebuild vocabularies in VQA_Model
5. Re-train all models

### Option B: Keep Old Data
1. Keep current annotations for baseline
2. Generate v2 for comparison
3. Train models on both datasets
4. Compare results in paper

---

## 📈 VALIDATION

### Before Running Full Generation:
```bash
# Test on 10 images first
cd Data_prep/code/scripts
python -c "
from generate_vqa_annotations_v2 import generate_qa_for_image
from pathlib import Path

images = list(Path('../../data/images').glob('*.jpg'))[:10]
for img in images:
    qa = generate_qa_for_image(img.name, str(img), 'dog')
    print(f'{img.name}: {len(qa)} Q&A pairs')
    for q in qa:
        print(f'  Q: {q[\"question\"]}')
        print(f'  A: {q[\"answer\"]}')
"
```

---

## ✅ QUALITY CHECKLIST

- [x] Color detection uses real image pixels
- [x] Question templates are diverse (33+)
- [x] Counting uses YOLO detection
- [x] Fallback mechanisms in place
- [x] Error handling implemented
- [x] Documentation complete
- [ ] Full dataset re-generated
- [ ] Vocabularies rebuilt
- [ ] Models re-trained

---

## 🎯 NEXT STEPS

1. **Test utilities** on sample images
2. **Run full generation** (10-30 minutes)
3. **Validate quality** of generated data
4. **Re-split dataset** (train/val/test)
5. **Rebuild vocabularies** in VQA_Model
6. **Re-train models** with improved data
7. **Compare results** (old vs new)

---

**Created**: 2026-02-03
**Status**: Phase 2A Complete - Ready for Generation ✅
