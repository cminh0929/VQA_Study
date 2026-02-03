# DATA PREPARATION FLOW ANALYSIS

## 📊 COMPLETE DATA PIPELINE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Download Images (COCO Dataset)
    ↓
STEP 2: Filter Animal Images  
    ↓
STEP 3: Generate VQA Annotations
    ↓
STEP 4: Split Dataset (Train/Val/Test)
    ↓
STEP 5: Build Vocabularies (in VQA_Model)
    ↓
READY FOR TRAINING
```

---

## 🔄 DETAILED FLOW BREAKDOWN

### **STEP 1: Download Images** 
**Script**: Manual download from COCO
**Location**: `Data_prep/data/images/`

```bash
# User manually downloads COCO images
# Places in: Data_prep/data/images/
# Result: ~118k images total
```

**Output**:
- `data/images/*.jpg` - Raw COCO images

---

### **STEP 2: Filter Animal Images**
**Script**: `code/filter/animal_filter.py`
**Purpose**: Extract only animal images from COCO

```python
# Flow:
1. Load COCO annotations (instances_train2017.json)
2. Filter categories: dog, cat, bird, horse, etc.
3. Extract image IDs with these animals
4. Copy filtered images to data/images/
5. Save metadata

# Command:
python code/filter/animal_filter.py
```

**Input**:
- COCO annotations (instances_train2017.json)
- COCO images (118k total)

**Output**:
- `data/images/*.jpg` - 20,732 animal images
- Filtered by 10 animal categories

**Current Status**: ✅ DONE (20,732 images)

---

### **STEP 3: Generate VQA Annotations**

#### **VERSION 1 (OLD - Low Quality)**
**Script**: `code/scripts/generate_vqa_annotations.py`

```python
# Problems:
❌ Random color assignment
❌ Fixed question templates  
❌ Random counting

# Flow:
for each image:
    1. Animal Recognition: "What animal is in the image?" → animal_name
    2. Color Recognition: "What color is the {animal}?" → RANDOM_COLOR ❌
    3. Yes/No (positive): "Is there a {animal}?" → "yes"
    4. Yes/No (negative): "Is there a {other_animal}?" → "no"
    5. Counting: "How many {animal}s?" → RANDOM(1,2,3) ❌

# Output: 5 Q&A pairs per image
```

**Output**:
- `data/annotations/annotations_complete.json`
- 103,660 Q&A pairs (20,732 images × 5)

**Current Status**: ✅ DONE (but low quality)

---

#### **VERSION 2 (NEW - High Quality)** ⭐
**Script**: `code/scripts/generate_vqa_annotations_v2.py`

```python
# Improvements:
✅ K-Means color detection (real colors)
✅ 33+ diverse templates
✅ YOLO-based counting

# Flow:
for each image:
    1. Animal Recognition: 
       - Template: random.choice(10 variations)
       - Answer: animal_name
    
    2. Color Recognition:
       - Detect real color: detect_animal_color(image) ✅
       - Template: random.choice(8 variations)
       - Answer: detected_color (not random!)
    
    3. Yes/No (positive):
       - Template: random.choice(8 variations)
       - Answer: "yes"
    
    4. Yes/No (negative):
       - Template: random.choice(8 variations)
       - Answer: "no"
    
    5. Counting:
       - Count: count_animals_in_image(image, YOLO) ✅
       - Template: random.choice(7 variations)
       - Answer: yolo_count (not random!)

# Output: 5 Q&A pairs per image (higher quality)
```

**Output**:
- `data/annotations/annotations_complete_v2.json`
- 103,660 Q&A pairs (same count, better quality)

**Current Status**: ⏳ NOT YET RUN (ready to generate)

---

### **STEP 4: Split Dataset**
**Script**: `code/scripts/split_dataset.py`
**Purpose**: Create train/val/test splits

```python
# Strategy: Split by IMAGE_ID (no data leakage)
# Ratio: 70% / 15% / 15%

# Flow:
1. Load annotations_complete.json
2. Get unique image IDs
3. Shuffle and split: 70/15/15
4. Assign Q&A pairs to splits based on image_id
5. Save 3 files

# Important: All Q&A for same image go to same split!
```

**Input**:
- `annotations_complete.json` (or v2)

**Output**:
```
data/annotations/
├── train.json      # 72,562 Q&A (14,512 images)
├── val.json        # 15,549 Q&A (3,110 images)  
└── test.json       # 15,549 Q&A (3,110 images)
```

**Current Status**: ✅ DONE (for v1 data)

---

### **STEP 5: Build Vocabularies**
**Location**: `VQA_Model/data/vocab.py`
**Purpose**: Create word-to-index mappings

```python
# Flow:
1. Load train.json
2. Tokenize all questions → question_vocab
3. Tokenize all answers → answer_vocab
4. Add special tokens: <PAD>, <UNK>, <SOS>, <EOS>
5. Save to JSON

# Command (in VQA_Model):
from data import Vocabulary
q_vocab = Vocabulary()
a_vocab = Vocabulary()

# Build from training data
for qa in train_data:
    q_vocab.add_sentence(qa['question'])
    a_vocab.add_sentence(qa['answer'])

q_vocab.save('data/question_vocab.json')
a_vocab.save('data/answer_vocab.json')
```

**Output**:
```
VQA_Model/data/
├── question_vocab.json  # 47 words
└── answer_vocab.json    # 29 words
```

**Current Status**: ✅ DONE (for v1 data)

---

## 📁 COMPLETE FILE STRUCTURE

```
Data_prep/
│
├── data/
│   ├── images/                    # 20,732 animal images
│   │   ├── 000000000001.jpg
│   │   ├── 000000000002.jpg
│   │   └── ...
│   │
│   └── annotations/
│       ├── annotations_complete.json     # V1: 103,660 Q&A (low quality)
│       ├── annotations_complete_v2.json  # V2: 103,660 Q&A (high quality) ⏳
│       ├── train.json                    # 72,562 Q&A
│       ├── val.json                      # 15,549 Q&A
│       └── test.json                     # 15,549 Q&A
│
├── code/
│   ├── filter/
│   │   └── animal_filter.py       # STEP 2: Filter animals
│   │
│   ├── scripts/
│   │   ├── generate_vqa_annotations.py      # STEP 3 (V1) ❌
│   │   ├── generate_vqa_annotations_v2.py   # STEP 3 (V2) ✅
│   │   └── split_dataset.py                 # STEP 4
│   │
│   └── utils/                     # NEW: Quality improvements
│       ├── __init__.py
│       ├── color_detection.py     # K-Means color
│       ├── question_templates.py  # 33+ templates
│       └── yolo_counter.py        # YOLO counting
│
├── DATA_QUALITY_IMPROVEMENTS.md
└── PREDATA_GUIDE.md
```

---

## 🔄 TWO POSSIBLE WORKFLOWS

### **WORKFLOW A: Use Existing Data (Current)**
```bash
# Already done:
✅ STEP 1: Images downloaded (20,732)
✅ STEP 2: Animals filtered
✅ STEP 3: Annotations generated (V1 - low quality)
✅ STEP 4: Dataset split (train/val/test)
✅ STEP 5: Vocabularies built

# Ready to train with existing data
cd VQA_Model
python train.py --model_id 2 --epochs 20
```

**Pros**: Immediate training
**Cons**: Low data quality (random colors, fixed templates)

---

### **WORKFLOW B: Regenerate with V2 (Recommended)** ⭐
```bash
# STEP 3 (V2): Generate improved annotations
cd Data_prep/code/scripts
python generate_vqa_annotations_v2.py
# → Creates annotations_complete_v2.json

# STEP 4: Re-split with V2 data
python split_dataset.py
# → Updates train/val/test.json

# STEP 5: Rebuild vocabularies
cd ../../../VQA_Model
python -c "
from data import Vocabulary
import json

# Load V2 training data
with open('../Data_prep/data/annotations/train.json') as f:
    train_data = json.load(f)

# Build vocabularies
q_vocab = Vocabulary()
a_vocab = Vocabulary()

for qa in train_data:
    q_vocab.add_sentence(qa['question'])
    a_vocab.add_sentence(qa['answer'])

q_vocab.save('data/question_vocab.json')
a_vocab.save('data/answer_vocab.json')
"

# STEP 6: Train with improved data
python train.py --model_id 2 --epochs 20
```

**Pros**: High data quality, meaningful metrics
**Cons**: Takes 10-30 minutes to regenerate

---

## 📊 DATA STATISTICS

### **Current (V1)**:
```
Images: 20,732
Total Q&A: 103,660
├── Animal Recognition: 20,732 (20.0%)
├── Color Recognition:  20,732 (20.0%) ❌ Random colors
├── Yes/No:            41,464 (40.0%)
└── Counting:          20,732 (20.0%) ❌ Random counts

Train: 72,562 Q&A (14,512 images)
Val:   15,549 Q&A (3,110 images)
Test:  15,549 Q&A (3,110 images)

Question Vocab: 47 words
Answer Vocab: 29 words
```

### **Expected (V2)**:
```
Images: 20,732 (same)
Total Q&A: 103,660 (same count)
├── Animal Recognition: 20,732 (20.0%) ✅ Diverse templates
├── Color Recognition:  20,732 (20.0%) ✅ Real K-Means colors
├── Yes/No:            41,464 (40.0%) ✅ Diverse templates
└── Counting:          20,732 (20.0%) ✅ YOLO detection

Question Vocab: ~60-80 words (more diverse)
Answer Vocab: ~35-40 words (more colors)
```

---

## 🎯 KEY INSIGHTS

### **Data Leakage Prevention**:
- ✅ Split by IMAGE_ID (not by Q&A)
- ✅ All Q&A for same image in same split
- ✅ No image appears in multiple splits

### **Quality Issues (V1)**:
- ❌ Colors: Random assignment
- ❌ Templates: Fixed structure
- ❌ Counting: Random numbers

### **Quality Fixes (V2)**:
- ✅ Colors: K-Means on pixels
- ✅ Templates: 33+ variations
- ✅ Counting: YOLO detection

---

## 🚀 RECOMMENDED ACTION

**Generate V2 data NOW** because:
1. Only takes 10-30 minutes
2. Huge quality improvement
3. Makes metrics meaningful
4. Better for paper/thesis

```bash
cd Data_prep/code/scripts
python generate_vqa_annotations_v2.py
```

---

**Created**: 2026-02-03
**Status**: Flow Analysis Complete ✅
