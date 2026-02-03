"""
Data Preparation Flow Visualizer
Creates a visual representation of the data pipeline
"""

def print_flow_diagram():
    """Print ASCII flow diagram"""
    
    diagram = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                     VQA DATA PREPARATION PIPELINE                         ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Download COCO Images                                            │
│ ────────────────────────────────────────────────────────────────────    │
│ Source: COCO Dataset (train2017)                                        │
│ Action: Manual download                                                 │
│ Output: ~118,000 images                                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Filter Animal Images                                            │
│ ────────────────────────────────────────────────────────────────────    │
│ Script: code/filter/animal_filter.py                                    │
│ Action: Extract 10 animal categories                                    │
│ Output: 20,732 animal images                                            │
│         Categories: dog, cat, bird, horse, sheep, cow,                  │
│                    elephant, bear, zebra, giraffe                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Generate VQA Annotations                                        │
│ ────────────────────────────────────────────────────────────────────    │
│                                                                          │
│  ┌──────────────────────────┐      ┌──────────────────────────┐        │
│  │ VERSION 1 (Current)      │      │ VERSION 2 (Improved) ⭐  │        │
│  │ ─────────────────────    │      │ ─────────────────────    │        │
│  │ ❌ Random colors         │      │ ✅ K-Means colors        │        │
│  │ ❌ Fixed templates       │      │ ✅ 33+ templates         │        │
│  │ ❌ Random counting       │      │ ✅ YOLO counting         │        │
│  │                          │      │                          │        │
│  │ Script:                  │      │ Script:                  │        │
│  │ generate_vqa_            │      │ generate_vqa_            │        │
│  │ annotations.py           │      │ annotations_v2.py        │        │
│  └──────────────────────────┘      └──────────────────────────┘        │
│                                                                          │
│ Output: 103,660 Q&A pairs (5 per image)                                 │
│         - Animal Recognition: 20,732                                    │
│         - Color Recognition:  20,732                                    │
│         - Yes/No Questions:   41,464                                    │
│         - Counting:           20,732                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Split Dataset (70/15/15)                                        │
│ ────────────────────────────────────────────────────────────────────    │
│ Script: code/scripts/split_dataset.py                                   │
│ Strategy: Split by IMAGE_ID (no data leakage)                           │
│                                                                          │
│ Output:                                                                  │
│   ├── train.json    → 72,562 Q&A (14,512 images)                       │
│   ├── val.json      → 15,549 Q&A (3,110 images)                        │
│   └── test.json     → 15,549 Q&A (3,110 images)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Build Vocabularies (in VQA_Model)                               │
│ ────────────────────────────────────────────────────────────────────    │
│ Script: VQA_Model/data/vocab.py                                         │
│ Action: Tokenize questions and answers                                  │
│                                                                          │
│ Output:                                                                  │
│   ├── question_vocab.json  → 47 words (V1) / ~70 words (V2)            │
│   └── answer_vocab.json    → 29 words (V1) / ~40 words (V2)            │
│                                                                          │
│ Special tokens: <PAD>=0, <UNK>=1, <SOS>=2, <EOS>=3                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ✅ READY FOR TRAINING                                                   │
│ ────────────────────────────────────────────────────────────────────    │
│ VQA_Model/train.py --model_id 2 --epochs 20 --batch_size 32            │
└─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
DATA QUALITY COMPARISON
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────┬─────────────────────┬─────────────────────┐
│ Aspect                  │ V1 (Current)        │ V2 (Improved)       │
├─────────────────────────┼─────────────────────┼─────────────────────┤
│ Color Detection         │ ❌ Random           │ ✅ K-Means          │
│ Color Accuracy          │ ~20%                │ ~60-80%             │
│                         │                     │                     │
│ Question Templates      │ ❌ Fixed (1 each)   │ ✅ Diverse (33+)    │
│ Template Diversity      │ 4 templates         │ 33+ templates       │
│                         │                     │                     │
│ Counting Method         │ ❌ Random(1,2,3)    │ ✅ YOLO Detection   │
│ Counting Accuracy       │ ~33%                │ ~70-90%             │
│                         │                     │                     │
│ Overall Data Quality    │ ⚠️  Low             │ ✅ High             │
│ Model Generalization    │ ⚠️  Poor            │ ✅ Good             │
│ Metrics Meaningfulness  │ ⚠️  Questionable    │ ✅ Reliable         │
└─────────────────────────┴─────────────────────┴─────────────────────┘


═══════════════════════════════════════════════════════════════════════════
UTILITIES (NEW)
═══════════════════════════════════════════════════════════════════════════

code/utils/
├── color_detection.py      → K-Means clustering on image pixels
├── question_templates.py   → 33+ diverse question phrasings
└── yolo_counter.py         → YOLOv8-based object counting

"""
    
    print(diagram)


def print_statistics():
    """Print data statistics"""
    
    stats = """
═══════════════════════════════════════════════════════════════════════════
DATASET STATISTICS
═══════════════════════════════════════════════════════════════════════════

Total Images:           20,732
Total Q&A Pairs:       103,660
Avg Q&A per Image:          5.0

Question Types:
  ├── Animal Recognition:  20,732 (20.0%)
  ├── Color Recognition:   20,732 (20.0%)
  ├── Yes/No Questions:    41,464 (40.0%)
  └── Counting:            20,732 (20.0%)

Data Splits:
  ├── Train:  72,562 Q&A  (14,512 images)  70%
  ├── Val:    15,549 Q&A  (3,110 images)   15%
  └── Test:   15,549 Q&A  (3,110 images)   15%

Vocabularies (V1):
  ├── Question Vocab:  47 words
  └── Answer Vocab:    29 words

Vocabularies (V2 - Expected):
  ├── Question Vocab:  ~70 words  (more diverse templates)
  └── Answer Vocab:    ~40 words  (more color variations)

═══════════════════════════════════════════════════════════════════════════
"""
    
    print(stats)


if __name__ == "__main__":
    print_flow_diagram()
    print_statistics()
    
    print("\n" + "="*79)
    print("📚 For detailed documentation, see:")
    print("  - DATA_FLOW_ANALYSIS.md")
    print("  - DATA_QUALITY_IMPROVEMENTS.md")
    print("="*79)
