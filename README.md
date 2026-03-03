# VQA Project - Visual Question Answering

A Visual Question Answering system that answers questions about images using CNN-LSTM architecture.

## Objective

Build a VQA system using **CNN-LSTM with LSTM Decoder** to answer questions about animal images.

**Primary goals:**
- ⭐ **Animal Recognition**: "What animal is in the image?" → dog, cat, bird...
- ⭐ **Color Recognition**: "What color is the dog?" → black, white, brown...
- ⭐ **Yes/No Questions**: "Is there a cat?" → yes, no
- ⚠️ **Simple Counting**: "How many dogs?" → 1, 2, 3

**Compare 8 model variants:**
- Attention vs No Attention
- Pretrained vs From-scratch
- ResNet50 vs VGG16

## Requirements

- **Python**: 3.10.x
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: Minimum 16GB

## Setup

```bash
# 1. Create Virtual Environment
py -3.10 -m venv venv
venv\Scripts\activate

# 2. Install Dependencies
pip install -r requirements.txt
```

## Project Structure

```
VQA_Workspace/
├── Data_prep/                  # Data preparation from COCO
│   ├── improve_annotations.py  # Generate V2 annotations (K-Means + YOLO)
│   ├── split_dataset.py        # Split train/val/test (small + full)
│   └── data/                   # Images and annotations
│       └── annotations/
│           ├── small/          # Dog + Cat only
│           └── full/           # 10 animal species
│
├── VQA_Model/                  # VQA Model
│   ├── main.py                 # Main entry point
│   ├── models/                 # CNN, LSTM, Attention, Decoder
│   ├── engine/                 # Trainer, Evaluator
│   ├── data/                   # Dataset, Vocabulary, Transforms
│   └── utils/                  # Metrics, Visualization
│
├── requirements.txt
├── PROJECT_PLAN.md             # Detailed project plan
└── README.md                   # Overview
```

## Usage

All operations are performed through `main.py`:

```bash
cd VQA_Model

# Build vocabulary (run once when switching datasets)
py -3.10 main.py build_vocab

# Train model (edit model_id in CONFIG)
py -3.10 main.py train

# Evaluate on test set
py -3.10 main.py evaluate

# Train + Evaluate in sequence
py -3.10 main.py both

# Compare all trained models
py -3.10 main.py compare
```

Edit configuration in the `CONFIG` section at the top of `main.py`:
```python
CONFIG = {
    'model_id': 2,           # 1-8
    'dataset': 'small',      # 'small' or 'full'
    'epochs': 5,
    'batch_size': 32,
    ...
}
```

## 8 Model Variants

| ID | CNN | Pretrained | Attention |
|----|-----|------------|-----------|
| 1 | ResNet50 | ✅ | ❌ |
| 2 | ResNet50 | ✅ | ✅ |
| 3 | ResNet50 | ❌ | ❌ |
| 4 | ResNet50 | ❌ | ✅ |
| 5 | VGG16 | ✅ | ❌ |
| 6 | VGG16 | ✅ | ✅ |
| 7 | VGG16 | ❌ | ❌ |
| 8 | VGG16 | ❌ | ✅ |

## Evaluation Metrics

- **Accuracy**: Exact match
- **BLEU-1 / BLEU-4**: N-gram precision
- **F1 Score**: Word-level precision & recall
- **Per-category**: Animal, Color, Yes/No, Counting

## Documentation

- **PROJECT_PLAN.md**: Detailed project plan, model architecture, metrics