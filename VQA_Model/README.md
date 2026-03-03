# VQA Model

Visual Question Answering model using CNN-LSTM Encoder-Decoder with Spatial Attention.

## Structure

```
VQA_Model/
├── main.py                 # ★ Main entry point (train, evaluate, compare)
├── train.py                # Training script (CLI)
├── evaluate.py             # Evaluation script (CLI)
├── inference.py            # Single image inference
├── analyze_results.py      # Results comparison
│
├── models/                 # Model architecture
│   ├── cnn_encoder.py      # ResNet50 / VGG16
│   ├── lstm_encoder.py     # Question encoder
│   ├── attention.py        # Spatial Attention mechanism
│   ├── lstm_decoder.py     # Answer decoder (autoregressive)
│   └── vqa_model.py        # VQA model (8 variants)
│
├── engine/                 # Training & evaluation loop
│   ├── trainer.py          # VQATrainer
│   └── evaluator.py        # VQAEvaluator
│
├── data/                   # Data pipeline
│   ├── vocab.py            # Vocabulary (build, encode, decode)
│   ├── transforms.py       # Image preprocessing
│   ├── dataset.py          # PyTorch Dataset & DataLoader
│   ├── question_vocab.json # Question vocabulary
│   └── answer_vocab.json   # Answer vocabulary
│
├── utils/                  # Utilities
│   ├── metrics.py          # Accuracy, BLEU, F1
│   └── visualization.py    # Training curves, model comparison
│
├── checkpoints/            # Model weights (gitignored)
├── logs/                   # Training logs (gitignored)
└── results/                # Predictions (gitignored)
```

## Usage

```bash
# Edit CONFIG in main.py then run:
py -3.10 main.py train          # Train model
py -3.10 main.py evaluate       # Evaluate on test set
py -3.10 main.py both           # Train + Evaluate
py -3.10 main.py compare        # Compare all trained models
py -3.10 main.py build_vocab    # Rebuild vocabulary
```

## 8 Model Variants

| ID | CNN | Pretrained | Attention |
|----|-----|------------|-----------|
| 1 | ResNet50 | ✅ Yes | ❌ No |
| 2 | ResNet50 | ✅ Yes | ✅ Yes |
| 3 | ResNet50 | ❌ No | ❌ No |
| 4 | ResNet50 | ❌ No | ✅ Yes |
| 5 | VGG16 | ✅ Yes | ❌ No |
| 6 | VGG16 | ✅ Yes | ✅ Yes |
| 7 | VGG16 | ❌ No | ❌ No |
| 8 | VGG16 | ❌ No | ✅ Yes |

## Evaluation Metrics

- **Accuracy** — Exact match
- **BLEU-1 / BLEU-4** — N-gram precision
- **F1 Score** — Word-level precision & recall
- **Per-category** — Animal, Color, Yes/No, Counting
