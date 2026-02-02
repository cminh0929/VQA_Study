# VQA Model - Visual Question Answering

Complete implementation of 8 VQA model variants using CNN-LSTM architecture.

## 📁 Project Structure

```
VQA_Model/
├── data/                      # Data pipeline
│   ├── vocab.py              # Vocabulary management
│   ├── transforms.py         # Image preprocessing
│   ├── dataset.py            # PyTorch Dataset
│   ├── question_vocab.json   # Question vocabulary (47 words)
│   └── answer_vocab.json     # Answer vocabulary (29 words)
│
├── models/                    # Model architecture
│   ├── cnn_encoder.py        # ResNet50/VGG16 encoder
│   ├── lstm_encoder.py       # Question encoder
│   ├── attention.py          # Attention mechanism
│   ├── lstm_decoder.py       # Answer decoder
│   └── vqa_model.py          # Full VQA model (8 variants)
│
├── engine/                    # Training & evaluation
│   ├── trainer.py            # Training loop
│   └── evaluator.py          # Evaluation metrics
│
├── utils/                     # Utilities
│   ├── metrics.py            # Accuracy, BLEU, F1
│   └── visualization.py      # Plot training curves, attention
│
├── configs/                   # Configuration
│   └── base_config.py        # Default hyperparameters
│
├── checkpoints/               # Model checkpoints
├── logs/                      # Training logs
├── results/                   # Evaluation results
├── visualizations/            # Generated plots
│
├── train.py                   # Main training script
├── evaluate.py                # Evaluation script
├── inference.py               # Single image inference
├── analyze_results.py         # Results analysis
└── README.md                  # This file
```

---

## 🎯 8 Model Variants

| ID | CNN | Pretrained | Attention | Params |
|----|-----|------------|-----------|--------|
| 1 | ResNet50 | ✅ | ❌ | 33.4M |
| 2 | ResNet50 | ✅ | ✅ | 34.8M |
| 3 | ResNet50 | ❌ | ❌ | 33.4M |
| 4 | ResNet50 | ❌ | ✅ | 34.8M |
| 5 | VGG16 | ✅ | ❌ | 145.2M |
| 6 | VGG16 | ✅ | ✅ | 147.6M |
| 7 | VGG16 | ❌ | ❌ | 145.2M |
| 8 | VGG16 | ❌ | ✅ | 147.6M |

---

## 🚀 Usage

### **1. Train a Single Model**

```bash
# Train Model 1 (ResNet50 + Pretrained + No Attention)
python train.py --model_id 1 --epochs 20 --batch_size 32 --lr 0.001

# Train Model 2 (with Attention)
python train.py --model_id 2 --epochs 20 --batch_size 32
```

### **2. Train All Models**

```bash
# Train all 8 models
python train.py --epochs 20 --batch_size 32
```

### **3. Evaluate a Model**

```bash
# Evaluate Model 1
python evaluate.py --model_id 1 --save_predictions

# Evaluate all models
python evaluate.py --save_predictions
```

### **4. Inference on Single Image**

```bash
# Single question
python inference.py --model_id 2 --image path/to/image.jpg --question "What animal is in the image?"

# Interactive mode
python inference.py --model_id 2 --interactive
```

### **5. Analyze Results**

```bash
# Analyze all trained models
python analyze_results.py --visualize

# Compare models
python analyze_results.py --results_dir results --output_dir visualizations
```

### **6. Custom Checkpoint**

```bash
# Evaluate with specific checkpoint
python evaluate.py --model_id 1 --checkpoint checkpoints/model_1/checkpoint_epoch_10.pth
```

---

## 📊 Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | None | Model ID (1-8). None = train all |
| `--epochs` | 20 | Number of epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--device` | cuda/cpu | Device to train on |

---

## 📈 Evaluation Metrics

- **Accuracy**: Exact match accuracy
- **BLEU-1**: Unigram BLEU score
- **BLEU-4**: 4-gram BLEU score
- **Precision**: Word-level precision
- **Recall**: Word-level recall
- **F1 Score**: Harmonic mean of precision and recall

**Per-category metrics:**
- Animal Recognition
- Color Recognition
- Yes/No Questions
- Counting (1-3)

---

## 💾 Outputs

### **Checkpoints** (`checkpoints/model_{id}/`)
- `best_model.pth` - Best model by validation accuracy
- `checkpoint_epoch_{N}.pth` - Periodic checkpoints

### **Logs** (`logs/model_{id}/`)
- `training_history.json` - Training/validation metrics per epoch

### **Results** (`results/`)
- `model_{id}_predictions.json` - Test set predictions and metrics

---

## 🧪 Testing

```bash
# Test Phase 1: Data Pipeline
python test_phase1.py

# Test Phase 2: Model Architecture
python test_phase2.py

# Comprehensive testing
python test_models_detailed.py
```

---

## 📝 Example: Quick Start

```python
# Load vocabularies
from data import Vocabulary
question_vocab = Vocabulary.load('data/question_vocab.json')
answer_vocab = Vocabulary.load('data/answer_vocab.json')

# Create model
from models import create_model_variant
model = create_model_variant(model_id=1, 
                             question_vocab_size=len(question_vocab),
                             answer_vocab_size=len(answer_vocab))

# Load checkpoint
import torch
checkpoint = torch.load('checkpoints/model_1/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Inference
model.eval()
with torch.no_grad():
    predictions, attention = model.generate_answer(images, questions, q_lengths)
```

---

## ✅ Phase Completion Status

- ✅ **Phase 1**: Data Pipeline (vocab, transforms, dataset)
- ✅ **Phase 2**: Model Architecture (8 variants)
- ✅ **Phase 3**: Training & Evaluation (trainer, evaluator, metrics)
- 🎯 **Ready for Training!**

---

## 📚 Dependencies

- Python 3.10+
- PyTorch 2.0+
- torchvision
- Pillow
- tqdm
- numpy

See `../requirements.txt` for full list.
