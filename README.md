# Visual Question Answering (VQA) on Animal Datasets

## 1. Overview
This repository implements a complete **Visual Question Answering (VQA)** system focused on the animal domain. The system bridges Computer Vision and Natural Language Processing to accurately answer text-based queries about images, including:
- **Species Recognition** (e.g., "What animal is this?")
- **Feature Identification** (e.g., "What color is the dog's fur?")
- **Object Counting** (e.g., "How many cats are in the room?")
- **Polar Queries** (e.g., "Is there a bird present?")

## 2. Technical Architecture
The system employs a tightly coupled **CNN-LSTM Architecture** to decode multi-modal features and generate autoregressive textual responses:

1. **Visual Feature Extraction (CNN):** 
   - **Pretrained Mode:** Utilizes standard `ResNet50` initialized with ImageNet weights to extract a 2048-D global mapping or a $7 \times 7$ spatial feature map.
   - **From-Scratch Mode:** Utilizes a custom, lightweight `ScratchCNN` built with 2-layer residual BasicBlocks (~11M parameters) to force pixel-level learning.
2. **Semantic Question Representation (LSTM):** 
   - Tokenizes and embeds grammatical structures into a localized dense Hidden Context Vector.
3. **Autoregressive Answer Generation (LSTM Decoder & Attention):** 
   - Experimental variants utilize **Spatial Attention**. This alignment dynamically computes a weighted contextual matrix, forcing the neural network to "look" at the specific 49 visual patches corresponding to the question's semantics before decoding the next sequence word.

## 3. Data Ecosystem
The architectural limits are comprehensively evaluated against two drastically different data distributions to test generalization (Generalization Gap):

- **Dataset A (Synthetic Templates):** ~100k question-answer pairs synthetically generated via syntactical rules bounding 10 specific mammal classes from Microsoft COCO 2014.
- **Dataset B (VQA v2.0 Unconstrained Subset):** ~54k complex, long-tail, and highly imbalanced human-authored queries algorithmically filtered directly from the official VQA v2.0 Benchmark.

## 4. Project Structure
The repository is modularized into mapping components and model iterations:

```text
VQA_Workspace/
├── Data_prep/                  # Pipeline for synthetic data creation & VQA v2.0 filtering
│   ├── annotations/            # JSON dataset formats
│   ├── improve_annotations.py  # V2 annotation generation (K-Means coloring + YOLO counting)
│   └── split_dataset.py        # Train/Validation/Test slicing
│
├── VQA_Model/                  # Initial models trained against structured synthetic dataset
│   ├── models/                 # Neural architectures (CNN, LSTM, Attention mechanism)
│   ├── run_all.py              # Automated full-pipeline execution
│   └── logs/                   # Training metrics tracking
│
└── VQA_Model_B/                # Architectures tuned for the chaotic VQA v2.0 Human Dataset
    ├── models/                 # Parallel complex architectures
    └── run_all.py              # Automated full-pipeline execution
```

## 5. Setup and Execution

**Requirements:** Python 3.10.x | PyTorch | NVIDIA GPU (CUDA Recommended)

**Installation:**
```bash
# 1. Initialize environment
py -3.10 -m venv venv
venv\Scripts\activate

# 2. Install neural pipeline dependencies
pip install -r requirements.txt
```

**Training execution:**
Navigate to either the synthetic iteration (`VQA_Model`) or natural language iteration (`VQA_Model_B`) and run the master script:
```bash
cd VQA_Model_B
python run_all.py
```
*Note: The script automatically handles vocabulary building, dataloader splitting, 4-variant model training, metric evaluation, and history plotting.*
