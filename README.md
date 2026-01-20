# VQA Project - Visual Question Answering with Ablation Study

Dự án VQA với 8 models để so sánh ablation study.

## Cấu Trúc

```
VQA/
├── configs/                # Cấu hình
│   ├── base_config.py
│   └── experiments.py      # 8 model configs
├── data/                   # Data processing
│   ├── scripts/            # Data generation
│   ├── dataset.py
│   ├── vocabulary.py
│   └── preprocessing.py
├── models/                 # Model architecture
│   ├── components/
│   │   ├── cnn_encoder.py
│   │   ├── rnn_encoder.py
│   │   └── attention.py
│   ├── vqa_model.py
│   └── vqa_factory.py
├── engine/                 # Training/Evaluation
│   ├── trainer.py
│   └── evaluator.py
└── utils/                  # Utilities
    ├── logger.py
    └── visualization.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Prepare data
python data/scripts/prepare_coco.py --max_images 5000
python data/scripts/download_images.py
python data/scripts/generate_qa.py

# Train
python main.py --model model_3

# Ablation study
python run_ablation.py
```
