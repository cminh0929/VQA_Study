# VQA Project - Visual Question Answering

Dự án VQA (Visual Question Answering) - Trả lời câu hỏi về hình ảnh.

## Mục Tiêu

Xây dựng hệ thống VQA trả lời 3 loại câu hỏi:
- **Animals**: "What animal is in the image?" → dog, cat, bird...
- **Colors**: "What color is the cat?" → black, white, brown...
- **Counting**: "How many dogs are there?" → 1, 2, 3...

## Cấu Trúc

```
VQA/
├── configs/                # Cấu hình
│   ├── base_config.py
│   └── experiments.py
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