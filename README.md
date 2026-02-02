# VQA Project - Visual Question Answering

Dự án VQA (Visual Question Answering) - Trả lời câu hỏi về hình ảnh.

## Mục Tiêu

Xây dựng hệ thống VQA sử dụng **CNN-LSTM với LSTM Decoder** để trả lời câu hỏi về hình ảnh động vật.

**Mục tiêu chính:**
- ⭐ **Nhận diện động vật**: "What animal is in the image?" → dog, cat, bird...
- ⭐ **Nhận diện màu sắc**: "What color is the dog?" → black, white, brown...
- ⭐ **Câu hỏi Yes/No**: "Is there a cat?" → yes, no

**Mục tiêu thử nghiệm:**
- ⚠️ **Đếm số lượng đơn giản**: "How many dogs?" → 1, 2, 3
  - *Lưu ý: Đây là task khó cho CNN-only models, kết quả thấp là expected*

**So sánh 8 biến thể mô hình:**
- Attention vs No Attention
- Pretrained vs From-scratch
- ResNet50 vs VGG16

## Requirements

### ⚠️ Python Version
**REQUIRED: Python 3.10.x**

Dự án này yêu cầu Python 3.10 để đảm bảo tương thích với tất cả dependencies.

### System Requirements
- **Python**: 3.10.x (REQUIRED)
- **GPU**: NVIDIA GPU với CUDA support (recommended cho training)
- **RAM**: Tối thiểu 16GB
- **Storage**: Tối thiểu 50GB cho data và checkpoints

## Setup Instructions

### 1. Tạo Virtual Environment với Python 3.10

```bash
# Windows
py -3.10 -m venv venv
venv\Scripts\activate

# Linux/Mac
python3.10 -m venv venv
source venv/bin/activate
```

### 2. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 3. Cài đặt spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 4. (Optional) Cài đặt PyTorch với CUDA

Nếu có GPU NVIDIA, cài PyTorch với CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Cấu Trúc Dự Án

```
VQA_Workspace/ (Root)
├── Data_prep/              # Chuẩn bị dữ liệu từ COCO
│   ├── code/               # Scripts xử lý/lọc dữ liệu
│   └── data/               # Chứa dữ liệu thô (đã bị ignore)
├── VQA_Model/              # Mã nguồn chính của mô hình VQA
│   ├── configs/            # Cấu hình thí nghiệm
│   ├── models/             # Kiến trúc mô hình (CNN, RNN, Attention)
│   ├── engine/             # Logic training và evaluation
│   ├── utils/              # Tiện ích bổ trợ (Logger, Visualization)
│   └── data/               # Dataloader và xử lý ngôn ngữ
├── requirements.txt        # Python dependencies
├── .gitignore              # Cấu hình Git toàn cục
├── README.md               # Hướng dẫn dự án
└── PROJECT_PLAN.md         # Kế hoạch phát triển
```

## Quick Start

### Data Preparation

```bash
# Bước 1: Lọc ảnh có động vật bằng YOLO
cd Data_prep/code/filter
py -3.10 animal_filter.py

# Bước 2: Sinh Q&A annotations (coming soon)
cd ../scripts
py -3.10 generate_vqa_annotations.py

# Bước 3: Chia train/val/test (coming soon)
py -3.10 split_dataset.py
```

### Model Training (Coming Soon)

```bash
cd VQA_Model
python train.py --config configs/model1_resnet50_pretrained_no_attn.yaml
```

## Documentation

- **PROJECT_PLAN.md**: Kế hoạch chi tiết dự án, kiến trúc mô hình, metrics
- **Data_prep/PREDATA_GUIDE.md**: Hướng dẫn về vai trò của YOLO trong data preparation