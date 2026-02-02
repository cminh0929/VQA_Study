# VISUAL QUESTION ANSWERING (VQA)

## 1. TỔNG QUAN DỰ ÁN

### 1.1. Yêu Cầu Đề Bài

**Mục tiêu**: Sử dụng **CNN** và **LSTM** để xây dựng kiến trúc cho bài toán Visual Question Answering.

**Input**: 
- Hình ảnh (Image)
- Câu hỏi (Question)

**Output**: 
- Câu trả lời (Answer) - **BẮT BUỘC** được sinh ra bởi **LSTM Decoder**

**Yêu cầu xây dựng**: Các loại kiến trúc khác nhau dựa trên:
1. **Attention Mechanism**: Không có vs Có dùng cơ chế Attention
2. **Training Strategy**: Train từ đầu (From-scratch) vs Sử dụng Pretrained model

**Yêu cầu đánh giá**: 
- Xác định các độ đo phù hợp
- So sánh các mô hình thông qua các độ đo này

### 1.2. Phạm Vi & Dữ Liệu

- **Nguồn dữ liệu**: MS COCO Dataset.
- **Quy mô**: ~3.000 - 5.000 hình ảnh chứa 10 loại động vật phổ biến.
- **Phương pháp sinh dữ liệu**: Sử dụng YOLO để detect objects và tự động sinh Q&A pairs.

**Loại câu hỏi (Theo thứ tự ưu tiên):**

1. **Nhận diện động vật (Animal Recognition)** - ⭐ MỤC TIÊU CHÍNH
   - Ví dụ: "What animal is in the image?" → "dog", "cat", "bird"
   - Phù hợp cho: **TẤT CẢ 8 models**
   - Expected accuracy: 50-90% (tùy model type)

2. **Nhận diện màu sắc (Color Recognition)** - ⭐ MỤC TIÊU CHÍNH
   - Ví dụ: "What color is the dog?" → "black", "white", "brown"
   - Phù hợp cho: **TẤT CẢ 8 models**
   - Expected accuracy: 40-80% (tùy model type)

3. **Câu hỏi Yes/No (Presence Detection)** - ⭐ MỤC TIÊU PHỤ
   - Ví dụ: "Is there a cat in the image?" → "yes", "no"
   - Phù hợp cho: **TẤT CẢ 8 models**
   - Expected accuracy: 60-95% (tùy model type)

4. **Đếm số lượng đơn giản (Simple Counting)** - ⚠️ MỤC TIÊU THỬ NGHIỆM
   - Ví dụ: "How many dogs are there?" → "1", "2", "3"
   - Phù hợp cho: **Pretrained models ONLY (1, 2, 5, 6)**
   - Expected accuracy: 45-65% (pretrained), ~10-20% (from-scratch)
   - **Lưu ý**: Đây là task KHÓ cho CNN-only models, kết quả thấp là expected

**⚠️ Lưu ý quan trọng về Counting:**
- CNN không có khả năng đếm tự nhiên (chỉ trích xuất features tổng thể)
- From-scratch models (3, 4, 7, 8) sẽ có accuracy RẤT THẤP (~10-20%) cho counting
- Điều này là **expected limitation**, không phải lỗi implementation
- Kết quả này sẽ được dùng để **chứng minh giá trị của Pretrained weights**

### 1.3. Kiến Trúc Mô Hình

#### 1.3.1. Tổng Quan Kiến Trúc (Theo Yêu Cầu Đề Bài)

Hệ thống VQA sử dụng kiến trúc **CNN-LSTM Encoder-Decoder**:

```
┌─────────────────────────────────────────────────────────────┐
│                         INPUT                                │
│  • Image (224x224x3)                                        │
│  • Question (text sequence)                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    ENCODER STAGE                             │
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  CNN Encoder     │         │  LSTM Encoder    │         │
│  │  (Image)         │         │  (Question)      │         │
│  │                  │         │                  │         │
│  │  ResNet50/VGG16  │         │  Embedding +     │         │
│  │  Pretrained/     │         │  2-layer LSTM    │         │
│  │  From-scratch    │         │  Hidden: 512-D   │         │
│  │                  │         │                  │         │
│  │  Output: 2048-D  │         │  Output: 512-D   │         │
│  └──────────────────┘         └──────────────────┘         │
│           ↓                            ↓                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    FUSION MODULE                             │
│                                                              │
│  Variant 1: Simple Concatenation (No Attention)            │
│  • Concat(image_feat, question_feat) → FC layers           │
│                                                              │
│  Variant 2: Attention Mechanism                             │
│  • Attention(image_feat, question_feat) → Weighted sum     │
│  • More focus on relevant image regions                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              DECODER STAGE (LSTM Decoder)                    │
│                                                              │
│  • 2-layer LSTM Decoder (Hidden: 512-D)                    │
│  • Input: Fused features + previous word embedding         │
│  • Teacher Forcing during training                          │
│  • Beam Search during inference                             │
│  • Output: Softmax over vocabulary (word-by-word)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                         OUTPUT                               │
│  Answer sequence (e.g., "dog", "black", "2")               │
└─────────────────────────────────────────────────────────────┘
```

#### 1.3.2. Ma Trận Các Mô Hình Cần Xây Dựng

Tổng cộng **8 mô hình** dựa trên 3 đặc điểm:
1. **CNN Architecture**: ResNet50 vs VGG16
2. **Pretrained vs From-scratch**: Pretrained trên ImageNet vs khởi tạo ngẫu nhiên
3. **Attention Mechanism**: Có vs Không

| Model ID | CNN Encoder | Pretrained | Attention | Question Encoder | Answer Decoder |
|----------|-------------|------------|-----------|------------------|----------------|
| **Model 1** | ResNet50 | ✅ Yes | ❌ No | LSTM | LSTM |
| **Model 2** | ResNet50 | ✅ Yes | ✅ Yes | LSTM | LSTM |
| **Model 3** | ResNet50 | ❌ No | ❌ No | LSTM | LSTM |
| **Model 4** | ResNet50 | ❌ No | ✅ Yes | LSTM | LSTM |
| **Model 5** | VGG16 | ✅ Yes | ❌ No | LSTM | LSTM |
| **Model 6** | VGG16 | ✅ Yes | ✅ Yes | LSTM | LSTM |
| **Model 7** | VGG16 | ❌ No | ❌ No | LSTM | LSTM |
| **Model 8** | VGG16 | ❌ No | ✅ Yes | LSTM | LSTM |

**Mục đích so sánh:**
- **Impact of Attention**: Model 1 vs 2, Model 3 vs 4, Model 5 vs 6, Model 7 vs 8
- **Impact of Pretrained CNN**: Model 1 vs 3, Model 2 vs 4, Model 5 vs 7, Model 6 vs 8
- **ResNet50 vs VGG16**: Model 1 vs 5, Model 2 vs 6, Model 3 vs 7, Model 4 vs 8

#### 1.3.3. Phân Tích Khả Năng Của Các Models

**⚠️ Lưu ý quan trọng về Question Types:**

| Model Type | Animal Recognition | Color Recognition | Counting (1-3) | Counting (4+) |
|------------|-------------------|-------------------|----------------|---------------|
| **Pretrained + Attention** (2, 6) | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐⭐ Good | ⭐⭐⭐ Fair |
| **Pretrained, No Attention** (1, 5) | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐ Good | ⭐⭐⭐ Fair | ⭐⭐ Poor |
| **From-scratch + Attention** (4, 8) | ⭐⭐⭐ Good | ⭐⭐⭐ Fair | ⭐⭐ Poor | ⭐ Very Poor |
| **From-scratch, No Attention** (3, 7) | ⭐⭐ Fair | ⭐⭐ Poor | ⭐ Very Poor | ❌ Fail |

**Giải thích:**
- **CNN không có khả năng đếm tự nhiên**: CNN chỉ trích xuất features tổng thể, không tách biệt từng object instance
- **Pretrained models**: Đã học visual features từ ImageNet, dễ transfer learning
- **From-scratch models**: Cần nhiều data hơn, khó học complex patterns như counting
- **Attention mechanism**: Giúp focus vào vùng quan trọng, cải thiện tất cả tasks

**Chiến lược Question Generation:**
- **Tier 1 (All models)**: Yes/No questions, Single-object recognition
- **Tier 2 (Pretrained only)**: Multi-class recognition, Simple counting (1-3)
- **Tier 3 (Pretrained + Attention)**: Complex counting (4+), Spatial reasoning

### 1.4. Thông Số Kỹ Thuật

- **Framework**: PyTorch
- **Image Encoder**: 
  - ResNet50 (output: 2048-D)
  - VGG16 (output: 4096-D)
- **Question Encoder**: 
  - LSTM (2 layers, hidden_size=512, bidirectional=False)
- **Answer Decoder**: 
  - LSTM (2 layers, hidden_size=512)
- **Word Embeddings**: 
  - GloVe 300-D (pretrained) hoặc Word2Vec
  - Vocabulary size: ~5000-10000 words
- **Attention Mechanism**: 
  - Additive Attention (Bahdanau-style)
  - Attention dimension: 512-D
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-5)
- **Loss Function**: Cross-Entropy Loss (for each word prediction)
- **Batch Size**: 32 hoặc 64
- **Max Sequence Length**: 
  - Question: 20 tokens
  - Answer: 10 tokens
- **Training Strategy**:
  - Teacher forcing ratio: 0.5 (50% of the time)
  - Gradient clipping: max_norm=5.0
  - Learning rate scheduler: ReduceLROnPlateau

### 1.5. Độ Đo Đánh Giá (Evaluation Metrics)

**Theo yêu cầu đề bài**: Xác định các độ đo phù hợp để so sánh các mô hình.

#### 1.5.1. Độ Đo Chính (Primary Metrics) - Trả Lời Yêu Cầu Đề Bài

**1. Accuracy (Độ chính xác) ⭐ QUAN TRỌNG NHẤT**
- **Định nghĩa**: Tỷ lệ câu trả lời khớp chính xác với ground truth
- **Công thức**: 
  ```
  Accuracy = (Số câu trả lời đúng) / (Tổng số câu hỏi)
  ```
- **Phân tích chi tiết**:
  - **Overall Accuracy**: Toàn bộ test set
  - **Per-category Accuracy**:
    - Animal Recognition Accuracy
    - Color Recognition Accuracy
    - Counting Accuracy (1-3)
    - Counting Accuracy (4+) - Expected to be low
- **Tại sao quan trọng**: 
  - ✅ Dễ hiểu, trực quan
  - ✅ Phản ánh trực tiếp khả năng trả lời đúng
  - ✅ Phù hợp cho VQA với câu trả lời ngắn (1-3 words)
- **Sử dụng để so sánh**:
  - Impact of Attention: Model 1 vs 2, 3 vs 4, 5 vs 6, 7 vs 8
  - Impact of Pretrained: Model 1 vs 3, 2 vs 4, 5 vs 7, 6 vs 8
  - ResNet50 vs VGG16: Model 1 vs 5, 2 vs 6, 3 vs 7, 4 vs 8

**2. BLEU Score (Bilingual Evaluation Understudy)**
- **Định nghĩa**: Đo lường sự tương đồng n-gram giữa câu trả lời sinh ra và ground truth
- **Variants sử dụng**:
  - **BLEU-1** (unigram): Phù hợp cho câu trả lời 1 từ ("dog", "cat")
  - **BLEU-4** (4-gram): Phù hợp cho câu trả lời dài hơn
- **Công thức**: Dựa trên precision của n-grams
- **Ưu điểm**: 
  - Phù hợp cho sequence generation (LSTM Decoder output)
  - Tính toán nhanh
  - Standard metric trong NLP
- **Nhược điểm**: 
  - Không xem xét synonyms ("dog" vs "puppy")
  - Quá strict với exact matching
- **Khi nào dùng**: So sánh quality của LSTM Decoder output

**3. F1 Score (Precision & Recall)**
- **Định nghĩa**: Harmonic mean của Precision và Recall
- **Công thức**:
  ```
  Precision = TP / (TP + FP)
  Recall = TP / (TP + FN)
  F1 = 2 * (Precision * Recall) / (Precision + Recall)
  ```
- **Ưu điểm**:
  - Cân bằng giữa precision và recall
  - Phù hợp khi dataset có class imbalance
- **Sử dụng**: Per-category evaluation (Animal, Color, Counting)

#### 1.5.2. Độ Đo Phụ (Secondary Metrics)

**4. METEOR (Metric for Evaluation of Translation)**
- **Định nghĩa**: Đo lường tương đồng với xem xét synonyms và stemming
- **Ưu điểm**: Correlation cao với human judgment
- **Khi nào dùng**: Nếu có multiple reference answers

**5. Training Efficiency Metrics**
- **Training time**: Thời gian huấn luyện (giờ)
- **Number of parameters**: Số lượng tham số (millions)
- **Convergence speed**: Số epochs để đạt best validation performance
- **GPU memory usage**: Lượng VRAM sử dụng (GB)
- **Mục đích**: So sánh trade-off giữa accuracy và computational cost

**6. Attention Visualization (Qualitative)**
- **Chỉ áp dụng**: Models 2, 4, 6, 8 (có Attention)
- **Đánh giá**:
  - Attention có focus vào vùng đúng không?
  - Attention có thay đổi theo question không?
- **Visualization**: Heatmap overlay trên ảnh gốc
- **Mục đích**: Giải thích interpretability của model

#### 1.5.3. Tóm Tắt: Độ Đo Nào Dùng Để So Sánh?

**Trả lời yêu cầu đề bài:**

| Mục đích so sánh | Độ đo chính | Độ đo phụ |
|------------------|-------------|-----------|
| **Overall Performance** | Overall Accuracy | BLEU-1, F1 Score |
| **Impact of Attention** | Accuracy (Model i vs i+1) | Training Time, Attention Viz |
| **Impact of Pretrained** | Accuracy (Pretrained vs Scratch) | Convergence Speed |
| **CNN Architecture** | Accuracy (ResNet50 vs VGG16) | Params, Memory Usage |
| **Per-task Performance** | Per-category Accuracy | F1 per category |
| **Decoder Quality** | BLEU-1, BLEU-4 | METEOR (optional) |

**Độ đo QUAN TRỌNG NHẤT**: 
1. **Overall Accuracy** - Đánh giá tổng thể
2. **Per-category Accuracy** - Phân tích chi tiết (Animal/Color/Counting)
3. **Training Time** - Đánh giá efficiency

#### 1.5.4. Kế Hoạch So Sánh (Theo Yêu Cầu Đề Bài)

**Bảng So Sánh Kết Quả:**

| Model | CNN | Pretrain | Attn | Overall Acc (%) | Animal Acc (%) | Color Acc (%) | Count Acc (%) | BLEU-1 | F1 Score | Time (h) |
|-------|-----|----------|------|-----------------|----------------|---------------|---------------|--------|----------|----------|
| Model 1 | ResNet50 | ✅ | ❌ | - | - | - | - | - | - | - |
| Model 2 | ResNet50 | ✅ | ✅ | - | - | - | - | - | - | - |
| Model 3 | ResNet50 | ❌ | ❌ | - | - | - | - | - | - | - |
| Model 4 | ResNet50 | ❌ | ✅ | - | - | - | - | - | - | - |
| Model 5 | VGG16 | ✅ | ❌ | - | - | - | - | - | - | - |
| Model 6 | VGG16 | ✅ | ✅ | - | - | - | - | - | - | - |
| Model 7 | VGG16 | ❌ | ❌ | - | - | - | - | - | - | - |
| Model 8 | VGG16 | ❌ | ✅ | - | - | - | - | - | - | - |

**Phân Tích So Sánh (Trả Lời Yêu Cầu Đề Bài):**

**1. Impact of Attention Mechanism:**
- **So sánh**: Model 1 vs 2, Model 3 vs 4, Model 5 vs 6, Model 7 vs 8
- **Câu hỏi**: Attention có cải thiện accuracy không? Cải thiện bao nhiêu %?
- **Độ đo**: Overall Accuracy, Per-category Accuracy, BLEU-1
- **Kỳ vọng**: Attention models (2, 4, 6, 8) sẽ tốt hơn, đặc biệt cho Color và Counting tasks

**2. Impact of Pretrained CNN:**
- **So sánh**: Model 1 vs 3, Model 2 vs 4, Model 5 vs 7, Model 6 vs 8
- **Câu hỏi**: Pretrained CNN giúp gì? Transfer learning có hiệu quả không?
- **Độ đo**: Overall Accuracy, Training Time, Convergence Speed
- **Kỳ vọng**: Pretrained models (1, 2, 5, 6) sẽ tốt hơn và hội tụ nhanh hơn

**3. ResNet50 vs VGG16:**
- **So sánh**: Model 1 vs 5, Model 2 vs 6, Model 3 vs 7, Model 4 vs 8
- **Câu hỏi**: CNN architecture nào tốt hơn cho VQA?
- **Độ đo**: Overall Accuracy, Number of Parameters
- **Kỳ vọng**: ResNet50 có thể tốt hơn nhờ residual connections

**Visualization:**
- **Bar chart**: Accuracy comparison across 8 models
- **Line plot**: Training curves (loss, accuracy over epochs)
- **Heatmap**: Attention visualization (Models 2, 4, 6, 8 only)

---

## 2. QUY TRÌNH THỰC HIỆN (ROADMAP)

Dự án được chia thành **5 giai đoạn chính**:

### Giai Đoạn 1: Chuẩn Bị Dữ Liệu (Data Preparation)

**Trạng thái hiện tại:**
- ✅ [Đã hoàn thành] Tải tệp chú thích (annotations) từ COCO
- ✅ [Đã hoàn thành] Lọc danh sách ảnh có chứa động vật
- ✅ [Đã hoàn thành] Tải hình ảnh về máy cục bộ

**Công việc cần làm:**
- 🔄 [Đang thực hiện] Tự động sinh câu hỏi và câu trả lời từ dữ liệu gốc
  - Generate 3 types of questions: Animal, Color, Counting
  - Create multiple reference answers for each question (for CIDEr, METEOR)
- ⏳ [Chờ thực hiện] Chia tập dữ liệu theo tỷ lệ: 70% Train / 15% Validation / 15% Test
- ⏳ [Chờ thực hiện] Data statistics và exploratory analysis
  - Distribution of question types
  - Answer length distribution
  - Vocabulary analysis

### Giai Đoạn 2: Tiền Xử Lý (Preprocessing)

**2.1. Xử lý hình ảnh:**
- Resize về kích thước 224x224
- Normalization: ImageNet mean & std
- Data augmentation (training only):
  - Random horizontal flip
  - Random rotation (±10 degrees)
  - Color jitter

**2.2. Xử lý văn bản:**
- **Tokenization**: Tách từ sử dụng NLTK hoặc spaCy
- **Vocabulary building**:
  - Build separate vocabularies for questions và answers
  - Add special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
  - Min frequency threshold: 2 (words appearing < 2 times → `<UNK>`)
- **Padding/Truncation**:
  - Questions: max_length = 20
  - Answers: max_length = 10
- **Word Embeddings**:
  - Load pretrained GloVe 300-D embeddings
  - Initialize unknown words randomly

**2.3. Mã hóa:**
- Convert text to token IDs
- Create attention masks for padded sequences

**2.4. DataLoader:**
- PyTorch Dataset class
- Batch collation với dynamic padding
- Data shuffling cho training set

### Giai Đoạn 3: Xây Dựng Mô Hình (Model Development)

**3.1. Implement Base Components:**

**A. CNN Encoder Module:**
```
- ResNet50Encoder (pretrained & from-scratch)
- VGG16Encoder (pretrained & from-scratch)
- Feature extraction layer
- Optional: Fine-tuning strategy
```

**B. Question Encoder Module:**
```
- Embedding layer (GloVe initialization)
- LSTM encoder (2 layers, hidden_size=512)
- Dropout for regularization
```

**C. Attention Module:**
```
- Additive Attention (Bahdanau-style)
- Attention weight computation
- Context vector generation
```

**D. Answer Decoder Module:**
```
- LSTM decoder (2 layers, hidden_size=512)
- Attention integration (for attention-based models)
- Output projection layer (vocab_size)
- Beam search implementation
```

**3.2. Assemble 8 Model Variants:**
- Model 1-8 theo ma trận đã định nghĩa
- Modular design để dễ dàng swap components
- Config files cho từng model variant

**3.3. Training Infrastructure:**
- Training loop với teacher forcing
- Validation loop
- Checkpoint saving/loading
- Early stopping mechanism
- Learning rate scheduling

**3.4. Inference Pipeline:**
- Beam search decoder
- Greedy decoding (baseline)
- Post-processing (remove special tokens)

### Giai Đoạn 4: Huấn Luyện Mô Hình (Training)

**4.1. Training Setup:**
- Huấn luyện từng biến thể mô hình (8 models)
- Số lượng epochs: 50-100 (với early stopping)
- Early stopping patience: 10 epochs
- Checkpoint frequency: Every 5 epochs
- Validation frequency: Every epoch

**4.2. Training Strategy:**
- **Teacher forcing ratio**: Start at 1.0, decay to 0.5
- **Gradient clipping**: max_norm = 5.0
- **Learning rate**: 
  - Initial: 0.001
  - Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- **Regularization**:
  - Dropout: 0.3
  - Weight decay: 1e-5

**4.3. Monitoring:**
- TensorBoard hoặc Weights & Biases
- Log metrics:
  - Training loss (per batch, per epoch)
  - Validation loss (per epoch)
  - Validation accuracy (per epoch)
  - Learning rate
- Save best model based on validation accuracy

**4.4. Experiment Tracking:**
- Log hyperparameters
- Save training logs
- Version control for model checkpoints

### Giai Đoạn 5: Đánh Giá & Phân Tích (Evaluation)

**5.1. Quantitative Evaluation:**
- Load best checkpoint cho mỗi model
- Evaluate trên test set
- Tính toán tất cả metrics:
  - Overall Accuracy
  - Per-category Accuracy (Animal, Color, Counting)
  - BLEU-1, BLEU-2, BLEU-3, BLEU-4
  - METEOR
  - CIDEr
  - ROUGE-L
- Statistical significance testing (t-test, ANOVA)

**5.2. Qualitative Analysis:**

**A. Training Curves:**
- Plot loss curves (training vs validation)
- Plot accuracy curves
- Identify overfitting/underfitting

**B. Attention Visualization:**
- Chỉ cho models 2, 4, 6, 8
- Generate attention heatmaps
- Overlay trên ảnh gốc
- Analyze attention patterns

**C. Sample Predictions:**
- Random sampling từ test set
- Show: Image + Question + Ground Truth + Prediction
- Highlight correct/incorrect predictions
- Error analysis

**D. Failure Case Analysis:**
- Identify common failure patterns
- Categorize errors:
  - Wrong object detection
  - Wrong color recognition
  - Wrong counting
  - Grammatical errors

**5.3. Comparative Analysis:**
- Create comparison tables
- Generate visualization charts
- Write analysis report:
  - Which model performs best?
  - Impact of attention?
  - Impact of pretrained CNN?
  - ResNet50 vs VGG16?
  - Trade-offs?

**5.4. Final Report:**
- Executive summary
- Methodology
- Results & Discussion
- Conclusions
- Future work recommendations
- Appendix: Sample predictions, attention maps

---

## 3. CẤU TRÚC THƯ MỤC DỰ ÁN

```
VQA_Workspace/ (Root)
├── Data_prep/                  # Giai đoạn 1: Chuẩn bị dữ liệu
│   ├── code/                   # Scripts lọc và xử lý ảnh COCO
│   └── data/                   # Chứa ảnh và annotations (local)
├── VQA_Model/                  # Giai đoạn chính: Phát triển model
│   ├── configs/                # Cấu hình cho 8 biến thể mô hình
│   ├── models/                 # Kiến trúc CNN, LSTM, Attention
│   ├── engine/                 # Logic Training & Evaluation
│   ├── utils/                  # Metrics, Visualization, Vocabulary
│   ├── data/                   # DataLoader và Q&A pairs đã xử lý
│   ├── checkpoints/            # Lưu trữ model (.pth)
│   ├── logs/                   # Log training (Tensorboard/txt)
│   └── results/                # Kết quả đánh giá cuối cùng
├── .gitignore                  # File ignore toàn cục
├── README.md                   # Giới thiệu tổng quan
└── PROJECT_PLAN.md             # Kế hoạch chi tiết (File này)
```

---

## 4. DEPENDENCIES & REQUIREMENTS

**Python Libraries:**
```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
nltk>=3.6.0
spacy>=3.1.0
Pillow>=8.3.0
tqdm>=4.62.0
tensorboard>=2.6.0
scikit-learn>=0.24.0

# For metrics
pycocoevalcap  # BLEU, METEOR, CIDEr, ROUGE
```

**Hardware Requirements:**
- GPU: NVIDIA GPU với ít nhất 8GB VRAM (recommended: RTX 3080 hoặc tốt hơn)
- RAM: Ít nhất 16GB
- Storage: Ít nhất 50GB cho data và checkpoints

---

## 5. TIMELINE DỰ KIẾN

| Giai Đoạn | Thời Gian | Mốc Quan Trọng |
|-----------|-----------|----------------|
| **Giai đoạn 1**: Data Preparation | 1 tuần | Q&A pairs generated, data splits ready |
| **Giai đoạn 2**: Preprocessing | 3-4 ngày | DataLoaders ready, embeddings loaded |
| **Giai đoạn 3**: Model Development | 1 tuần | All 8 models implemented & tested |
| **Giai đoạn 4**: Training | 1 tuần | All models trained (có thể parallel) |
| **Giai đoạn 5**: Evaluation | 1 tuần | All metrics computed, report written |

---


---

## 6. EXPECTED OUTCOMES

**Deliverables:**
1. ✅ 8 trained VQA models với different architectures
2. ✅ Comprehensive evaluation report với all metrics
3. ✅ Attention visualization cho attention-based models
4. ✅ Comparative analysis report
5. ✅ Code repository với documentation
6. ✅ Presentation slides/poster

**Research Questions Answered:**
- Does attention mechanism improve VQA performance?
- How much does pretrained CNN help?
- Which CNN architecture (ResNet50 vs VGG16) is better for VQA?
- What are the trade-offs between accuracy and computational cost?

**Potential Extensions:**
- Multi-modal fusion techniques (beyond simple concatenation)
- Transformer-based models (BERT for questions, ViT for images)
- Larger datasets (full VQA v2.0)
- More question types
- Real-time inference optimization
