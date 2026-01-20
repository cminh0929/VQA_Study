"""
Cấu hình tập trung cho mô hình VQA
Central configuration for VQA model
"""

import os
import torch

class Config:
    """Cấu hình cho mô hình VQA"""
    
    # ==================== Đường dẫn dữ liệu / Data Paths ====================
    DATA_DIR = "data"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    IMAGES_DIR = os.path.join(RAW_DATA_DIR, "images")
    
    # File chứa câu hỏi và câu trả lời (CSV/JSON)
    QA_FILE = os.path.join(RAW_DATA_DIR, "questions_answers.json")
    
    # Vocabulary files
    QUESTION_VOCAB_FILE = os.path.join(PROCESSED_DATA_DIR, "question_vocab.pkl")
    ANSWER_VOCAB_FILE = os.path.join(PROCESSED_DATA_DIR, "answer_vocab.pkl")
    
    # ==================== Cấu hình mô hình / Model Configuration ====================
    # CNN Feature Extractor
    CNN_BACKBONE = "resnet50"  # Options: "vgg16", "resnet50", "efficientnet"
    FREEZE_CNN = True  # Đóng băng các lớp CNN pretrained
    USE_SPATIAL_FEATURES = True  # Sử dụng spatial feature maps (7x7) thay vì vector
    SPATIAL_FEATURE_SIZE = 7  # Kích thước spatial grid (7x7)
    CNN_FEATURE_DIM = 2048  # Kích thước feature cho mỗi spatial location (ResNet50: 2048, VGG16: 512)
    IMAGE_FEATURE_DIM = 1024  # Kích thước sau khi project (để giảm chiều)
    
    # Question Encoder (LSTM)
    EMBEDDING_DIM = 300  # Kích thước word embedding
    LSTM_HIDDEN_SIZE = 512  # Kích thước hidden state của LSTM
    LSTM_NUM_LAYERS = 2  # Số lớp LSTM
    LSTM_DROPOUT = 0.3  # Dropout cho LSTM
    BIDIRECTIONAL = False  # Sử dụng bidirectional LSTM
    
    # Attention Mechanism (Spatial Attention)
    USE_ATTENTION = True  # Bật/tắt attention
    ATTENTION_TYPE = "spatial"  # Options: "spatial", "multi_head", "stacked"
    NUM_ATTENTION_HEADS = 8  # Số heads cho multi-head attention
    ATTENTION_DIM = 512  # Kích thước attention
    ATTENTION_DROPOUT = 0.2  # Dropout cho attention layer
    
    # Fusion & Classifier
    FUSION_DIM = 1024  # Kích thước sau khi fusion
    HIDDEN_DIMS = [1024, 512]  # Các lớp fully connected
    DROPOUT = 0.5  # Dropout cho classifier
    
    # Answer Vocabulary
    MAX_ANSWERS = 1000  # Số lượng câu trả lời tối đa (top-K most common)
    
    # ==================== Tiền xử lý / Preprocessing ====================
    # Image preprocessing
    IMAGE_SIZE = 224  # Resize ảnh về 224x224
    IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
    IMAGE_STD = [0.229, 0.224, 0.225]  # ImageNet std
    
    # Text preprocessing
    MAX_QUESTION_LENGTH = 20  # Độ dài tối đa của câu hỏi (padding)
    MIN_WORD_FREQ = 2  # Tần suất tối thiểu để từ được thêm vào vocabulary
    
    # ==================== Training Configuration ====================
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    # Learning rate scheduler
    LR_SCHEDULER = "ReduceLROnPlateau"  # Options: "ReduceLROnPlateau", "StepLR"
    LR_PATIENCE = 3  # Số epochs không cải thiện trước khi giảm LR
    LR_FACTOR = 0.5  # Factor để giảm learning rate
    
    # Early stopping
    EARLY_STOPPING = True
    PATIENCE = 10  # Số epochs không cải thiện trước khi dừng
    
    # ==================== Checkpointing ====================
    CHECKPOINT_DIR = "checkpoints"
    SAVE_BEST_ONLY = True  # Chỉ lưu model tốt nhất
    
    # ==================== Reinforcement Learning (Optional) ====================
    USE_RL = False  # Bật/tắt reinforcement learning
    RL_PRETRAIN_EPOCHS = 20  # Số epochs pretrain với supervised learning
    RL_FINETUNE_EPOCHS = 10  # Số epochs fine-tune với RL
    RL_REWARD_CORRECT = 1.0  # Reward cho câu trả lời đúng
    RL_REWARD_INCORRECT = 0.0  # Reward cho câu trả lời sai
    
    # ==================== Device Configuration ====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Số workers cho DataLoader
    
    # ==================== Logging ====================
    LOG_DIR = "logs"
    LOG_INTERVAL = 10  # Log mỗi N batches
    
    # ==================== Evaluation ====================
    EVAL_BATCH_SIZE = 64
    
    # ==================== Data Split ====================
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    @classmethod
    def display(cls):
        """Hiển thị tất cả cấu hình"""
        print("=" * 60)
        print("VQA MODEL CONFIGURATION")
        print("=" * 60)
        for key, value in cls.__dict__.items():
            if not key.startswith("_") and not callable(value):
                print(f"{key}: {value}")
        print("=" * 60)


if __name__ == "__main__":
    # Test configuration
    Config.display()
