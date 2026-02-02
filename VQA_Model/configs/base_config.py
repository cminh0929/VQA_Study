"""
Base Configuration for VQA Training
Default hyperparameters and paths
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseConfig:
    """Base configuration for VQA models"""
    
    # Paths
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root: str = os.path.join(project_root, '..', 'Data_prep', 'data')
    
    train_json: str = os.path.join(data_root, 'annotations', 'train.json')
    val_json: str = os.path.join(data_root, 'annotations', 'val.json')
    test_json: str = os.path.join(data_root, 'annotations', 'test.json')
    image_dir: str = os.path.join(data_root, 'images')
    
    question_vocab: str = 'data/question_vocab.json'
    answer_vocab: str = 'data/answer_vocab.json'
    
    # Model architecture
    embed_dim: int = 300
    lstm_hidden_dim: int = 512
    attn_dim: int = 512
    num_lstm_layers: int = 2
    dropout: float = 0.5
    
    # Data
    max_question_len: int = 20
    max_answer_len: int = 10
    image_size: int = 224
    
    # Training
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    teacher_forcing_ratio: float = 0.5
    gradient_clip: float = 5.0
    
    # Optimization
    optimizer: str = 'adam'  # 'adam' or 'sgd'
    lr_scheduler: str = 'plateau'  # 'plateau', 'step', or None
    lr_patience: int = 3
    lr_factor: float = 0.5
    
    # Hardware
    device: str = 'cuda'  # 'cuda' or 'cpu'
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    save_every: int = 5
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    
    # Evaluation
    eval_batch_size: int = 32
    save_predictions: bool = True
    
    def __post_init__(self):
        """Validate and create directories"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def get_model_config(self, model_id: int):
        """Get configuration for specific model variant"""
        configs = {
            1: {'cnn_arch': 'resnet50', 'cnn_pretrained': True, 'use_attention': False},
            2: {'cnn_arch': 'resnet50', 'cnn_pretrained': True, 'use_attention': True},
            3: {'cnn_arch': 'resnet50', 'cnn_pretrained': False, 'use_attention': False},
            4: {'cnn_arch': 'resnet50', 'cnn_pretrained': False, 'use_attention': True},
            5: {'cnn_arch': 'vgg16', 'cnn_pretrained': True, 'use_attention': False},
            6: {'cnn_arch': 'vgg16', 'cnn_pretrained': True, 'use_attention': True},
            7: {'cnn_arch': 'vgg16', 'cnn_pretrained': False, 'use_attention': False},
            8: {'cnn_arch': 'vgg16', 'cnn_pretrained': False, 'use_attention': True},
        }
        return configs.get(model_id, configs[1])
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# Default config instance
default_config = BaseConfig()


# Example usage
if __name__ == "__main__":
    config = BaseConfig()
    
    print("Base Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Device: {config.device}")
    
    print("\nModel 2 config:")
    model_config = config.get_model_config(2)
    print(f"  {model_config}")
