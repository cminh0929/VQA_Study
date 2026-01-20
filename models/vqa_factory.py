"""
Model Factory
Tạo VQA models với các cấu hình khác nhau
"""

import torch.nn as nn
from .image_encoder import ImageEncoder
from .question_encoder import QuestionEncoder
from .attention import SpatialAttention, MultiHeadSpatialAttention, StackedAttention
from .vqa_model import VQAModel


class ModelFactory:
    """Factory để tạo VQA models với các cấu hình khác nhau"""
    
    # Định nghĩa 8 model configs
    MODEL_CONFIGS = {
        'model_1': {
            'name': 'CNN(Scratch) + RNN + No Attention',
            'cnn_type': 'scratch',
            'cnn_pretrained': False,
            'freeze_cnn': False,
            'rnn_type': 'rnn',
            'bidirectional': False,
            'use_attention': False,
            'attention_type': None
        },
        'model_2': {
            'name': 'CNN(Scratch) + LSTM + No Attention',
            'cnn_type': 'scratch',
            'cnn_pretrained': False,
            'freeze_cnn': False,
            'rnn_type': 'lstm',
            'bidirectional': False,
            'use_attention': False,
            'attention_type': None
        },
        'model_3': {
            'name': 'CNN(Pretrained) + LSTM + No Attention',
            'cnn_type': 'resnet50',
            'cnn_pretrained': True,
            'freeze_cnn': True,
            'rnn_type': 'lstm',
            'bidirectional': False,
            'use_attention': False,
            'attention_type': None
        },
        'model_4': {
            'name': 'CNN(Pretrained) + LSTM + Spatial Attention',
            'cnn_type': 'resnet50',
            'cnn_pretrained': True,
            'freeze_cnn': True,
            'rnn_type': 'lstm',
            'bidirectional': False,
            'use_attention': True,
            'attention_type': 'spatial'
        },
        'model_5': {
            'name': 'CNN(Pretrained) + Bi-LSTM + Spatial Attention',
            'cnn_type': 'resnet50',
            'cnn_pretrained': True,
            'freeze_cnn': True,
            'rnn_type': 'lstm',
            'bidirectional': True,
            'use_attention': True,
            'attention_type': 'spatial'
        },
        'model_6': {
            'name': 'CNN(Pretrained) + Bi-LSTM + Multi-Head Attention',
            'cnn_type': 'resnet50',
            'cnn_pretrained': True,
            'freeze_cnn': True,
            'rnn_type': 'lstm',
            'bidirectional': True,
            'use_attention': True,
            'attention_type': 'multi_head'
        },
        'model_7': {
            'name': 'CNN(Fine-tune) + Bi-LSTM + Multi-Head Attention',
            'cnn_type': 'resnet50',
            'cnn_pretrained': True,
            'freeze_cnn': False,  # Fine-tune!
            'rnn_type': 'lstm',
            'bidirectional': True,
            'use_attention': True,
            'attention_type': 'multi_head'
        },
        'model_8': {
            'name': 'CNN(Pretrained) + Bi-LSTM + Stacked Attention',
            'cnn_type': 'resnet50',
            'cnn_pretrained': True,
            'freeze_cnn': True,
            'rnn_type': 'lstm',
            'bidirectional': True,
            'use_attention': True,
            'attention_type': 'stacked'
        }
    }
    
    @staticmethod
    def create_model(model_name, vocab_sizes, **kwargs):
        """
        Tạo model theo tên
        
        Args:
            model_name: Tên model ('model_1', 'model_2', ...)
            vocab_sizes: Dict {'question': int, 'answer': int}
            **kwargs: Override config parameters
            
        Returns:
            VQAModel instance
        """
        if model_name not in ModelFactory.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Available: {list(ModelFactory.MODEL_CONFIGS.keys())}")
        
        # Get config
        config = ModelFactory.MODEL_CONFIGS[model_name].copy()
        config.update(kwargs)  # Override với kwargs
        
        print(f"\nCreating {config['name']}...")
        print(f"  CNN: {config['cnn_type']} "
              f"(pretrained={config['cnn_pretrained']}, "
              f"freeze={config['freeze_cnn']})")
        print(f"  RNN: {config['rnn_type']} "
              f"(bidirectional={config['bidirectional']})")
        print(f"  Attention: {config['attention_type'] if config['use_attention'] else 'None'}")
        
        # Create model
        model = VQAModel(
            question_vocab_size=vocab_sizes['question'],
            answer_vocab_size=vocab_sizes['answer'],
            cnn_backbone=config['cnn_type'],
            cnn_pretrained=config['cnn_pretrained'],
            freeze_cnn=config['freeze_cnn'],
            bidirectional=config['bidirectional'],
            use_attention=config['use_attention'],
            attention_type=config['attention_type'],
            # Các params khác từ kwargs
            **{k: v for k, v in kwargs.items() 
               if k not in config}
        )
        
        return model
    
    @staticmethod
    def create_custom_model(vocab_sizes, 
                          cnn_type='resnet50',
                          cnn_pretrained=True,
                          freeze_cnn=True,
                          rnn_type='lstm',
                          bidirectional=False,
                          use_attention=False,
                          attention_type=None,
                          **kwargs):
        """
        Tạo model custom với config tùy chỉnh
        
        Args:
            vocab_sizes: Dict {'question': int, 'answer': int}
            cnn_type: 'scratch', 'resnet50', 'resnet101', 'vgg16'
            cnn_pretrained: Dùng pretrained weights
            freeze_cnn: Freeze CNN layers
            rnn_type: 'rnn' hoặc 'lstm'
            bidirectional: Bi-directional RNN
            use_attention: Có dùng attention không
            attention_type: 'spatial', 'multi_head', 'stacked'
            **kwargs: Các params khác
            
        Returns:
            VQAModel instance
        """
        print(f"\nCreating custom model...")
        print(f"  CNN: {cnn_type} (pretrained={cnn_pretrained}, freeze={freeze_cnn})")
        print(f"  RNN: {rnn_type} (bidirectional={bidirectional})")
        print(f"  Attention: {attention_type if use_attention else 'None'}")
        
        model = VQAModel(
            question_vocab_size=vocab_sizes['question'],
            answer_vocab_size=vocab_sizes['answer'],
            cnn_backbone=cnn_type,
            cnn_pretrained=cnn_pretrained,
            freeze_cnn=freeze_cnn,
            bidirectional=bidirectional,
            use_attention=use_attention,
            attention_type=attention_type,
            **kwargs
        )
        
        return model
    
    @staticmethod
    def list_models():
        """List tất cả models có sẵn"""
        print("\nAvailable Models:")
        print("=" * 80)
        for model_id, config in ModelFactory.MODEL_CONFIGS.items():
            print(f"{model_id:10s}: {config['name']}")
        print("=" * 80)
    
    @staticmethod
    def get_config(model_name):
        """Lấy config của model"""
        return ModelFactory.MODEL_CONFIGS.get(model_name, {}).copy()
    
    @staticmethod
    def compare_models():
        """So sánh configs của các models"""
        print("\nModel Comparison:")
        print("=" * 100)
        print(f"{'Model':<10} {'CNN':<15} {'Pretrain':<10} {'Freeze':<8} "
              f"{'RNN':<8} {'Bi-dir':<8} {'Attention':<15}")
        print("-" * 100)
        
        for model_id, config in ModelFactory.MODEL_CONFIGS.items():
            print(f"{model_id:<10} "
                  f"{config['cnn_type']:<15} "
                  f"{str(config['cnn_pretrained']):<10} "
                  f"{str(config['freeze_cnn']):<8} "
                  f"{config['rnn_type']:<8} "
                  f"{str(config['bidirectional']):<8} "
                  f"{config['attention_type'] or 'None':<15}")
        print("=" * 100)


# Convenience functions
def create_model(model_name, vocab_sizes, **kwargs):
    """Shortcut để tạo model"""
    return ModelFactory.create_model(model_name, vocab_sizes, **kwargs)


def create_all_models(vocab_sizes, **kwargs):
    """
    Tạo tất cả 8 models
    
    Args:
        vocab_sizes: Dict {'question': int, 'answer': int}
        **kwargs: Override params cho tất cả models
        
    Returns:
        Dict {model_name: model}
    """
    models = {}
    for model_name in ModelFactory.MODEL_CONFIGS.keys():
        models[model_name] = ModelFactory.create_model(
            model_name, 
            vocab_sizes, 
            **kwargs
        )
    return models


if __name__ == "__main__":
    # Demo
    print("VQA Model Factory")
    print("=" * 80)
    
    # List models
    ModelFactory.list_models()
    
    # Compare
    ModelFactory.compare_models()
    
    # Example usage
    print("\nExample Usage:")
    print("-" * 80)
    print("""
# Tạo model từ preset
model = create_model('model_3', vocab_sizes={'question': 1000, 'answer': 30})

# Tạo custom model
model = ModelFactory.create_custom_model(
    vocab_sizes={'question': 1000, 'answer': 30},
    cnn_type='resnet50',
    cnn_pretrained=True,
    freeze_cnn=True,
    use_attention=True,
    attention_type='spatial'
)

# Tạo tất cả models
models = create_all_models(vocab_sizes={'question': 1000, 'answer': 30})
    """)
