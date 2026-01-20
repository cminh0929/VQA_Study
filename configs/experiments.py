"""
Experiment Configurations
Định nghĩa 8 models cho ablation study
"""

from .base_config import BaseConfig


class ExperimentConfigs:
    """8 model configurations cho ablation study"""
    
    EXPERIMENTS = {
        'model_1': {
            'name': 'CNN(Scratch) + RNN + No Attention',
            'description': 'Baseline - Train CNN from scratch',
            'cnn_backbone': 'scratch',
            'cnn_pretrained': False,
            'freeze_cnn': False,
            'rnn_type': 'rnn',
            'bidirectional': False,
            'use_attention': False,
            'attention_type': None,
            # Inherit from BaseConfig
            **BaseConfig.get_training_params()
        },
        
        'model_2': {
            'name': 'CNN(Scratch) + LSTM + No Attention',
            'description': 'Test RNN vs LSTM',
            'cnn_backbone': 'scratch',
            'cnn_pretrained': False,
            'freeze_cnn': False,
            'rnn_type': 'lstm',
            'bidirectional': False,
            'use_attention': False,
            'attention_type': None,
            **BaseConfig.get_training_params()
        },
        
        'model_3': {
            'name': 'CNN(Pretrained) + LSTM + No Attention',
            'description': 'Test transfer learning effect',
            'cnn_backbone': 'resnet50',
            'cnn_pretrained': True,
            'freeze_cnn': True,
            'rnn_type': 'lstm',
            'bidirectional': False,
            'use_attention': False,
            'attention_type': None,
            **BaseConfig.get_training_params()
        },
        
        'model_4': {
            'name': 'CNN(Pretrained) + LSTM + Spatial Attention',
            'description': 'Test attention mechanism',
            'cnn_backbone': 'resnet50',
            'cnn_pretrained': True,
            'freeze_cnn': True,
            'rnn_type': 'lstm',
            'bidirectional': False,
            'use_attention': True,
            'attention_type': 'spatial',
            **BaseConfig.get_training_params()
        },
        
        'model_5': {
            'name': 'CNN(Pretrained) + Bi-LSTM + Spatial Attention',
            'description': 'Test bidirectional RNN',
            'cnn_backbone': 'resnet50',
            'cnn_pretrained': True,
            'freeze_cnn': True,
            'rnn_type': 'lstm',
            'bidirectional': True,
            'use_attention': True,
            'attention_type': 'spatial',
            **BaseConfig.get_training_params()
        },
        
        'model_6': {
            'name': 'CNN(Pretrained) + Bi-LSTM + Multi-Head Attention',
            'description': 'Test multi-head attention',
            'cnn_backbone': 'resnet50',
            'cnn_pretrained': True,
            'freeze_cnn': True,
            'rnn_type': 'lstm',
            'bidirectional': True,
            'use_attention': True,
            'attention_type': 'multi_head',
            **BaseConfig.get_training_params()
        },
        
        'model_7': {
            'name': 'CNN(Fine-tune) + Bi-LSTM + Multi-Head Attention',
            'description': 'Test fine-tuning CNN',
            'cnn_backbone': 'resnet50',
            'cnn_pretrained': True,
            'freeze_cnn': False,  # Fine-tune!
            'rnn_type': 'lstm',
            'bidirectional': True,
            'use_attention': True,
            'attention_type': 'multi_head',
            **BaseConfig.get_training_params()
        },
        
        'model_8': {
            'name': 'CNN(Pretrained) + Bi-LSTM + Stacked Attention',
            'description': 'Test stacked attention',
            'cnn_backbone': 'resnet50',
            'cnn_pretrained': True,
            'freeze_cnn': True,
            'rnn_type': 'lstm',
            'bidirectional': True,
            'use_attention': True,
            'attention_type': 'stacked',
            **BaseConfig.get_training_params()
        }
    }
    
    @staticmethod
    def get_config(model_name):
        """Lấy config của một model"""
        return ExperimentConfigs.EXPERIMENTS.get(model_name, {}).copy()
    
    @staticmethod
    def list_experiments():
        """List tất cả experiments"""
        print("\n" + "="*80)
        print("ABLATION STUDY EXPERIMENTS")
        print("="*80)
        for model_id, config in ExperimentConfigs.EXPERIMENTS.items():
            print(f"\n{model_id}: {config['name']}")
            print(f"  Description: {config['description']}")
        print("="*80)
    
    @staticmethod
    def get_all_configs():
        """Lấy tất cả configs"""
        return ExperimentConfigs.EXPERIMENTS.copy()


if __name__ == "__main__":
    # Demo
    ExperimentConfigs.list_experiments()
