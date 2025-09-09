#!/usr/bin/env python3
"""
Utility functions for MTS
Includes logging setup, configuration loading, and helper functions
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import yaml


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to file"""
    config_path = Path(config_path)
    os.makedirs(config_path.parent, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, indent=2)
        else:
            json.dump(config, f, indent=2)


def get_device(device: str = 'auto') -> torch.device:
    """Get appropriate device for PyTorch"""
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def ensure_dir(path: str):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_model_size(model: torch.nn.Module) -> int:
    """Get model size in parameters"""
    return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multi-task Thermometer System (MTS)")
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--log_level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to log file')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, 
                       default='/mnt/sharedata/ssd_large/common/datasets/cais___mmlu',
                       help='Path to MMLU dataset')
    parser.add_argument('--model_path', type=str,
                       default='/mnt/sharedata/ssd_large/common/LLMs/Llama-2-7b-chat-hf',
                       help='Path to language model')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for feature extraction')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum sequence length')
    
    # Mode selection
    parser.add_argument('--mode', choices=['preprocess', 'extract', 'train', 'evaluate', 'all'],
                       default='all', help='Mode of operation')
    
    return parser.parse_args()


class ConfigManager:
    """Configuration manager for MTS"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.config_path = config_path
        
        # Set default values
        self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration values"""
        defaults = {
            'data': {
                'input_dir': '/mnt/sharedata/ssd_large/common/datasets/cais___mmlu',
                'output_dir': 'outputs/data_5shot',
                'max_length': 1024
            },
            'model': {
                'path': '/mnt/sharedata/ssd_large/common/LLMs/Llama-2-7b-chat-hf',
                'device': 'auto',
                'batch_size': 1
            },
            'training': {
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'num_epochs': 20,
                'batch_size': 64,
                'hidden_dim': 256
            },
            'output': {
                'features_dir': 'outputs/features_5shot',
                'models_dir': 'outputs/models_5shot',
                'results_dir': 'outputs/results_5shot'
            },
            'logging': {
                'level': 'INFO',
                'file': None
            }
        }
        
        # Merge defaults with existing config
        for section, values in defaults.items():
            if section not in self.config:
                self.config[section] = values
            else:
                for key, value in values.items():
                    if key not in self.config[section]:
                        self.config[section][key] = value
    
    def get(self, key: str, default=None):
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value with dot notation support"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str = None):
        """Save configuration to file"""
        if path is None:
            path = self.config_path
        save_config(self.config, path)
    
    def create_directories(self):
        """Create all necessary directories"""
        dirs = [
            self.get('data.output_dir'),
            self.get('output.features_dir'),
            self.get('output.models_dir'),
            self.get('output.results_dir')
        ]
        
        for dir_path in dirs:
            if dir_path:
                ensure_dir(dir_path)


def create_sample_config(output_path: str):
    """Create a sample configuration file"""
    sample_config = {
        'data': {
            'input_dir': '/path/to/mmlu/dataset',
            'output_dir': 'outputs/data_5shot',
            'max_length': 1024
        },
        'model': {
            'path': '/path/to/language/model',
            'device': 'auto',
            'batch_size': 1
        },
        'training': {
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'num_epochs': 20,
            'batch_size': 64,
            'hidden_dim': 256
        },
        'output': {
            'features_dir': 'outputs/features_5shot',
            'models_dir': 'outputs/models_5shot',
            'results_dir': 'outputs/results_5shot'
        },
        'logging': {
            'level': 'INFO',
            'file': None
        }
    }
    
    save_config(sample_config, output_path)


if __name__ == "__main__":
    # Create sample configuration
    create_sample_config('config/sample_config.yaml')
    print("Sample configuration created: config/sample_config.yaml")