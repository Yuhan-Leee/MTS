#!/usr/bin/env python3
"""
Multi-task Thermometer System (MTS)
A comprehensive system for temperature scaling and calibration of language models
"""

__version__ = "1.0.0"
__author__ = "MTS Team"

# Core modules
from .models import ThermometerModel
from .training import Trainer, ModelEvaluator
from .evaluation import ModelEvaluator as FullEvaluator, MetricsCalculator
from .utils import setup_logging, load_config, ConfigManager, get_device

# Data processing modules (re-export for convenience)
try:
    from data.processing import DataPreprocessor, DataLoader
    from data.features import FeatureExtractor, FeatureLoader
    from data.utils import DataValidator, DataStats, PathManager
    
    __all__ = [
        # Core modules
        'ThermometerModel',
        'Trainer',
        'ModelEvaluator',
        'FullEvaluator',
        'MetricsCalculator',
        'setup_logging',
        'load_config',
        'ConfigManager',
        'get_device',
        # Data processing modules
        'DataPreprocessor',
        'DataLoader',
        'FeatureExtractor',
        'FeatureLoader',
        'DataValidator',
        'DataStats',
        'PathManager'
    ]
except ImportError:
    # If data modules are not available, export only core modules
    __all__ = [
        'ThermometerModel',
        'Trainer',
        'ModelEvaluator',
        'FullEvaluator',
        'MetricsCalculator',
        'setup_logging',
        'load_config',
        'ConfigManager',
        'get_device'
    ]