#!/usr/bin/env python3
"""
Main script for Multi-task Thermometer System (MTS)
This script orchestrates the entire pipeline from data preprocessing to model evaluation
"""

import sys
import os

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils import ConfigManager, setup_logging, get_device
from data.processing import DataPreprocessor
from data.features import FeatureExtractor
from src.training import ModelEvaluator
from src.evaluation import MetricsCalculator
import torch
import logging

logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-task Thermometer System (MTS)")
    parser.add_argument('--config', type=str, default='../config/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', choices=['preprocess', 'extract', 'train', 'all'],
                       default='all', help='Mode of operation')
    parser.add_argument('--log_level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager(args.config)
    config.create_directories()
    
    # Setup logging
    setup_logging(args.log_level, config.get('logging.file'))
    
    # Get device
    device = get_device(config.get('model.device'))
    logger.info(f"Using device: {device}")
    
    # Execute pipeline
    if args.mode in ['preprocess', 'all']:
        logger.info("=== Starting Data Preprocessing ===")
        preprocessor = DataPreprocessor(
            config.get('data.input_dir'),
            config.get('data.output_dir')
        )
        preprocessor.process_all_subjects()
    
    if args.mode in ['extract', 'all']:
        logger.info("=== Starting Feature Extraction ===")
        extractor = FeatureExtractor(config.get('model.path'), device)
        
        for split in ['train', 'val', 'test']:
            logger.info(f"Extracting features for {split} split...")
            extractor.extract_features(
                data_path=f"{config.get('data.output_dir')}/{split}.jsonl",
                dev_path=f"{config.get('data.output_dir')}/dev_by_subject.json",
                output_path=f"{config.get('output.features_dir')}/{split}_features.pkl",
                batch_size=config.get('model.batch_size'),
                max_length=config.get('data.max_length')
            )
    
    if args.mode in ['train', 'all']:
        logger.info("=== Starting Model Training ===")
        
        # Load features
        def load_features(path):
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
            return (data['features'].float(), data['labels'], data['logits'].float(), 
                    data['domain_ids'], data['subjects'])
        
        train_data = load_features(f"{config.get('output.features_dir')}/train_features.pkl")
        val_data = load_features(f"{config.get('output.features_dir')}/val_features.pkl")
        test_data = load_features(f"{config.get('output.features_dir')}/test_features.pkl")
        
        # Train and evaluate
        evaluator = ModelEvaluator(device)
        results = evaluator.compare_models(
            train_data, val_data, test_data,
            config.get('output.results_dir'),
            config.get('training.num_epochs')
        )
        
        logger.info("=== Pipeline Complete ===")
        logger.info(f"Results saved to: {config.get('output.results_dir')}/results.json")


if __name__ == "__main__":
    main()