#!/usr/bin/env python3
"""
Quick start script for MTS
Provides a simple interface to run the entire pipeline with default settings
"""

import sys
import os

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils import setup_logging, get_device
from data.processing import DataPreprocessor
from data.features import FeatureExtractor
from src.training import ModelEvaluator
import logging

logger = logging.getLogger(__name__)


def main():
    """Quick start with default settings"""
    # Setup logging
    setup_logging('INFO')
    
    # Default paths (modify these as needed)
    data_path = "/mnt/sharedata/ssd_large/common/datasets/cais___mmlu"
    model_path = "/mnt/sharedata/ssd_large/common/LLMs/Llama-2-7b-chat-hf"
    output_dir = "outputs"
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(f"{output_dir}/data_5shot", exist_ok=True)
    os.makedirs(f"{output_dir}/features_5shot", exist_ok=True)
    os.makedirs(f"{output_dir}/results_5shot", exist_ok=True)
    
    print("🔬 Starting MTS Quick Start Pipeline")
    print("=" * 50)
    
    # 1. Preprocess data
    print("📊 Step 1: Preprocessing data...")
    preprocessor = DataPreprocessor(data_path, f"{output_dir}/data_5shot")
    preprocessor.process_all_subjects()
    
    # 2. Extract features
    print("🧠 Step 2: Extracting features...")
    extractor = FeatureExtractor(model_path, device)
    
    for split in ['train', 'val', 'test']:
        print(f"   Extracting {split} features...")
        extractor.extract_features(
            data_path=f"{output_dir}/data_5shot/{split}.jsonl",
            dev_path=f"{output_dir}/data_5shot/dev_by_subject.json",
            output_path=f"{output_dir}/features_5shot/{split}_features.pkl",
            batch_size=1
        )
    
    # 3. Train and evaluate
    print("🎯 Step 3: Training and evaluating models...")
    
    # Load features
    import pickle
    def load_features(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return (data['features'].float(), data['labels'], data['logits'].float(), 
                data['domain_ids'], data['subjects'])
    
    train_data = load_features(f"{output_dir}/features_5shot/train_features.pkl")
    val_data = load_features(f"{output_dir}/features_5shot/val_features.pkl")
    test_data = load_features(f"{output_dir}/features_5shot/test_features.pkl")
    
    # Train and evaluate
    evaluator = ModelEvaluator(device)
    results = evaluator.compare_models(
        train_data, val_data, test_data,
        f"{output_dir}/results_5shot",
        num_epochs=20
    )
    
    print("\n✅ Pipeline Complete!")
    print(f"📁 Results saved to: {output_dir}/results_5shot/")
    print("\n🎉 Key Results:")
    print(f"   Vanilla Accuracy: {results['vanilla']['accuracy']:.4f}")
    print(f"   Vanilla ECE: {results['vanilla']['ece']:.4f}")
    print(f"   Thermometer Accuracy: {results['thermometer_global']['accuracy']:.4f}")
    print(f"   Thermometer ECE: {results['thermometer_global']['ece']:.4f}")
    
    improvement = results['vanilla']['ece'] - results['thermometer_global']['ece']
    print(f"   ECE Improvement: {improvement:.4f}")


if __name__ == "__main__":
    main()