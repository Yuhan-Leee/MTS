#!/usr/bin/env python3
"""
Training script for MTS
Standalone script for model training and evaluation
"""

import sys
import os

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.training import ModelEvaluator
from src.utils import setup_logging, get_device

def main():
    """Run model training and evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and evaluate MTS models")
    parser.add_argument('--features_dir', required=True, help="Directory containing features")
    parser.add_argument('--output_dir', required=True, help="Output directory for results")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Get device
    device = get_device()
    
    # Load features
    import pickle
    def load_features(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return (data['features'].float(), data['labels'], data['logits'].float(), 
                data['domain_ids'], data['subjects'])
    
    print("📊 Loading features...")
    train_data = load_features(f"{args.features_dir}/train_features.pkl")
    val_data = load_features(f"{args.features_dir}/val_features.pkl")
    test_data = load_features(f"{args.features_dir}/test_features.pkl")
    
    print(f"   Train: {len(train_data[0])} samples")
    print(f"   Val: {len(val_data[0])} samples")
    print(f"   Test: {len(test_data[0])} samples")
    
    # Train and evaluate
    print("🎯 Training and evaluating models...")
    evaluator = ModelEvaluator(device)
    results = evaluator.compare_models(
        train_data, val_data, test_data,
        args.output_dir,
        args.num_epochs
    )
    
    print("\n✅ Training and evaluation complete!")
    print(f"📁 Results saved to: {args.output_dir}")
    print("\n🎉 Key Results:")
    print(f"   Vanilla Accuracy: {results['vanilla']['accuracy']:.4f}")
    print(f"   Vanilla ECE: {results['vanilla']['ece']:.4f}")
    print(f"   Thermometer Accuracy: {results['thermometer']['accuracy']:.4f}")
    print(f"   Thermometer ECE: {results['thermometer']['ece']:.4f}")
    
    improvement = results['vanilla']['ece'] - results['thermometer']['ece']
    print(f"   ECE Improvement: {improvement:.4f}")


if __name__ == "__main__":
    main()