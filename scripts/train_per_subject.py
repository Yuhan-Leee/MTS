#!/usr/bin/env python3
"""
Per-subject Thermometer training script
Compares global vs per-subject Thermometer models
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
    """Run per-subject Thermometer experiments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare global vs per-subject Thermometer models")
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
    
    # Get unique subjects for analysis
    unique_subjects = list(set(train_data[4]))
    print(f"   Subjects: {len(unique_subjects)}")
    print(f"   Average samples per subject: {len(train_data[0]) / len(unique_subjects):.1f}")
    
    # Run comparison
    print("\n🎯 Comparing Global vs Per-subject Thermometer models...")
    evaluator = ModelEvaluator(device)
    results = evaluator.compare_global_vs_subject_wise(
        train_data, val_data, test_data,
        args.output_dir,
        args.num_epochs
    )
    
    print("\n✅ Per-subject experiment completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    
    # Print key insights
    print("\n🎯 Key Insights:")
    comparison = results['comparison']
    
    print(f"• Global Thermometer improves over Vanilla by {comparison['global_improvement_over_vanilla']:.4f} ECE")
    print(f"• Subject-wise Thermometer improves over Vanilla by {comparison['subject_wise_improvement_over_vanilla']:.4f} ECE")
    print(f"• Subject-wise vs Global improvement: {comparison['subject_wise_improvement_over_global']:.4f} ECE")
    print(f"• Parameter increase ratio: {comparison['parameter_increase_ratio']:.2f}x")
    
    if comparison['subject_wise_improvement_over_global'] > 0.01:
        print("✅ Subject-wise approach is recommended for better calibration")
    else:
        print("ℹ️ Global approach is sufficient - no significant benefit from per-subject models")
    
    print(f"\n📁 Check the detailed results in:")
    print(f"   {args.output_dir}/comparison_results.json")
    print(f"   {args.output_dir}/subject_wise/subject_wise_results.json")


if __name__ == "__main__":
    main()