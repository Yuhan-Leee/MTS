#!/usr/bin/env python3
"""
Quick test script for MTS
This script runs a minimal test to verify the system works
"""

import sys
import os

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils import setup_logging, get_device
from data.processing import DataPreprocessor
import logging

def main():
    """Run quick test"""
    print("🧪 MTS Quick Test")
    print("=" * 30)
    
    # Setup logging
    setup_logging('INFO')
    
    # Get device
    device = get_device()
    print(f"Device: {device}")
    
    # Test data preprocessing with minimal data
    print("\n📊 Testing data preprocessing...")
    
    # Use a small subset for testing
    test_input_dir = "/mnt/sharedata/ssd_large/common/datasets/cais___mmlu"
    test_output_dir = "test_outputs"
    
    if not os.path.exists(test_input_dir):
        print(f"❌ Test data directory not found: {test_input_dir}")
        print("Please update the path in the script")
        return False
    
    # Create test preprocessor
    preprocessor = DataPreprocessor(test_input_dir, test_output_dir)
    
    # Process just first few subjects for testing
    subject_dirs = [d for d in preprocessor.input_dir.iterdir() 
                   if d.is_dir() and d.name != 'all'][:3]  # Only first 3 subjects
    
    print(f"Processing {len(subject_dirs)} subjects for testing...")
    
    all_train, all_val, all_test = [], [], []
    dev_by_subject = {}
    
    for i, subject_dir in enumerate(subject_dirs):
        subject = subject_dir.name
        print(f"  Processing {subject}...")
        
        subject_data = preprocessor.load_subject_data(subject_dir, i)
        
        if not subject_data:
            continue
            
        new_train, new_val, new_test, new_dev = preprocessor.redistribute_data(subject_data)
        
        all_train.extend(new_train[:10])  # Only 10 samples for testing
        all_val.extend(new_val[:5])       # Only 5 samples for testing
        all_test.extend(new_test[:5])      # Only 5 samples for testing
        dev_by_subject[subject] = new_dev
        
        print(f"    {subject}: {len(new_train[:10])} train, {len(new_val[:5])} val, {len(new_test[:5])} test")
    
    # Save test data
    os.makedirs(test_output_dir, exist_ok=True)
    
    def save_jsonl(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    import json
    save_jsonl(all_train, os.path.join(test_output_dir, "train.jsonl"))
    save_jsonl(all_val, os.path.join(test_output_dir, "val.jsonl"))
    save_jsonl(all_test, os.path.join(test_output_dir, "test.jsonl"))
    
    with open(os.path.join(test_output_dir, "dev_by_subject.json"), 'w', encoding='utf-8') as f:
        json.dump(dev_by_subject, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Test data preprocessing completed!")
    print(f"   Train: {len(all_train)} samples")
    print(f"   Val: {len(all_val)} samples")
    print(f"   Test: {len(all_test)} samples")
    
    # Test feature extraction (optional, can be skipped due to time)
    print("\n🧠 Testing feature extraction (optional)...")
    print("   Note: This step may take time with a real model")
    print("   To test feature extraction, run:")
    print("   python scripts/extract_features.py --model_path /path/to/model --data_dir test_outputs --output_dir test_features")
    
    print("\n✅ Quick test completed successfully!")
    print("\nNext steps:")
    print("1. Full pipeline: python scripts/quick_start.py")
    print("2. Individual steps:")
    print("   - python scripts/preprocess_data.py --input_dir /path/to/mmlu --output_dir data/processed")
    print("   - python scripts/extract_features.py --model_path /path/to/model --data_dir data/processed --output_dir data/features")
    print("   - python scripts/train_models.py --features_dir data/features --output_dir outputs/results")
    
    return True

if __name__ == "__main__":
    main()