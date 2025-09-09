#!/usr/bin/env python3
"""
Feature extraction script for MTS
Standalone script for feature extraction
"""

import sys
import os

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data.features import FeatureExtractor
from src.utils import setup_logging, get_device

def main():
    """Run feature extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features for MTS")
    parser.add_argument('--model_path', required=True, help="Path to the language model")
    parser.add_argument('--data_dir', required=True, help="Directory containing processed data")
    parser.add_argument('--output_dir', required=True, help="Output directory for features")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--max_length', type=int, default=1024, help="Maximum sequence length")
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Get device
    device = get_device()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract features
    extractor = FeatureExtractor(args.model_path, device)
    
    for split in ['train', 'val', 'test']:
        print(f"🧠 Extracting {split} features...")
        extractor.extract_features(
            data_path=f"{args.data_dir}/{split}.jsonl",
            dev_path=f"{args.data_dir}/dev_by_subject.json",
            output_path=f"{args.output_dir}/{split}_features.pkl",
            batch_size=args.batch_size,
            max_length=args.max_length
        )
    
    print(f"✅ Feature extraction complete!")
    print(f"📁 Features saved to: {args.output_dir}")


if __name__ == "__main__":
    main()