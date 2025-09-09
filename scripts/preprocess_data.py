#!/usr/bin/env python3
"""
Data preprocessing script for MTS
Standalone script for data preprocessing
"""

import sys
import os

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data.processing import DataPreprocessor
from src.utils import setup_logging

def main():
    """Run data preprocessing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess MMLU data for MTS")
    parser.add_argument('--input_dir', required=True, help="Input directory containing MMLU data")
    parser.add_argument('--output_dir', required=True, help="Output directory for processed data")
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Process data
    preprocessor = DataPreprocessor(args.input_dir, args.output_dir)
    preprocessor.process_all_subjects()
    
    print(f"✅ Data preprocessing complete!")
    print(f"📁 Processed data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()