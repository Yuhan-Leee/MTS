#!/usr/bin/env python3
"""
Data processing utilities for MTS
Handles loading and preprocessing of MMLU dataset for 5-shot learning
"""

import os
import json
import datasets
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessor for MMLU 5-shot learning"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_subject_data(self, subject_dir: Path, domain_id: int) -> Dict[str, List[Dict]]:
        """Load data for a single subject"""
        subject = subject_dir.name
        arrow_files = list(subject_dir.glob("**/*.arrow"))
        
        subject_data = {}
        for arrow_file in arrow_files:
            try:
                file_name = arrow_file.name.lower()
                if 'test' in file_name:
                    split = 'test'
                elif 'validation' in file_name:
                    split = 'validation'  
                elif 'dev' in file_name:
                    split = 'dev'
                else:
                    continue
                
                dataset = datasets.load_dataset('arrow', data_files=str(arrow_file), split='train')
                items = []
                for item in dataset:
                    processed = {
                        "question": item.get("question", ""),
                        "choices": item.get("choices", []),
                        "answer": str(item.get("answer", "")),
                        "subject": item.get("subject", subject),
                        "domain_id": domain_id
                    }
                    items.append(processed)
                
                subject_data[split] = items
                logger.info(f"  {subject}/{split}: {len(items)} 样本")
                
            except Exception as e:
                logger.error(f"处理 {arrow_file} 出错: {e}")
        
        return subject_data
    
    def redistribute_data(self, subject_data: Dict[str, List[Dict]]) -> Tuple[List, List, List]:
        """Redistribute data: test->train, validation->test, dev->5shot"""
        original_test = subject_data.get('test', [])
        original_val = subject_data.get('validation', [])
        original_dev = subject_data.get('dev', [])
        
        # New allocation
        new_train = original_test  # Use original test for training
        new_test = original_val    # Use original validation for testing
        new_dev = original_dev     # Use dev as 5-shot examples
        
        # Split validation set from training
        train_size = len(new_train)
        if train_size > 10:
            val_size = max(5, train_size // 10)
            new_val = new_train[-val_size:]
            new_train = new_train[:-val_size]
        else:
            new_val = new_train[:1] if new_train else []
        
        return new_train, new_val, new_test, new_dev
    
    def save_jsonl(self, data: List[Dict], path: Path):
        """Save data as JSONL format"""
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def process_all_subjects(self):
        """Process all subjects in the dataset"""
        subject_dirs = [d for d in self.input_dir.iterdir() if d.is_dir() and d.name != 'all']
        logger.info(f"找到 {len(subject_dirs)} 个学科")
        
        all_train, all_val, all_test = [], [], []
        dev_by_subject = {}
        
        for i, subject_dir in enumerate(subject_dirs):
            subject = subject_dir.name
            subject_data = self.load_subject_data(subject_dir, i)
            
            if not subject_data:
                continue
                
            new_train, new_val, new_test, new_dev = self.redistribute_data(subject_data)
            
            all_train.extend(new_train)
            all_val.extend(new_val) 
            all_test.extend(new_test)
            dev_by_subject[subject] = new_dev
            
            logger.info(f"  {subject} 重分配: 训练={len(new_train)}, 验证={len(new_val)}, 测试={len(new_test)}, 5-shot={len(new_dev)}")
        
        # Save processed data
        self.save_jsonl(all_train, self.output_dir / "train.jsonl")
        self.save_jsonl(all_val, self.output_dir / "val.jsonl")
        self.save_jsonl(all_test, self.output_dir / "test.jsonl")
        
        with open(self.output_dir / "dev_by_subject.json", 'w', encoding='utf-8') as f:
            json.dump(dev_by_subject, f, indent=2, ensure_ascii=False)
        
        # Save configuration
        config = {
            'subjects': list(dev_by_subject.keys()),
            'total_subjects': len(dev_by_subject),
            'data_stats': {
                'total_train': len(all_train),
                'total_val': len(all_val),
                'total_test': len(all_test)
            }
        }
        
        with open(self.output_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"\n预处理完成!")
        logger.info(f"训练: {len(all_train)}, 验证: {len(all_val)}, 测试: {len(all_test)}")
        
        return config


class DataLoader:
    """Utility class for loading processed data"""
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        """Load JSONL file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    
    @staticmethod
    def load_json(file_path: str) -> Dict:
        """Load JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def load_config(data_dir: str) -> Dict:
        """Load dataset configuration"""
        return DataLoader.load_json(os.path.join(data_dir, "config.json"))
    
    @staticmethod
    def load_dev_examples(data_dir: str) -> Dict:
        """Load dev examples by subject"""
        return DataLoader.load_json(os.path.join(data_dir, "dev_by_subject.json"))


def main():
    parser = argparse.ArgumentParser(description="Preprocess MMLU data for 5-shot learning")
    parser.add_argument('--input_dir', required=True, help="Input directory containing MMLU data")
    parser.add_argument('--output_dir', required=True, help="Output directory for processed data")
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process data
    preprocessor = DataPreprocessor(args.input_dir, args.output_dir)
    preprocessor.process_all_subjects()


if __name__ == "__main__":
    main()