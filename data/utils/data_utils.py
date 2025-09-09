#!/usr/bin/env python3
"""
Data utilities for MTS
Common utilities for data handling and processing
"""

import os
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data integrity and format"""
    
    @staticmethod
    def validate_mmlu_structure(data_dir: str) -> bool:
        """Validate MMLU dataset structure"""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            return False
        
        # Check for subject directories
        subject_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name != 'all']
        if not subject_dirs:
            logger.error("No subject directories found")
            return False
        
        # Check for required files in each subject
        for subject_dir in subject_dirs[:3]:  # Sample first 3 subjects
            arrow_files = list(subject_dir.glob("**/*.arrow"))
            required_splits = ['test', 'validation', 'dev']
            
            for split in required_splits:
                split_files = [f for f in arrow_files if split in f.name.lower()]
                if not split_files:
                    logger.warning(f"No {split} file found in {subject_dir.name}")
        
        logger.info(f"Found {len(subject_dirs)} subject directories")
        return True
    
    @staticmethod
    def validate_processed_data(data_dir: str) -> bool:
        """Validate processed data structure"""
        required_files = ['train.jsonl', 'val.jsonl', 'test.jsonl', 'dev_by_subject.json', 'config.json']
        
        for file_name in required_files:
            file_path = os.path.join(data_dir, file_name)
            if not os.path.exists(file_path):
                logger.error(f"Missing required file: {file_path}")
                return False
        
        logger.info("All required processed data files found")
        return True
    
    @staticmethod
    def validate_features(features_dir: str) -> bool:
        """Validate extracted features"""
        required_files = ['train_features.pkl', 'val_features.pkl', 'test_features.pkl']
        
        for file_name in required_files:
            file_path = os.path.join(features_dir, file_name)
            if not os.path.exists(file_path):
                logger.error(f"Missing feature file: {file_path}")
                return False
        
        logger.info("All required feature files found")
        return True


class DataStats:
    """Calculate and display data statistics"""
    
    @staticmethod
    def calculate_data_stats(data_dir: str) -> Dict[str, Any]:
        """Calculate comprehensive statistics for processed data"""
        stats = {}
        
        # Load configuration
        with open(os.path.join(data_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        stats['subjects'] = config['subjects']
        stats['total_subjects'] = config['total_subjects']
        stats['data_stats'] = config['data_stats']
        
        # Load dev examples
        with open(os.path.join(data_dir, 'dev_by_subject.json'), 'r') as f:
            dev_examples = json.load(f)
        
        # Calculate per-subject statistics
        subject_stats = {}
        for subject, examples in dev_examples.items():
            subject_stats[subject] = {
                'dev_examples': len(examples),
                'has_dev_examples': len(examples) >= 5
            }
        
        stats['subject_stats'] = subject_stats
        
        # Calculate answer distribution
        answer_dist = {0: 0, 1: 0, 2: 0, 3: 0}
        for split in ['train', 'val', 'test']:
            with open(os.path.join(data_dir, f'{split}.jsonl'), 'r') as f:
                for line in f:
                    item = json.loads(line)
                    answer = item.get('answer', 'A')
                    if isinstance(answer, str) and answer.isdigit():
                        answer_idx = int(answer)
                    elif isinstance(answer, str):
                        answer_idx = ord(answer.upper()) - ord('A')
                    else:
                        answer_idx = int(answer)
                    
                    if 0 <= answer_idx <= 3:
                        answer_dist[answer_idx] += 1
        
        stats['answer_distribution'] = answer_dist
        
        return stats
    
    @staticmethod
    def print_data_stats(stats: Dict[str, Any]):
        """Print formatted data statistics"""
        print("\n" + "="*50)
        print("DATA STATISTICS")
        print("="*50)
        
        print(f"Total subjects: {stats['total_subjects']}")
        print(f"Total train samples: {stats['data_stats']['total_train']}")
        print(f"Total val samples: {stats['data_stats']['total_val']}")
        print(f"Total test samples: {stats['data_stats']['total_test']}")
        
        print("\nAnswer distribution:")
        for i, count in stats['answer_distribution'].items():
            print(f"  {chr(65+i)}: {count}")
        
        print("\nSubject-wise dev examples:")
        subjects_with_enough_dev = sum(1 for s in stats['subject_stats'].values() if s['has_dev_examples'])
        print(f"  Subjects with >=5 dev examples: {subjects_with_enough_dev}/{stats['total_subjects']}")
        
        print("="*50)


class DataSplitter:
    """Utilities for splitting data"""
    
    @staticmethod
    def split_train_val(train_data: List[Dict], val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        """Split training data into train and validation sets"""
        val_size = max(1, int(len(train_data) * val_ratio))
        val_data = train_data[-val_size:]
        train_data = train_data[:-val_size]
        
        return train_data, val_data
    
    @staticmethod
    def stratified_split(data: List[Dict], val_ratio: float = 0.1, stratify_key: str = 'subject') -> Tuple[List[Dict], List[Dict]]:
        """Stratified split based on subject"""
        from collections import defaultdict
        
        # Group by subject
        subject_groups = defaultdict(list)
        for item in data:
            subject_groups[item[stratify_key]].append(item)
        
        train_data, val_data = [], []
        
        for subject, items in subject_groups.items():
            val_size = max(1, int(len(items) * val_ratio))
            val_data.extend(items[-val_size:])
            train_data.extend(items[:-val_size])
        
        return train_data, val_data


class DataExporter:
    """Export data to different formats"""
    
    @staticmethod
    def export_to_csv(data_dir: str, output_dir: str):
        """Export processed data to CSV format"""
        import pandas as pd
        
        os.makedirs(output_dir, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            data = []
            with open(os.path.join(data_dir, f'{split}.jsonl'), 'r') as f:
                for line in f:
                    item = json.loads(line)
                    data.append({
                        'question': item['question'],
                        'choices': '|'.join(item['choices']),
                        'answer': item['answer'],
                        'subject': item['subject'],
                        'domain_id': item['domain_id']
                    })
            
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(output_dir, f'{split}.csv'), index=False)
            logger.info(f"Exported {split} data to CSV: {len(data)} samples")
    
    @staticmethod
    def export_features_to_hdf5(features_dir: str, output_path: str):
        """Export features to HDF5 format"""
        import h5py
        
        with h5py.File(output_path, 'w') as f:
            for split in ['train', 'val', 'test']:
                feature_path = os.path.join(features_dir, f'{split}_features.pkl')
                if os.path.exists(feature_path):
                    with open(feature_path, 'rb') as pf:
                        data = pickle.load(pf)
                    
                    group = f.create_group(split)
                    group.create_dataset('features', data=data['features'].numpy())
                    group.create_dataset('labels', data=data['labels'].numpy())
                    group.create_dataset('logits', data=data['logits'].numpy())
                    group.create_dataset('domain_ids', data=data['domain_ids'].numpy())
                    
                    # Store subjects as strings
                    subjects_encoded = [s.encode('utf-8') for s in data['subjects']]
                    group.create_dataset('subjects', data=subjects_encoded)
        
        logger.info(f"Exported features to HDF5: {output_path}")


class PathManager:
    """Manages file paths for the project"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        
    def get_data_dir(self, create: bool = True) -> Path:
        """Get data directory"""
        path = self.base_dir / 'data'
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_processed_data_dir(self, create: bool = True) -> Path:
        """Get processed data directory"""
        path = self.get_data_dir() / 'processed'
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_features_dir(self, create: bool = True) -> Path:
        """Get features directory"""
        path = self.get_data_dir() / 'features'
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_outputs_dir(self, create: bool = True) -> Path:
        """Get outputs directory"""
        path = self.base_dir / 'outputs'
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_models_dir(self, create: bool = True) -> Path:
        """Get models directory"""
        path = self.get_outputs_dir() / 'models'
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_results_dir(self, create: bool = True) -> Path:
        """Get results directory"""
        path = self.get_outputs_dir() / 'results'
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path


def create_data_directories(base_dir: str = None):
    """Create all necessary data directories"""
    path_manager = PathManager(base_dir)
    
    directories = [
        path_manager.get_data_dir(),
        path_manager.get_processed_data_dir(),
        path_manager.get_features_dir(),
        path_manager.get_outputs_dir(),
        path_manager.get_models_dir(),
        path_manager.get_results_dir()
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


if __name__ == "__main__":
    # Test data utilities
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    path_manager = PathManager()
    print(f"Data directory: {path_manager.get_data_dir()}")
    print(f"Processed data directory: {path_manager.get_processed_data_dir()}")
    print(f"Features directory: {path_manager.get_features_dir()}")
    print(f"Outputs directory: {path_manager.get_outputs_dir()}")
    
    create_data_directories()