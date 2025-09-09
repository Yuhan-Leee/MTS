#!/usr/bin/env python3
"""
Feature extraction utilities for MTS
Extracts features from language models for 5-shot learning
"""

import os
import json
import torch
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class Dataset5Shot(Dataset):
    """Dataset for 5-shot feature extraction"""
    
    def __init__(self, data_path: str, dev_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load main data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        
        # Load 5-shot examples
        with open(dev_path, 'r', encoding='utf-8') as f:
            self.dev_examples = json.load(f)
        
        logger.info(f"加载 {len(self.data)} 样本, {len(self.dev_examples)} 学科5-shot")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        subject = item.get('subject', 'unknown')
        
        # Construct 5-shot prompt
        prompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
        
        # Add 5-shot examples
        if subject in self.dev_examples:
            for ex in self.dev_examples[subject][:5]:
                prompt += f"{ex['question']}\n"
                for j, choice in enumerate(ex['choices']):
                    prompt += f"{chr(65+j)}. {choice}\n"
                
                # Add answer
                answer = ex.get('answer', 'A')
                if isinstance(answer, str) and answer.isdigit():
                    answer_letter = chr(65 + int(answer))
                elif isinstance(answer, int):
                    answer_letter = chr(65 + answer)
                else:
                    answer_letter = answer.upper()
                prompt += f"Answer: {answer_letter}\n\n"
        
        # Add current question
        prompt += f"{item['question']}\n"
        for i, choice in enumerate(item['choices']):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        
        # Encode
        inputs = self.tokenizer(prompt, truncation=True, max_length=self.max_length, 
                               padding=False, return_tensors="pt")
        
        # Process answer
        answer = item.get('answer', 'A')
        if isinstance(answer, str) and answer.isdigit():
            answer_idx = int(answer)
        elif isinstance(answer, str):
            answer_idx = ord(answer.upper()) - ord('A')
        else:
            answer_idx = int(answer)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': answer_idx,
            'domain_id': item.get('domain_id', 0),
            'subject': subject
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader"""
    max_len = max([item['input_ids'].size(0) for item in batch])
    
    input_ids, attention_masks, labels, domain_ids, subjects = [], [], [], [], []
    
    for item in batch:
        input_id = item['input_ids']
        attention_mask = item['attention_mask']
        
        pad_length = max_len - input_id.size(0)
        if pad_length > 0:
            input_id = F.pad(input_id, (0, pad_length), value=0)
            attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
        
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append(item['label'])
        domain_ids.append(item['domain_id'])
        subjects.append(item['subject'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.tensor(labels),
        'domain_ids': torch.tensor(domain_ids),
        'subjects': subjects
    }


class FeatureExtractor:
    """Feature extractor for language models"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = self._get_device(device)
        self.model_path = model_path
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        logger.info(f"模型加载完成: {model_path}")
        logger.info(f"设备: {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def extract_features(self, data_path: str, dev_path: str, output_path: str, 
                        batch_size: int = 1, max_length: int = 1024):
        """Extract features from the model"""
        # Create dataset and dataloader
        dataset = Dataset5Shot(data_path, dev_path, self.tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                               collate_fn=collate_fn, num_workers=0)
        
        # Extract features
        self.model.eval()
        all_features, all_labels, all_logits, all_domain_ids, all_subjects = [], [], [], [], []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="提取5-shot特征"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, 
                                   output_hidden_states=True)
                
                hidden_states = outputs.hidden_states[-1]
                sequence_lengths = attention_mask.sum(dim=1) - 1
                features = hidden_states[range(len(sequence_lengths)), sequence_lengths]
                
                logits = outputs.logits[range(len(sequence_lengths)), sequence_lengths]
                choice_logits = logits[:, :4]  # A,B,C,D
                
                all_features.append(features.cpu())
                all_labels.append(batch['labels'])
                all_logits.append(choice_logits.cpu())
                all_domain_ids.append(batch['domain_ids'])
                all_subjects.extend(batch['subjects'])
        
        # Save results
        result = {
            'features': torch.cat(all_features, dim=0),
            'labels': torch.cat(all_labels, dim=0),
            'logits': torch.cat(all_logits, dim=0),
            'domain_ids': torch.cat(all_domain_ids, dim=0),
            'subjects': all_subjects
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        
        logger.info(f"特征提取完成: {result['features'].shape}")
        return result


class FeatureLoader:
    """Utility class for loading extracted features"""
    
    @staticmethod
    def load_features(feature_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """Load extracted features from pickle file"""
        with open(feature_path, 'rb') as f:
            data = pickle.load(f)
        
        return (
            data['features'].float(),
            data['labels'],
            data['logits'].float(),
            data['domain_ids'],
            data['subjects']
        )
    
    @staticmethod
    def load_all_features(features_dir: str) -> Dict[str, Tuple]:
        """Load all features from a directory"""
        features = {}
        for split in ['train', 'val', 'test']:
            feature_path = os.path.join(features_dir, f"{split}_features.pkl")
            if os.path.exists(feature_path):
                features[split] = FeatureLoader.load_features(feature_path)
        
        return features


def main():
    parser = argparse.ArgumentParser(description="Extract features for 5-shot learning")
    parser.add_argument('--model_path', required=True, help="Path to the model")
    parser.add_argument('--data_path', required=True, help="Path to the data file")
    parser.add_argument('--dev_path', required=True, help="Path to the dev examples")
    parser.add_argument('--output_path', required=True, help="Path to save features")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--max_length', type=int, default=1024, help="Maximum sequence length")
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Extract features
    extractor = FeatureExtractor(args.model_path)
    extractor.extract_features(
        args.data_path, args.dev_path, args.output_path, 
        args.batch_size, args.max_length
    )


if __name__ == "__main__":
    main()