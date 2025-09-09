#!/usr/bin/env python3
"""
Quick test script for per-subject functionality
"""

import sys
import os

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.training import SubjectWiseTrainer
from src.utils import setup_logging, get_device
import torch
import numpy as np

def test_subject_wise_trainer():
    """Test SubjectWiseTrainer with synthetic data"""
    print("🧪 Testing SubjectWiseTrainer...")
    
    # Setup
    setup_logging('INFO')
    device = get_device()
    
    # Create synthetic data
    num_samples = 100
    feature_dim = 64
    num_classes = 4
    num_subjects = 5
    
    # Generate synthetic features, labels, and subjects
    features = torch.randn(num_samples, feature_dim)
    labels = torch.randint(0, num_classes, (num_samples,))
    logits = torch.randn(num_samples, num_classes)
    domain_ids = torch.zeros(num_samples)
    subjects = [f"subject_{i % num_subjects}" for i in range(num_samples)]
    
    # Split data
    train_size = 60
    val_size = 20
    test_size = 20
    
    train_data = (features[:train_size], labels[:train_size], logits[:train_size], 
                 domain_ids[:train_size], subjects[:train_size])
    val_data = (features[train_size:train_size+val_size], labels[train_size:train_size+val_size], 
                logits[train_size:train_size+val_size], domain_ids[train_size:train_size+val_size], 
                subjects[train_size:train_size+val_size])
    test_data = (features[train_size+val_size:], labels[train_size+val_size:], 
                 logits[train_size+val_size:], domain_ids[train_size+val_size:], 
                 subjects[train_size+val_size:])
    
    print(f"Created synthetic data:")
    print(f"  Train: {len(train_data[0])} samples")
    print(f"  Val: {len(val_data[0])} samples")
    print(f"  Test: {len(test_data[0])} samples")
    print(f"  Subjects: {len(set(subjects))}")
    
    # Test SubjectWiseTrainer
    trainer = SubjectWiseTrainer(device)
    
    # Test training a single subject
    print("\n🎯 Testing single subject training...")
    subject = subjects[0]
    model = trainer.train_subject_model(subject, train_data, val_data, num_epochs=2)
    
    if model is not None:
        print(f"✅ Successfully trained model for {subject}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    else:
        print(f"❌ Failed to train model for {subject}")
        return False
    
    # Test ensemble evaluation
    print("\n📊 Testing ensemble evaluation...")
    try:
        subject_models = {subject: model}
        acc, ece = trainer._evaluate_ensemble(subject_models, test_data)
        print(f"✅ Ensemble evaluation successful")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   ECE: {ece:.4f}")
    except Exception as e:
        print(f"❌ Ensemble evaluation failed: {e}")
        return False
    
    print("\n✅ SubjectWiseTrainer test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_subject_wise_trainer()
    if success:
        print("\n🎉 All tests passed! Subject-wise functionality is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")