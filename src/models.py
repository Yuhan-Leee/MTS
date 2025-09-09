#!/usr/bin/env python3
"""
Model definitions for MTS
Contains the Thermometer model and related neural network components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ThermometerModel(nn.Module):
    """
    Thermometer model for temperature scaling and calibration
    
    This model learns to predict optimal temperature scaling parameters
    for improving calibration of language model outputs.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 4):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Temperature head
        self.temperature_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
        logger.info(f"ThermometerModel initialized: input_dim={input_dim}, hidden_dim={hidden_dim}, num_classes={num_classes}")
    
    def forward(self, features: torch.Tensor, logits: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            features: Input features from the language model
            logits: Optional logits to apply temperature scaling
            
        Returns:
            Tuple of (calibrated_logits, temperature) or (class_logits, temperature)
        """
        x = self.feature_extractor(features)
        
        # Predict temperature
        temperature = self.temperature_head(x) + 0.1
        temperature = torch.clamp(temperature, 0.1, 25.0)
        
        # Get class logits
        class_logits = self.classifier(x)
        
        if logits is not None:
            # Apply temperature scaling
            calibrated_logits = logits / temperature.expand_as(logits)
            return calibrated_logits, temperature, class_logits
        else:
            return class_logits, temperature


class FeatureDataset(torch.utils.data.Dataset):
    """Dataset for features extracted from language models"""
    
    def __init__(self, features, labels, logits, domain_ids, subjects):
        self.features = features
        self.labels = labels
        self.logits = logits
        self.domain_ids = domain_ids
        self.subjects = subjects
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx],
            'logits': self.logits[idx],
            'domain_id': self.domain_ids[idx],
            'subject': self.subjects[idx]
        }


class BaselineModels:
    """Collection of baseline models for comparison"""
    
    @staticmethod
    def vanilla_baseline(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        """Vanilla baseline without temperature scaling"""
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(labels.numpy(), preds.numpy())
        correct_mask = labels.numpy() == preds.numpy()
        
        ece = BaselineModels.expected_calibration_error(correct_mask.astype(float), max_probs.numpy())
        
        return accuracy, ece
    
    @staticmethod
    def oracle_baseline(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float]:
        """Oracle baseline with optimal temperature scaling"""
        def ece_loss(temperature):
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            
            correct_mask = labels.numpy() == preds.numpy()
            ece = BaselineModels.expected_calibration_error(correct_mask.astype(float), max_probs.numpy())
            return ece
        
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(ece_loss, bounds=(0.1, 25.0), method='bounded')
        optimal_temp = result.x
        
        scaled_logits = logits / optimal_temp
        probs = torch.softmax(scaled_logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(labels.numpy(), preds.numpy())
        correct_mask = labels.numpy() == preds.numpy()
        ece = BaselineModels.expected_calibration_error(correct_mask.astype(float), max_probs.numpy())
        
        return accuracy, ece, optimal_temp
    
    @staticmethod
    def expected_calibration_error(y_true, y_prob, n_bins=10):
        """Calculate Expected Calibration Error (ECE)"""
        import numpy as np
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece