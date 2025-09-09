#!/usr/bin/env python3
"""
Training module for MTS
Handles training of the Thermometer model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import logging
import os
import json

from .models import ThermometerModel, BaselineModels
from data.features import FeatureLoader

logger = logging.getLogger(__name__)


class SubjectWiseTrainer:
    """Trainer for per-subject Thermometer models"""
    
    def __init__(self, device: torch.device, learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        logger.info(f"SubjectWiseTrainer initialized: lr={learning_rate}, weight_decay={weight_decay}")
    
    def train_subject_model(self, subject: str, train_data: Tuple, val_data: Tuple, 
                           num_epochs: int = 20) -> ThermometerModel:
        """Train a separate Thermometer model for a specific subject"""
        
        train_features, train_labels, train_logits, train_domain_ids, train_subjects = train_data
        val_features, val_labels, val_logits, val_domain_ids, val_subjects = val_data
        
        # Filter data for this subject
        subject_mask = np.array([s == subject for s in train_subjects])
        subject_val_mask = np.array([s == subject for s in val_subjects])
        
        if np.sum(subject_mask) == 0:
            logger.warning(f"No training data found for subject: {subject}")
            return None
        
        # Extract subject-specific data
        subject_train_features = train_features[subject_mask]
        subject_train_labels = train_labels[subject_mask]
        subject_train_logits = train_logits[subject_mask]
        
        subject_val_features = val_features[subject_val_mask]
        subject_val_labels = val_labels[subject_val_mask]
        subject_val_logits = val_logits[subject_val_mask]
        
        logger.info(f"Training model for {subject}: {len(subject_train_features)} train, {len(subject_val_features)} val samples")
        
        # Create model
        model = ThermometerModel(train_features.shape[1]).to(self.device)
        trainer = Trainer(model, self.device, self.learning_rate, self.weight_decay)
        
        # Create data loaders
        from .models import FeatureDataset
        train_dataset = FeatureDataset(
            subject_train_features, subject_train_labels, subject_train_logits,
            train_domain_ids[subject_mask], [subject] * len(subject_train_features)
        )
        val_dataset = FeatureDataset(
            subject_val_features, subject_val_labels, subject_val_logits,
            val_domain_ids[subject_val_mask], [subject] * len(subject_val_features)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Train model
        model = trainer.train(train_loader, val_loader, num_epochs, save_dir=None)
        
        return model
    
    def train_all_subjects(self, train_data: Tuple, val_data: Tuple, test_data: Tuple,
                          output_dir: str, num_epochs: int = 20) -> Dict[str, Any]:
        """Train separate models for all subjects"""
        
        train_features, train_labels, train_logits, train_domain_ids, train_subjects = train_data
        test_features, test_labels, test_logits, test_domain_ids, test_subjects = test_data
        
        # Get unique subjects
        unique_subjects = list(set(train_subjects))
        logger.info(f"Found {len(unique_subjects)} unique subjects")
        
        results = {
            'subject_models': {},
            'overall_results': {},
            'comparison': {}
        }
        
        # Train model for each subject
        subject_models = {}
        subject_results = {}
        
        for subject in tqdm(unique_subjects, desc="Training subject models"):
            model = self.train_subject_model(subject, train_data, (val_data[0], val_data[1], val_data[2], val_data[3], val_data[4]), num_epochs)
            
            if model is not None:
                subject_models[subject] = model
                
                # Evaluate on subject-specific test data
                subject_test_mask = np.array([s == subject for s in test_subjects])
                if np.sum(subject_test_mask) > 0:
                    subject_test_features = test_features[subject_test_mask]
                    subject_test_labels = test_labels[subject_test_mask]
                    subject_test_logits = test_logits[subject_test_mask]
                    
                    # Evaluate
                    evaluator = ModelEvaluator(self.device)
                    subject_acc, subject_ece = evaluator.evaluate_thermometer(
                        model, 
                        self._create_subject_loader(subject_test_features, subject_test_labels, subject_test_logits, subject)
                    )
                    
                    subject_results[subject] = {
                        'accuracy': float(subject_acc),
                        'ece': float(subject_ece),
                        'num_samples': int(np.sum(subject_test_mask))
                    }
                    
                    logger.info(f"   {subject}: Acc={subject_acc:.4f}, ECE={subject_ece:.4f}, Samples={np.sum(subject_test_mask)}")
        
        # Calculate overall results (ensemble approach)
        overall_acc, overall_ece = self._evaluate_ensemble(subject_models, test_data)
        
        results['subject_models'] = subject_results
        results['overall_results'] = {
            'accuracy': float(overall_acc),
            'ece': float(overall_ece),
            'num_subjects': len(subject_models)
        }
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'subject_wise_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Subject-wise training completed. Overall: Acc={overall_acc:.4f}, ECE={overall_ece:.4f}")
        
        return results
    
    def _create_subject_loader(self, features, labels, logits, subject):
        """Create a data loader for subject-specific evaluation"""
        from .models import FeatureDataset
        dataset = FeatureDataset(features, labels, logits, 
                               torch.zeros(len(features)), [subject] * len(features))
        return DataLoader(dataset, batch_size=64, shuffle=False)
    
    def _evaluate_ensemble(self, subject_models: Dict, test_data: Tuple) -> Tuple[float, float]:
        """Evaluate ensemble of subject-specific models"""
        
        test_features, test_labels, test_logits, test_domain_ids, test_subjects = test_data
        
        all_preds, all_labels, all_probs = [], [], []
        
        # Set all models to eval mode
        for model in subject_models.values():
            model.eval()
        
        with torch.no_grad():
            for i in range(len(test_features)):
                subject = test_subjects[i]
                features = test_features[i:i+1].to(self.device)
                labels = test_labels[i:i+1]
                logits = test_logits[i:i+1].to(self.device)
                
                if subject in subject_models:
                    model = subject_models[subject]
                    calibrated_logits, _, _ = model(features, logits)
                    probs = torch.softmax(calibrated_logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(torch.max(probs, dim=1)[0].cpu().numpy())
                else:
                    # Fallback to vanilla prediction for unseen subjects
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(torch.max(probs, dim=1)[0].cpu().numpy())
                
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        ece = BaselineModels.expected_calibration_error(
            np.array(all_labels) == np.array(all_preds), 
            np.array(all_probs)
        )
        
        return accuracy, ece


class Trainer:
    """Trainer for the Thermometer model"""
    
    def __init__(self, model: ThermometerModel, device: torch.device, 
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=0.7)
        
        logger.info(f"Trainer initialized: lr={learning_rate}, weight_decay={weight_decay}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            logits = batch['logits'].to(self.device)
            
            self.optimizer.zero_grad()
            
            calibrated_logits, temperature, class_logits = self.model(features, logits)
            
            # Loss components
            class_loss = self.criterion(class_logits, labels)
            cal_loss = self.criterion(calibrated_logits, labels)
            temp_reg = 0.01 * torch.mean((temperature.squeeze() - 3.0) ** 2)
            
            loss = class_loss + cal_loss + temp_reg
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        val_preds, val_labels, val_probs = [], [], []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                logits = batch['logits'].to(self.device)
                
                calibrated_logits, _, _ = self.model(features, logits)
                probs = torch.softmax(calibrated_logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(torch.max(probs, dim=1)[0].cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_ece = BaselineModels.expected_calibration_error(
            np.array(val_labels) == np.array(val_preds), 
            np.array(val_probs)
        )
        
        return val_acc, val_ece
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 20, save_dir: str = None) -> ThermometerModel:
        """Full training loop"""
        best_val_ece = float('inf')
        best_state = None
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_acc, val_ece = self.validate(val_loader)
            self.scheduler.step(val_ece)
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val Acc={val_acc:.4f}, Val ECE={val_ece:.4f}")
            
            # Save best model
            if val_ece < best_val_ece:
                best_val_ece = val_ece
                best_state = self.model.state_dict().copy()
                
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(best_state, os.path.join(save_dir, 'best_model.pth'))
                    logger.info(f"Saved best model with ECE: {val_ece:.4f}")
        
        # Load best model
        if best_state:
            self.model.load_state_dict(best_state)
            logger.info(f"Loaded best model with ECE: {best_val_ece:.4f}")
        
        return self.model


class ModelEvaluator:
    """Evaluator for comparing different models"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def evaluate_thermometer(self, model: ThermometerModel, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate Thermometer model"""
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                logits = batch['logits'].to(self.device)
                
                calibrated_logits, _, _ = model(features, logits)
                probs = torch.softmax(calibrated_logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.max(probs, dim=1)[0].cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        ece = BaselineModels.expected_calibration_error(
            np.array(all_labels) == np.array(all_preds), 
            np.array(all_probs)
        )
        
        return accuracy, ece
    
    def compare_models(self, train_data: Tuple, val_data: Tuple, test_data: Tuple, 
                      output_dir: str, num_epochs: int = 20) -> Dict[str, Any]:
        """Compare all models"""
        train_features, train_labels, train_logits, train_domain_ids, train_subjects = train_data
        test_features, test_labels, test_logits, test_domain_ids, test_subjects = test_data
        
        logger.info("Starting model comparison...")
        
        results = {}
        
        # 1. Vanilla baseline
        logger.info("1. Evaluating Vanilla baseline...")
        vanilla_acc, vanilla_ece = BaselineModels.vanilla_baseline(test_logits, test_labels)
        results['vanilla'] = {'accuracy': float(vanilla_acc), 'ece': float(vanilla_ece)}
        logger.info(f"   Vanilla - Accuracy: {vanilla_acc:.4f}, ECE: {vanilla_ece:.4f}")
        
        # 2. Oracle baseline
        logger.info("2. Evaluating Oracle baseline...")
        oracle_acc, oracle_ece, oracle_temp = BaselineModels.oracle_baseline(test_logits, test_labels)
        results['oracle'] = {'accuracy': float(oracle_acc), 'ece': float(oracle_ece), 'temperature': float(oracle_temp)}
        logger.info(f"   Oracle - Accuracy: {oracle_acc:.4f}, ECE: {oracle_ece:.4f}, T: {oracle_temp:.3f}")
        
        # 3. Global Thermometer model
        logger.info("3. Training and evaluating Global Thermometer model...")
        model = ThermometerModel(train_features.shape[1]).to(self.device)
        
        # Create data loaders
        from .models import FeatureDataset
        train_dataset = FeatureDataset(*train_data)
        val_dataset = FeatureDataset(*val_data)
        test_dataset = FeatureDataset(*test_data)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # Train model
        trainer = Trainer(model, self.device)
        model = trainer.train(train_loader, val_loader, num_epochs, output_dir)
        
        # Evaluate
        thermo_acc, thermo_ece = self.evaluate_thermometer(model, test_loader)
        results['thermometer_global'] = {'accuracy': float(thermo_acc), 'ece': float(thermo_ece)}
        logger.info(f"   Global Thermometer - Accuracy: {thermo_acc:.4f}, ECE: {thermo_ece:.4f}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        logger.info("\n=== Model Comparison Results ===")
        logger.info(f"{'Method':<20} {'Accuracy':<10} {'ECE':<10}")
        logger.info("-" * 40)
        logger.info(f"{'Vanilla':<20} {vanilla_acc:.4f}   {vanilla_ece:.4f}")
        logger.info(f"{'Oracle':<20} {oracle_acc:.4f}   {oracle_ece:.4f}")
        logger.info(f"{'Global Thermometer':<20} {thermo_acc:.4f}   {thermo_ece:.4f}")
        
        improvement = vanilla_ece - thermo_ece
        logger.info(f"\nECE Improvement (Global): {improvement:.4f}")
        
        if improvement > 0.05:
            logger.info("✓ Global Thermometer significantly improves calibration")
        else:
            logger.info("⚠ Limited improvement - may need debugging")
        
        return results
    
    def compare_global_vs_subject_wise(self, train_data: Tuple, val_data: Tuple, test_data: Tuple,
                                     output_dir: str, num_epochs: int = 20) -> Dict[str, Any]:
        """Compare global vs per-subject Thermometer models"""
        train_features, train_labels, train_logits, train_domain_ids, train_subjects = train_data
        test_features, test_labels, test_logits, test_domain_ids, test_subjects = test_data
        
        logger.info("Starting Global vs Subject-wise comparison...")
        
        results = {
            'baselines': {},
            'global': {},
            'subject_wise': {},
            'comparison': {}
        }
        
        # 1. Evaluate baselines
        logger.info("1. Evaluating baselines...")
        vanilla_acc, vanilla_ece = BaselineModels.vanilla_baseline(test_logits, test_labels)
        oracle_acc, oracle_ece, oracle_temp = BaselineModels.oracle_baseline(test_logits, test_labels)
        
        results['baselines'] = {
            'vanilla': {'accuracy': float(vanilla_acc), 'ece': float(vanilla_ece)},
            'oracle': {'accuracy': float(oracle_acc), 'ece': float(oracle_ece), 'temperature': float(oracle_temp)}
        }
        
        logger.info(f"   Vanilla - Accuracy: {vanilla_acc:.4f}, ECE: {vanilla_ece:.4f}")
        logger.info(f"   Oracle - Accuracy: {oracle_acc:.4f}, ECE: {oracle_ece:.4f}, T: {oracle_temp:.3f}")
        
        # 2. Train and evaluate global Thermometer
        logger.info("2. Training and evaluating Global Thermometer...")
        global_model = ThermometerModel(train_features.shape[1]).to(self.device)
        
        from .models import FeatureDataset
        train_dataset = FeatureDataset(*train_data)
        val_dataset = FeatureDataset(*val_data)
        test_dataset = FeatureDataset(*test_data)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        global_trainer = Trainer(global_model, self.device)
        global_model = global_trainer.train(train_loader, val_loader, num_epochs, 
                                           os.path.join(output_dir, 'global'))
        
        global_acc, global_ece = self.evaluate_thermometer(global_model, test_loader)
        results['global'] = {
            'accuracy': float(global_acc),
            'ece': float(global_ece),
            'num_parameters': sum(p.numel() for p in global_model.parameters())
        }
        
        logger.info(f"   Global Thermometer - Accuracy: {global_acc:.4f}, ECE: {global_ece:.4f}")
        
        # 3. Train and evaluate subject-wise models
        logger.info("3. Training and evaluating Subject-wise Thermometer models...")
        subject_trainer = SubjectWiseTrainer(self.device)
        subject_results = subject_trainer.train_all_subjects(
            train_data, val_data, test_data,
            os.path.join(output_dir, 'subject_wise'),
            num_epochs
        )
        
        results['subject_wise'] = subject_results
        
        # 4. Comparison analysis
        logger.info("4. Analysis results...")
        
        subject_wise_acc = subject_results['overall_results']['accuracy']
        subject_wise_ece = subject_results['overall_results']['ece']
        num_subjects = subject_results['overall_results']['num_subjects']
        
        # Calculate total parameters for subject-wise models
        # Since models aren't stored in results, estimate based on feature dimension
        feature_dim = train_features.shape[1]
        single_model_params = feature_dim * 64 + 64 * 64 + 64 * 2  # Thermometer model params
        total_subject_params = single_model_params * num_subjects
        
        results['comparison'] = {
            'global_improvement_over_vanilla': float(vanilla_ece - global_ece),
            'subject_wise_improvement_over_vanilla': float(vanilla_ece - subject_wise_ece),
            'subject_wise_improvement_over_global': float(global_ece - subject_wise_ece),
            'parameter_increase_ratio': float(total_subject_params / results['global']['num_parameters']),
            'num_subjects': num_subjects
        }
        
        # Save all results
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print comprehensive summary
        logger.info("\n" + "="*80)
        logger.info("GLOBAL vs SUBJECT-WISE THERMOMETER COMPARISON")
        logger.info("="*80)
        
        logger.info(f"\n📊 BASELINES:")
        logger.info(f"{'Method':<15} {'Accuracy':<10} {'ECE':<10}")
        logger.info("-" * 35)
        logger.info(f"{'Vanilla':<15} {vanilla_acc:.4f}   {vanilla_ece:.4f}")
        logger.info(f"{'Oracle':<15} {oracle_acc:.4f}   {oracle_ece:.4f}")
        
        logger.info(f"\n🎯 THERMOMETER MODELS:")
        logger.info(f"{'Method':<20} {'Accuracy':<10} {'ECE':<10} {'Params':<12}")
        logger.info("-" * 52)
        logger.info(f"{'Global':<20} {global_acc:.4f}   {global_ece:.4f}   {results['global']['num_parameters']:,}")
        logger.info(f"{'Subject-wise':<20} {subject_wise_acc:.4f}   {subject_wise_ece:.4f}   {total_subject_params:,}")
        
        logger.info(f"\n📈 IMPROVEMENT ANALYSIS:")
        logger.info(f"Global vs Vanilla ECE improvement: {results['comparison']['global_improvement_over_vanilla']:.4f}")
        logger.info(f"Subject-wise vs Vanilla ECE improvement: {results['comparison']['subject_wise_improvement_over_vanilla']:.4f}")
        logger.info(f"Subject-wise vs Global ECE improvement: {results['comparison']['subject_wise_improvement_over_global']:.4f}")
        logger.info(f"Parameter increase ratio: {results['comparison']['parameter_increase_ratio']:.2f}x")
        
        logger.info(f"\n📝 SUBJECT-WISE DETAILS:")
        logger.info(f"Number of subjects: {num_subjects}")
        logger.info(f"Average samples per subject: {len(test_features) / num_subjects:.1f}")
        
        # Show top 5 and bottom 5 subjects
        subject_results_list = list(subject_results['subject_wise']['subject_models'].items())
        subject_results_list.sort(key=lambda x: x[1]['ece'])
        
        logger.info(f"\n🏆 TOP 5 SUBJECTS (by ECE):")
        for subject, metrics in subject_results_list[:5]:
            logger.info(f"   {subject:<20} Acc: {metrics['accuracy']:.4f}, ECE: {metrics['ece']:.4f}, Samples: {metrics['num_samples']}")
        
        logger.info(f"\n📊 BOTTOM 5 SUBJECTS (by ECE):")
        for subject, metrics in subject_results_list[-5:]:
            logger.info(f"   {subject:<20} Acc: {metrics['accuracy']:.4f}, ECE: {metrics['ece']:.4f}, Samples: {metrics['num_samples']}")
        
        # Final recommendations
        logger.info(f"\n🎯 RECOMMENDATIONS:")
        if results['comparison']['subject_wise_improvement_over_global'] > 0.01:
            logger.info("✓ Subject-wise approach shows significant improvement")
            if results['comparison']['parameter_increase_ratio'] > 5:
                logger.info("⚠ But requires substantially more parameters")
        else:
            logger.info("ℹ️ Global approach is sufficient - subject-wise doesn't provide significant benefit")
        
        return results