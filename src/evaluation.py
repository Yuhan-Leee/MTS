#!/usr/bin/env python3
"""
Evaluation module for MTS
Handles evaluation and analysis of model performance
"""

import json
import pickle
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive evaluator for model performance"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.results = {}
        
    def load_results(self, results_path: str) -> Dict[str, Any]:
        """Load evaluation results"""
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def analyze_by_subject(self, features_path: str, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance by subject"""
        with open(features_path, 'rb') as f:
            data = pickle.load(f)
        
        subjects = data['subjects']
        labels = data['labels']
        unique_subjects = list(set(subjects))
        
        subject_results = {}
        for subject in unique_subjects:
            subject_mask = np.array([s == subject for s in subjects])
            subject_labels = np.array(labels)[subject_mask]
            
            # Analyze performance for this subject
            subject_results[subject] = {
                'count': int(np.sum(subject_mask)),
                'label_distribution': {str(i): int(np.sum(subject_labels == i)) for i in range(4)}
            }
        
        return subject_results
    
    def calculate_reliability_diagram(self, probs: np.ndarray, correct: np.ndarray, 
                                    n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate reliability diagram data"""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        accuracies = []
        confidences = []
        counts = []
        
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if np.sum(mask) > 0:
                acc = np.mean(correct[mask])
                conf = np.mean(probs[mask])
                count = np.sum(mask)
            else:
                acc = 0
                conf = 0
                count = 0
            
            accuracies.append(acc)
            confidences.append(conf)
            counts.append(count)
        
        return np.array(bin_centers), np.array(accuracies), np.array(confidences), np.array(counts)
    
    def plot_reliability_diagram(self, probs: np.ndarray, correct: np.ndarray, 
                               title: str = "Reliability Diagram", save_path: Optional[str] = None):
        """Plot reliability diagram"""
        bin_centers, accuracies, confidences, counts = self.calculate_reliability_diagram(probs, correct)
        
        plt.figure(figsize=(10, 8))
        
        # Plot reliability curve
        plt.plot(bin_centers, accuracies, 'bo-', label='Model', linewidth=2, markersize=8)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        # Plot histogram
        ax2 = plt.gca().twinx()
        ax2.bar(bin_centers, counts, width=0.08, alpha=0.3, color='gray', label='Count')
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_ylim(0, max(counts) * 1.1)
        
        plt.xlabel('Confidence', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved reliability diagram to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str] = None, title: str = "Confusion Matrix",
                            save_path: Optional[str] = None):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = ['A', 'B', 'C', 'D']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title, fontsize=14)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.show()
    
    def generate_report(self, results_path: str, features_path: str, output_dir: str):
        """Generate comprehensive evaluation report"""
        logger.info("Generating evaluation report...")
        
        # Load results
        results = self.load_results(results_path)
        
        # Analyze by subject
        subject_analysis = self.analyze_by_subject(features_path, results)
        
        # Create report
        report = {
            'summary': results,
            'subject_analysis': subject_analysis,
            'recommendations': self.generate_recommendations(results)
        }
        
        # Save report
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {report_path}")
        return report
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        vanilla_ece = results.get('vanilla', {}).get('ece', 0)
        thermo_ece = results.get('thermometer', {}).get('ece', 0)
        oracle_ece = results.get('oracle', {}).get('ece', 0)
        
        # Check if Thermometer helps
        if vanilla_ece > 0 and thermo_ece < vanilla_ece:
            improvement = (vanilla_ece - thermo_ece) / vanilla_ece * 100
            recommendations.append(f"Thermometer improves calibration by {improvement:.1f}%")
        else:
            recommendations.append("Thermometer shows limited improvement - consider model architecture changes")
        
        # Check gap to Oracle
        if oracle_ece > 0 and thermo_ece > oracle_ece:
            gap = thermo_ece - oracle_ece
            recommendations.append(f"Gap to Oracle: {gap:.3f} - potential for further improvement")
        
        # Check overall calibration
        if thermo_ece > 0.1:
            recommendations.append("ECE is still high - consider more training or architectural changes")
        elif thermo_ece < 0.05:
            recommendations.append("Good calibration achieved - model is ready for deployment")
        
        return recommendations


class MetricsCalculator:
    """Calculate various evaluation metrics"""
    
    @staticmethod
    def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
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
    
    @staticmethod
    def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Brier score"""
        return np.mean((y_prob - y_true) ** 2)
    
    @staticmethod
    def calculate_nll(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Negative Log Likelihood"""
        # Avoid log(0)
        y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'ece': MetricsCalculator.calculate_ece(y_true, y_prob),
            'brier_score': MetricsCalculator.calculate_brier_score(y_true, y_prob),
            'nll': MetricsCalculator.calculate_nll(y_true, y_prob),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        return metrics