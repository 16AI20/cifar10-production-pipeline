"""
Advanced Evaluation Module for CIFAR-10 Project

This module provides comprehensive evaluation capabilities including:
- Advanced metrics beyond accuracy (precision, recall, F1, AUC)
- Calibration analysis and reliability diagrams
- Class-wise performance analysis
- Uncertainty quantification
- Enhanced visualizations and reporting
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for comprehensive evaluation metrics."""
    
    # Basic metrics
    accuracy: float
    loss: float
    
    # Per-class metrics
    precision_macro: float
    precision_micro: float
    precision_weighted: float
    recall_macro: float
    recall_micro: float
    recall_weighted: float
    f1_macro: float
    f1_micro: float
    f1_weighted: float
    
    # Advanced metrics
    auc_macro: Optional[float] = None
    auc_weighted: Optional[float] = None
    average_precision: Optional[float] = None
    
    # Calibration metrics
    calibration_error: Optional[float] = None
    reliability_score: Optional[float] = None
    
    # Uncertainty metrics
    mean_entropy: Optional[float] = None
    mean_confidence: Optional[float] = None
    
    # Per-class detailed metrics
    class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    
    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None


@dataclass
class ModelPredictions:
    """Container for model predictions and analysis."""
    
    predictions: np.ndarray
    probabilities: np.ndarray
    true_labels: np.ndarray
    class_names: List[str]
    
    # Confidence and uncertainty measures
    max_probabilities: np.ndarray
    entropies: np.ndarray
    
    # Prediction correctness
    correct_mask: np.ndarray
    
    # Sample indices for analysis
    most_confident_correct: np.ndarray
    most_confident_incorrect: np.ndarray
    least_confident_correct: np.ndarray
    least_confident_incorrect: np.ndarray


class AdvancedEvaluator:
    """
    Comprehensive evaluation class with advanced metrics and visualizations.
    """
    
    def __init__(self, class_names: List[str], save_dir: str = "images/evaluation"):
        """
        Initialize the advanced evaluator.
        
        Args:
            class_names: List of class names
            save_dir: Directory to save visualizations
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Advanced evaluator initialized for {self.num_classes} classes")
    
    def comprehensive_evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device
    ) -> Tuple[EvaluationMetrics, ModelPredictions]:
        """
        Perform comprehensive evaluation of the model.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader for evaluation
            criterion: Loss criterion
            device: Device for computation
            
        Returns:
            Tuple of (evaluation metrics, model predictions)
        """
        logger.info("Starting comprehensive evaluation...")
        
        # Get predictions and probabilities
        predictions = self._get_model_predictions(model, dataloader, device)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            predictions, criterion, device
        )
        
        logger.info(f"Evaluation completed. Accuracy: {metrics.accuracy:.4f}")
        return metrics, predictions
    
    def _get_model_predictions(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ) -> ModelPredictions:
        """Get model predictions with probabilities and uncertainty measures."""
        
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.append(predictions.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions)
        probabilities = np.concatenate(all_probabilities)
        true_labels = np.concatenate(all_labels)
        
        # Calculate confidence and uncertainty measures
        max_probabilities = np.max(probabilities, axis=1)
        entropies = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        
        # Determine correctness
        correct_mask = predictions == true_labels
        
        # Find interesting samples for analysis
        confidence_indices = np.argsort(max_probabilities)
        correct_indices = np.where(correct_mask)[0]
        incorrect_indices = np.where(~correct_mask)[0]
        
        # Most/least confident correct/incorrect samples
        most_confident_correct = np.intersect1d(
            confidence_indices[-50:], correct_indices
        )[-10:] if len(correct_indices) > 0 else np.array([])
        
        most_confident_incorrect = np.intersect1d(
            confidence_indices[-50:], incorrect_indices
        )[-10:] if len(incorrect_indices) > 0 else np.array([])
        
        least_confident_correct = np.intersect1d(
            confidence_indices[:50], correct_indices
        )[:10] if len(correct_indices) > 0 else np.array([])
        
        least_confident_incorrect = np.intersect1d(
            confidence_indices[:50], incorrect_indices
        )[:10] if len(incorrect_indices) > 0 else np.array([])
        
        return ModelPredictions(
            predictions=predictions,
            probabilities=probabilities,
            true_labels=true_labels,
            class_names=self.class_names,
            max_probabilities=max_probabilities,
            entropies=entropies,
            correct_mask=correct_mask,
            most_confident_correct=most_confident_correct,
            most_confident_incorrect=most_confident_incorrect,
            least_confident_correct=least_confident_correct,
            least_confident_incorrect=least_confident_incorrect
        )
    
    def _calculate_comprehensive_metrics(
        self,
        predictions: ModelPredictions,
        criterion: nn.Module,
        device: torch.device
    ) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics."""
        
        pred = predictions.predictions
        prob = predictions.probabilities
        true = predictions.true_labels
        
        # Basic metrics
        accuracy = accuracy_score(true, pred)
        
        # Calculate loss (approximate)
        with torch.no_grad():
            log_prob = torch.log(torch.tensor(prob) + 1e-8)
            loss = F.nll_loss(log_prob, torch.tensor(true)).item()
        
        # Per-class metrics
        precision_macro = precision_score(true, pred, average='macro', zero_division=0)
        precision_micro = precision_score(true, pred, average='micro', zero_division=0)
        precision_weighted = precision_score(true, pred, average='weighted', zero_division=0)
        
        recall_macro = recall_score(true, pred, average='macro', zero_division=0)
        recall_micro = recall_score(true, pred, average='micro', zero_division=0)
        recall_weighted = recall_score(true, pred, average='weighted', zero_division=0)
        
        f1_macro = f1_score(true, pred, average='macro', zero_division=0)
        f1_micro = f1_score(true, pred, average='micro', zero_division=0)
        f1_weighted = f1_score(true, pred, average='weighted', zero_division=0)
        
        # Multi-class AUC
        auc_macro = None
        auc_weighted = None
        average_precision = None
        
        try:
            if self.num_classes == 2:
                auc_macro = roc_auc_score(true, prob[:, 1])
                average_precision = average_precision_score(true, prob[:, 1])
            else:
                auc_macro = roc_auc_score(true, prob, multi_class='ovr', average='macro')
                auc_weighted = roc_auc_score(true, prob, multi_class='ovr', average='weighted')
                # For multi-class, calculate macro-averaged precision
                average_precision = average_precision_score(
                    np.eye(self.num_classes)[true], prob, average='macro'
                )
        except ValueError as e:
            logger.warning(f"Could not calculate AUC/AP: {e}")
        
        # Calibration metrics
        calibration_error = self._calculate_calibration_error(prob, true)
        reliability_score = self._calculate_reliability_score(prob, true)
        
        # Uncertainty metrics
        mean_entropy = np.mean(predictions.entropies)
        mean_confidence = np.mean(predictions.max_probabilities)
        
        # Per-class detailed metrics
        class_metrics = self._calculate_per_class_metrics(true, pred, prob)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true, pred)
        
        return EvaluationMetrics(
            accuracy=accuracy,
            loss=loss,
            precision_macro=precision_macro,
            precision_micro=precision_micro,
            precision_weighted=precision_weighted,
            recall_macro=recall_macro,
            recall_micro=recall_micro,
            recall_weighted=recall_weighted,
            f1_macro=f1_macro,
            f1_micro=f1_micro,
            f1_weighted=f1_weighted,
            auc_macro=auc_macro,
            auc_weighted=auc_weighted,
            average_precision=average_precision,
            calibration_error=calibration_error,
            reliability_score=reliability_score,
            mean_entropy=mean_entropy,
            mean_confidence=mean_confidence,
            class_metrics=class_metrics,
            confusion_matrix=conf_matrix
        )
    
    def _calculate_calibration_error(self, probabilities: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        
        max_probs = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        accuracies = (predictions == true_labels)
        
        # Bin predictions by confidence
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_reliability_score(self, probabilities: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate reliability score based on Brier score."""
        
        # Convert to one-hot encoding
        one_hot = np.eye(self.num_classes)[true_labels]
        
        # Calculate Brier score
        brier_score = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        
        # Convert to reliability score (lower Brier score = higher reliability)
        reliability = 1.0 - brier_score
        return max(0.0, reliability)
    
    def _calculate_per_class_metrics(
        self,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Calculate detailed per-class metrics."""
        
        class_metrics = {}
        
        # Get per-class precision, recall, f1
        precision_per_class = precision_score(true_labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(true_labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            # Basic metrics
            class_metrics[class_name] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1_score': f1_per_class[i],
                'support': np.sum(true_labels == i)
            }
            
            # Class-specific accuracy
            class_mask = true_labels == i
            if np.any(class_mask):
                class_predictions = predictions[class_mask]
                class_accuracy = np.mean(class_predictions == i)
                class_metrics[class_name]['accuracy'] = class_accuracy
                
                # Average confidence for this class
                class_probs = probabilities[class_mask, i]
                class_metrics[class_name]['avg_confidence'] = np.mean(class_probs)
            else:
                class_metrics[class_name]['accuracy'] = 0.0
                class_metrics[class_name]['avg_confidence'] = 0.0
        
        return class_metrics
    
    def create_comprehensive_visualizations(
        self,
        metrics: EvaluationMetrics,
        predictions: ModelPredictions,
        model_name: str = "Model"
    ) -> Dict[str, str]:
        """
        Create comprehensive evaluation visualizations.
        
        Args:
            metrics: Evaluation metrics
            predictions: Model predictions
            model_name: Name of the model for titles
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        logger.info("Creating comprehensive evaluation visualizations...")
        
        viz_paths = {}
        
        # 1. Enhanced confusion matrix
        viz_paths['confusion_matrix'] = self._plot_enhanced_confusion_matrix(
            metrics.confusion_matrix, model_name
        )
        
        # 2. Class-wise performance analysis
        viz_paths['class_performance'] = self._plot_class_performance_analysis(
            metrics, model_name
        )
        
        # 3. Calibration plots
        viz_paths['calibration'] = self._plot_calibration_analysis(
            predictions, model_name
        )
        
        # 4. Confidence distribution analysis
        viz_paths['confidence_analysis'] = self._plot_confidence_analysis(
            predictions, model_name
        )
        
        # 5. ROC curves (if applicable)
        if self.num_classes <= 10:  # Avoid cluttered plots for too many classes
            viz_paths['roc_curves'] = self._plot_roc_curves(
                predictions, model_name
            )
        
        # 6. Precision-Recall curves
        if self.num_classes <= 10:
            viz_paths['pr_curves'] = self._plot_precision_recall_curves(
                predictions, model_name
            )
        
        # 7. Error analysis
        viz_paths['error_analysis'] = self._plot_error_analysis(
            predictions, model_name
        )
        
        # 8. Uncertainty analysis
        viz_paths['uncertainty_analysis'] = self._plot_uncertainty_analysis(
            predictions, model_name
        )
        
        logger.info(f"Created {len(viz_paths)} evaluation visualizations")
        return viz_paths
    
    def _plot_enhanced_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        model_name: str
    ) -> str:
        """Create enhanced confusion matrix with additional statistics."""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Normalized confusion matrix
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # Plot raw counts
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[0])
        axes[0].set_title(f'{model_name} - Confusion Matrix (Counts)')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Plot normalized percentages
        sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[1])
        axes[1].set_title(f'{model_name} - Confusion Matrix (Normalized)')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        
        save_path = self.save_dir / f'enhanced_confusion_matrix_{model_name.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_class_performance_analysis(
        self,
        metrics: EvaluationMetrics,
        model_name: str
    ) -> str:
        """Create comprehensive class performance analysis."""
        
        if metrics.class_metrics is None:
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data
        classes = list(metrics.class_metrics.keys())
        precision_scores = [metrics.class_metrics[cls]['precision'] for cls in classes]
        recall_scores = [metrics.class_metrics[cls]['recall'] for cls in classes]
        f1_scores = [metrics.class_metrics[cls]['f1_score'] for cls in classes]
        support_counts = [metrics.class_metrics[cls]['support'] for cls in classes]
        
        # 1. Precision, Recall, F1 comparison
        x = np.arange(len(classes))
        width = 0.25
        
        axes[0, 0].bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        axes[0, 0].bar(x, recall_scores, width, label='Recall', alpha=0.8)
        axes[0, 0].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        axes[0, 0].set_xlabel('Classes')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Per-Class Performance Metrics')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Support distribution
        axes[0, 1].bar(classes, support_counts, color='skyblue', alpha=0.7)
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].set_title('Class Support Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision vs Recall scatter plot
        axes[1, 0].scatter(recall_scores, precision_scores, s=100, alpha=0.7)
        for i, cls in enumerate(classes):
            axes[1, 0].annotate(cls, (recall_scores[i], precision_scores[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision vs Recall by Class')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add diagonal line for reference
        min_val = min(min(recall_scores), min(precision_scores))
        max_val = max(max(recall_scores), max(precision_scores))
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        # 4. Performance vs Support correlation
        axes[1, 1].scatter(support_counts, f1_scores, s=100, alpha=0.7, color='green')
        for i, cls in enumerate(classes):
            axes[1, 1].annotate(cls, (support_counts[i], f1_scores[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[1, 1].set_xlabel('Class Support (# samples)')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_title('Performance vs Class Support')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr_coef = np.corrcoef(support_counts, f1_scores)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr_coef:.3f}',
                        transform=axes[1, 1].transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'{model_name} - Class Performance Analysis', fontsize=16)
        plt.tight_layout()
        
        save_path = self.save_dir / f'class_performance_analysis_{model_name.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_calibration_analysis(
        self,
        predictions: ModelPredictions,
        model_name: str
    ) -> str:
        """Create model calibration analysis plots."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        prob = predictions.probabilities
        true = predictions.true_labels
        pred = predictions.predictions
        
        # 1. Reliability diagram
        max_probs = np.max(prob, axis=1)
        accuracies = (pred == true)
        
        # Bin predictions by confidence
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.sum()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(prop_in_bin)
            else:
                bin_accuracies.append(0)
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_counts.append(0)
        
        # Plot reliability diagram
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        axes[0].plot(bin_confidences, bin_accuracies, 'ro-', label='Model Calibration')
        
        # Add bars for bin counts
        bar_width = 0.1
        for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
            if count > 0:
                axes[0].bar(conf, count / max(bin_counts) * 0.2, bar_width, 
                           bottom=0, alpha=0.3, color='gray')
        
        axes[0].set_xlabel('Mean Predicted Probability')
        axes[0].set_ylabel('Fraction of Positives')
        axes[0].set_title('Reliability Diagram')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Confidence histogram
        axes[1].hist(max_probs[accuracies], bins=20, alpha=0.7, label='Correct', density=True)
        axes[1].hist(max_probs[~accuracies], bins=20, alpha=0.7, label='Incorrect', density=True)
        axes[1].set_xlabel('Confidence (Max Probability)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Confidence Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Calibration error by confidence
        ece_per_bin = []
        for bin_lower, bin_upper, acc, conf in zip(bin_lowers, bin_uppers, bin_accuracies, bin_confidences):
            ece_per_bin.append(abs(conf - acc))
        
        axes[2].bar(range(len(ece_per_bin)), ece_per_bin, alpha=0.7, color='orange')
        axes[2].set_xlabel('Confidence Bin')
        axes[2].set_ylabel('Calibration Error')
        axes[2].set_title('Calibration Error per Bin')
        axes[2].set_xticks(range(len(ece_per_bin)))
        axes[2].set_xticklabels([f'{l:.1f}-{u:.1f}' for l, u in zip(bin_lowers, bin_uppers)],
                               rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Calibration Analysis', fontsize=16)
        plt.tight_layout()
        
        save_path = self.save_dir / f'calibration_analysis_{model_name.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_confidence_analysis(
        self,
        predictions: ModelPredictions,
        model_name: str
    ) -> str:
        """Create confidence and uncertainty analysis plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        confidence = predictions.max_probabilities
        entropy = predictions.entropies
        correct = predictions.correct_mask
        
        # 1. Confidence vs Accuracy
        confidence_bins = np.linspace(0, 1, 11)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (confidence >= confidence_bins[i]) & (confidence < confidence_bins[i + 1])
            if np.sum(mask) > 0:
                bin_accuracies.append(np.mean(correct[mask]))
                bin_counts.append(np.sum(mask))
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        axes[0, 0].bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Confidence Bin')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs Confidence')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add count labels
        for i, (center, acc, count) in enumerate(zip(bin_centers, bin_accuracies, bin_counts)):
            if count > 0:
                axes[0, 0].text(center, acc + 0.02, f'{count}', ha='center', va='bottom', fontsize=9)
        
        # 2. Entropy distribution
        axes[0, 1].hist(entropy[correct], bins=30, alpha=0.7, label='Correct', density=True)
        axes[0, 1].hist(entropy[~correct], bins=30, alpha=0.7, label='Incorrect', density=True)
        axes[0, 1].set_xlabel('Entropy')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Entropy Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confidence vs Entropy scatter
        scatter = axes[1, 0].scatter(confidence, entropy, c=correct, alpha=0.6, 
                                    cmap='RdYlGn', s=20)
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].set_title('Confidence vs Entropy')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Correct Prediction')
        
        # 4. Per-class confidence analysis
        class_confidences = []
        class_accuracies = []
        
        for i, class_name in enumerate(predictions.class_names):
            class_mask = predictions.true_labels == i
            if np.sum(class_mask) > 0:
                class_conf = np.mean(confidence[class_mask])
                class_acc = np.mean(correct[class_mask])
                class_confidences.append(class_conf)
                class_accuracies.append(class_acc)
            else:
                class_confidences.append(0)
                class_accuracies.append(0)
        
        bars = axes[1, 1].bar(range(len(predictions.class_names)), class_confidences, 
                             alpha=0.7, color='lightcoral')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Mean Confidence')
        axes[1, 1].set_title('Mean Confidence per Class')
        axes[1, 1].set_xticks(range(len(predictions.class_names)))
        axes[1, 1].set_xticklabels(predictions.class_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add accuracy as text on bars
        for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(f'{model_name} - Confidence & Uncertainty Analysis', fontsize=16)
        plt.tight_layout()
        
        save_path = self.save_dir / f'confidence_analysis_{model_name.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_roc_curves(self, predictions: ModelPredictions, model_name: str) -> str:
        """Create ROC curves for multi-class classification."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        prob = predictions.probabilities
        true = predictions.true_labels
        
        # Convert to binary format for ROC curves
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(true, classes=range(self.num_classes))
        
        # Calculate ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], prob[:, i])
            roc_auc[i] = roc_auc_score(y_bin[:, i], prob[:, i])
        
        # Plot individual class ROC curves
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_classes))
        
        for i, color in enumerate(colors):
            ax1.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{predictions.class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', lw=2)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'{model_name} - ROC Curves per Class')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Macro and micro average ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), prob.ravel())
        roc_auc["micro"] = roc_auc_score(y_bin, prob, average='micro', multi_class='ovr')
        
        # Macro average
        from scipy import interp
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.num_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= self.num_classes
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = roc_auc_score(y_bin, prob, average='macro', multi_class='ovr')
        
        # Plot average ROC curves
        ax2.plot(fpr["micro"], tpr["micro"], 
                label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
                color='deeppink', linestyle=':', linewidth=4)
        
        ax2.plot(fpr["macro"], tpr["macro"],
                label=f'macro-average ROC curve (area = {roc_auc["macro"]:.2f})',
                color='navy', linestyle=':', linewidth=4)
        
        ax2.plot([0, 1], [0, 1], 'k--', lw=2)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title(f'{model_name} - Average ROC Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.save_dir / f'roc_curves_{model_name.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_precision_recall_curves(self, predictions: ModelPredictions, model_name: str) -> str:
        """Create Precision-Recall curves for multi-class classification."""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        prob = predictions.probabilities
        true = predictions.true_labels
        
        # Convert to binary format
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(true, classes=range(self.num_classes))
        
        # Calculate PR curve for each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        
        for i in range(self.num_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], prob[:, i])
            average_precision[i] = average_precision_score(y_bin[:, i], prob[:, i])
        
        # Plot individual class PR curves
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_classes))
        
        for i, color in enumerate(colors):
            axes[0].plot(recall[i], precision[i], color=color, lw=2,
                        label=f'{predictions.class_names[i]} (AP = {average_precision[i]:.2f})')
        
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_title(f'{model_name} - Precision-Recall Curves per Class')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Micro-average PR curve
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_bin.ravel(), prob.ravel()
        )
        average_precision["micro"] = average_precision_score(y_bin, prob, average="micro")
        
        axes[1].plot(recall["micro"], precision["micro"],
                    label=f'micro-average PR curve (area = {average_precision["micro"]:.2f})',
                    color='gold', linestyle='-', linewidth=3)
        
        # Add macro-average
        macro_ap = np.mean([average_precision[i] for i in range(self.num_classes)])
        axes[1].axhline(y=macro_ap, color='navy', linestyle='--', linewidth=2,
                       label=f'macro-average AP = {macro_ap:.2f}')
        
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title(f'{model_name} - Average Precision-Recall Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.save_dir / f'pr_curves_{model_name.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_error_analysis(self, predictions: ModelPredictions, model_name: str) -> str:
        """Create error analysis visualization."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Error rate by class
        error_rates = []
        for i, class_name in enumerate(predictions.class_names):
            class_mask = predictions.true_labels == i
            if np.sum(class_mask) > 0:
                error_rate = 1 - np.mean(predictions.correct_mask[class_mask])
                error_rates.append(error_rate)
            else:
                error_rates.append(0)
        
        bars = axes[0, 0].bar(predictions.class_names, error_rates, 
                             alpha=0.7, color='lightcoral')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Error Rate')
        axes[0, 0].set_title('Error Rate by Class')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, error_rates):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{rate:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Most confused class pairs
        conf_matrix = confusion_matrix(predictions.true_labels, predictions.predictions)
        
        # Find most confused pairs (excluding diagonal)
        conf_pairs = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:
                    conf_pairs.append((i, j, conf_matrix[i, j]))
        
        # Sort by confusion count and take top 10
        conf_pairs.sort(key=lambda x: x[2], reverse=True)
        top_confusions = conf_pairs[:10]
        
        if top_confusions:
            pair_labels = [f"{predictions.class_names[i]}→{predictions.class_names[j]}" 
                          for i, j, _ in top_confusions]
            confusion_counts = [count for _, _, count in top_confusions]
            
            axes[0, 1].barh(range(len(pair_labels)), confusion_counts, alpha=0.7, color='orange')
            axes[0, 1].set_yticks(range(len(pair_labels)))
            axes[0, 1].set_yticklabels(pair_labels)
            axes[0, 1].set_xlabel('Number of Confusions')
            axes[0, 1].set_title('Most Confused Class Pairs')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error distribution by confidence
        incorrect_mask = ~predictions.correct_mask
        if np.any(incorrect_mask):
            incorrect_confidences = predictions.max_probabilities[incorrect_mask]
            correct_confidences = predictions.max_probabilities[predictions.correct_mask]
            
            axes[1, 0].hist(correct_confidences, bins=20, alpha=0.7, label='Correct', 
                           density=True, color='green')
            axes[1, 0].hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', 
                           density=True, color='red')
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Confidence Distribution: Correct vs Incorrect')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction entropy analysis
        if np.any(incorrect_mask):
            incorrect_entropy = predictions.entropies[incorrect_mask]
            correct_entropy = predictions.entropies[predictions.correct_mask]
            
            axes[1, 1].boxplot([correct_entropy, incorrect_entropy], 
                              labels=['Correct', 'Incorrect'])
            axes[1, 1].set_ylabel('Entropy')
            axes[1, 1].set_title('Entropy Distribution: Correct vs Incorrect')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Error Analysis', fontsize=16)
        plt.tight_layout()
        
        save_path = self.save_dir / f'error_analysis_{model_name.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_uncertainty_analysis(self, predictions: ModelPredictions, model_name: str) -> str:
        """Create uncertainty quantification analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        confidence = predictions.max_probabilities
        entropy = predictions.entropies
        correct = predictions.correct_mask
        
        # 1. Uncertainty vs accuracy relationship
        # Bin samples by entropy
        entropy_bins = np.linspace(entropy.min(), entropy.max(), 10)
        bin_centers = (entropy_bins[:-1] + entropy_bins[1:]) / 2
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(entropy_bins) - 1):
            mask = (entropy >= entropy_bins[i]) & (entropy < entropy_bins[i + 1])
            if np.sum(mask) > 0:
                bin_accuracies.append(np.mean(correct[mask]))
                bin_counts.append(np.sum(mask))
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        axes[0, 0].bar(bin_centers, bin_accuracies, width=(entropy_bins[1] - entropy_bins[0]) * 0.8,
                      alpha=0.7, color='purple')
        axes[0, 0].set_xlabel('Entropy (Uncertainty)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs Uncertainty')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Calibration vs uncertainty
        # Show how calibration varies with uncertainty
        low_entropy_mask = entropy <= np.median(entropy)
        high_entropy_mask = entropy > np.median(entropy)
        
        # Calculate calibration for low and high entropy predictions
        low_ent_conf = confidence[low_entropy_mask]
        low_ent_acc = correct[low_entropy_mask]
        high_ent_conf = confidence[high_entropy_mask]
        high_ent_acc = correct[high_entropy_mask]
        
        # Plot calibration for both groups
        conf_bins = np.linspace(0, 1, 11)
        
        for conf_group, acc_group, label, color in [
            (low_ent_conf, low_ent_acc, 'Low Entropy', 'blue'),
            (high_ent_conf, high_ent_acc, 'High Entropy', 'red')
        ]:
            bin_conf_means = []
            bin_acc_means = []
            
            for i in range(len(conf_bins) - 1):
                mask = (conf_group >= conf_bins[i]) & (conf_group < conf_bins[i + 1])
                if np.sum(mask) > 0:
                    bin_conf_means.append(np.mean(conf_group[mask]))
                    bin_acc_means.append(np.mean(acc_group[mask]))
            
            if bin_conf_means and bin_acc_means:
                axes[0, 1].plot(bin_conf_means, bin_acc_means, 'o-', 
                               label=label, color=color, alpha=0.7)
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        axes[0, 1].set_xlabel('Mean Confidence')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Calibration by Uncertainty Level')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Uncertainty heatmap by predicted vs true class
        uncertainty_matrix = np.zeros((self.num_classes, self.num_classes))
        count_matrix = np.zeros((self.num_classes, self.num_classes))
        
        for i in range(len(predictions.true_labels)):
            true_class = predictions.true_labels[i]
            pred_class = predictions.predictions[i]
            uncertainty_matrix[true_class, pred_class] += entropy[i]
            count_matrix[true_class, pred_class] += 1
        
        # Average uncertainty per cell
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_uncertainty = uncertainty_matrix / count_matrix
            avg_uncertainty = np.nan_to_num(avg_uncertainty)
        
        im = axes[1, 0].imshow(avg_uncertainty, cmap='Reds', interpolation='nearest')
        axes[1, 0].set_xlabel('Predicted Class')
        axes[1, 0].set_ylabel('True Class')
        axes[1, 0].set_title('Average Uncertainty by Prediction')
        axes[1, 0].set_xticks(range(self.num_classes))
        axes[1, 0].set_xticklabels(predictions.class_names, rotation=45, ha='right')
        axes[1, 0].set_yticks(range(self.num_classes))
        axes[1, 0].set_yticklabels(predictions.class_names)
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Predictive confidence vs ground truth confidence
        # For each class, show the distribution of confidences for correct predictions
        class_confidences = []
        class_names_for_box = []
        
        for i, class_name in enumerate(predictions.class_names):
            class_mask = (predictions.true_labels == i) & correct
            if np.sum(class_mask) > 5:  # Only include classes with enough correct predictions
                class_conf = confidence[class_mask]
                class_confidences.append(class_conf)
                class_names_for_box.append(class_name)
        
        if class_confidences:
            axes[1, 1].boxplot(class_confidences, labels=class_names_for_box)
            axes[1, 1].set_xlabel('Class')
            axes[1, 1].set_ylabel('Confidence (Correct Predictions)')
            axes[1, 1].set_title('Confidence Distribution per Class')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Uncertainty Analysis', fontsize=16)
        plt.tight_layout()
        
        save_path = self.save_dir / f'uncertainty_analysis_{model_name.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def generate_evaluation_report(
        self,
        metrics: EvaluationMetrics,
        predictions: ModelPredictions,
        model_name: str,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive HTML evaluation report.
        
        Args:
            metrics: Evaluation metrics
            predictions: Model predictions
            model_name: Name of the model
            save_path: Optional path to save report
            
        Returns:
            Path to saved report
        """
        if save_path is None:
            save_path = self.save_dir / f'evaluation_report_{model_name.lower()}.html'
        
        # Generate comprehensive HTML report content
        html_content = self._create_html_report_content(metrics, predictions, model_name)
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Comprehensive evaluation report saved to {save_path}")
        return str(save_path)
    
    def _create_html_report_content(
        self,
        metrics: EvaluationMetrics,
        predictions: ModelPredictions,
        model_name: str
    ) -> str:
        """Create HTML content for evaluation report."""
        
        # Calculate additional statistics
        total_samples = len(predictions.true_labels)
        correct_predictions = np.sum(predictions.correct_mask)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{model_name} - Comprehensive Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #667eea; 
                           background-color: #f8f9fa; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                               gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: white; padding: 15px; border-radius: 8px; 
                               box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
                .metric-label {{ font-size: 0.9em; color: #666; text-transform: uppercase; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #667eea; color: white; }}
                .good {{ color: #28a745; font-weight: bold; }}
                .warning {{ color: #ffc107; font-weight: bold; }}
                .poor {{ color: #dc3545; font-weight: bold; }}
                .summary-stats {{ background: #e9ecef; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 {model_name} Evaluation Report</h1>
                <p>Comprehensive Performance Analysis</p>
            </div>
            
            <div class="section">
                <h2>📊 Overall Performance</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{metrics.accuracy:.3f}</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.f1_weighted:.3f}</div>
                        <div class="metric-label">Weighted F1</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.loss:.3f}</div>
                        <div class="metric-label">Loss</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{correct_predictions}/{total_samples}</div>
                        <div class="metric-label">Correct Predictions</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 Detailed Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Macro Avg</th>
                        <th>Micro Avg</th>
                        <th>Weighted Avg</th>
                    </tr>
                    <tr>
                        <td><strong>Precision</strong></td>
                        <td>{metrics.precision_macro:.3f}</td>
                        <td>{metrics.precision_micro:.3f}</td>
                        <td>{metrics.precision_weighted:.3f}</td>
                    </tr>
                    <tr>
                        <td><strong>Recall</strong></td>
                        <td>{metrics.recall_macro:.3f}</td>
                        <td>{metrics.recall_micro:.3f}</td>
                        <td>{metrics.recall_weighted:.3f}</td>
                    </tr>
                    <tr>
                        <td><strong>F1-Score</strong></td>
                        <td>{metrics.f1_macro:.3f}</td>
                        <td>{metrics.f1_micro:.3f}</td>
                        <td>{metrics.f1_weighted:.3f}</td>
                    </tr>
                </table>
            </div>
        """
        
        # Add calibration and uncertainty metrics if available
        if metrics.calibration_error is not None:
            html_content += f"""
            <div class="section">
                <h2>🎚️ Calibration & Uncertainty</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{metrics.calibration_error:.3f}</div>
                        <div class="metric-label">Calibration Error</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.mean_confidence:.3f}</div>
                        <div class="metric-label">Mean Confidence</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.mean_entropy:.3f}</div>
                        <div class="metric-label">Mean Entropy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.reliability_score:.3f}</div>
                        <div class="metric-label">Reliability Score</div>
                    </div>
                </div>
            </div>
            """
        
        # Add per-class performance table
        if metrics.class_metrics:
            html_content += """
            <div class="section">
                <h2>📋 Per-Class Performance</h2>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                        <th>Accuracy</th>
                    </tr>
            """
            
            for class_name, class_metrics in metrics.class_metrics.items():
                precision = class_metrics['precision']
                recall = class_metrics['recall']
                f1 = class_metrics['f1_score']
                support = class_metrics['support']
                accuracy = class_metrics.get('accuracy', 0)
                
                # Color code based on F1 score
                f1_class = 'good' if f1 >= 0.8 else 'warning' if f1 >= 0.6 else 'poor'
                
                html_content += f"""
                <tr>
                    <td><strong>{class_name}</strong></td>
                    <td>{precision:.3f}</td>
                    <td>{recall:.3f}</td>
                    <td class="{f1_class}">{f1:.3f}</td>
                    <td>{support}</td>
                    <td>{accuracy:.3f}</td>
                </tr>
                """
            
            html_content += "</table></div>"
        
        # Add summary insights
        html_content += """
        <div class="section">
            <h2>💡 Key Insights</h2>
            <div class="summary-stats">
        """
        
        # Generate insights based on metrics
        insights = self._generate_performance_insights(metrics, predictions)
        for insight in insights:
            html_content += f"<p>• {insight}</p>"
        
        html_content += """
            </div>
        </div>
        
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_performance_insights(
        self,
        metrics: EvaluationMetrics,
        predictions: ModelPredictions
    ) -> List[str]:
        """Generate automated insights from performance metrics."""
        
        insights = []
        
        # Overall performance assessment
        if metrics.accuracy >= 0.9:
            insights.append("🎉 Excellent overall performance with >90% accuracy")
        elif metrics.accuracy >= 0.8:
            insights.append("✅ Good performance with >80% accuracy")
        elif metrics.accuracy >= 0.7:
            insights.append("⚠️ Moderate performance - consider improvements")
        else:
            insights.append("🚨 Poor performance - significant improvements needed")
        
        # Calibration assessment
        if metrics.calibration_error is not None:
            if metrics.calibration_error < 0.05:
                insights.append("🎯 Well-calibrated model with low calibration error")
            elif metrics.calibration_error < 0.1:
                insights.append("📊 Reasonably calibrated model")
            else:
                insights.append("⚖️ Poor calibration - confidence scores may be unreliable")
        
        # Uncertainty analysis
        if metrics.mean_entropy is not None:
            max_entropy = np.log(len(predictions.class_names))
            relative_entropy = metrics.mean_entropy / max_entropy
            
            if relative_entropy < 0.3:
                insights.append("🔒 Model shows high confidence in predictions")
            elif relative_entropy < 0.7:
                insights.append("🤔 Moderate uncertainty in predictions")
            else:
                insights.append("❓ High uncertainty - model may need more training")
        
        # Class balance analysis
        if metrics.class_metrics:
            f1_scores = [cls_metrics['f1_score'] for cls_metrics in metrics.class_metrics.values()]
            f1_std = np.std(f1_scores)
            
            if f1_std < 0.1:
                insights.append("⚖️ Consistent performance across all classes")
            elif f1_std < 0.2:
                insights.append("📈 Some variation in per-class performance")
            else:
                insights.append("📊 Significant performance variation - some classes need attention")
                
                # Find worst performing classes
                worst_classes = sorted(
                    metrics.class_metrics.items(),
                    key=lambda x: x[1]['f1_score']
                )[:3]
                
                worst_names = [name for name, _ in worst_classes]
                insights.append(f"🎯 Focus improvement efforts on: {', '.join(worst_names)}")
        
        # Precision vs Recall trade-off
        precision_recall_diff = abs(metrics.precision_macro - metrics.recall_macro)
        if precision_recall_diff > 0.1:
            if metrics.precision_macro > metrics.recall_macro:
                insights.append("🎯 Model favors precision over recall - few false positives")
            else:
                insights.append("🔍 Model favors recall over precision - few false negatives")
        else:
            insights.append("⚖️ Well-balanced precision and recall")
        
        return insights


# Convenience function for quick evaluation
def quick_advanced_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
    model_name: str = "Model"
) -> Tuple[EvaluationMetrics, Dict[str, str]]:
    """
    Perform quick advanced evaluation with visualizations.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation data loader
        criterion: Loss criterion
        device: Computation device
        class_names: List of class names
        model_name: Name for the model
        
    Returns:
        Tuple of (metrics, visualization paths)
    """
    evaluator = AdvancedEvaluator(class_names)
    metrics, predictions = evaluator.comprehensive_evaluate(model, dataloader, criterion, device)
    viz_paths = evaluator.create_comprehensive_visualizations(metrics, predictions, model_name)
    evaluator.generate_evaluation_report(metrics, predictions, model_name)
    
    return metrics, viz_paths