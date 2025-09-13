"""
Enhanced EDA Module for CIFAR-10 Project

This module provides programmatic EDA utilities to complement the existing eda.ipynb notebook.
It focuses on automated analysis functions that can be integrated into the training pipeline
and provide real-time insights during development.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """
    Automated analyzer to complement the interactive EDA notebook.
    Provides functions that can be called programmatically during training.
    """
    
    def __init__(self, save_dir: str = "images/eda"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dataset analyzer initialized. Save directory: {self.save_dir}")
    
    def quick_dataset_summary(self, dataset: CIFAR10) -> Dict[str, Any]:
        """
        Generate a quick dataset summary for logging/monitoring.
        
        Args:
            dataset: CIFAR-10 dataset instance
            
        Returns:
            Dictionary with key dataset metrics
        """
        # Basic info
        total_samples = len(dataset)
        num_classes = len(dataset.classes)
        
        # Sample for shape analysis
        sample_img, _ = dataset[0]
        if hasattr(sample_img, 'size'):  # PIL Image
            img_shape = sample_img.size[::-1] + (len(sample_img.getbands()),)
        else:  # Tensor
            img_shape = sample_img.shape
        
        # Quick class balance check (sample subset for speed)
        sample_size = min(1000, total_samples)
        indices = torch.randperm(total_samples)[:sample_size]
        
        class_counts = defaultdict(int)
        for idx in indices:
            _, label = dataset[idx]
            class_counts[label] += 1
        
        counts = list(class_counts.values())
        balance_cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
        
        summary = {
            'total_samples': total_samples,
            'num_classes': num_classes,
            'class_names': dataset.classes,
            'image_shape': img_shape,
            'balance_coefficient_of_variation': balance_cv,
            'is_balanced': balance_cv < 0.1,
            'sample_size_analyzed': sample_size
        }
        
        logger.info(f"Dataset summary: {total_samples} samples, {num_classes} classes, balanced: {summary['is_balanced']}")
        return summary
    
    def analyze_data_loader_batch(self, dataloader: DataLoader, num_batches: int = 3) -> Dict[str, Any]:
        """
        Analyze a few batches from a DataLoader to validate preprocessing.
        
        Args:
            dataloader: DataLoader to analyze
            num_batches: Number of batches to analyze
            
        Returns:
            Analysis results dictionary
        """
        batch_stats = []
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            # Calculate batch statistics
            batch_mean = images.mean(dim=[0, 2, 3])  # Per channel mean
            batch_std = images.std(dim=[0, 2, 3])    # Per channel std
            batch_min = images.min().item()
            batch_max = images.max().item()
            
            batch_stats.append({
                'batch_idx': batch_idx,
                'batch_size': images.size(0),
                'image_shape': images.shape[1:],
                'mean_per_channel': batch_mean.tolist(),
                'std_per_channel': batch_std.tolist(),
                'global_min': batch_min,
                'global_max': batch_max,
                'has_nan': torch.isnan(images).any().item(),
                'has_inf': torch.isinf(images).any().item(),
                'unique_labels': len(torch.unique(labels))
            })
        
        # Aggregate statistics
        if batch_stats:
            overall_means = [stats['mean_per_channel'] for stats in batch_stats]
            overall_stds = [stats['std_per_channel'] for stats in batch_stats]
            
            aggregated = {
                'num_batches_analyzed': len(batch_stats),
                'avg_batch_size': np.mean([stats['batch_size'] for stats in batch_stats]),
                'mean_stability': np.std(overall_means, axis=0).tolist(),
                'std_stability': np.std(overall_stds, axis=0).tolist(),
                'value_range': {
                    'min': min(stats['global_min'] for stats in batch_stats),
                    'max': max(stats['global_max'] for stats in batch_stats)
                },
                'data_quality': {
                    'has_nan': any(stats['has_nan'] for stats in batch_stats),
                    'has_inf': any(stats['has_inf'] for stats in batch_stats)
                },
                'batch_details': batch_stats
            }
            
            logger.info(f"DataLoader analysis: {len(batch_stats)} batches, "
                       f"range [{aggregated['value_range']['min']:.3f}, {aggregated['value_range']['max']:.3f}]")
            return aggregated
        
        return {}
    
    def create_class_distribution_chart(self, dataset: CIFAR10, save_path: Optional[str] = None) -> str:
        """
        Create an enhanced class distribution chart.
        
        Args:
            dataset: CIFAR-10 dataset
            save_path: Optional path to save the chart
            
        Returns:
            Path to saved chart
        """
        # Count all classes
        class_counts = defaultdict(int)
        for _, label in dataset:
            class_counts[dataset.classes[label]] += 1
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart with enhanced styling
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = sns.color_palette("husl", len(classes))
        
        bars = ax1.bar(classes, counts, color=colors)
        ax1.set_title('CIFAR-10 Class Distribution', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Classes', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add statistics annotation
        total = sum(counts)
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        cv = std_count / mean_count if mean_count > 0 else 0
        
        stats_text = (
            f"Total: {total:,}\n"
            f"Per class: {mean_count:.0f} ± {std_count:.1f}\n"
            f"CV: {cv:.3f}\n"
            f"Balance: {'Good' if cv < 0.1 else 'Poor'}"
        )
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Pie chart for proportions
        ax2.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Class Proportions', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = self.save_dir / 'automated_class_distribution.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        logger.info(f"Class distribution chart saved to {save_path}")
        return str(save_path)
    
    def create_pixel_analysis_chart(self, dataset: CIFAR10, sample_size: int = 2000, save_path: Optional[str] = None) -> str:
        """
        Create pixel intensity analysis chart with correlation heatmap.
        
        Args:
            dataset: CIFAR-10 dataset
            sample_size: Number of images to sample for analysis
            save_path: Optional path to save the chart
            
        Returns:
            Path to saved chart
        """
        # Sample images for pixel analysis
        indices = torch.randperm(len(dataset))[:sample_size]
        pixel_data = {'R': [], 'G': [], 'B': []}
        
        for idx in indices:
            img, _ = dataset[idx]
            np_img = np.array(img)
            
            # Extract channel data
            pixel_data['R'].extend(np_img[:, :, 0].flatten())
            pixel_data['G'].extend(np_img[:, :, 1].flatten())
            pixel_data['B'].extend(np_img[:, :, 2].flatten())
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 10))
        
        # Channel histograms
        for i, (channel, color) in enumerate(zip(['R', 'G', 'B'], ['red', 'green', 'blue'])):
            ax = plt.subplot(2, 3, i + 1)
            data = pixel_data[channel]
            
            ax.hist(data, bins=50, alpha=0.7, color=color, edgecolor='black')
            ax.set_title(f'{channel} Channel Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Pixel Intensity')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            skew_val = self._calculate_skewness(np.array(data))
            
            stats_text = f'μ={mean_val:.1f}\nσ={std_val:.1f}\nskew={skew_val:.2f}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
        
        # Combined distribution
        ax = plt.subplot(2, 3, 4)
        for channel, color in zip(['R', 'G', 'B'], ['red', 'green', 'blue']):
            ax.hist(pixel_data[channel], bins=50, alpha=0.4, color=color, label=f'{channel} Channel')
        ax.set_title('Combined Channel Distributions', fontsize=12, fontweight='bold')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Channel correlation heatmap
        ax = plt.subplot(2, 3, 5)
        
        # Sample data for correlation (limit for performance)
        sample_data = {channel: data[:10000] for channel, data in pixel_data.items()}
        df_sample = pd.DataFrame(sample_data)
        corr_matrix = df_sample.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, cbar_kws={"shrink": .8})
        ax.set_title('Channel Correlations', fontsize=12, fontweight='bold')
        
        # Channel statistics summary
        ax = plt.subplot(2, 3, 6)
        ax.axis('off')
        
        # Calculate comprehensive statistics
        channel_stats = {}
        for channel in ['R', 'G', 'B']:
            data = np.array(pixel_data[channel])
            channel_stats[channel] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'skewness': self._calculate_skewness(data)
            }
        
        # Create statistics table
        stats_text = "Channel Statistics:\n\n"
        stats_text += f"{'Channel':<8} {'Mean':<6} {'Std':<6} {'Skew':<6}\n"
        stats_text += "-" * 32 + "\n"
        
        for channel, stats in channel_stats.items():
            stats_text += f"{channel:<8} {stats['mean']:<6.1f} {stats['std']:<6.1f} {stats['skewness']:<6.2f}\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11, fontfamily='monospace',
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'CIFAR-10 Pixel Analysis (Sample: {sample_size:,} images)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = self.save_dir / 'automated_pixel_analysis.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Pixel analysis chart saved to {save_path}")
        return str(save_path)
    
    def analyze_augmentation_impact(
        self, 
        original_loader: DataLoader, 
        augmented_loader: DataLoader,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare original vs augmented data to show augmentation impact.
        
        Args:
            original_loader: DataLoader with original data
            augmented_loader: DataLoader with augmented data
            save_path: Optional path to save comparison chart
            
        Returns:
            Analysis results
        """
        # Analyze both loaders
        orig_stats = self.analyze_data_loader_batch(original_loader, num_batches=5)
        aug_stats = self.analyze_data_loader_batch(augmented_loader, num_batches=5)
        
        comparison = {
            'original_stats': orig_stats,
            'augmented_stats': aug_stats,
            'differences': {}
        }
        
        # Calculate differences
        if orig_stats and aug_stats:
            orig_means = np.array(orig_stats['batch_details'][0]['mean_per_channel'])
            aug_means = np.mean([batch['mean_per_channel'] for batch in aug_stats['batch_details']], axis=0)
            
            orig_range = orig_stats['value_range']
            aug_range = aug_stats['value_range']
            
            comparison['differences'] = {
                'mean_shift': (aug_means - orig_means).tolist(),
                'range_expansion': {
                    'min_change': aug_range['min'] - orig_range['min'],
                    'max_change': aug_range['max'] - orig_range['max']
                }
            }
            
            # Create visualization
            if save_path is None:
                save_path = self.save_dir / 'augmentation_impact.png'
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Mean comparison
            channels = ['Red', 'Green', 'Blue']
            x = range(len(channels))
            
            axes[0].bar([i - 0.2 for i in x], orig_means, 0.4, label='Original', alpha=0.7)
            axes[0].bar([i + 0.2 for i in x], aug_means, 0.4, label='Augmented', alpha=0.7)
            axes[0].set_xlabel('Channel')
            axes[0].set_ylabel('Mean Value')
            axes[0].set_title('Channel Means: Original vs Augmented')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(channels)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Range comparison
            ranges_orig = [orig_range['max'] - orig_range['min']]
            ranges_aug = [aug_range['max'] - aug_range['min']]
            
            axes[1].bar(['Original', 'Augmented'], [ranges_orig[0], ranges_aug[0]], 
                       color=['blue', 'orange'], alpha=0.7)
            axes[1].set_ylabel('Value Range')
            axes[1].set_title('Dynamic Range Comparison')
            axes[1].grid(True, alpha=0.3)
            
            # Mean shift visualization
            axes[2].bar(channels, comparison['differences']['mean_shift'], 
                       color=['red', 'green', 'blue'], alpha=0.7)
            axes[2].set_xlabel('Channel')
            axes[2].set_ylabel('Mean Shift')
            axes[2].set_title('Augmentation-Induced Mean Shift')
            axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Augmentation impact analysis saved to {save_path}")
        
        return comparison
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3) if std > 0 else 0
    
    def create_data_quality_report(self, dataset: CIFAR10, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report.
        
        Args:
            dataset: Raw dataset
            dataloader: Processed dataloader
            
        Returns:
            Quality report dictionary
        """
        report = {
            'dataset_summary': self.quick_dataset_summary(dataset),
            'preprocessing_analysis': self.analyze_data_loader_batch(dataloader),
            'visualizations': {},
            'recommendations': []
        }
        
        # Generate visualizations
        report['visualizations']['class_distribution'] = self.create_class_distribution_chart(dataset)
        report['visualizations']['pixel_analysis'] = self.create_pixel_analysis_chart(dataset)
        
        # Generate recommendations based on analysis
        dataset_summary = report['dataset_summary']
        
        if not dataset_summary['is_balanced']:
            report['recommendations'].append(
                f"Dataset shows class imbalance (CV: {dataset_summary['balance_coefficient_of_variation']:.3f}). "
                "Consider using weighted sampling or class weights."
            )
        
        preprocessing_analysis = report['preprocessing_analysis']
        if preprocessing_analysis.get('data_quality', {}).get('has_nan', False):
            report['recommendations'].append("NaN values detected in preprocessing. Check transform pipeline.")
        
        if preprocessing_analysis.get('data_quality', {}).get('has_inf', False):
            report['recommendations'].append("Infinite values detected in preprocessing. Check normalization.")
        
        # Check value range
        value_range = preprocessing_analysis.get('value_range', {})
        if value_range:
            range_span = value_range['max'] - value_range['min']
            if range_span > 10:  # Unnormalized data
                report['recommendations'].append(
                    f"Large value range detected ({range_span:.2f}). Ensure proper normalization."
                )
        
        logger.info(f"Data quality report generated with {len(report['recommendations'])} recommendations")
        return report


def validate_dataset_preprocessing(dataset: CIFAR10, dataloader: DataLoader) -> bool:
    """
    Quick validation function to check if dataset preprocessing is correct.
    
    Args:
        dataset: Raw CIFAR-10 dataset
        dataloader: Preprocessed dataloader
        
    Returns:
        True if preprocessing appears correct
    """
    analyzer = DatasetAnalyzer()
    
    try:
        # Quick checks
        dataset_summary = analyzer.quick_dataset_summary(dataset)
        loader_analysis = analyzer.analyze_data_loader_batch(dataloader, num_batches=2)
        
        # Validation criteria
        checks = []
        
        # 1. Dataset should be balanced
        checks.append(dataset_summary['is_balanced'])
        
        # 2. No NaN or Inf values
        if loader_analysis:
            data_quality = loader_analysis.get('data_quality', {})
            checks.append(not data_quality.get('has_nan', True))
            checks.append(not data_quality.get('has_inf', True))
            
            # 3. Reasonable value range (normalized data should be roughly [-3, 3])
            value_range = loader_analysis.get('value_range', {})
            if value_range:
                range_reasonable = (-5 <= value_range['min'] <= 5) and (-5 <= value_range['max'] <= 5)
                checks.append(range_reasonable)
        
        all_passed = all(checks)
        
        if all_passed:
            logger.info("Dataset preprocessing validation PASSED")
        else:
            logger.warning(f"Dataset preprocessing validation FAILED. Checks: {checks}")
        
        return all_passed
        
    except Exception as e:
        logger.error(f"Dataset preprocessing validation failed with error: {e}")
        return False