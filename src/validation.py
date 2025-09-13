"""
Data Validation Module for CIFAR-10 Project

This module provides comprehensive validation utilities for data preprocessing,
transform outputs, and dataset statistics.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None


@dataclass
class DatasetValidationReport:
    """Comprehensive report of dataset validation."""
    dataset_name: str
    total_samples: int
    shape_validation: ValidationResult
    range_validation: ValidationResult
    statistics_validation: ValidationResult
    class_distribution: ValidationResult
    overall_passed: bool


class DatasetValidator:
    """Validates dataset properties and preprocessing outputs."""
    
    def __init__(self, tolerance: float = 0.1):
        """
        Initialize validator with tolerance settings.
        
        Args:
            tolerance: Tolerance for statistical comparisons
        """
        self.tolerance = tolerance
    
    def validate_cifar10_dataset(
        self, 
        dataset: CIFAR10,
        expected_classes: int = 10,
        expected_shape: Tuple[int, int, int] = (3, 32, 32),
        expected_size: int = 50000
    ) -> DatasetValidationReport:
        """
        Comprehensive validation of CIFAR-10 dataset.
        
        Args:
            dataset: CIFAR-10 dataset instance
            expected_classes: Expected number of classes
            expected_shape: Expected image shape (C, H, W)
            expected_size: Expected dataset size
            
        Returns:
            Comprehensive validation report
        """
        logger.info(f"Validating CIFAR-10 dataset...")
        
        # Initialize results
        results = {}
        
        # 1. Basic properties validation
        actual_size = len(dataset)
        if actual_size != expected_size:
            results["size"] = ValidationResult(
                passed=False,
                message=f"Dataset size mismatch: expected {expected_size}, got {actual_size}"
            )
        else:
            results["size"] = ValidationResult(
                passed=True,
                message=f"Dataset size correct: {actual_size} samples"
            )
        
        # 2. Shape validation
        sample_image, _ = dataset[0]
        if hasattr(sample_image, 'size'):  # PIL Image
            actual_shape = (len(sample_image.getbands()),) + sample_image.size[::-1]  # (C, H, W)
        else:  # Already tensor
            actual_shape = sample_image.shape
        
        shape_valid = actual_shape == expected_shape
        results["shape"] = ValidationResult(
            passed=shape_valid,
            message=f"Shape validation: expected {expected_shape}, got {actual_shape}",
            details={"expected": expected_shape, "actual": actual_shape}
        )
        
        # 3. Class distribution validation
        class_counts = self._count_classes(dataset, expected_classes)
        results["classes"] = self._validate_class_distribution(class_counts, expected_classes)
        
        # 4. Pixel value range validation (for PIL images)
        results["range"] = self._validate_pixel_range(dataset)
        
        # Create overall report
        overall_passed = all(result.passed for result in results.values())
        
        return DatasetValidationReport(
            dataset_name="CIFAR-10",
            total_samples=actual_size,
            shape_validation=results["shape"],
            range_validation=results["range"],
            statistics_validation=ValidationResult(True, "N/A for raw dataset"),
            class_distribution=results["classes"],
            overall_passed=overall_passed
        )
    
    def validate_dataloader_output(
        self,
        dataloader: DataLoader,
        expected_shape: Tuple[int, int, int],
        expected_mean: Optional[List[float]] = None,
        expected_std: Optional[List[float]] = None,
        num_batches_to_check: int = 5
    ) -> ValidationResult:
        """
        Validate DataLoader output properties.
        
        Args:
            dataloader: DataLoader to validate
            expected_shape: Expected tensor shape (C, H, W)
            expected_mean: Expected mean values per channel
            expected_std: Expected std values per channel
            num_batches_to_check: Number of batches to sample
            
        Returns:
            Validation result
        """
        logger.info("Validating DataLoader output...")
        
        issues = []
        warnings_list = []
        
        try:
            # Sample batches
            batch_count = 0
            all_samples = []
            
            for batch_idx, (images, labels) in enumerate(dataloader):
                if batch_count >= num_batches_to_check:
                    break
                
                # Check batch properties
                if batch_idx == 0:  # First batch checks
                    if images.shape[1:] != expected_shape:
                        issues.append(f"Shape mismatch: expected {expected_shape}, got {images.shape[1:]}")
                    
                    if images.dtype != torch.float32:
                        issues.append(f"Data type should be float32, got {images.dtype}")
                
                # Check for invalid values
                if torch.isnan(images).any():
                    issues.append(f"NaN values found in batch {batch_idx}")
                
                if torch.isinf(images).any():
                    issues.append(f"Infinite values found in batch {batch_idx}")
                
                all_samples.append(images)
                batch_count += 1
            
            # Statistical validation
            if expected_mean is not None and expected_std is not None:
                combined_samples = torch.cat(all_samples, dim=0)
                actual_mean = combined_samples.mean(dim=[0, 2, 3])
                actual_std = combined_samples.std(dim=[0, 2, 3])
                
                # Check means
                mean_diff = torch.abs(actual_mean - torch.tensor(expected_mean))
                if (mean_diff > self.tolerance).any():
                    warnings_list.append(
                        f"Mean deviation: expected {expected_mean}, "
                        f"got {actual_mean.tolist()}, diff {mean_diff.tolist()}"
                    )
                
                # Check stds
                std_diff = torch.abs(actual_std - torch.tensor(expected_std))
                if (std_diff > self.tolerance).any():
                    warnings_list.append(
                        f"Std deviation: expected {expected_std}, "
                        f"got {actual_std.tolist()}, diff {std_diff.tolist()}"
                    )
            
            # Create result
            passed = len(issues) == 0
            message = "DataLoader validation passed" if passed else f"DataLoader validation failed: {'; '.join(issues)}"
            
            return ValidationResult(
                passed=passed,
                message=message,
                warnings=warnings_list if warnings_list else None,
                details={
                    "batches_checked": batch_count,
                    "issues_found": len(issues),
                    "warnings_count": len(warnings_list)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"DataLoader validation failed with error: {str(e)}"
            )
    
    def _count_classes(self, dataset: CIFAR10, expected_classes: int) -> Dict[int, int]:
        """Count samples per class in dataset."""
        class_counts = {i: 0 for i in range(expected_classes)}
        
        # Sample subset for counting (for performance)
        sample_size = min(len(dataset), 10000)
        indices = torch.randperm(len(dataset))[:sample_size]
        
        for idx in indices:
            _, label = dataset[idx]
            if label < expected_classes:
                class_counts[label] += 1
        
        return class_counts
    
    def _validate_class_distribution(
        self, 
        class_counts: Dict[int, int], 
        expected_classes: int
    ) -> ValidationResult:
        """Validate class distribution balance."""
        
        if len(class_counts) != expected_classes:
            return ValidationResult(
                passed=False,
                message=f"Wrong number of classes: expected {expected_classes}, found {len(class_counts)}"
            )
        
        # Check balance
        counts = list(class_counts.values())
        min_count, max_count = min(counts), max(counts)
        
        # Allow 10% imbalance
        imbalance_threshold = 0.1
        if min_count > 0:
            imbalance_ratio = (max_count - min_count) / min_count
            if imbalance_ratio > imbalance_threshold:
                return ValidationResult(
                    passed=False,
                    message=f"Class imbalance detected: ratio {imbalance_ratio:.3f} > {imbalance_threshold}",
                    details={"class_counts": class_counts, "imbalance_ratio": imbalance_ratio}
                )
        
        return ValidationResult(
            passed=True,
            message=f"Class distribution balanced: {counts}",
            details={"class_counts": class_counts}
        )
    
    def _validate_pixel_range(self, dataset: CIFAR10) -> ValidationResult:
        """Validate pixel value ranges in dataset."""
        
        # Sample a few images
        sample_indices = torch.randperm(len(dataset))[:100]
        min_vals, max_vals = [], []
        
        for idx in sample_indices:
            image, _ = dataset[idx]
            
            if hasattr(image, 'getdata'):  # PIL Image
                pixels = list(image.getdata())
                if isinstance(pixels[0], tuple):  # RGB
                    flat_pixels = [val for pixel in pixels for val in pixel]
                else:  # Grayscale
                    flat_pixels = pixels
                
                min_vals.append(min(flat_pixels))
                max_vals.append(max(flat_pixels))
            else:  # Tensor
                min_vals.append(image.min().item())
                max_vals.append(image.max().item())
        
        overall_min = min(min_vals)
        overall_max = max(max_vals)
        
        # Check expected ranges
        expected_min, expected_max = 0, 255  # For PIL images
        
        if overall_min < expected_min or overall_max > expected_max:
            return ValidationResult(
                passed=False,
                message=f"Pixel values out of range: [{overall_min}, {overall_max}] not in [{expected_min}, {expected_max}]",
                details={"min": overall_min, "max": overall_max}
            )
        
        return ValidationResult(
            passed=True,
            message=f"Pixel values in valid range: [{overall_min}, {overall_max}]",
            details={"min": overall_min, "max": overall_max}
        )


class StatisticsValidator:
    """Validates dataset and transform statistics."""
    
    @staticmethod
    def calculate_dataset_statistics(
        dataloader: DataLoader,
        num_batches: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate mean and std statistics for a dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            num_batches: Limit number of batches to process
            
        Returns:
            Tuple of (mean, std) tensors for each channel
        """
        logger.info("Calculating dataset statistics...")
        
        mean = torch.zeros(3)  # Assume RGB
        std = torch.zeros(3)
        total_samples = 0
        
        batch_count = 0
        for images, _ in dataloader:
            if num_batches and batch_count >= num_batches:
                break
                
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_samples += batch_samples
            batch_count += 1
        
        mean /= total_samples
        std /= total_samples
        
        logger.info(f"Dataset statistics: mean={mean.tolist()}, std={std.tolist()}")
        return mean, std
    
    @staticmethod
    def compare_statistics(
        actual_mean: torch.Tensor,
        actual_std: torch.Tensor,
        expected_mean: List[float],
        expected_std: List[float],
        tolerance: float = 0.05
    ) -> ValidationResult:
        """
        Compare actual vs expected statistics.
        
        Args:
            actual_mean: Calculated mean values
            actual_std: Calculated std values
            expected_mean: Expected mean values
            expected_std: Expected std values
            tolerance: Tolerance for comparisons
            
        Returns:
            Validation result
        """
        expected_mean_tensor = torch.tensor(expected_mean)
        expected_std_tensor = torch.tensor(expected_std)
        
        mean_diff = torch.abs(actual_mean - expected_mean_tensor)
        std_diff = torch.abs(actual_std - expected_std_tensor)
        
        mean_ok = (mean_diff < tolerance).all()
        std_ok = (std_diff < tolerance).all()
        
        if mean_ok and std_ok:
            return ValidationResult(
                passed=True,
                message="Statistics validation passed",
                details={
                    "actual_mean": actual_mean.tolist(),
                    "actual_std": actual_std.tolist(),
                    "expected_mean": expected_mean,
                    "expected_std": expected_std,
                    "mean_diff": mean_diff.tolist(),
                    "std_diff": std_diff.tolist()
                }
            )
        else:
            issues = []
            if not mean_ok:
                issues.append(f"Mean mismatch: {mean_diff.tolist()}")
            if not std_ok:
                issues.append(f"Std mismatch: {std_diff.tolist()}")
            
            return ValidationResult(
                passed=False,
                message=f"Statistics validation failed: {'; '.join(issues)}",
                details={
                    "actual_mean": actual_mean.tolist(),
                    "actual_std": actual_std.tolist(),
                    "expected_mean": expected_mean,
                    "expected_std": expected_std,
                    "mean_diff": mean_diff.tolist(),
                    "std_diff": std_diff.tolist()
                }
            )
    
    @staticmethod
    def plot_statistics_comparison(
        actual_mean: torch.Tensor,
        actual_std: torch.Tensor,
        expected_mean: List[float],
        expected_std: List[float],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison of actual vs expected statistics.
        
        Args:
            actual_mean: Calculated mean values
            actual_std: Calculated std values
            expected_mean: Expected mean values
            expected_std: Expected std values
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        channels = ['Red', 'Green', 'Blue']
        x = range(len(channels))
        
        # Mean comparison
        ax1.bar([i - 0.2 for i in x], actual_mean.tolist(), 0.4, label='Actual', alpha=0.7)
        ax1.bar([i + 0.2 for i in x], expected_mean, 0.4, label='Expected', alpha=0.7)
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('Mean')
        ax1.set_title('Mean Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(channels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Std comparison
        ax2.bar([i - 0.2 for i in x], actual_std.tolist(), 0.4, label='Actual', alpha=0.7)
        ax2.bar([i + 0.2 for i in x], expected_std, 0.4, label='Expected', alpha=0.7)
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Standard Deviation Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(channels)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Statistics comparison plot saved to {save_path}")
        
        plt.show()


def validate_preprocessing_pipeline(
    dataloader: DataLoader,
    model_name: str,
    expected_shape: Tuple[int, int, int],
    expected_mean: List[float],
    expected_std: List[float],
    tolerance: float = 0.1
) -> ValidationResult:
    """
    Comprehensive validation of preprocessing pipeline.
    
    Args:
        dataloader: DataLoader with preprocessing applied
        model_name: Name of the model for logging
        expected_shape: Expected tensor shape
        expected_mean: Expected normalization mean
        expected_std: Expected normalization std
        tolerance: Statistical tolerance
        
    Returns:
        Overall validation result
    """
    logger.info(f"Validating preprocessing pipeline for {model_name}")
    
    validator = DatasetValidator(tolerance=tolerance)
    result = validator.validate_dataloader_output(
        dataloader, expected_shape, expected_mean, expected_std
    )
    
    if result.passed:
        logger.info(f"Preprocessing validation passed for {model_name}")
    else:
        logger.error(f"Preprocessing validation failed for {model_name}: {result.message}")
        if result.warnings:
            for warning in result.warnings:
                logger.warning(warning)
    
    return result