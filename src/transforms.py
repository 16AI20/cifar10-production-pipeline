"""
Enhanced Transform Management for CIFAR-10 Project

This module provides configurable, reusable transform pipelines with validation
and advanced augmentation strategies.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import yaml
import torch
import numpy as np
from pathlib import Path

import torchvision.transforms as T
from torchvision.transforms import (
    Compose, Normalize, ToTensor, Resize, RandomCrop, RandomHorizontalFlip,
    ColorJitter, RandomRotation, RandomAffine, RandomErasing, GaussianBlur
)

from .enums import ModelType
from .augmentations import create_advanced_transforms

logger = logging.getLogger(__name__)


class TransformValidationError(Exception):
    """Custom exception for transform validation failures."""
    pass


class DatasetStats:
    """Container for dataset statistics."""
    
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std
        self.validate()
    
    def validate(self) -> None:
        """Validate dataset statistics."""
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("Dataset statistics must have exactly 3 values (RGB)")
        
        if any(s <= 0 for s in self.std):
            raise ValueError("Standard deviations must be positive")
        
        if any(abs(m) > 2.0 for m in self.mean):
            warnings.warn(f"Unusual mean values detected: {self.mean}")


class TransformManager:
    """Manages transform configurations and creates transform pipelines."""
    
    def __init__(self, config_path: str = "conf/transforms.yaml"):
        """
        Initialize TransformManager with configuration.
        
        Args:
            config_path: Path to transforms configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load transform configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded transform configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Transform config file not found: {self.config_path}")
            # Return default configuration
            return self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file loading fails."""
        return {
            "dataset_stats": {
                "imagenet": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "cifar10": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]}
            },
            "input_sizes": {"pretrained_cnn": 224, "custom_cnn": 32, "vit": 224},
            "augmentation_strategies": {"basic": {}},
            "model_configs": {
                "resnet": {"dataset_stats": "imagenet", "input_size": "pretrained_cnn", "augmentation_strategy": "basic"},
                "efficientnet": {"dataset_stats": "imagenet", "input_size": "pretrained_cnn", "augmentation_strategy": "basic"},
                "vit": {"dataset_stats": "imagenet", "input_size": "vit", "augmentation_strategy": "basic"}
            }
        }
    
    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        required_keys = ["dataset_stats", "input_sizes", "augmentation_strategies", "model_configs"]
        
        for key in required_keys:
            if key not in self.config:
                raise TransformValidationError(f"Missing required config key: {key}")
        
        # Validate dataset stats
        for stats_name, stats in self.config["dataset_stats"].items():
            try:
                DatasetStats(stats["mean"], stats["std"])
            except (KeyError, ValueError) as e:
                raise TransformValidationError(f"Invalid dataset stats '{stats_name}': {e}")
    
    def get_dataset_stats(self, stats_name: str) -> DatasetStats:
        """Get dataset statistics by name."""
        if stats_name not in self.config["dataset_stats"]:
            raise ValueError(f"Unknown dataset stats: {stats_name}")
        
        stats = self.config["dataset_stats"][stats_name]
        return DatasetStats(stats["mean"], stats["std"])
    
    def get_input_size(self, size_name: str) -> int:
        """Get input size by name."""
        if size_name not in self.config["input_sizes"]:
            raise ValueError(f"Unknown input size: {size_name}")
        
        return self.config["input_sizes"][size_name]
    
    def create_transforms(
        self, 
        model_type: Union[ModelType, str],
        train: bool = True,
        custom_strategy: Optional[str] = None
    ) -> Compose:
        """
        Create transform pipeline for a given model type and mode.
        
        Args:
            model_type: Model type (enum or string)
            train: Whether to create training transforms (with augmentation)
            custom_strategy: Override augmentation strategy
            
        Returns:
            Composed transform pipeline
        """
        # Convert to string if ModelType enum
        if isinstance(model_type, ModelType):
            model_name = model_type.value
        else:
            model_name = str(model_type).lower()
        
        # Get model configuration
        if model_name not in self.config["model_configs"]:
            logger.warning(f"No config for model '{model_name}', using defaults")
            model_config = {
                "dataset_stats": "imagenet",
                "input_size": "pretrained_cnn", 
                "augmentation_strategy": "basic"
            }
        else:
            model_config = self.config["model_configs"][model_name]
        
        # Get components
        stats = self.get_dataset_stats(model_config["dataset_stats"])
        input_size = self.get_input_size(model_config["input_size"])
        strategy_name = custom_strategy or model_config["augmentation_strategy"]
        
        logger.info(f"Creating transforms for {model_name} | "
                   f"Stats: {model_config['dataset_stats']} | "
                   f"Size: {input_size} | Strategy: {strategy_name}")
        
        if train:
            return self._create_train_transforms(input_size, stats, strategy_name)
        else:
            return self._create_eval_transforms(input_size, stats)
    
    def _create_train_transforms(
        self, 
        input_size: int, 
        stats: DatasetStats, 
        strategy_name: str
    ) -> Compose:
        """Create training transforms with augmentation."""
        transforms = []
        
        # Basic resizing
        transforms.append(Resize(input_size))
        
        # Get augmentation strategy
        if strategy_name not in self.config["augmentation_strategies"]:
            logger.warning(f"Unknown strategy '{strategy_name}', using basic")
            strategy_name = "basic"
        
        strategy = self.config["augmentation_strategies"][strategy_name]
        
        # Add augmentations based on strategy
        self._add_augmentations(transforms, strategy, input_size)
        
        # Convert to tensor and normalize
        transforms.append(ToTensor())
        
        # Add post-tensor augmentations
        self._add_post_tensor_augmentations(transforms, strategy)
        
        # Add advanced augmentations if enabled
        advanced_transforms = self._get_advanced_augmentations(strategy)
        transforms.extend(advanced_transforms)
        
        # Normalize
        transforms.append(Normalize(mean=stats.mean, std=stats.std))
        
        return Compose(transforms)
    
    def _create_eval_transforms(self, input_size: int, stats: DatasetStats) -> Compose:
        """Create evaluation transforms without augmentation."""
        transforms = [
            Resize(input_size),
            ToTensor(),
            Normalize(mean=stats.mean, std=stats.std)
        ]
        return Compose(transforms)
    
    def _add_augmentations(
        self, 
        transforms: List, 
        strategy: Dict[str, Any], 
        input_size: int
    ) -> None:
        """Add pre-tensor augmentations to transform list."""
        
        # Random Crop
        if strategy.get("random_crop", {}).get("enabled", False):
            crop_config = strategy["random_crop"]
            padding = crop_config.get("padding", 4)
            transforms.append(RandomCrop(input_size, padding=padding))
        
        # Random Horizontal Flip
        if strategy.get("random_horizontal_flip", {}).get("enabled", False):
            flip_config = strategy["random_horizontal_flip"]
            p = flip_config.get("p", 0.5)
            transforms.append(RandomHorizontalFlip(p=p))
        
        # Color Jitter
        if strategy.get("color_jitter", {}).get("enabled", False):
            jitter_config = strategy["color_jitter"]
            transforms.append(ColorJitter(
                brightness=jitter_config.get("brightness", 0.0),
                contrast=jitter_config.get("contrast", 0.0),
                saturation=jitter_config.get("saturation", 0.0),
                hue=jitter_config.get("hue", 0.0)
            ))
        
        # Random Rotation
        if strategy.get("random_rotation", {}).get("enabled", False):
            rotation_config = strategy["random_rotation"]
            degrees = rotation_config.get("degrees", 0)
            transforms.append(RandomRotation(degrees))
        
        # Random Affine
        if strategy.get("random_affine", {}).get("enabled", False):
            affine_config = strategy["random_affine"]
            transforms.append(RandomAffine(
                degrees=affine_config.get("degrees", 0),
                translate=affine_config.get("translate", None),
                scale=affine_config.get("scale", None),
                shear=affine_config.get("shear", None)
            ))
        
        # Gaussian Blur
        if strategy.get("gaussian_blur", {}).get("enabled", False):
            blur_config = strategy["gaussian_blur"]
            kernel_size = blur_config.get("kernel_size", [3, 5])
            sigma = blur_config.get("sigma", [0.1, 2.0])
            if torch.rand(1).item() < blur_config.get("p", 0.2):
                transforms.append(GaussianBlur(kernel_size, sigma))
    
    def _add_post_tensor_augmentations(self, transforms: List, strategy: Dict[str, Any]) -> None:
        """Add post-tensor augmentations (that require tensor input)."""
        
        # Random Erasing
        if strategy.get("random_erasing", {}).get("enabled", False):
            erasing_config = strategy["random_erasing"]
            transforms.append(RandomErasing(
                p=erasing_config.get("p", 0.2),
                scale=erasing_config.get("scale", (0.02, 0.33)),
                ratio=erasing_config.get("ratio", (0.3, 3.3))
            ))
    
    def _get_advanced_augmentations(self, strategy: Dict[str, Any]) -> List:
        """Get advanced augmentation transforms if enabled."""
        advanced_transforms = []
        
        # Create advanced transforms using the strategy config
        # This includes RandAugment, TrivialAugment, etc.
        try:
            advanced_list = create_advanced_transforms(strategy)
            advanced_transforms.extend(advanced_list)
            
            if advanced_list:
                logger.info(f"Added {len(advanced_list)} advanced augmentation(s)")
        except Exception as e:
            logger.warning(f"Failed to create advanced augmentations: {e}")
        
        return advanced_transforms


class TransformValidator:
    """Validates transform outputs and dataset statistics."""
    
    @staticmethod
    def validate_transform_output(
        sample_batch: torch.Tensor,
        expected_shape: Tuple[int, ...],
        expected_range: Tuple[float, float] = (-3.0, 3.0)
    ) -> bool:
        """
        Validate transform output properties.
        
        Args:
            sample_batch: Batch of transformed samples
            expected_shape: Expected tensor shape (C, H, W)
            expected_range: Expected value range after normalization
            
        Returns:
            True if validation passes
        """
        try:
            # Check shape
            if sample_batch.shape[1:] != expected_shape:
                logger.error(f"Shape mismatch: expected {expected_shape}, got {sample_batch.shape[1:]}")
                return False
            
            # Check data type
            if sample_batch.dtype != torch.float32:
                logger.error(f"Data type mismatch: expected torch.float32, got {sample_batch.dtype}")
                return False
            
            # Check value range
            min_val, max_val = sample_batch.min().item(), sample_batch.max().item()
            if not (expected_range[0] <= min_val and max_val <= expected_range[1]):
                logger.warning(f"Values outside expected range {expected_range}: [{min_val:.3f}, {max_val:.3f}]")
            
            # Check for NaN/Inf
            if torch.isnan(sample_batch).any() or torch.isinf(sample_batch).any():
                logger.error("Found NaN or Inf values in transformed data")
                return False
            
            logger.debug("Transform validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Transform validation failed: {e}")
            return False
    
    @staticmethod
    def validate_normalization(
        sample_batch: torch.Tensor,
        expected_mean: List[float],
        expected_std: List[float],
        tolerance: float = 0.1
    ) -> bool:
        """
        Validate that batch statistics match expected normalization.
        
        Args:
            sample_batch: Batch of normalized samples (N, C, H, W)
            expected_mean: Expected mean for each channel
            expected_std: Expected standard deviation for each channel
            tolerance: Tolerance for mean/std validation
            
        Returns:
            True if validation passes
        """
        try:
            # Calculate actual statistics
            actual_mean = sample_batch.mean(dim=[0, 2, 3])  # Average over batch, height, width
            actual_std = sample_batch.std(dim=[0, 2, 3])
            
            expected_mean_tensor = torch.tensor(expected_mean)
            expected_std_tensor = torch.tensor(expected_std)
            
            # Check means
            mean_diff = torch.abs(actual_mean - expected_mean_tensor)
            if (mean_diff > tolerance).any():
                logger.warning(f"Mean validation failed. Expected: {expected_mean}, "
                             f"Actual: {actual_mean.tolist()}, Diff: {mean_diff.tolist()}")
                return False
            
            # Check standard deviations
            std_diff = torch.abs(actual_std - expected_std_tensor)
            if (std_diff > tolerance).any():
                logger.warning(f"Std validation failed. Expected: {expected_std}, "
                             f"Actual: {actual_std.tolist()}, Diff: {std_diff.tolist()}")
                return False
            
            logger.debug("Normalization validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Normalization validation failed: {e}")
            return False


# Legacy compatibility function
def get_transforms(model_type: ModelType = ModelType.RESNET, train: bool = True) -> Compose:
    """
    Legacy compatibility function for existing code.
    
    Args:
        model_type: Type of model (enum)
        train: Whether to create training transforms
        
    Returns:
        Transform pipeline
    """
    manager = TransformManager()
    return manager.create_transforms(model_type, train)