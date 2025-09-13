"""
CIFAR-10 dataset loading and preprocessing utilities.

This module provides functions for loading the CIFAR-10 dataset with
appropriate transformations based on the target model architecture.
It handles data splitting, preprocessing, and DataLoader creation.
"""

import logging
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from .enums import ModelType
from .preprocessing import get_transforms

logger = logging.getLogger(__name__)

# --- MAIN DATA LOADER FOR TRAINING AND VALIDATION ---
def load_data(
    model_type: ModelType = ModelType.RESNET,
    batch_size: int = 64,
    num_workers: int = 2,
    val_ratio: float = 0.1,
    data_dir: str = "data/cifar-10-batches-py"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset with model-specific preprocessing and data splitting.
    
    This function creates training, validation, and test DataLoaders for the CIFAR-10
    dataset with transformations appropriate for the specified model architecture.
    The training set is split into train/validation subsets based on val_ratio.
    
    Args:
        model_type: Target model architecture type, determines preprocessing strategy
        batch_size: Number of samples per batch for all DataLoaders
        num_workers: Number of subprocesses for data loading (auto-reduced in containers)
        val_ratio: Fraction of training data to reserve for validation (0.0 to 1.0)
        data_dir: Directory path for CIFAR-10 dataset storage and download
    
    Returns:
        Tuple containing:
            - train_loader: Training DataLoader with augmentation
            - val_loader: Validation DataLoader without augmentation  
            - test_loader: Test DataLoader without augmentation
    
    Raises:
        ValueError: If val_ratio is not in valid range [0.0, 1.0]
        RuntimeError: If CIFAR-10 dataset download fails
    
    Example:
        >>> from src.data_loader import load_data
        >>> from src.enums import ModelType
        >>> 
        >>> train_loader, val_loader, test_loader = load_data(
        ...     model_type=ModelType.VIT,
        ...     batch_size=32,
        ...     val_ratio=0.15
        ... )
        >>> 
        >>> # Check batch shapes
        >>> train_batch = next(iter(train_loader))
        >>> print(f"Training batch shape: {train_batch[0].shape}")  # (32, 3, 224, 224) for ViT
    
    Note:
        - Training data uses augmentation transforms (random crops, flips, etc.)
        - Validation and test data use only normalization and resizing
        - Input images are automatically resized to match model requirements
        - CIFAR-10 dataset is automatically downloaded if not present
    """

    # Reduce memory usage if running in container
    if is_container():
        num_workers = 0

    logger.info(f"Loading CIFAR-10 dataset with model type: {model_type.name}")
    logger.info(f"Parameters - batch_size: {batch_size}, num_workers: {num_workers}, val_ratio: {val_ratio}")

    # Get train and test transforms based on model architecture
    train_transform = get_transforms(model_type=model_type, train=True)
    test_transform = get_transforms(model_type=model_type, train=False)
    logger.debug("Transforms loaded.")

    # Load full training dataset with training transforms
    logger.info("Downloading and preparing training dataset...")
    full_train_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    logger.debug(f"Total training samples: {len(full_train_dataset)}")

    # Determine train/val split sizes
    total_len = len(full_train_dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len
    logger.debug(f"Splitting training set: {train_len} train / {val_len} val")

    # Perform split
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        lengths=[train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    logger.debug("Training and validation split complete.")

    # Override transform on val set (remove augmentation)
    val_dataset.dataset.transform = test_transform
    logger.debug("Validation transform set (no augmentation).")

    # Load test dataset with test transforms
    logger.info("Downloading and preparing test dataset...")
    test_dataset = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    # Wrap in DataLoaders
    logger.info("Wrapping datasets in DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    logger.info("DataLoaders ready.")
    return train_loader, val_loader, test_loader


# --- RAW DATA LOADER FOR EDA (No Transforms Applied) ---
def load_raw_cifar10(data_dir: str = "data/cifar-10-batches-py") -> Tuple[CIFAR10, CIFAR10]:
    """
    Load raw CIFAR-10 datasets without any transformations for analysis.
    
    This function loads the original CIFAR-10 training and test datasets
    without applying any preprocessing transformations. The raw data is
    useful for exploratory data analysis, visualizations, and understanding
    the original data distribution.
    
    Args:
        data_dir: Directory path for CIFAR-10 dataset storage and download
    
    Returns:
        Tuple containing:
            - cifar10_train: Raw training dataset (50,000 samples)
            - cifar10_test: Raw test dataset (10,000 samples)
    
    Raises:
        RuntimeError: If CIFAR-10 dataset download fails
        
    Example:
        >>> from src.data_loader import load_raw_cifar10
        >>> train_raw, test_raw = load_raw_cifar10()
        >>> 
        >>> # Access raw sample data
        >>> image, label = train_raw[0]  # PIL Image and integer label
        >>> print(f"Image size: {image.size}")  # (32, 32)
        >>> print(f"Label: {label}")  # 0-9 class index
        
    Note:
        - Images are returned as PIL Image objects in RGB format
        - Labels are integer class indices (0-9)
        - No normalization or resizing is applied
        - Dataset is automatically downloaded if not present
    """
    logger.info("Loading raw CIFAR-10 datasets for EDA...")

    cifar10_train = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None
    )
    logger.debug(f"Loaded raw training set: {len(cifar10_train)} samples.")

    cifar10_test = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=None
    )
    logger.debug(f"Loaded raw test set: {len(cifar10_test)} samples.")

    return cifar10_train, cifar10_test

def is_container() -> bool:
    """
    Detect if the code is running inside a containerized environment.
    
    This function checks for the presence of container-specific files
    to determine if the current process is running inside a Docker
    container or similar containerized environment. This is used to
    optimize resource usage (e.g., reducing num_workers for DataLoaders).
    
    Returns:
        True if running in a container environment, False otherwise
        
    Example:
        >>> from src.data_loader import is_container
        >>> if is_container():
        ...     print("Running in container - using optimized settings")
        ... else:
        ...     print("Running on host system")
        
    Note:
        - Checks for Docker-specific files (/.dockerenv)
        - Checks for Podman-specific files (/run/.containerenv)
        - Used to automatically adjust num_workers in DataLoaders
    """
    return os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
