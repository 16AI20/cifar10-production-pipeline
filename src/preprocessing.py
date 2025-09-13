"""
Legacy preprocessing utilities for CIFAR-10 data transformation.

This module provides backward compatibility for existing code while
redirecting to the enhanced transform system. It contains deprecated
functions that are maintained for compatibility but should not be
used in new code.

Note:
    This module is deprecated. Use src.transforms.TransformManager
    for new implementations with advanced augmentation strategies.
"""

import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize
from .enums import ModelType
from .transforms import TransformManager, get_transforms as new_get_transforms
import logging
import warnings

logger = logging.getLogger(__name__)

def get_transforms(model_type: ModelType = ModelType.RESNET, train: bool = True) -> Compose:
    """
    DEPRECATED: Use TransformManager.create_transforms() for new code.
    
    Legacy function for backward compatibility. Uses enhanced transform system.
    
    Args:
        model_type (ModelType): Type of model to be used (e.g., RESNET, VIT).
        train (bool): If True, returns training transforms with augmentation.
                      If False, returns test/validation transforms without augmentation.

    Returns:
        transforms.Compose: A composition of torchvision transforms.
    """
    
    warnings.warn(
        "get_transforms() is deprecated. Use TransformManager.create_transforms() for enhanced functionality.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Use new transform system for consistency
    return new_get_transforms(model_type, train)
