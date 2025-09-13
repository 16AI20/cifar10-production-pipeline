"""
Enumeration classes for the CIFAR-10 project.

This module defines enumeration types used throughout the project
to ensure type safety and consistent naming conventions.
"""

from enum import Enum


class ModelType(Enum):
    """
    Enumeration of supported deep learning model architectures.
    
    This enum defines the available model types that can be used
    for CIFAR-10 image classification. Each value corresponds to
    a specific architecture implementation.
    
    Values:
        RESNET: ResNet-50 architecture with residual connections
        VGG: VGG-16 architecture with deep convolutional layers  
        EFFICIENTNET: EfficientNet-B0 with compound scaling
        VIT: Vision Transformer with patch-based attention
        
    Example:
        >>> from src.enums import ModelType
        >>> model_type = ModelType.RESNET
        >>> print(model_type.value)  # 'resnet'
    """
    RESNET = 'resnet'
    VGG = 'vgg'
    EFFICIENTNET = 'efficientnet'
    VIT = 'vit'