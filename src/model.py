"""
Model factory and configuration management for CIFAR-10 classification.

This module provides high-level functions for creating and configuring
deep learning models with support for various architectures, pre-trained
weights, and advanced layer freezing strategies.
"""

from typing import Optional, Dict, Any
from torch import nn
from .enums import ModelType
from .networks import ResNetBuilder, EfficientNetBuilder, ViTBuilder
from .layer_freezing import FreezingConfig, create_freezing_config_from_dict
import logging

logger = logging.getLogger(__name__)

def get_model(
    model_type: ModelType, 
    num_classes: int = 10, 
    pretrained: bool = True,
    freezing_config: Optional[FreezingConfig] = None,
    **kwargs
) -> nn.Module:
    """
    Create a deep learning model with the specified architecture and configuration.
    
    This function serves as the main factory for creating models with support
    for different architectures, pre-trained weights, and advanced layer freezing
    strategies. It delegates to specialized builders for each model type.
    
    Args:
        model_type: Target model architecture (ResNet, VGG, EfficientNet, or ViT)
        num_classes: Number of output classes for classification head (default: 10 for CIFAR-10)
        pretrained: Whether to load ImageNet pre-trained weights (default: True)
        freezing_config: Optional layer freezing strategy configuration
        **kwargs: Additional model-specific parameters passed to builders
    
    Returns:
        Configured PyTorch nn.Module ready for training or inference
        
    Raises:
        ValueError: If model_type is not supported
        RuntimeError: If model creation fails
        
    Example:
        >>> from src.model import get_model
        >>> from src.enums import ModelType
        >>> 
        >>> # Create basic ResNet model
        >>> model = get_model(ModelType.RESNET, num_classes=10, pretrained=True)
        >>> 
        >>> # Create model with freezing configuration
        >>> from src.layer_freezing import FreezingStrategy, FreezingConfig
        >>> freezing_config = FreezingConfig(strategy=FreezingStrategy.FREEZE_BACKBONE)
        >>> model = get_model(ModelType.VIT, freezing_config=freezing_config)
        
    Note:
        - Uses builder pattern for flexible model construction
        - Supports progressive unfreezing during training
        - Pre-trained weights are automatically adapted for CIFAR-10 input size
        - Each architecture has specialized preprocessing requirements
    """
    logger.info(f"Loading model: {model_type.name} | Pretrained: {pretrained} | Classes: {num_classes}")
    
    if freezing_config:
        logger.info(f"Using freezing strategy: {freezing_config.strategy.value}")

    if model_type == ModelType.RESNET:
        return ResNetBuilder(
            num_classes=num_classes, 
            pretrained=pretrained, 
            freezing_config=freezing_config,
            **kwargs
        ).build()
    elif model_type == ModelType.EFFICIENTNET:
        return EfficientNetBuilder(
            num_classes=num_classes, 
            pretrained=pretrained, 
            freezing_config=freezing_config,
            **kwargs
        ).build()
    elif model_type == ModelType.VIT:
        return ViTBuilder(
            num_classes=num_classes, 
            pretrained=pretrained, 
            freezing_config=freezing_config,
            **kwargs
        ).build()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def create_model_with_freezing_from_config(
    model_type: ModelType,
    model_config: Dict[str, Any],
    freezing_config_dict: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Create a model from configuration dictionaries (typically loaded from YAML).
    
    This function provides a convenient way to create models using configuration
    dictionaries, making it easy to integrate with configuration management
    systems like Hydra or direct YAML file loading.
    
    Args:
        model_type: Target model architecture type
        model_config: Dictionary containing model parameters such as:
            - num_classes: Number of output classes (default: 10)
            - pretrained: Whether to use pre-trained weights (default: True)
            - Additional model-specific parameters
        freezing_config_dict: Optional dictionary with freezing configuration:
            - strategy: Freezing strategy name
            - freeze_layers: List of layer names to freeze
            - progressive_unfreezing: Progressive unfreezing settings
            
    Returns:
        Configured PyTorch model ready for training
        
    Raises:
        ValueError: If configuration parameters are invalid
        KeyError: If required configuration keys are missing
        
    Example:
        >>> from src.model import create_model_with_freezing_from_config
        >>> from src.enums import ModelType
        >>> 
        >>> model_config = {
        ...     'num_classes': 10,
        ...     'pretrained': True,
        ...     'dropout': 0.2
        ... }
        >>> 
        >>> freezing_config = {
        ...     'strategy': 'freeze_backbone',
        ...     'unfreeze_last_n': 2
        ... }
        >>> 
        >>> model = create_model_with_freezing_from_config(
        ...     model_type=ModelType.RESNET,
        ...     model_config=model_config,
        ...     freezing_config_dict=freezing_config
        ... )
        
    Note:
        - Automatically extracts standard parameters (num_classes, pretrained)
        - Passes additional parameters as keyword arguments to model builders
        - Supports all freezing strategies defined in FreezingConfig
    """
    # Parse freezing config if provided
    freezing_config = None
    if freezing_config_dict:
        freezing_config = create_freezing_config_from_dict(freezing_config_dict)
    
    # Extract model parameters
    num_classes = model_config.get('num_classes', 10)
    pretrained = model_config.get('pretrained', True)
    
    # Remove these from kwargs to avoid duplication
    kwargs = {k: v for k, v in model_config.items() 
              if k not in ['num_classes', 'pretrained']}
    
    return get_model(
        model_type=model_type,
        num_classes=num_classes,
        pretrained=pretrained,
        freezing_config=freezing_config,
        **kwargs
    )
