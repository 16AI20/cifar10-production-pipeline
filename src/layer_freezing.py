"""
Advanced Layer Freezing and Progressive Unfreezing for CIFAR-10 Project

This module provides sophisticated layer freezing strategies including:
- Configurable freezing patterns
- Progressive unfreezing during training
- Layer-wise learning rate scheduling
- Freezing strategy validation and monitoring
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
from torch.optim import Optimizer
import re

logger = logging.getLogger(__name__)


class FreezingStrategy(Enum):
    """Enumeration of available freezing strategies."""
    NONE = "none"                    # No freezing
    FULL_FREEZE = "full_freeze"      # Freeze all except classifier
    PARTIAL_FREEZE = "partial_freeze" # Freeze based on layer patterns
    PROGRESSIVE = "progressive"       # Progressive unfreezing during training
    LAYER_WISE_LR = "layer_wise_lr"  # Different learning rates per layer group


@dataclass
class FreezingConfig:
    """Configuration for layer freezing strategy."""
    strategy: FreezingStrategy
    freeze_patterns: List[str] = None  # Regex patterns for layers to freeze
    unfreeze_patterns: List[str] = None  # Regex patterns for layers to unfreeze
    progressive_schedule: List[Tuple[int, List[str]]] = None  # (epoch, layers_to_unfreeze)
    layer_lr_multipliers: Dict[str, float] = None  # Pattern -> LR multiplier
    freeze_bn: bool = True  # Whether to freeze batch norm layers
    monitor_gradients: bool = True  # Monitor gradient flow


class LayerFreezingManager:
    """
    Manages layer freezing strategies for deep learning models.
    
    Supports various freezing patterns and progressive unfreezing schedules
    to optimize transfer learning performance.
    """
    
    def __init__(self, model: nn.Module, config: FreezingConfig):
        """
        Initialize the freezing manager.
        
        Args:
            model: PyTorch model to manage
            config: Freezing configuration
        """
        self.model = model
        self.config = config
        self.original_requires_grad = {}  # Store original states
        self.layer_groups = {}  # Group layers by patterns
        self.gradient_stats = {}  # Track gradient statistics
        
        self._analyze_model_structure()
        self._initialize_freezing()
    
    def _analyze_model_structure(self):
        """Analyze model structure and group layers."""
        logger.info("Analyzing model structure for freezing management...")
        
        # Get all named modules
        named_modules = dict(self.model.named_modules())
        self.named_parameters = dict(self.model.named_parameters())
        
        # Store original requires_grad states
        for name, param in self.named_parameters.items():
            self.original_requires_grad[name] = param.requires_grad
        
        # Group layers by common patterns
        self._group_layers_by_patterns()
        
        logger.info(f"Model structure analyzed: {len(self.named_parameters)} parameters, "
                   f"{len(self.layer_groups)} layer groups identified")
    
    def _group_layers_by_patterns(self):
        """Group layers by common architectural patterns."""
        # Common patterns for different architectures
        patterns = {
            'backbone_early': [r'^(conv1|bn1)', r'^layer1', r'^features\.[0-4]'],
            'backbone_mid': [r'^layer2', r'^features\.[5-9]'],
            'backbone_late': [r'^layer3', r'^features\.1[0-4]'],
            'backbone_final': [r'^layer4', r'^features\.1[5-9]'],
            'classifier': [r'^(fc|classifier|head)', r'\.classifier', r'\.head'],
            'batch_norm': [r'\.bn\d*$', r'\.norm', r'batch_norm'],
            'attention': [r'\.attn', r'\.attention', r'\.self_attn'],
            'embeddings': [r'^(patch_embed|pos_embed|cls_token)'],
            'encoder': [r'^encoder', r'^blocks\.[0-5]'],
            'decoder': [r'^decoder', r'^blocks\.[6-9]', r'^blocks\.1[0-1]'],
        }
        
        # Initialize groups
        for group_name in patterns:
            self.layer_groups[group_name] = []
        self.layer_groups['other'] = []
        
        # Classify parameters
        for param_name in self.named_parameters:
            assigned = False
            
            for group_name, group_patterns in patterns.items():
                if any(re.search(pattern, param_name) for pattern in group_patterns):
                    self.layer_groups[group_name].append(param_name)
                    assigned = True
                    break
            
            if not assigned:
                self.layer_groups['other'].append(param_name)
        
        # Log group sizes
        for group_name, params in self.layer_groups.items():
            if params:
                logger.debug(f"Layer group '{group_name}': {len(params)} parameters")
    
    def _initialize_freezing(self):
        """Initialize freezing based on strategy."""
        logger.info(f"Initializing freezing strategy: {self.config.strategy.value}")
        
        if self.config.strategy == FreezingStrategy.NONE:
            self._unfreeze_all()
        
        elif self.config.strategy == FreezingStrategy.FULL_FREEZE:
            self._freeze_all_except_classifier()
        
        elif self.config.strategy == FreezingStrategy.PARTIAL_FREEZE:
            self._apply_partial_freezing()
        
        elif self.config.strategy == FreezingStrategy.PROGRESSIVE:
            self._initialize_progressive_freezing()
        
        elif self.config.strategy == FreezingStrategy.LAYER_WISE_LR:
            # Layer-wise LR is handled in optimizer creation
            pass
        
        self._log_freezing_status()
    
    def _freeze_all_except_classifier(self):
        """Freeze all parameters except classifier layers."""
        # Freeze everything first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier layers
        classifier_params = self.layer_groups.get('classifier', [])
        for param_name in classifier_params:
            if param_name in self.named_parameters:
                self.named_parameters[param_name].requires_grad = True
        
        logger.info(f"Frozen all layers except {len(classifier_params)} classifier parameters")
    
    def _apply_partial_freezing(self):
        """Apply partial freezing based on patterns."""
        freeze_patterns = self.config.freeze_patterns or []
        unfreeze_patterns = self.config.unfreeze_patterns or []
        
        # First, apply freeze patterns
        frozen_count = 0
        for param_name, param in self.named_parameters.items():
            for pattern in freeze_patterns:
                if re.search(pattern, param_name):
                    param.requires_grad = False
                    frozen_count += 1
                    break
        
        # Then, apply unfreeze patterns (override freeze patterns)
        unfrozen_count = 0
        for param_name, param in self.named_parameters.items():
            for pattern in unfreeze_patterns:
                if re.search(pattern, param_name):
                    param.requires_grad = True
                    unfrozen_count += 1
                    break
        
        logger.info(f"Partial freezing applied: {frozen_count} frozen, {unfrozen_count} explicitly unfrozen")
    
    def _initialize_progressive_freezing(self):
        """Initialize progressive freezing (start with everything frozen)."""
        # Start by freezing everything except classifier
        self._freeze_all_except_classifier()
        
        if self.config.progressive_schedule:
            logger.info(f"Progressive unfreezing scheduled for {len(self.config.progressive_schedule)} stages")
        else:
            logger.warning("Progressive freezing requested but no schedule provided")
    
    def _unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        
        logger.info("All parameters unfrozen")
    
    def update_freezing_for_epoch(self, epoch: int):
        """
        Update freezing state for progressive unfreezing.
        
        Args:
            epoch: Current training epoch
        """
        if self.config.strategy != FreezingStrategy.PROGRESSIVE:
            return
        
        if not self.config.progressive_schedule:
            return
        
        unfrozen_this_epoch = []
        
        for unfreeze_epoch, layer_patterns in self.config.progressive_schedule:
            if epoch == unfreeze_epoch:
                for pattern in layer_patterns:
                    unfrozen_count = self._unfreeze_by_pattern(pattern)
                    unfrozen_this_epoch.append((pattern, unfrozen_count))
        
        if unfrozen_this_epoch:
            total_unfrozen = sum(count for _, count in unfrozen_this_epoch)
            logger.info(f"Epoch {epoch}: Progressive unfreezing applied - {total_unfrozen} parameters unfrozen")
            for pattern, count in unfrozen_this_epoch:
                logger.debug(f"  Pattern '{pattern}': {count} parameters")
            
            # Log current freezing status
            self._log_freezing_status()
    
    def _unfreeze_by_pattern(self, pattern: str) -> int:
        """
        Unfreeze parameters matching a pattern.
        
        Args:
            pattern: Regex pattern to match parameter names
            
        Returns:
            Number of parameters unfrozen
        """
        unfrozen_count = 0
        
        for param_name, param in self.named_parameters.items():
            if re.search(pattern, param_name) and not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1
        
        return unfrozen_count
    
    def create_layer_wise_optimizer(self, base_lr: float, optimizer_class=torch.optim.AdamW, **kwargs) -> Optimizer:
        """
        Create optimizer with layer-wise learning rates.
        
        Args:
            base_lr: Base learning rate
            optimizer_class: Optimizer class to use
            **kwargs: Additional optimizer arguments
            
        Returns:
            Configured optimizer
        """
        if self.config.strategy != FreezingStrategy.LAYER_WISE_LR:
            # Create standard optimizer
            return optimizer_class(self.model.parameters(), lr=base_lr, **kwargs)
        
        # Group parameters by learning rate multipliers
        param_groups = []
        lr_multipliers = self.config.layer_lr_multipliers or {}
        
        # Default group (base learning rate)
        default_params = []
        grouped_params = set()
        
        # Create groups for each multiplier pattern
        for pattern, multiplier in lr_multipliers.items():
            group_params = []
            
            for param_name, param in self.named_parameters.items():
                if param.requires_grad and re.search(pattern, param_name):
                    group_params.append(param)
                    grouped_params.add(param_name)
            
            if group_params:
                param_groups.append({
                    'params': group_params,
                    'lr': base_lr * multiplier,
                    'name': pattern
                })
                
                logger.debug(f"Layer group '{pattern}': {len(group_params)} params, LR={base_lr * multiplier:.2e}")
        
        # Add remaining parameters to default group
        for param_name, param in self.named_parameters.items():
            if param.requires_grad and param_name not in grouped_params:
                default_params.append(param)
        
        if default_params:
            param_groups.append({
                'params': default_params,
                'lr': base_lr,
                'name': 'default'
            })
            
            logger.debug(f"Default group: {len(default_params)} params, LR={base_lr:.2e}")
        
        # Create optimizer
        optimizer = optimizer_class(param_groups, **kwargs)
        
        logger.info(f"Layer-wise optimizer created with {len(param_groups)} parameter groups")
        return optimizer
    
    def monitor_gradients(self, log_interval: int = 100) -> Dict[str, float]:
        """
        Monitor gradient flow through different layer groups.
        
        Args:
            log_interval: How often to log gradient statistics
            
        Returns:
            Dictionary of gradient statistics per layer group
        """
        if not self.config.monitor_gradients:
            return {}
        
        grad_stats = {}
        
        for group_name, param_names in self.layer_groups.items():
            if not param_names:
                continue
            
            group_grads = []
            active_params = 0
            
            for param_name in param_names:
                param = self.named_parameters[param_name]
                if param.requires_grad and param.grad is not None:
                    group_grads.append(param.grad.detach().cpu().abs().mean().item())
                    active_params += 1
            
            if group_grads:
                grad_stats[group_name] = {
                    'mean_grad': sum(group_grads) / len(group_grads),
                    'max_grad': max(group_grads),
                    'min_grad': min(group_grads),
                    'active_params': active_params,
                    'total_params': len(param_names)
                }
        
        self.gradient_stats = grad_stats
        
        # Log periodically
        if hasattr(self, '_grad_monitor_counter'):
            self._grad_monitor_counter += 1
        else:
            self._grad_monitor_counter = 1
        
        if self._grad_monitor_counter % log_interval == 0:
            self._log_gradient_stats()
        
        return grad_stats
    
    def _log_gradient_stats(self):
        """Log gradient statistics."""
        if not self.gradient_stats:
            return
        
        logger.debug("Gradient flow statistics:")
        for group_name, stats in self.gradient_stats.items():
            if stats['active_params'] > 0:
                logger.debug(f"  {group_name}: "
                           f"mean_grad={stats['mean_grad']:.2e}, "
                           f"active={stats['active_params']}/{stats['total_params']}")
    
    def _log_freezing_status(self):
        """Log current freezing status."""
        total_params = len(self.named_parameters)
        frozen_count = sum(1 for param in self.model.parameters() if not param.requires_grad)
        trainable_count = total_params - frozen_count
        
        logger.info(f"Freezing status: {trainable_count}/{total_params} parameters trainable "
                   f"({trainable_count/total_params*100:.1f}%)")
        
        # Log by groups
        for group_name, param_names in self.layer_groups.items():
            if param_names:
                group_trainable = sum(1 for name in param_names 
                                    if self.named_parameters[name].requires_grad)
                group_total = len(param_names)
                
                if group_total > 0:
                    logger.debug(f"  {group_name}: {group_trainable}/{group_total} trainable")
    
    def get_freezing_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive freezing summary.
        
        Returns:
            Dictionary with freezing statistics
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        summary = {
            'strategy': self.config.strategy.value,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'trainable_percentage': trainable_params / total_params * 100,
            'layer_groups': {}
        }
        
        # Group-wise statistics
        for group_name, param_names in self.layer_groups.items():
            if param_names:
                group_params = [self.named_parameters[name] for name in param_names]
                group_total_params = sum(p.numel() for p in group_params)
                group_trainable_params = sum(p.numel() for p in group_params if p.requires_grad)
                
                summary['layer_groups'][group_name] = {
                    'total_parameters': group_total_params,
                    'trainable_parameters': group_trainable_params,
                    'trainable_percentage': group_trainable_params / group_total_params * 100 if group_total_params > 0 else 0
                }
        
        return summary
    
    def reset_to_original(self):
        """Reset all parameters to their original requires_grad state."""
        for name, param in self.named_parameters.items():
            original_state = self.original_requires_grad.get(name, True)
            param.requires_grad = original_state
        
        logger.info("All parameters reset to original requires_grad state")


def create_freezing_config_from_dict(config_dict: Dict[str, Any]) -> FreezingConfig:
    """
    Create FreezingConfig from dictionary (e.g., from YAML).
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        FreezingConfig instance
    """
    strategy_str = config_dict.get('strategy', 'none')
    try:
        strategy = FreezingStrategy(strategy_str)
    except ValueError:
        logger.warning(f"Unknown freezing strategy '{strategy_str}', using 'none'")
        strategy = FreezingStrategy.NONE
    
    # Parse progressive schedule
    progressive_schedule = None
    if 'progressive_schedule' in config_dict:
        schedule_data = config_dict['progressive_schedule']
        progressive_schedule = []
        for item in schedule_data:
            if isinstance(item, dict) and 'epoch' in item and 'patterns' in item:
                progressive_schedule.append((item['epoch'], item['patterns']))
    
    return FreezingConfig(
        strategy=strategy,
        freeze_patterns=config_dict.get('freeze_patterns'),
        unfreeze_patterns=config_dict.get('unfreeze_patterns'),
        progressive_schedule=progressive_schedule,
        layer_lr_multipliers=config_dict.get('layer_lr_multipliers'),
        freeze_bn=config_dict.get('freeze_bn', True),
        monitor_gradients=config_dict.get('monitor_gradients', True)
    )


# Predefined freezing configurations for common scenarios
PRESET_CONFIGS = {
    'no_freezing': FreezingConfig(strategy=FreezingStrategy.NONE),
    
    'full_freeze': FreezingConfig(strategy=FreezingStrategy.FULL_FREEZE),
    
    'freeze_early_layers': FreezingConfig(
        strategy=FreezingStrategy.PARTIAL_FREEZE,
        freeze_patterns=[r'^(conv1|bn1|layer1)', r'^features\.[0-4]']
    ),
    
    'progressive_resnet': FreezingConfig(
        strategy=FreezingStrategy.PROGRESSIVE,
        progressive_schedule=[
            (5, [r'^layer4']),
            (10, [r'^layer3']),
            (15, [r'^layer2'])
        ]
    ),
    
    'layer_wise_lr': FreezingConfig(
        strategy=FreezingStrategy.LAYER_WISE_LR,
        layer_lr_multipliers={
            r'^(conv1|bn1|layer1)': 0.1,
            r'^layer2': 0.3,
            r'^layer3': 0.5,
            r'^layer4': 0.8,
            r'^(fc|classifier)': 1.0
        }
    )
}