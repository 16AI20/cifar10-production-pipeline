# src/networks/base_builder.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from torch import nn
from ..layer_freezing import FreezingConfig, LayerFreezingManager, create_freezing_config_from_dict


class BaseModelBuilder(ABC):
    """
    Abstract base class for all model builders.
    Enforces implementation of the `build()` method and supports advanced freezing strategies.
    """

    def __init__(
        self, 
        num_classes: int, 
        pretrained: bool = True,
        freezing_config: Optional[FreezingConfig] = None,
        **kwargs
    ):
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freezing_config = freezing_config
        self.extra_config = kwargs

    @abstractmethod
    def build(self) -> nn.Module:
        """Return a constructed model."""
        
    def apply_freezing_strategy(self, model: nn.Module) -> Optional[LayerFreezingManager]:
        """
        Apply freezing strategy to the model.
        
        Args:
            model: Model to apply freezing to
            
        Returns:
            LayerFreezingManager instance if freezing is applied, None otherwise
        """
        if self.freezing_config is None:
            return None
        
        freezing_manager = LayerFreezingManager(model, self.freezing_config)
        return freezing_manager
