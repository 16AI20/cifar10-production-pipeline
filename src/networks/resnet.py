from torchvision import models
from torchvision.models import ResNet50_Weights
from torch import nn
import logging
from .base_builder import BaseModelBuilder

logger = logging.getLogger(__name__)

class ResNetBuilder(BaseModelBuilder):
    """
    Builder class for constructing a ResNet-50 model customized for image classification.

    This class loads a torchvision ResNet-50 architecture and optionally:
    - Uses pretrained ImageNet weights.
    - Replaces the final fully connected (fc) layer with a new classifier matching
      the number of output classes.
    - Freezes all layers except the new classifier head if `pretrained=True`.

    Attributes:
        num_classes (int): Number of output classes for classification.
        pretrained (bool): Whether to load ImageNet pretrained weights.
    """

    def build(self) -> nn.Module:
        """
        Builds and returns a ResNet-50 model with a custom classification head.

        Returns:
            nn.Module: The modified ResNet-50 model.
        """
        # Load pretrained weights if requested
        weights = ResNet50_Weights.DEFAULT if self.pretrained else None
        model: nn.Module = models.resnet50(weights=weights)

        # Replace final classification layer
        in_features: int = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)
        logger.debug("Replaced ResNet50 final layer.")

        # Apply advanced freezing strategy if configured
        if self.freezing_config is not None:
            freezing_manager = self.apply_freezing_strategy(model)
            if freezing_manager:
                logger.info("Advanced freezing strategy applied to ResNet50")
                # Store freezing manager in model for later use
                model._freezing_manager = freezing_manager
        elif self.pretrained:
            # Fallback to legacy freezing behavior
            logger.debug("Using legacy freezing: all layers except classifier.")
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

        logger.info(f"ResNet50 model ready | Pretrained: {self.pretrained} | Classes: {self.num_classes}")
        return model
