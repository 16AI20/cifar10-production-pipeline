from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from torch import nn
import logging
from .base_builder import BaseModelBuilder

logger = logging.getLogger(__name__)

class EfficientNetBuilder(BaseModelBuilder):
    """
    Builder class for creating an EfficientNet-B0 model for image classification.

    This class customizes the EfficientNet architecture from torchvision by:
    - Loading pretrained weights (optional).
    - Replacing the final classifier layer to match the specified number of output classes.
    - Optionally freezing all layers except the final classifier for transfer learning.

    Attributes:
        num_classes (int): Number of output classes for the classification task.
        pretrained (bool): Whether to use pretrained ImageNet weights.
    """

    def build(self) -> nn.Module:
        """
        Builds and returns the EfficientNet-B0 model with a modified classification head.

        Returns:
            nn.Module: A PyTorch EfficientNet model ready for training or inference.
        """
        # Load pretrained weights if specified
        weights = EfficientNet_B0_Weights.DEFAULT if self.pretrained else None
        model = models.efficientnet_b0(weights=weights)

        # Replace the final classifier layer to match num_classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, self.num_classes)
        logger.debug("Replaced EfficientNet final classifier layer.")

        # Freeze all layers except the classifier if using pretrained weights
        if self.pretrained:
            logger.debug("Freezing all EfficientNet layers except classifier.")
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier[1].parameters():
                param.requires_grad = True

        logger.info(f"EfficientNet-B0 model ready | Pretrained: {self.pretrained} | Classes: {self.num_classes}")
        return model
