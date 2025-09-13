import timm
from torch import nn
import logging
from .base_builder import BaseModelBuilder

logger = logging.getLogger(__name__)

class ViTBuilder(BaseModelBuilder):
    """
    Builder class for constructing a Vision Transformer (ViT) model using the timm library.

    This class initializes a ViT model (`vit_base_patch16_224`), optionally with ImageNet-pretrained
    weights, and modifies it for a specified number of output classes. If pretrained weights are used,
    all layers except the classification head are frozen.

    Attributes:
        num_classes (int): Number of output classes for classification.
        pretrained (bool): Whether to load pretrained ImageNet weights.
    """

    def build(self) -> nn.Module:
        """
        Constructs and returns the Vision Transformer model.

        Returns:
            nn.Module: The modified ViT model suitable for fine-tuning on custom datasets.
        """
        model: nn.Module = timm.create_model(
            'vit_base_patch16_224',
            pretrained=self.pretrained,
            num_classes=self.num_classes
        )
        logger.debug("Initialized ViT from timm.")

        # Freeze all layers except classification head if pretrained
        if self.pretrained:
            logger.debug("Freezing all ViT layers except head.")
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze head if it exists
            if hasattr(model, 'head'):
                for param in model.head.parameters():
                    param.requires_grad = True
                logger.debug("Unfroze ViT classifier head.")
            else:
                logger.warning("ViT model does not have a `.head` attribute.")

        logger.info(f"ViT model ready | Pretrained: {self.pretrained} | Classes: {self.num_classes}")
        return model
