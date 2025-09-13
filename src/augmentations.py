"""
Advanced Augmentation Strategies for CIFAR-10 Project

This module implements state-of-the-art augmentation techniques including
RandAugment, CutMix, MixUp, and other advanced strategies.
"""

import logging
from typing import Tuple, List, Optional, Union, Callable, Any
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import numpy as np

logger = logging.getLogger(__name__)


class RandAugment(nn.Module):
    """
    RandAugment implementation based on the paper:
    "RandAugment: Practical automated data augmentation with a reduced search space"
    
    Applies N random augmentation operations with magnitude M.
    """
    
    def __init__(
        self, 
        num_ops: int = 2, 
        magnitude: int = 10,
        magnitude_std: float = 0.5,
        interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
        fill: Optional[Union[int, float, List[float]]] = None
    ):
        """
        Initialize RandAugment.
        
        Args:
            num_ops: Number of augmentation operations to apply
            magnitude: Base magnitude of augmentations (0-30)
            magnitude_std: Standard deviation for magnitude sampling
            interpolation: Interpolation mode for geometric transforms
            fill: Fill value for geometric transforms
        """
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.magnitude_std = magnitude_std
        self.interpolation = interpolation
        self.fill = fill
        
        # Define augmentation operations
        self.ops = [
            self._identity,
            self._auto_contrast,
            self._equalize,
            self._rotate,
            self._solarize,
            self._color,
            self._posterize,
            self._contrast,
            self._brightness,
            self._sharpness,
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y,
        ]
        
        logger.debug(f"RandAugment initialized: num_ops={num_ops}, magnitude={magnitude}")
    
    def forward(self, img: Tensor) -> Tensor:
        """Apply random augmentations to image."""
        for _ in range(self.num_ops):
            # Sample magnitude with noise
            mag = max(0, min(30, self.magnitude + random.gauss(0, self.magnitude_std)))
            
            # Select random operation
            op = random.choice(self.ops)
            img = op(img, mag)
        
        return img
    
    def _get_magnitude(self, magnitude: float, max_val: float) -> float:
        """Convert magnitude to actual parameter value."""
        return magnitude / 30.0 * max_val
    
    # Augmentation operations
    def _identity(self, img: Tensor, magnitude: float) -> Tensor:
        return img
    
    def _auto_contrast(self, img: Tensor, magnitude: float) -> Tensor:
        return TF.autocontrast(img)
    
    def _equalize(self, img: Tensor, magnitude: float) -> Tensor:
        return TF.equalize(img)
    
    def _rotate(self, img: Tensor, magnitude: float) -> Tensor:
        degrees = self._get_magnitude(magnitude, 30.0)
        if random.random() < 0.5:
            degrees = -degrees
        return TF.rotate(img, degrees, interpolation=self.interpolation, fill=self.fill)
    
    def _solarize(self, img: Tensor, magnitude: float) -> Tensor:
        threshold = 256 - self._get_magnitude(magnitude, 256)
        return TF.solarize(img, threshold)
    
    def _color(self, img: Tensor, magnitude: float) -> Tensor:
        factor = 1.0 + self._get_magnitude(magnitude, 0.9) * random.choice([-1, 1])
        return TF.adjust_saturation(img, factor)
    
    def _posterize(self, img: Tensor, magnitude: float) -> Tensor:
        bits = int(8 - self._get_magnitude(magnitude, 4))
        return TF.posterize(img, bits)
    
    def _contrast(self, img: Tensor, magnitude: float) -> Tensor:
        factor = 1.0 + self._get_magnitude(magnitude, 0.9) * random.choice([-1, 1])
        return TF.adjust_contrast(img, factor)
    
    def _brightness(self, img: Tensor, magnitude: float) -> Tensor:
        factor = 1.0 + self._get_magnitude(magnitude, 0.9) * random.choice([-1, 1])
        return TF.adjust_brightness(img, factor)
    
    def _sharpness(self, img: Tensor, magnitude: float) -> Tensor:
        factor = 1.0 + self._get_magnitude(magnitude, 0.9) * random.choice([-1, 1])
        return TF.adjust_sharpness(img, factor)
    
    def _shear_x(self, img: Tensor, magnitude: float) -> Tensor:
        shear = self._get_magnitude(magnitude, 0.3) * random.choice([-1, 1])
        return TF.affine(img, angle=0, translate=[0, 0], scale=1, shear=[math.degrees(shear), 0],
                        interpolation=self.interpolation, fill=self.fill)
    
    def _shear_y(self, img: Tensor, magnitude: float) -> Tensor:
        shear = self._get_magnitude(magnitude, 0.3) * random.choice([-1, 1])
        return TF.affine(img, angle=0, translate=[0, 0], scale=1, shear=[0, math.degrees(shear)],
                        interpolation=self.interpolation, fill=self.fill)
    
    def _translate_x(self, img: Tensor, magnitude: float) -> Tensor:
        translate = self._get_magnitude(magnitude, 150) * random.choice([-1, 1])
        return TF.affine(img, angle=0, translate=[translate, 0], scale=1, shear=[0, 0],
                        interpolation=self.interpolation, fill=self.fill)
    
    def _translate_y(self, img: Tensor, magnitude: float) -> Tensor:
        translate = self._get_magnitude(magnitude, 150) * random.choice([-1, 1])
        return TF.affine(img, angle=0, translate=[0, translate], scale=1, shear=[0, 0],
                        interpolation=self.interpolation, fill=self.fill)


class CutMix:
    """
    CutMix augmentation implementation based on the paper:
    "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    
    Note: This is applied to batches, not individual images.
    """
    
    def __init__(self, alpha: float = 1.0, cutmix_prob: float = 0.5):
        """
        Initialize CutMix.
        
        Args:
            alpha: Beta distribution parameter for sampling mixing ratio
            cutmix_prob: Probability of applying CutMix to a batch
        """
        self.alpha = alpha
        self.cutmix_prob = cutmix_prob
        logger.debug(f"CutMix initialized: alpha={alpha}, prob={cutmix_prob}")
    
    def __call__(
        self, 
        batch: Tensor, 
        target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, float]:
        """
        Apply CutMix to a batch.
        
        Args:
            batch: Input batch (N, C, H, W)
            target: Target labels (N,)
            
        Returns:
            Tuple of (mixed_batch, target_a, target_b, lambda)
        """
        if random.random() > self.cutmix_prob:
            return batch, target, target, 1.0
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        
        target_a = target
        target_b = target[index]
        
        # Generate random bounding box
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(batch.size(), lam)
        
        # Mix images
        batch[:, :, bbx1:bbx2, bby1:bby2] = batch[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch.size()[-1] * batch.size()[-2]))
        
        return batch, target_a, target_b, lam
    
    def _rand_bbox(self, size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
        """Generate random bounding box."""
        W = size[2]
        H = size[3]
        
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


class MixUp:
    """
    MixUp augmentation implementation based on the paper:
    "mixup: Beyond Empirical Risk Minimization"
    
    Note: This is applied to batches, not individual images.
    """
    
    def __init__(self, alpha: float = 0.2, mixup_prob: float = 0.5):
        """
        Initialize MixUp.
        
        Args:
            alpha: Beta distribution parameter for sampling mixing ratio
            mixup_prob: Probability of applying MixUp to a batch
        """
        self.alpha = alpha
        self.mixup_prob = mixup_prob
        logger.debug(f"MixUp initialized: alpha={alpha}, prob={mixup_prob}")
    
    def __call__(
        self, 
        batch: Tensor, 
        target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, float]:
        """
        Apply MixUp to a batch.
        
        Args:
            batch: Input batch (N, C, H, W)
            target: Target labels (N,)
            
        Returns:
            Tuple of (mixed_batch, target_a, target_b, lambda)
        """
        if random.random() > self.mixup_prob:
            return batch, target, target, 1.0
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        
        # Mix images
        mixed_batch = lam * batch + (1 - lam) * batch[index, :]
        
        target_a = target
        target_b = target[index]
        
        return mixed_batch, target_a, target_b, lam


def mixup_criterion(
    criterion: Callable, 
    pred: Tensor, 
    target_a: Tensor, 
    target_b: Tensor, 
    lam: float
) -> Tensor:
    """
    Compute loss for mixed samples.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        target_a: First set of targets
        target_b: Second set of targets
        lam: Mixing ratio
        
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, target_a) + (1 - lam) * criterion(pred, target_b)


class TrivialAugmentWide(nn.Module):
    """
    TrivialAugmentWide implementation - a simpler alternative to RandAugment
    that applies only one augmentation per image.
    """
    
    def __init__(
        self,
        interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
        fill: Optional[Union[int, float, List[float]]] = None
    ):
        super().__init__()
        self.interpolation = interpolation
        self.fill = fill
        
        # Define ops with their magnitude ranges
        self.ops = [
            ("Identity", 0, 1, False),
            ("ShearX", 0.0, 0.99, True),
            ("ShearY", 0.0, 0.99, True),
            ("TranslateX", 0.0, 32.0, True),
            ("TranslateY", 0.0, 32.0, True),
            ("Rotate", 0.0, 135.0, True),
            ("Brightness", 0.01, 0.99, False),
            ("Color", 0.01, 0.99, False),
            ("Contrast", 0.01, 0.99, False),
            ("Sharpness", 0.01, 0.99, False),
            ("Posterize", 2, 8, False),
            ("Solarize", 0, 256, False),
            ("AutoContrast", 0, 1, False),
            ("Equalize", 0, 1, False),
        ]
    
    def forward(self, img: Tensor) -> Tensor:
        """Apply one random augmentation."""
        op_name, min_val, max_val, signed = random.choice(self.ops)
        val = random.uniform(min_val, max_val)
        
        if signed and random.random() < 0.5:
            val = -val
        
        return self._apply_op(img, op_name, val)
    
    def _apply_op(self, img: Tensor, op_name: str, magnitude: float) -> Tensor:
        """Apply specific augmentation operation."""
        if op_name == "Identity":
            return img
        elif op_name == "ShearX":
            return TF.affine(img, angle=0, translate=[0, 0], scale=1, 
                           shear=[math.degrees(magnitude), 0],
                           interpolation=self.interpolation, fill=self.fill)
        elif op_name == "ShearY":
            return TF.affine(img, angle=0, translate=[0, 0], scale=1, 
                           shear=[0, math.degrees(magnitude)],
                           interpolation=self.interpolation, fill=self.fill)
        elif op_name == "TranslateX":
            return TF.affine(img, angle=0, translate=[magnitude, 0], scale=1, shear=[0, 0],
                           interpolation=self.interpolation, fill=self.fill)
        elif op_name == "TranslateY":
            return TF.affine(img, angle=0, translate=[0, magnitude], scale=1, shear=[0, 0],
                           interpolation=self.interpolation, fill=self.fill)
        elif op_name == "Rotate":
            return TF.rotate(img, magnitude, interpolation=self.interpolation, fill=self.fill)
        elif op_name == "Brightness":
            return TF.adjust_brightness(img, 1 + magnitude)
        elif op_name == "Color":
            return TF.adjust_saturation(img, 1 + magnitude)
        elif op_name == "Contrast":
            return TF.adjust_contrast(img, 1 + magnitude)
        elif op_name == "Sharpness":
            return TF.adjust_sharpness(img, 1 + magnitude)
        elif op_name == "Posterize":
            return TF.posterize(img, int(magnitude))
        elif op_name == "Solarize":
            return TF.solarize(img, int(magnitude))
        elif op_name == "AutoContrast":
            return TF.autocontrast(img)
        elif op_name == "Equalize":
            return TF.equalize(img)
        else:
            return img


class AugmentationMixer:
    """
    Utility class to mix different augmentation strategies and 
    integrate them with the transform pipeline.
    """
    
    def __init__(self, config: dict):
        """Initialize with augmentation configuration."""
        self.config = config
        self.mixup = None
        self.cutmix = None
        self.randaugment = None
        self.trivialaugment = None
        
        self._init_augmentations()
    
    def _init_augmentations(self):
        """Initialize augmentation strategies based on config."""
        
        # MixUp
        if self.config.get("mixup", {}).get("enabled", False):
            mixup_config = self.config["mixup"]
            self.mixup = MixUp(
                alpha=mixup_config.get("alpha", 0.2),
                mixup_prob=mixup_config.get("mixup_prob", 0.5)
            )
            logger.info("MixUp initialized")
        
        # CutMix
        if self.config.get("cutmix", {}).get("enabled", False):
            cutmix_config = self.config["cutmix"]
            self.cutmix = CutMix(
                alpha=cutmix_config.get("alpha", 1.0),
                cutmix_prob=cutmix_config.get("cutmix_prob", 0.5)
            )
            logger.info("CutMix initialized")
        
        # RandAugment
        if self.config.get("randaugment", {}).get("enabled", False):
            ra_config = self.config["randaugment"]
            self.randaugment = RandAugment(
                num_ops=ra_config.get("num_ops", 2),
                magnitude=ra_config.get("magnitude", 10),
                magnitude_std=ra_config.get("magnitude_std", 0.5)
            )
            logger.info("RandAugment initialized")
    
    def apply_batch_augmentation(
        self, 
        batch: Tensor, 
        target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, float]:
        """
        Apply batch-level augmentations (MixUp/CutMix).
        
        Returns:
            Tuple of (augmented_batch, target_a, target_b, lambda)
        """
        # Randomly choose between MixUp and CutMix if both are enabled
        available_augs = []
        if self.mixup:
            available_augs.append(self.mixup)
        if self.cutmix:
            available_augs.append(self.cutmix)
        
        if available_augs:
            aug_method = random.choice(available_augs)
            return aug_method(batch, target)
        else:
            return batch, target, target, 1.0
    
    def get_image_transform(self) -> Optional[nn.Module]:
        """Get image-level augmentation transform."""
        if self.randaugment:
            return self.randaugment
        return None


def create_advanced_transforms(config: dict) -> List[nn.Module]:
    """
    Create list of advanced transform modules based on configuration.
    
    Args:
        config: Augmentation configuration dictionary
        
    Returns:
        List of transform modules
    """
    transforms = []
    
    # RandAugment
    if config.get("randaugment", {}).get("enabled", False):
        ra_config = config["randaugment"]
        transforms.append(RandAugment(
            num_ops=ra_config.get("num_ops", 2),
            magnitude=ra_config.get("magnitude", 10),
            magnitude_std=ra_config.get("magnitude_std", 0.5)
        ))
    
    # TrivialAugmentWide
    if config.get("trivialaugment", {}).get("enabled", False):
        transforms.append(TrivialAugmentWide())
    
    return transforms