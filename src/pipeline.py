import logging
import torch
from typing import Optional

from dotenv import load_dotenv

from torch import nn, optim
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

from .enums import ModelType
from .model import get_model
from .data_loader import load_data
from .train import train_model
from .evaluate import evaluate, final_evaluation
from .general_utils import setup_logging, set_global_seed
from .optuna_sweep import run_optuna

logger = logging.getLogger(__name__)


class CIFAR10Pipeline:
    """
    A pipeline for training and evaluating deep learning models on the CIFAR-10 dataset.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the pipeline with configuration settings.

        Args:
            cfg (DictConfig): Hydra-config object containing all experiment parameters.
        """

        load_dotenv()

        setup_logging(cfg.logging.config)
        self.cfg: DictConfig = cfg

        self.device: torch.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.model_type: ModelType = ModelType[cfg.model_cfg.model_type.upper()]
        self.num_classes: int = cfg.model_cfg.get("num_classes", 10)

        self.model: Optional[nn.Module] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None

    def setup_model(self) -> None:
        """
        Initializes the model, loss function, optimizer, and learning rate scheduler.
        """
        logger.info(f"Initializing model: {self.model_type}")
        self.model = get_model(
            model_type=self.model_type,
            num_classes=self.num_classes,
            pretrained=self.cfg.model_cfg.pretrained
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg.model_cfg.lr
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.cfg.scheduler.mode,
            patience=self.cfg.scheduler.patience,
            factor=self.cfg.scheduler.factor
        )

    def setup_data(self) -> None:
        """
        Loads CIFAR-10 training, validation, and test datasets.
        """
        logger.info("Loading CIFAR-10 data...")
        self.train_loader, self.val_loader, self.test_loader = load_data(
            model_type=self.model_type,
            batch_size=self.cfg.model_cfg.batch_size
        )

    def train(self) -> None:
        """
        Trains the model using the configured training and validation data.
        """
        logger.info("Starting training...")
        train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            scheduler=self.scheduler,
            cfg=self.cfg
        )
        logger.info("Training complete.")

    def evaluate(self) -> None:
        """
        Evaluates the trained model on the test dataset and logs the results.
        """
        logger.info("Evaluating on test set...")
        test_loss, test_acc, test_f1 = evaluate(
            model=self.model,
            val_loader=self.test_loader,
            criterion=self.criterion,
            device=self.device
        )
        logger.info(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, F1 Score: {test_f1:.4f}")

    def final_evaluation(self) -> None:
        """
        Runs a final comparison across all saved models (ResNet, EfficientNet, ViT)
        and logs the evaluation table.
        """
        logger.info("Evaluating on test set...")
        df_results = final_evaluation()
        logger.info("Final evaluation complete.\n" + df_results.to_markdown(index=False))


@hydra.main(config_path="../conf", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig) -> None:
    """
    Entrypoint for the CIFAR-10 pipeline using Hydra configuration.

    Args:
        cfg (DictConfig): Full Hydra configuration loaded from YAML and overrides.
    """
    logger.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")
    set_global_seed(cfg.seed)

    if cfg.get("mode") == "sweep":
        run_optuna(cfg)
        return

    pipeline = CIFAR10Pipeline(cfg)
    pipeline.setup_data()
    pipeline.setup_model()
    pipeline.train()
    pipeline.evaluate()
    pipeline.final_evaluation()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter