"""
Comprehensive training orchestration for CIFAR-10 deep learning models.

This module provides a robust training framework with the following features:
- MLflow experiment tracking and model registration
- Checkpoint saving and resuming functionality
- Early stopping with configurable patience
- Comprehensive metric tracking (loss, accuracy, F1-score)
- Input validation and sanity checks
- Progress tracking with visual feedback
- Automatic artifact logging and visualization
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import mlflow
from typing import Optional, Any, Dict, Tuple, List
from .general_utils import (
    mlflow_init,
    mlflow_log,
    mlflow_pytorch_call,
)
import os
from .evaluate import evaluate, plot_training_curves
import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Set up MLflow tracking constants

MLFLOW_LOG_PARAM = "log_param"
MLFLOW_LOG_PARAMS = "log_params"
MLFLOW_LOG_METRIC = "log_metric"
MLFLOW_LOG_METRICS = "log_metrics"
MLFLOW_LOG_ARTIFACT = "log_artifact"
MLFLOW_LOG_ARTIFACTS = "log_artifacts"
MLFLOW_LOG_DICT = "log_dict"
MLFLOW_LOG_FIGURE = "log_figure"
MLFLOW_LOG_TABLE = "log_table"
MLFLOW_LOG_IMAGE = "log_image"
MLFLOW_LOG_INPUT = "log_input"
MLFLOW_LOG_INPUTS = "log_inputs"
MLFLOW_LOG_MODEL = "log_model"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    cfg: DictConfig,
    scheduler: Optional[Any] = None
) -> None:
    """
    Orchestrates the model training, validation, checkpointing, logging, and early stopping.
    """
    _validate_inputs(model, train_loader, val_loader)
    _log_training_start(device)

    # Config extraction
    seed, exp_name, resume, run_name, checkpoint_path, artifact_dir, epochs, patience = _extract_cfg(cfg)

    # History initialization
    train_hist, val_hist = _init_metric_histories()
    best_val_loss, epochs_no_improve, start_epoch = float("inf"), 0, 0

    # MLflow setup
    mlflow_init_status, mlflow_run, step_offset = _init_mlflow(exp_name, run_name, cfg, resume)
    _resume_checkpoint_if_needed(
        resume, checkpoint_path, device, model, optimizer, train_hist, val_hist
    )

    _log_static_hyperparams(mlflow_init_status, train_loader, val_loader, optimizer, seed, epochs, cfg)

    # Main training loop
    for epoch in range(start_epoch, epochs):
        train_metrics = _train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
        val_metrics = _validate(model, val_loader, criterion, device)

        _update_histories(train_hist, val_hist, train_metrics, val_metrics)

        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        improved = val_metrics["loss"] < best_val_loss
        if improved:
            _save_best_model_and_log_metrics(
                model, optimizer, checkpoint_path, epoch,
                train_hist, val_hist, train_metrics, val_metrics,
                mlflow_init_status
            )
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                logger.warning("Early stopping triggered.")
                break

    _finalize_training(
        train_hist, val_hist, mlflow_init_status, artifact_dir, model, cfg.model_cfg.registered_model_name
    )

def _validate_inputs(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
    """
    Validates the model and dataloaders to ensure compatibility with training.

    This function performs the following sanity checks:
    - Ensures the model is an instance of `torch.nn.Module`.
    - Ensures both train and validation loaders are iterable.
    - Checks that the loaders are not empty.
    - Verifies each batch is a (images, labels) tuple or list.
    - Checks that both `images` and `labels` are `torch.Tensor` instances.
    - Ensures `images` have shape (B, C, H, W) and `labels` have shape (B,).
    - Verifies that neither `images` nor `labels` contain NaNs.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_loader (DataLoader): Dataloader for training data.
        val_loader (DataLoader): Dataloader for validation data.

    Raises:
        TypeError: If model is not an `nn.Module` or data is not in tensor form.
        ValueError: If dataloader is empty, returns malformed batches, or has invalid tensor shapes.
        RuntimeError: If a sample cannot be fetched from a loader due to an unexpected error.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("Provided model is not a valid PyTorch nn.Module.")

    for loader_name, loader in zip(["train_loader", "val_loader"], [train_loader, val_loader]):
        if not hasattr(loader, "__iter__"):
            raise TypeError(f"{loader_name} must be iterable.")

        try:
            sample = next(iter(loader))
        except StopIteration:
            raise ValueError(f"{loader_name} is empty.")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch sample from {loader_name}: {e}")

        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise ValueError(f"{loader_name} should return (images, labels).")

        images, labels = sample

        if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError(f"{loader_name} must return tensors.")

        if images.dim() != 4:
            raise ValueError(f"Images in {loader_name} should be 4D (B, C, H, W), got {images.shape}.")

        if labels.dim() != 1:
            raise ValueError(f"Labels in {loader_name} should be 1D (B,), got {labels.shape}.")

        if torch.isnan(images).any() or torch.isnan(labels).any():
            raise ValueError(f"{loader_name} contains NaNs.")

    logger.info("Sanity checks on model and data passed.")

def _extract_cfg(cfg: DictConfig) -> Tuple[int, str, bool, str, str, str, int, int]:
    """
    Extracts key configuration values from a Hydra DictConfig object.

    This function pulls out the following settings:
    - `seed`: Random seed for reproducibility.
    - `exp_name`: Name of the experiment for logging/tracking.
    - `resume`: Flag indicating whether to resume from a checkpoint.
    - `run_name`: Name of the model for MLflow and checkpointing.
    - `checkpoint_path`: Full path to the checkpoint file.
    - `artifact_dir`: Directory to store logs and other artifacts.
    - `epochs`: Number of training epochs.
    - `patience`: Number of epochs with no improvement before early stopping.

    Args:
        cfg (DictConfig): The experiment configuration object, typically loaded via Hydra.

    Returns:
        Tuple[
            int,   # seed
            str,   # exp_name
            bool,  # resume
            str,   # run_name
            str,   # checkpoint_path
            str,   # artifact_dir
            int,   # epochs
            int    # patience
        ]
    """
    seed: int = cfg.seed
    exp_name: str = cfg.exp_name
    resume: bool = cfg.resume
    run_name: str = cfg.model_cfg.registered_model_name
    checkpoint_path: str = os.path.join(cfg.checkpoint.dir, run_name + ".pth")
    artifact_dir: str = cfg.logging.artifact_dir
    epochs: int = cfg.model_cfg.epochs
    patience: int = cfg.model_cfg.patience

    return seed, exp_name, resume, run_name, checkpoint_path, artifact_dir, epochs, patience

def _init_metric_histories() -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Initializes empty metric history containers for training and validation.

    This function creates and returns two dictionaries—one for training metrics and one for validation metrics.
    Each dictionary contains lists to accumulate values for:
    - "loss": Loss values over epochs.
    - "acc": Accuracy values over epochs.
    - "f1": F1-score values over epochs.

    Returns:
        Tuple[
            Dict[str, List[float]],  # Training metrics: loss, acc, f1
            Dict[str, List[float]]   # Validation metrics: loss, acc, f1
        ]
    """
    train_metrics: Dict[str, List[float]] = {"loss": [], "acc": [], "f1": []}
    val_metrics: Dict[str, List[float]] = {"loss": [], "acc": [], "f1": []}
    return train_metrics, val_metrics

def _log_training_start(device: torch.device) -> None:
    logger.info(f"Starting training on device: {device}")

def _init_mlflow(exp_name: str, run_name: str, cfg: DictConfig, resume: bool) -> Tuple[bool, Any, int]:
    """
    Initializes MLflow tracking for the experiment.

    This function configures MLflow by setting the tracking URI, experiment name, and run name.
    It also forwards setup options such as whether to enable autologging and whether to resume
    a previous run.

    Args:
        exp_name (str): The name of the MLflow experiment.
        run_name (str): The name of the MLflow run.
        cfg (DictConfig): The configuration object containing MLflow settings.
                          Expected keys under `cfg.mlflow`:
                          - `setup` (bool): Whether to explicitly configure MLflow experiment/run.
                          - `autolog` (bool): Whether to enable automatic logging.
        resume (bool): Whether to resume from an existing MLflow run (if any).

    Returns:
        Tuple[bool, Any, int]:
            - bool: Indicates if MLflow was successfully initialized.
            - Any: The MLflow run object (could be `mlflow.active_run()` or a custom wrapper).
            - int: Step offset in case of resuming a previous run.
    """
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI")

    return mlflow_init(
        tracking_uri=tracking_uri,
        exp_name=exp_name,
        run_name=run_name,
        setup_mlflow=cfg.mlflow.setup,
        autolog=cfg.mlflow.autolog,
        resume=resume,
    )

def _resume_checkpoint_if_needed(
    resume: bool,
    checkpoint_path: str,
    device: torch.device,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_hist: Dict[str, List[float]],
    val_hist: Dict[str, List[float]]
) -> None:
    """
    Loads model and optimizer state from a checkpoint file if resume is enabled and the file exists.
    Also restores metric histories for training and validation.

    Args:
        resume (bool): Whether to resume training from a saved checkpoint.
        checkpoint_path (str): Full path to the checkpoint file (typically .pth).
        device (torch.device): The device on which to load the checkpoint (e.g., "cpu", "cuda", "mps").
        model (nn.Module): The model instance to load weights into.
        optimizer (optim.Optimizer): The optimizer to restore state for.
        train_hist (Dict[str, List[float]]): Dictionary holding training metric histories. This will be updated in-place.
        val_hist (Dict[str, List[float]]): Dictionary holding validation metric histories. This will be updated in-place.

    Raises:
        RuntimeError: If the checkpoint file exists but is corrupted or incompatible with the model/optimizer.
    """
    if resume and os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimiser_state_dict"])

        for key in train_hist:
            train_hist[key] = checkpoint.get(f"train_{key}_history", [])
            val_hist[key] = checkpoint.get(f"val_{key}_history", [])
    else:
        logger.warning("No checkpoint found or resume disabled. Starting from scratch.")

def _log_static_hyperparams(
    status: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    seed: int,
    epochs: int,
    cfg: DictConfig
) -> None:
    """
    Logs static training hyperparameters to MLflow.

    This function constructs a dictionary of key training hyperparameters and logs them
    to MLflow for experiment tracking. It includes learning rate, batch sizes,
    random seed, number of epochs, and model type.

    Args:
        status (bool): Whether MLflow logging is enabled. If False, logging is skipped.
        train_loader (DataLoader): Dataloader for the training dataset, used to extract batch size.
        val_loader (DataLoader): Dataloader for the validation dataset, used to extract batch size.
        optimizer (optim.Optimizer): Optimizer used during training, used to extract learning rate.
        seed (int): Random seed for reproducibility.
        epochs (int): Number of training epochs.
        cfg (DictConfig): Configuration object with model details, specifically `cfg.model_cfg.model_type`.

    Returns:
        None
    """
    params = {
        "learning_rate": optimizer.param_groups[0]["lr"],
        "train_batch_size": train_loader.batch_size,
        "val_batch_size": val_loader.batch_size,
        "seed": seed,
        "epochs": epochs,
        "model": cfg.model_cfg.model_type,
    }

    mlflow_log(status, MLFLOW_LOG_PARAMS, params=params)

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> Dict[str, float]:
    """
    Performs one epoch of training on the provided dataset.

    The function sets the model to training mode and iterates through the provided DataLoader.
    It performs forward and backward passes, updates model weights, and accumulates metrics.
    It returns average loss, accuracy, and F1-score over the epoch.

    Args:
        model (nn.Module): The PyTorch model to train.
        loader (DataLoader): The training data loader.
        criterion (nn.Module): The loss function to optimize.
        optimizer (optim.Optimizer): The optimizer for updating model weights.
        device (torch.device): The device on which to perform training ("cuda", "cpu", "mps").
        epoch (int): The current epoch index (0-based).
        total_epochs (int): Total number of epochs (used for display in progress bar).

    Returns:
        Dict[str, float]: A dictionary containing:
            - "loss": Average training loss for the epoch.
            - "acc": Training accuracy (%) for the epoch.
            - "f1": Macro-averaged F1 score for the epoch.
    """
    model.train()
    running_loss: float = 0.0
    correct: int = 0
    total: int = 0
    y_true: list = []
    y_pred: list = []

    loop = tqdm(loader, desc=f"Epoch [{epoch + 1}/{total_epochs}]", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

        loop.set_postfix(loss=loss.item())

    avg_loss: float = running_loss / len(loader)
    accuracy: float = 100 * correct / total
    f1: float = f1_score(y_true, y_pred, average="macro")

    logger.info(f"Train - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, F1: {f1:.4f}")

    return {"loss": avg_loss, "acc": accuracy, "f1": f1}

def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates the model on the validation dataset and computes performance metrics.

    This function sets the model to evaluation mode and invokes the `evaluate` utility,
    which is expected to return the average loss, accuracy, and F1-score over the validation set.
    The results are logged and returned as a dictionary.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        loader (DataLoader): The validation data loader.
        criterion (nn.Module): The loss function used during evaluation.
        device (torch.device): The device to run evaluation on ("cuda", "cpu", or "mps").

    Returns:
        Dict[str, float]: A dictionary containing:
            - "loss": Average validation loss.
            - "acc": Validation accuracy (percentage).
            - "f1": Macro-averaged F1 score.
    """
    loss: float
    accuracy: float
    f1: float

    loss, accuracy, f1 = evaluate(model, loader, criterion, device)

    logger.info(f"Validation - Loss: {loss:.4f}, Acc: {accuracy:.2f}%, F1: {f1:.4f}")

    return {"loss": loss, "acc": accuracy, "f1": f1}

def _update_histories(
    train_hist: Dict[str, List[float]],
    val_hist: Dict[str, List[float]],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float]
) -> None:
    """
    Appends the current epoch's training and validation metrics to their respective history containers.

    This function updates the running history of metrics (such as loss, accuracy, and F1 score) by
    appending the new values from the current epoch's results. The histories are modified in place.

    Args:
        train_hist (Dict[str, List[float]]): A dictionary mapping metric names to lists of historical
                                             training values (e.g., {"loss": [...], "acc": [...], "f1": [...]}).
        val_hist (Dict[str, List[float]]): A dictionary mapping metric names to lists of historical
                                           validation values with the same structure as `train_hist`.
        train_metrics (Dict[str, float]): A dictionary containing the latest training metric values
                                          for the current epoch.
        val_metrics (Dict[str, float]): A dictionary containing the latest validation metric values
                                        for the current epoch.

    Returns:
        None
    """
    for k in train_metrics:
        train_hist[k].append(train_metrics[k])
        val_hist[k].append(val_metrics[k])

def _save_best_model_and_log_metrics(
    model: nn.Module,
    optimizer: optim.Optimizer,
    checkpoint_path: str,
    epoch: int,
    train_hist: Dict[str, List[float]],
    val_hist: Dict[str, List[float]],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    mlflow_status: bool
) -> None:
    """
    Saves the current best model checkpoint and logs metrics to MLflow.

    This function performs the following tasks:
    - Serializes and saves the model and optimizer state dictionaries to the specified checkpoint path.
    - Stores the current training and validation metric histories in the checkpoint.
    - Logs the model checkpoint as an MLflow artifact (if MLflow logging is enabled).
    - Logs individual metric values (loss, accuracy, F1, etc.) for training and validation to MLflow for the current epoch.

    Args:
        model (nn.Module): The PyTorch model to save.
        optimizer (optim.Optimizer): The optimizer whose state should be checkpointed.
        checkpoint_path (str): The file path where the model checkpoint will be saved.
        epoch (int): The current epoch number (used for checkpoint metadata and MLflow step logging).
        train_hist (Dict[str, List[float]]): Historical training metrics accumulated over epochs.
        val_hist (Dict[str, List[float]]): Historical validation metrics accumulated over epochs.
        train_metrics (Dict[str, float]): Training metrics for the current epoch (e.g., loss, accuracy, F1).
        val_metrics (Dict[str, float]): Validation metrics for the current epoch.
        mlflow_status (bool): Whether MLflow logging is enabled. If False, no metrics or artifacts will be logged.

    Returns:
        None
    """
    logger.info("New best model found, saving...")

    # Save model checkpoint including epoch and metric histories
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "optimiser_state_dict": optimizer.state_dict(),
        **{f"train_{k}_history": v for k, v in train_hist.items()},
        **{f"val_{k}_history": v for k, v in val_hist.items()},
    }, checkpoint_path)

    # Log checkpoint as artifact
    mlflow_log(mlflow_status, MLFLOW_LOG_ARTIFACT, local_path=checkpoint_path)

    # Log training and validation metrics for this epoch
    for k in train_metrics:
        mlflow_log(mlflow_status, MLFLOW_LOG_METRIC, key=f"train_{k}", value=train_metrics[k], step=epoch)
        mlflow_log(mlflow_status, MLFLOW_LOG_METRIC, key=f"validation_{k}", value=val_metrics[k], step=epoch)

def _finalize_training(
    train_hist: Dict[str, List[float]],
    val_hist: Dict[str, List[float]],
    mlflow_status: bool,
    artifact_dir: str,
    model: nn.Module,
    model_name: str
) -> None:
    """
    Finalizes the training process by plotting metrics, logging artifacts, and registering the model.

    This function performs several post-training tasks:
    - Plots training and validation curves for loss, accuracy, and F1-score.
    - Logs configuration and output directories as artifacts to MLflow.
    - Registers the trained model in MLflow under the specified model name.
    - Logs the final MLflow artifact URI and ends the MLflow run, if logging is enabled.

    Args:
        train_hist (Dict[str, List[float]]): Dictionary of training metric histories (e.g., "loss", "acc", "f1").
        val_hist (Dict[str, List[float]]): Dictionary of validation metric histories.
        mlflow_status (bool): Whether MLflow logging is enabled.
        artifact_dir (str): Directory containing training artifacts to be logged (e.g., logs, plots).
        model (nn.Module): The trained PyTorch model to register.
        model_name (str): The name under which the model will be registered in MLflow.

    Returns:
        None
    """
    # Plot training and validation curves
    plot_training_curves(
        train_hist["loss"], val_hist["loss"],
        train_hist["acc"], val_hist["acc"],
        train_f1=train_hist["f1"],
        val_f1=val_hist["f1"]
    )
    logger.info("Plotted training curves.")

    # Log config and artifacts to MLflow
    mlflow_log(mlflow_status, MLFLOW_LOG_DICT, dictionary={}, artifact_file="configs/params.json")
    mlflow_log(mlflow_status, MLFLOW_LOG_ARTIFACTS, local_dir=artifact_dir, artifact_path="logs")

    # Register the trained model
    mlflow_pytorch_call(
        mlflow_status,
        pytorch_function=MLFLOW_LOG_MODEL,
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=model_name
    )

    # End the MLflow run and log final artifact URI
    if mlflow_status:
        artifact_uri = mlflow.get_artifact_uri()
        mlflow_log(mlflow_status, "log_params", params={"artifact_uri": artifact_uri})
        mlflow.end_run()
        logger.info("MLflow run ended.")
