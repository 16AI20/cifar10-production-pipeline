# optuna_sweep.py

import yaml
import torch
import optuna

from typing import Any, Dict

from omegaconf import OmegaConf, DictConfig
from optuna.trial import Trial

from .enums import ModelType
from .model import get_model
from .data_loader import load_data
from .train import train_model
from .evaluate import evaluate
from .general_utils import set_global_seed


def load_search_space(path: str = "conf/optuna_config.yaml") -> Dict[str, Any]:
    """
    Load the Optuna hyperparameter search space from a YAML config file.

    Args:
        path (str): Path to the YAML file defining search space.

    Returns:
        Dict[str, Any]: Dictionary containing search space parameters.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)["optuna"]["search_space"]


def sample_param(trial: Trial, name: str, spec: Dict[str, Any]) -> Any:
    """
    Sample a hyperparameter value from the trial using the specified strategy.

    Args:
        trial (Trial): Optuna trial object.
        name (str): Name of the hyperparameter.
        spec (Dict[str, Any]): Specification dict with keys: type, low/high/choices.

    Returns:
        Any: Sampled value appropriate for the parameter type.

    Raises:
        ValueError: If the parameter type is unsupported.
    """
    if spec["type"] == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    elif spec["type"] == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    elif spec["type"] == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    else:
        raise ValueError(f"Unsupported parameter type: {spec['type']}")


def objective(trial: Trial) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    This function:
    - Loads base config
    - Samples hyperparameters
    - Trains the model
    - Evaluates the model

    Args:
        trial (Trial): Current Optuna trial instance.

    Returns:
        float: Evaluation metric to be minimized (e.g., validation loss).
    """
    cfg: DictConfig = OmegaConf.load("conf/config.yaml")
    search_space = load_search_space()

    # Sample trial hyperparameters
    cfg.lr = sample_param(trial, "lr", search_space["lr"])
    cfg.batch_size = sample_param(trial, "batch_size", search_space["batch_size"])
    cfg.model_cfg = sample_param(trial, "model_type", search_space["model_type"])
    cfg.run_name = f"optuna-{trial.number}"
    cfg.registered_model_name = f"optuna-{cfg.model_cfg}-{trial.number}"
    cfg.resume = False  # Disable resume for sweeps

    # Set global seed and device
    set_global_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    model_type = ModelType[cfg.model_cfg.upper()]
    train_loader, val_loader, _ = load_data(model_type=model_type, batch_size=cfg.batch_size)
    model = get_model(model_type=model_type, num_classes=10, pretrained=True).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Train the model with current hyperparameters
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        cfg=cfg,
        scheduler=None
    )

    # Evaluate and return validation loss (or negative F1 to maximize)
    val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
    return val_loss


def run_optuna(n_trials: int = 20, direction: str = "minimize") -> optuna.study.Study:
    """
    Run Optuna hyperparameter optimization sweep.

    Args:
        n_trials (int): Number of trials to run.
        direction (str): Optimization direction — "minimize" or "maximize".

    Returns:
        optuna.study.Study: The completed Optuna study object.
    """
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    print(study.best_trial)

    return study


if __name__ == "__main__":
    run_optuna()
