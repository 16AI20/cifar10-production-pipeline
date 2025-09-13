"""
General utilities and helper functions for the CIFAR-10 project.

This module provides cross-cutting concerns and utilities that are used
throughout the project including:
- Logging configuration and setup
- MLflow experiment tracking integration
- Reproducibility utilities (seed setting)
- Environment and device detection
- General helper functions for data processing
"""

import logging
import logging.config
import os
import time
import random
from typing import Any, Optional, Tuple, List

import yaml

import torch

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

import numpy as np

# Set up basic logging configuration
logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path: str = "./conf/logging.yaml",
    default_level: int = logging.INFO,
    log_dir: Optional[str] = None
) -> None:
    """
    Set up logging configuration from a YAML file. If the configuration file
    is missing, malformed, or inaccessible, falls back to a basic logging setup.

    Parameters
    ----------
    logging_config_path : str, optional
        Path to a YAML file that specifies the logging configuration.
        Default is './conf/logging.yaml'.

    default_level : int, optional
        Logging level to use in the fallback basic configuration.
        Default is logging.INFO.

    log_dir : str or None, optional
        Directory to save log files. If provided, filenames in the config
        will be redirected to this directory. Default is None (uses paths as-is).
    """
    try:
        # Attempt to read the YAML config file
        with open(logging_config_path, "rt", encoding="utf-8") as file:
            log_config = yaml.safe_load(file)

        # If a custom log directory is provided, update handler file paths
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists
            for handler in log_config.get("handlers", {}).values():
                if "filename" in handler:
                    # Extract filename and recompose it with the new log directory
                    filename = os.path.basename(handler["filename"])
                    handler["filename"] = os.path.join(log_dir, filename)

        # Apply logging configuration from the YAML dict
        logging.config.dictConfig(log_config)

    except (FileNotFoundError, PermissionError) as file_err:
        # If file is not found or unreadable, use fallback logging
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logging.getLogger().warning("Logging config file not found or inaccessible. Using basic config.")
        logging.getLogger().exception(file_err)

    except (yaml.YAMLError, ValueError, TypeError) as parse_err:
        # If the YAML is malformed or incompatible, use fallback logging
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logging.getLogger().warning("Error parsing logging config. Using basic config.")
        logging.getLogger().exception(parse_err)


def _set_mlflow_tags() -> None:
    """
    Set MLflow tags from environment variables to help identify and filter runs.
    """
    def set_tag(env_var: str, tag_name: str = "") -> None:
        """
        Helper function to set a tag from an environment variable.
        If `tag_name` is not provided, the environment variable name (lowercased) is used.

        Args:
            env_var (str): Name of the environment variable to check.
            tag_name (str, optional): MLflow tag name to use. Defaults to "".
        """
        if env_var in os.environ:
            key = tag_name if tag_name else env_var.lower()
            mlflow.set_tag(key, os.environ.get(env_var))

    # Common tags for hyperparameter tuning and job tracking
    set_tag("MLFLOW_HP_TUNING_TAG", "hptuning_tag")
    set_tag("JOB_UUID")
    set_tag("JOB_NAME")


def _resume_previous_run(
    client: MlflowClient,
    exp_name: str,
    base_run_name: str
) -> Tuple[Optional[Run], int]:
    """
    Attempt to resume the most recent MLflow run with the same base name.

    Args:
        client (MlflowClient): Initialized MLflow client.
        exp_name (str): Name of the MLflow experiment.
        base_run_name (str): Prefix used to identify run names to resume.

    Returns:
        Tuple[Optional[Run], int]: The resumed run (or None) and the max step from previous metrics.
    """
    experiment = client.get_experiment_by_name(exp_name)
    if not experiment:
        return None, 0

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName LIKE '{base_run_name}-%'",
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None, 0

    run = runs[0]
    run_id = run.info.run_id
    mlflow.start_run(run_id=run_id)
    logger.info(f"Resuming previous run: {run.info.run_name}")

    max_step = _get_max_logged_step(client, run_id)
    return run, max_step


def _get_metric_keys(client: MlflowClient, run_id: str) -> List[str]:
    """Extract metric keys from a run, trying multiple fallback strategies."""
    try:
        run_data = client.get_run(run_id).data
        keys = getattr(run_data.metrics, 'keys', lambda: [])()
        if keys:
            return list(keys)

        keys = [m.key for m in getattr(run_data.metrics, '__iter__', lambda: [])()]
        if keys:
            return keys

        return [m.key for m in client.list_metrics(run_id)]
    except Exception as e:
        logger.warning(f"Failed to extract metric keys: {e}")
        return []


def _get_max_step_for_key(client: MlflowClient, run_id: str, key: str) -> Optional[int]:
    """Return the max step for a given metric key, if available."""
    try:
        history = client.get_metric_history(run_id, key)
        if history:
            return max(m.step for m in history)
    except Exception as e:
        logger.warning(f"Error retrieving metric history for '{key}': {e}")
    return None


def _get_max_logged_step(client: MlflowClient, run_id: str) -> int:
    """
    Get the maximum step value from all logged metrics for a given run.
    """
    metric_keys = _get_metric_keys(client, run_id)
    max_steps = filter(None, (_get_max_step_for_key(client, run_id, k) for k in metric_keys))
    return max(max_steps, default=0)



def _start_new_run(base_run_name: str) -> str:
    """
    Start a new MLflow run with a timestamped name.

    Args:
        base_run_name (str): Prefix name for the new run.

    Returns:
        str: Full name of the new MLflow run.
    """
    run_name = f"{base_run_name}-{int(time.time())}"
    mlflow.start_run(run_name=run_name)
    logger.info(f"Starting new run: {run_name}")
    return run_name


def mlflow_init(
    tracking_uri: str,
    exp_name: str,
    run_name: str,
    setup_mlflow: bool = False,
    autolog: bool = False,
    resume: bool = False,
) -> Tuple[bool, Optional[Run], int]:
    """
    Initialise MLflow connection and optionally resume a previous run.

    Args:
        tracking_uri (str): URI for MLflow tracking server.
        exp_name (str): Name of the MLflow experiment.
        run_name (str): Name of the run or run prefix.
        setup_mlflow (bool, optional): If True, perform MLflow setup. Defaults to False.
        autolog (bool, optional): If True, enable MLflow autologging. Defaults to False.
        resume (bool, optional): If True, resume from last run with matching prefix. Defaults to False.

    Returns:
        Tuple[bool, Optional[Run], int]: 
            - Success flag,
            - The active MLflow run (if any),
            - Step offset for continuing metric logging.
    """
    init_success = False
    mlflow_run = None
    step_offset = 0

    if not setup_mlflow:
        return init_success, mlflow_run, step_offset

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(exp_name)
        mlflow.enable_system_metrics_logging()

        if autolog:
            mlflow.autolog()

        # Add suffix if this is a hyperparameter tuning run
        if "MLFLOW_HPTUNING_TAG" in os.environ:
            run_name += "-hp"

        base_run_name = run_name
        client = mlflow.tracking.MlflowClient()

        # Resume from previous run or start a new one
        if resume:
            resumed_run, step_offset = _resume_previous_run(client, exp_name, base_run_name)
            if not resumed_run:
                run_name = _start_new_run(base_run_name)
        else:
            run_name = _start_new_run(base_run_name)

        _set_mlflow_tags()

        mlflow_run = mlflow.active_run()
        init_success = True
        logger.info("MLflow initialisation has succeeded.")
        logger.info(f"UUID for MLflow run: {mlflow_run.info.run_id}")

    except Exception as e:
        logger.error("MLflow initialisation has failed.")
        logger.error(e)

    return init_success, mlflow_run, step_offset


def mlflow_log(mlflow_init_status, log_function, **kwargs):
    """Custom function for utilising MLflow's logging functions.

    This function is only relevant when the function `mlflow_init`
    returns a "True" value, translating to a successful initialisation
    of a connection with an MLflow server.

    Parameters
    ----------
    mlflow_init_status : bool
        Boolean value indicative of success of intialising connection
        with MLflow server.
    log_function : str
        Name of MLflow logging function to be used.
        See https://www.mlflow.org/docs/latest/python_api/mlflow.html
    **kwargs
        Keyword arguments passed to `log_function`.
    """
    if mlflow_init_status:
        try:
            method = getattr(mlflow, log_function)
            method(
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key in method.__code__.co_varnames
                }
            )
        except Exception as error:
            logger.error(error)


def mlflow_pytorch_call(
    mlflow_init_status: bool, pytorch_function: str, **kwargs
) -> Optional[Any]:
    """
    Convenience wrapper around the ``mlflow.pytorch`` API.

    This helper is intended to be used **only** after an MLflow tracking
    server / experiment has been successfully initialised, i.e. when the
    function that establishes the MLflow connection returns ``True``.
    It dynamically resolves the required *PyTorch–specific* MLflow
    function (e.g. ``log_model``, ``save_model``, ``load_model``) and
    forwards the provided keyword arguments if – and only if – they
    are part of the function’s formal parameter list.

    Parameters
    ----------
    mlflow_init_status : bool
        Flag indicating whether a connection to an MLflow tracking
        server has been successfully established.
    pytorch_function : str
        Name of the ``mlflow.pytorch`` function to be invoked.
        Refer to the official documentation:
        https://www.mlflow.org/docs/latest/api_reference/python_api/mlflow.pytorch.html
    **kwargs
        Arbitrary keyword arguments to be forwarded to the chosen
        ``mlflow.pytorch`` function.

    Returns
    -------
    Any or None
        Whatever the invoked ``mlflow.pytorch`` function returns.
        If ``mlflow_init_status`` is ``False`` or an exception is raised,
        ``None`` is returned.

    Notes
    -----
    • Only keyword arguments that appear in the target function’s
      signature are forwarded.
    • Errors are logged but **not** re‑raised to avoid interrupting the
      calling workflow.

    Examples
    --------
    >>> mlflow_pytorch_call(
    ...     mlflow_init_status=mlflow_init(...),
    ...     pytorch_function="log_model",
    ...     pytorch_model=model,
    ...     artifact_path="models/sketch"
    ... )
    """
    if not mlflow_init_status:
        return None

    try:
        method = getattr(mlflow.pytorch, pytorch_function)
    except AttributeError as err:
        logger.error(
            "Function '%s' does not exist in mlflow.pytorch: %s",
            pytorch_function,
            err,
        )
        return None

    try:
        # Forward only those kwargs that are accepted by the target method
        valid_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in method.__code__.co_varnames
        }
        return method(**valid_kwargs)
    except Exception as err:  # noqa: BLE001
        logger.error(
            "mlflow.pytorch.%s failed with error: %s",
            pytorch_function,
            err,
        )
        return None
    
def set_global_seed(seed: int = 42) -> None:
    """
    Sets global random seeds for Python, NumPy, and PyTorch to ensure reproducible behavior.

    Parameters:
        config_path (Optional[str]): Path to a YAML configuration file containing a 'seed' key.
                                    If None, defaults to 'config.yaml' in the current directory.

    Behavior:
        - Loads the seed from the config file (defaults to 42 if not found).
        - Sets the seed for Python's `random` module, NumPy, and PyTorch (CPU and CUDA).
        - Configures PyTorch's cuDNN backend for deterministic behavior.

    Raises:
        FileNotFoundError: If the provided config file path does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    # if config_path is None:
    #     config_path = "conf/config.yaml"

    # with open(config_path, "r") as f:
    #     config = yaml.safe_load(f)

    # seed = config.get("seed", 42)

    # Set random seeds
    random.seed(seed)                             # Python built-in RNG
    np.random.seed(seed)                          # NumPy RNG
    torch.manual_seed(seed)                       # PyTorch CPU RNG
    torch.cuda.manual_seed(seed)                  # For single-GPU CUDA
    torch.cuda.manual_seed_all(seed)              # For multi-GPU setups
    torch.backends.cudnn.deterministic = True     # Ensures deterministic convolution operations
    torch.backends.cudnn.benchmark = False        # Disables CUDNN autotuner for reproducibility

    return seed
