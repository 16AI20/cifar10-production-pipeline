#!/bin/bash

EPOCHS=${1:-}
MODEL=${2:-}
MODE=${3:-train}  # Default to 'train' if not provided

# Check if uv is available, fallback to python if not
if command -v uv &> /dev/null; then
    PYTHON_CMD="uv run python"
    echo "Using uv for execution..."
else
    PYTHON_CMD="python"
    echo "uv not found, falling back to python..."
fi

if [ "$MODE" == "sweep" ]; then
  echo "Running Optuna sweep..."
  $PYTHON_CMD -m src.pipeline mode=sweep
else
  echo "Running training pipeline..."
  CMD="$PYTHON_CMD -m src.pipeline mode=$MODE"
  [ -n "$EPOCHS" ] && CMD="$CMD model_cfg.epochs=$EPOCHS"
  [ -n "$MODEL" ] && CMD="$CMD model_cfg=$MODEL"
  eval $CMD
fi
