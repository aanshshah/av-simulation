#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=".venv"
PYTHON_BIN="python3"

if [[ ! -d "$ENV_DIR" ]]; then
  echo "[1/6] Creating virtual environment in $ENV_DIR"
  $PYTHON_BIN -m venv "$ENV_DIR"
else
  echo "[1/6] Virtual environment $ENV_DIR already exists"
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  source "$ENV_DIR"/bin/activate
elif [[ "$(uname -s)" == "Linux" ]]; then
  source "$ENV_DIR"/bin/activate
else
  source "$ENV_DIR"/Scripts/activate
fi

echo "[2/6] Upgrading pip"
pip install --upgrade pip

echo "[3/6] Installing pinned dependencies (requirements-local.txt)"
pip install -r requirements-local.txt

echo "[4/6] Installing project in editable mode"
pip install -e .

echo "[5/6] Running simulation import smoke test"
python -m av_simulation.core.simulation --help

echo "[6/6] Running pytest suite"
pytest tests

echo "All validation steps completed successfully."
