#!/bin/bash
# ALFWorld text environment in its own interpreter (used by the CPU environment process).
# usage: bash scripts/setup_alfworld.sh [env_dir] [data_dir]
set -euo pipefail
ENV_DIR=${1:-envs/alfworld}; DATA_DIR=${2:-data_env/alfworld}
python3.11 -m venv "$ENV_DIR"
"$ENV_DIR/bin/pip" install alfworld==0.4.2 textworld==1.7.0 pyyaml
ALFWORLD_DATA="$DATA_DIR" "$ENV_DIR/bin/alfworld-download"
echo "export ALFWORLD_DATA=$(cd "$DATA_DIR" && pwd) ALFWORLD_PYTHON=$(cd "$ENV_DIR" && pwd)/bin/python"
