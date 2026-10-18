#!/bin/bash
# ScienceWorld simulator in its own interpreter (requires Java 11 or newer on PATH).
# usage: bash scripts/setup_scienceworld.sh [env_dir]
set -euo pipefail
ENV_DIR=${1:-envs/scienceworld}
python3.11 -m venv "$ENV_DIR"
"$ENV_DIR/bin/pip" install scienceworld==1.2.3
echo "export SCIENCEWORLD_PYTHON=$(cd "$ENV_DIR" && pwd)/bin/python"
