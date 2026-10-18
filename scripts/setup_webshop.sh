#!/bin/bash
# WebShop environment (small product set, 1,000 products). Run inside a fresh Python 3.8 conda environment.
# usage: conda create -n webshop python=3.8 -y && conda activate webshop && bash scripts/setup_webshop.sh [dir]
set -euo pipefail
WS_HOME=${1:-third_party/WebShop}
git clone https://github.com/princeton-nlp/WebShop.git "$WS_HOME"
git -C "$WS_HOME" checkout 64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
(cd "$WS_HOME" && bash setup.sh -d small)   # dependencies, Java 11, product data, search index
echo "export WS_HOME=$(cd "$WS_HOME" && pwd) WEBSHOP_PYTHON=$(which python)"
