#!/usr/bin/env bash
set -euo pipefail
BATCH="/home/risegpu/sunjiahao/Alignins/research_results/RAMP_grid/zero1to5_penalty2to8_20261112"
exec "${PYTHON:-/data/miniconda3/envs/sunPytorch/bin/python}" "$BATCH/controller.py" row 4 --device "${1:-cuda:1}"
