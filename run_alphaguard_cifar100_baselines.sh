#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RAMP_PYTHON="${RAMP_PYTHON:-/data/miniconda3/envs/sunPytorch/bin/python}"
exec "$RAMP_PYTHON" -u research/alphaguard_baseline_batch.py launch "$@"
