#!/usr/bin/env bash
# GPU 1: C_random_repair, then D_full_clean. Use the same BATCH_NAME as GPU 0.
set -euo pipefail
export TZ="${TZ:-Asia/Shanghai}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
BATCH_NAME="${BATCH_NAME:-oracle_v1}"
BASE="$ROOT/research_results/RAMP_oracle/cifar10_resnet9/noniid_beta0.5/clients20_malicious4_20pct/badnet"
if [[ "$BATCH_NAME" == */* || "$BATCH_NAME" == . || "$BATCH_NAME" == .. ]]; then
  echo 'BATCH_NAME must be a simple directory name.' >&2; exit 2
fi
exec "$PYTHON" -u "$ROOT/research_oracle.py" lane --lane 1 \
  --device "${DEVICE:-cuda:1}" --rounds "${ROUNDS:-100}" --local-ep "${LOCAL_EP:-2}" \
  --topk-ratio "${TOPK_RATIO:-0.001}" --epsilon "${EPSILON:-0.001}" \
  --reference "${REFERENCE:-median}" --batch-root "${RESULT_ROOT:-$BASE/$BATCH_NAME}" "$@"
