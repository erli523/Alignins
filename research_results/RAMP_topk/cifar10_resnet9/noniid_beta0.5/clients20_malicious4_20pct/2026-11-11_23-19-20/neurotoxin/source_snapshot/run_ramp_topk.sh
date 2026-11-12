#!/usr/bin/env bash
# One attack: bash run_ramp_topk.sh badnet
# All three sequentially: bash run_ramp_topk.sh all
set -euo pipefail
export TZ="${TZ:-Asia/Shanghai}"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
ROUNDS="${ROUNDS:-100}"
LOCAL_EP="${LOCAL_EP:-2}"
TOPK_RATIO="${TOPK_RATIO:-0.001}"
EPSILON="${EPSILON:-0.001}"
REFERENCE="${REFERENCE:-median}"
RANDOM_REPEATS="${RANDOM_REPEATS:-30}"
RANDOM_SEED="${RANDOM_SEED:-20261111}"
MODE="${1:-all}"

case "$MODE" in
  all) ATTACKS=(badnet pgd neurotoxin) ;;
  badnet|badnets|BadNets) ATTACKS=(badnet) ;;
  pgd|PGD) ATTACKS=(pgd) ;;
  neurotoxin|Neurotoxin) ATTACKS=(neurotoxin) ;;
  *) echo "Usage: bash run_ramp_topk.sh [all|badnet|pgd|neurotoxin]" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
  echo "Use ROUNDS, DEVICE, LOCAL_EP, TOPK_RATIO, EPSILON, REFERENCE, RANDOM_REPEATS, RANDOM_SEED for overrides." >&2
  exit 2
fi

"$PYTHON" -c 'import sys, torch, numpy, matplotlib; print("Python:",sys.executable); print("Torch:",torch.__version__)'
BASE="${RESULT_ROOT:-$ROOT/research_results/RAMP_topk/cifar10_resnet9/noniid_beta0.5/clients20_malicious4_20pct}"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
BATCH="$BASE/$STAMP"
mkdir -p -- "$BASE"
# mkdir without -p deliberately rejects an existing batch directory.
mkdir -- "$BATCH"
echo "Independent experiment batch: $BATCH"
echo "Attacks: ${ATTACKS[*]}; rounds=$ROUNDS; local_ep=$LOCAL_EP; device=$DEVICE"
echo "Layer-matched random control: $RANDOM_REPEATS repeats; independent seed=$RANDOM_SEED"

# If training fails, retain completed data and try to build a partial batch report.
trap 'code=$?; if [[ $code -ne 0 ]]; then "$PYTHON" "$ROOT/research_topk.py" --compare "$BATCH" || true; echo "Batch stopped; inspect $BATCH" >&2; fi' EXIT

for attack in "${ATTACKS[@]}"; do
  echo "===== Starting $attack ====="
  "$PYTHON" -u "$ROOT/research_topk.py" \
    --output "$BATCH/$attack" \
    --topk-ratio "$TOPK_RATIO" --epsilon "$EPSILON" --reference "$REFERENCE" \
    --random-repeats "$RANDOM_REPEATS" --random-seed "$RANDOM_SEED" \
    --save-indices selected --snapshot-every 25 \
    --data cifar10 --model resnet9 --attack "$attack" --aggr alphaguard \
    --num_agents 20 --num_corrupt 4 --agent_frac 1.0 --poison_frac 0.3 \
    --non_iid --beta 0.5 --rounds "$ROUNDS" --local_ep "$LOCAL_EP" \
    --client_lr 0.1 --server_lr 1.0 --lr_decay 0.99 --bs 64 \
    --ag_topk "$TOPK_RATIO" --ag_l2_lower 25 --ag_l2_upper 75 \
    --ag_gref_strategy median --snap 1 --num_workers 0 --device "$DEVICE"
done

"$PYTHON" "$ROOT/research_topk.py" --compare "$BATCH"
echo "Finished. Open: $BATCH/README.md"
