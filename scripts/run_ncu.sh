#!/usr/bin/env bash
# Linux equivalent of scripts/run_ncu.ps1.
# Usage: scripts/run_ncu.sh [impl] [label] [extra profile_target.py args...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
IMPL="${1:-attn_naive}"
LABEL="${2:-$IMPL}"
shift $(( $# > 2 ? 2 : $# ))
SHAPE="--B 2 --H 16 --N 2048 --D 64 --dtype fp16"
mkdir -p "$ROOT/results/ncu"
REP="$ROOT/results/ncu/$LABEL"

EXTRA="smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
ncu --set full --metrics "$EXTRA" --import-source yes --nvtx --nvtx-include "kf_profile/" \
    --target-processes all -f -o "$REP" \
    "$PY" "$ROOT/bench/profile_target.py" --impl "$IMPL" $SHAPE "$@"

"$PY" "$ROOT/scripts/ncu_summary.py" "$REP.ncu-rep" --label "$LABEL" --impl "$IMPL" --shape "$SHAPE $*"
