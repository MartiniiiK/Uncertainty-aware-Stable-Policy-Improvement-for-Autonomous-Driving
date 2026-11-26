#!/usr/bin/env bash
# Optional future USPI comparison, NOT a record of completed V1 experiments.
set -euo pipefail
ACTOR=${ACTOR:-da_runs/strong42_eta1/dql_epoch0200.pth}
CRITICS=${CRITICS:-da_runs/uspi_fqe42/critics.pt}
OUT=${OUT:-da_runs/uspi42}
test -s "$ACTOR"; test -s "$CRITICS"
if [ -e "$OUT" ]; then echo 'Choose a fresh OUT; refusing overwrite' >&2; exit 1; fi
mkdir -p "$OUT" logs
exec 9>logs/uspi42.lock; flock -n 9
for mode in uspi rule dql; do
  extra=(); case "$mode" in uspi) extra=(--checkpoint "$ACTOR" --control-checkpoint "$CRITICS" --uspi-config configs/uspi.json);; dql) extra=(--checkpoint "$ACTOR");; esac
  python -u evaluate_uspi.py --mode "$mode" "${extra[@]}" --out "$OUT/eval_$mode" --episodes 100 --seed 1000 --town Town03 --vehicles 20 --steps 1000 --device cuda > "logs/uspi42_eval_${mode}.log" 2>&1
done
python scripts/summarize.py "$OUT"
