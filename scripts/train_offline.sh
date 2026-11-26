#!/usr/bin/env bash
# Optional reproducibility only. Does not connect CARLA. Never launched by packaging.
set -euo pipefail
DATA=${DATA:-data/easycarla_offline_dataset.hdf5}
DQL=${DQL:-da_runs/strong42_eta1}
FQE=${FQE:-da_runs/uspi_fqe42}
mkdir -p logs
if [ -e "$DQL" ] || [ -e "$FQE" ]; then echo 'Refusing to overwrite existing training directories' >&2; exit 1; fi
python -u train_dql_v2.py --data "$DATA" --out "$DQL" --phase dql --epochs 200 --updates-per-epoch 1000 --batch 256 --eta 1.0 --seed 42 --device cuda > logs/uspi_dql_train.log 2>&1
python -u train_da_control.py --data "$DATA" --out "$FQE" --steps 66667 --heads 3 --batch 256 --width 256 --seed 42 --split-seed 2026 --gamma .99 --tau .005 --dt .1 --lr .0003 --log-every 500 --device cuda > logs/uspi_fqe_train.log 2>&1
