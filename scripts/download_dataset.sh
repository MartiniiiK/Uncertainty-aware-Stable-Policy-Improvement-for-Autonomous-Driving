#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
file=data/easycarla_offline_dataset.hdf5
if [ -s "$file" ]; then echo 'Dataset exists; refusing overwrite'; exit 1; fi
curl -fL --retry 5 -C - -o "$file.part" https://huggingface.co/datasets/silverwingsbot/easycarla/resolve/main/easycarla_offline_dataset.hdf5
python - "$file.part" <<'PY'
import h5py,sys
with h5py.File(sys.argv[1]) as f:
    n=len(f['observations']); assert n>0 and f['observations'].shape==(n,307) and f['next_observations'].shape==(n,307) and f['actions'].shape==(n,3)
    for k in ('rewards','costs','done','info/is_collision','info/is_off_road'):assert len(f[k])==n,k
    print('Dataset shape verified:',n)
PY
mv "$file.part" "$file"
sha256sum "$file" > "$file.sha256"
