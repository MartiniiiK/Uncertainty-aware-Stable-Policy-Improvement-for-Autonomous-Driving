#!/usr/bin/env bash
set -euo pipefail
# Python 3.11; use a separate venv or known GPU container, not the host interpreter.
python -c 'import sys; assert sys.version_info[:2] == (3,11), sys.version'
python -m pip install --index-url https://download.pytorch.org/whl/cu121 'torch==2.2.1'
python -m pip install 'gym==0.26.2' 'numpy==1.24.4' 'pandas==2.0.3' 'tqdm==4.64.1' 'matplotlib==3.7.2' 'Pillow==10.2.0' 'h5py==3.10.0' 'scipy==1.11.4' 'setuptools<70' wheel 'tensorboard' 'tabulate'
CARLA_ROOT=${CARLA_ROOT:?Set CARLA_ROOT to extracted CARLA 0.9.16}
mapfile -t wheels < <(find "$CARLA_ROOT/PythonAPI" -name 'carla-0.9.16-cp311*linux*x86_64.whl')
if [ "${#wheels[@]}" -ne 1 ]; then echo 'Expected one CARLA cp311 wheel' >&2; exit 1; fi
python -m pip install --no-deps "${wheels[0]}"
python - <<'PY'
import torch,carla,gym,numpy,h5py
print('PyTorch:',torch.__version__,'CUDA available:',torch.cuda.is_available())
print('Gym:',gym.__version__,'NumPy:',numpy.__version__)
PY
