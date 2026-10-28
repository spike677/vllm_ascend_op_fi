#!/usr/bin/env bash
set -e
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /work/venv/bin/activate
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
export XDG_CACHE_HOME=/evidence/cache ASCEND_CACHE_PATH=/evidence/cache/ascend
mkdir -p "$XDG_CACHE_HOME" "$ASCEND_CACHE_PATH"
export PYTHONPATH=/audit:/msprobe_packages:${PYTHONPATH:-}
