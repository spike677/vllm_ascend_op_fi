#!/usr/bin/env bash
set -e
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
source /work/venv/bin/activate
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/dcmi:${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
export ACLNN_CACHE_LIMIT=0
export PYTHONPATH=/audit:/msprobe_packages:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export XDG_CACHE_HOME=/private_batch/cache ASCEND_CACHE_PATH=/private_batch/cache/ascend
export HF_HOME=/private_batch/cache/huggingface VLLM_CACHE_ROOT=/private_batch/cache/vllm
export TRITON_CACHE_DIR=/private_batch/cache/triton TORCHINDUCTOR_CACHE_DIR=/private_batch/cache/inductor
export TOKENIZERS_PARALLELISM=false VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
export ASCEND_PROCESS_LOG_PATH=/private_batch/native_logs
mkdir -p "$XDG_CACHE_HOME" "$ASCEND_CACHE_PATH" "$ASCEND_PROCESS_LOG_PATH" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
# The controller sets DEBUG/stdout logging per capture command. No speculative
# GE_DUMP_TILING variable or opaque executor/buffer modifications are used.
