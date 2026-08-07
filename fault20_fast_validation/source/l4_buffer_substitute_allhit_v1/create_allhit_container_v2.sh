#!/usr/bin/env bash
set -euo pipefail

container="fault20-qwen3-32b-allhit-v2"

if docker inspect "$container" >/dev/null 2>&1; then
    echo "REFUSE: container already exists: $container" >&2
    exit 2
fi

docker create \
  --name "$container" \
  --net=host \
  --shm-size=16g \
  --device /dev/davinci2:/dev/davinci2 \
  --device /dev/davinci5:/dev/davinci5 \
  --device /dev/davinci_manager:/dev/davinci_manager \
  --device /dev/devmm_svm:/dev/devmm_svm \
  --device /dev/hisi_hdc:/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /home/haoran/models/Qwen3-32B-w8a8-nopdmix:/AIdata/haoran/Qwen3-32B-w8a8-nopdmix:ro \
  -v /home/haoran/fault20_fast_validation:/fault20:rw \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
  -e FAULT20_METADATA_LOG_PATH=/fault20/runtime/l4_allhit_logs_v2/metadata.jsonl \
  -e FAULT20_BUFFER_LOG_PATH=/fault20/runtime/l4_allhit_logs_v2/run_trace.jsonl \
  --entrypoint /bin/bash \
  quay.io/ascend/vllm-ascend:v0.12.0rc1 \
  -lc "source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
       export LD_LIBRARY_PATH='/fault20/runtime/l4_metadata_lifecycle_proxy':\$LD_LIBRARY_PATH && \
       export LD_PRELOAD='/fault20/runtime/l4_allhit_v1/libfault20_buffer_substitute_abi0.so' && \
       exec vllm serve '/AIdata/haoran/Qwen3-32B-w8a8-nopdmix' \
         --served-model-name qwen3-32b-w8a8 \
         --host 127.0.0.1 \
         --port 18005 \
         --tensor-parallel-size 2 \
         --data-parallel-size 1 \
         --distributed-executor-backend mp \
         --trust-remote-code \
         --enforce-eager \
         --quantization ascend \
         --max-model-len 8192 \
         --max-num-seqs 2 \
         --gpu-memory-utilization 0.90"

docker start "$container"
