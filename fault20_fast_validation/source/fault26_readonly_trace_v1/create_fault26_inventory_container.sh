#!/usr/bin/env bash
set -euo pipefail

docker_host="${FAULT26_DOCKER_HOST:-unix:///run/fault26-docker-20260817.sock}"
image="${FAULT26_IMAGE:-quay.io/ascend/vllm-ascend:v0.18.0rc1}"
run_id="${FAULT26_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
container="fault26-qwen3-32b-readonly-inventory-${run_id,,}"
server_port="${FAULT26_SERVER_PORT:-18026}"
host_root="/home/haoran/fault39_dev_20260812"
container_root="/fault39"
model_host="/home/haoran/models/Qwen3-32B-w8a8-nopdmix"
model_container="/AIdata/haoran/Qwen3-32B-w8a8-nopdmix"
runtime_version="${FAULT26_RUNTIME_VERSION:-fault26_readonly_trace_v1}"
runtime_dir="$container_root/runtime/$runtime_version"
host_log="$host_root/runtime/$runtime_version/inventory_${run_id}.jsonl"
container_log="$runtime_dir/inventory_${run_id}.jsonl"
capture_tensors="${FAULT26_CAPTURE_TENSORS:-0}"
trace_all_apis="${FAULT26_TRACE_ALL_APIS_VALUE:-1}"
trace_filter="${FAULT26_TRACE_FILTER_VALUE:-}"
max_events="${FAULT26_TRACE_MAX_EVENTS_VALUE:-200000}"
use_opapi_proxy="${FAULT26_USE_OPAPI_PROXY:-0}"
preload_opapi_proxy="${FAULT26_PRELOAD_OPAPI_PROXY:-0}"
use_split_opapi_nn_proxy="${FAULT26_USE_SPLIT_OPAPI_NN_PROXY:-0}"
use_typed_extension="${FAULT26_USE_TYPED_EXTENSION:-0}"
use_write_gate="${FAULT26_USE_WRITE_GATE:-0}"
use_hccl_trace="${FAULT26_USE_HCCL_TRACE:-0}"
use_qmm_residual_stomp="${FAULT26_USE_QMM_RESIDUAL_STOMP:-0}"
use_writer_audit="${FAULT26_USE_WRITER_AUDIT:-0}"
use_fia_writer_proxy="${FAULT26_USE_FIA_WRITER_PROXY:-0}"
use_writer_topology="${FAULT26_USE_WRITER_TOPOLOGY:-0}"
use_allocator_audit="${FAULT26_USE_ALLOCATOR_AUDIT:-0}"
use_qmm_underflow="${FAULT26_USE_QMM_UNDERFLOW:-0}"
runtime_prefix=""
preload="$runtime_dir/libfault26_readonly_trace_abi0.so"
pythonpath_prefix=""
worker_extension_arg=""
typed_log="$host_root/runtime/$runtime_version/typed_${run_id}.jsonl"
container_typed_log="$runtime_dir/typed_${run_id}.jsonl"
write_gate_log="$host_root/runtime/$runtime_version/write_gate_${run_id}.jsonl"
container_write_gate_log="$runtime_dir/write_gate_${run_id}.jsonl"
qmm_log="$host_root/runtime/$runtime_version/qmm_${run_id}.jsonl"
container_qmm_log="$runtime_dir/qmm_${run_id}.jsonl"
hccl_log="$host_root/runtime/$runtime_version/hccl_${run_id}.jsonl"
container_hccl_log="$runtime_dir/hccl_${run_id}.jsonl"
qmm_stomp_log="$host_root/runtime/$runtime_version/qmm_stomp_${run_id}.jsonl"
container_qmm_stomp_log="$runtime_dir/qmm_stomp_${run_id}.jsonl"
fia_writer_log="$host_root/runtime/$runtime_version/fia_writer_${run_id}.jsonl"
container_fia_writer_log="$runtime_dir/fia_writer_${run_id}.jsonl"
qmm_underflow_log="$host_root/runtime/$runtime_version/qmm_underflow_${run_id}.jsonl"
container_qmm_underflow_log="$runtime_dir/qmm_underflow_${run_id}.jsonl"
custom_opapi_container="${FAULT26_REAL_CUSTOM_OPAPI:-/vllm-workspace/vllm-ascend/csrc/build/libcust_opapi.so}"
custom_opapi_image_path="/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so"
custom_proxy_mount_args=()
split_opapi_mount_args=()

if [[ "$capture_tensors" != 0 && "$capture_tensors" != 1 ]]; then
  echo "REFUSE: FAULT26_CAPTURE_TENSORS must be 0 or 1" >&2
  exit 1
fi
if [[ "$trace_all_apis" != 0 && "$trace_all_apis" != 1 ]]; then
  echo "REFUSE: FAULT26_TRACE_ALL_APIS_VALUE must be 0 or 1" >&2
  exit 1
fi
if [[ ! "$max_events" =~ ^[1-9][0-9]*$ ]]; then
  echo "REFUSE: FAULT26_TRACE_MAX_EVENTS_VALUE must be a positive integer" >&2
  exit 1
fi
if [[ ! "$server_port" =~ ^[1-9][0-9]*$ ]] ||
   (( server_port > 65535 )); then
  echo "REFUSE: FAULT26_SERVER_PORT must be in [1, 65535]" >&2
  exit 1
fi
if [[ "$use_opapi_proxy" != 0 && "$use_opapi_proxy" != 1 ]]; then
  echo "REFUSE: FAULT26_USE_OPAPI_PROXY must be 0 or 1" >&2
  exit 1
fi
if [[ "$preload_opapi_proxy" != 0 && "$preload_opapi_proxy" != 1 ]]; then
  echo "REFUSE: FAULT26_PRELOAD_OPAPI_PROXY must be 0 or 1" >&2
  exit 1
fi
if [[ "$preload_opapi_proxy" == 1 && "$use_opapi_proxy" != 1 ]]; then
  echo "REFUSE: FAULT26_PRELOAD_OPAPI_PROXY requires FAULT26_USE_OPAPI_PROXY=1" >&2
  exit 1
fi
if [[ "$use_split_opapi_nn_proxy" != 0 &&
      "$use_split_opapi_nn_proxy" != 1 ]]; then
  echo "REFUSE: FAULT26_USE_SPLIT_OPAPI_NN_PROXY must be 0 or 1" >&2
  exit 1
fi
if [[ "$use_split_opapi_nn_proxy" == 1 && "$use_opapi_proxy" != 1 ]]; then
  echo "REFUSE: split opapi_nn proxy requires FAULT26_USE_OPAPI_PROXY=1" >&2
  exit 1
fi
if [[ "$use_typed_extension" != 0 && "$use_typed_extension" != 1 ]]; then
  echo "REFUSE: FAULT26_USE_TYPED_EXTENSION must be 0 or 1" >&2
  exit 1
fi
if [[ "$use_write_gate" != 0 && "$use_write_gate" != 1 ]]; then
  echo "REFUSE: FAULT26_USE_WRITE_GATE must be 0 or 1" >&2
  exit 1
fi
if [[ "$use_hccl_trace" != 0 && "$use_hccl_trace" != 1 ]]; then
  echo "REFUSE: FAULT26_USE_HCCL_TRACE must be 0 or 1" >&2
  exit 1
fi
if [[ "$use_qmm_residual_stomp" != 0 && "$use_qmm_residual_stomp" != 1 ]]; then
  echo "REFUSE: FAULT26_USE_QMM_RESIDUAL_STOMP must be 0 or 1" >&2
  exit 1
fi
if [[ "$use_writer_audit" != 0 && "$use_writer_audit" != 1 ]]; then
  echo "REFUSE: FAULT26_USE_WRITER_AUDIT must be 0 or 1" >&2
  exit 1
fi
if [[ "$use_fia_writer_proxy" != 0 && "$use_fia_writer_proxy" != 1 ]]; then
  echo "REFUSE: FAULT26_USE_FIA_WRITER_PROXY must be 0 or 1" >&2
  exit 1
fi
if [[ "$use_writer_topology" != 0 && "$use_writer_topology" != 1 ]]; then
  echo "REFUSE: FAULT26_USE_WRITER_TOPOLOGY must be 0 or 1" >&2
  exit 1
fi
if [[ "$use_allocator_audit" != 0 && "$use_allocator_audit" != 1 ]]; then
  echo "REFUSE: FAULT26_USE_ALLOCATOR_AUDIT must be 0 or 1" >&2
  exit 1
fi
if [[ "$use_qmm_underflow" != 0 && "$use_qmm_underflow" != 1 ]]; then
  echo "REFUSE: FAULT26_USE_QMM_UNDERFLOW must be 0 or 1" >&2
  exit 1
fi
if [[ "$use_qmm_underflow" == 1 && "$use_typed_extension" != 1 ]]; then
  echo "REFUSE: QMM underflow requires the developer-mode worker extension" >&2
  exit 1
fi
if [[ "$use_allocator_audit" == 1 && "$use_typed_extension" != 1 ]]; then
  echo "REFUSE: allocator audit requires the developer-mode worker extension" >&2
  exit 1
fi
if [[ "$use_writer_topology" == 1 ]] &&
   [[ "$capture_tensors" != 1 || "$use_opapi_proxy" != 1 ||
      "$preload_opapi_proxy" != 1 || "$use_split_opapi_nn_proxy" != 1 ]]; then
  echo "REFUSE: writer topology requires tensor capture plus preloaded main/split typed opapi proxies" >&2
  exit 1
fi
if [[ "$use_fia_writer_proxy" == 1 ]] &&
   [[ "$use_opapi_proxy" != 1 || "$use_typed_extension" != 1 ]]; then
  echo "REFUSE: FIA writer proxy requires opapi proxy and developer worker extension" >&2
  exit 1
fi
if [[ "$use_fia_writer_proxy" == 1 ]] &&
   [[ "$use_write_gate" == 1 || "$use_qmm_residual_stomp" == 1 ]]; then
  echo "REFUSE: Phase C FIA lossless gate cannot run with another write gate" >&2
  exit 1
fi
if [[ "$use_writer_audit" == 1 && "$use_typed_extension" != 1 ]]; then
  echo "REFUSE: residual writer audit requires the developer-mode worker extension" >&2
  exit 1
fi
if [[ "$use_write_gate" == 1 && "$use_qmm_residual_stomp" == 1 ]]; then
  echo "REFUSE: legacy AddRms write gate and QMM residual stomp are mutually exclusive" >&2
  exit 1
fi
if [[ "$use_qmm_underflow" == 1 ]] &&
   [[ "$use_write_gate" == 1 || "$use_qmm_residual_stomp" == 1 ||
      "$use_fia_writer_proxy" == 1 ]]; then
  echo "REFUSE: QMM underflow cannot co-load another write-capable gate" >&2
  exit 1
fi

for path in \
  /dev/davinci2 /dev/davinci5 /dev/davinci_manager /dev/devmm_svm \
  /dev/hisi_hdc "$host_root" "$model_host" \
  "$host_root/runtime/$runtime_version/libfault26_readonly_trace_abi0.so"; do
  if [[ ! -e "$path" ]]; then
    echo "REFUSE: required path is missing: $path" >&2
    exit 1
  fi
done

if [[ -e "$host_log" ]]; then
  echo "REFUSE: inventory log already exists: $host_log" >&2
  exit 1
fi
if [[ "$use_typed_extension" == 1 && -e "$typed_log" ]]; then
  echo "REFUSE: typed probe log already exists: $typed_log" >&2
  exit 1
fi
if [[ "$use_write_gate" == 1 && -e "$write_gate_log" ]]; then
  echo "REFUSE: write-gate log already exists: $write_gate_log" >&2
  exit 1
fi
if [[ -e "$qmm_log" ]]; then
  echo "REFUSE: QMM trace log already exists: $qmm_log" >&2
  exit 1
fi
if [[ "$use_hccl_trace" == 1 && -e "$hccl_log" ]]; then
  echo "REFUSE: HCCL trace log already exists: $hccl_log" >&2
  exit 1
fi
if [[ "$use_qmm_residual_stomp" == 1 && -e "$qmm_stomp_log" ]]; then
  echo "REFUSE: QMM residual stomp log already exists: $qmm_stomp_log" >&2
  exit 1
fi
if [[ "$use_fia_writer_proxy" == 1 && -e "$fia_writer_log" ]]; then
  echo "REFUSE: FIA writer log already exists: $fia_writer_log" >&2
  exit 1
fi
if [[ "$use_qmm_underflow" == 1 && -e "$qmm_underflow_log" ]]; then
  echo "REFUSE: QMM underflow log already exists: $qmm_underflow_log" >&2
  exit 1
fi

if ! docker -H "$docker_host" image inspect "$image" >/dev/null 2>&1; then
  echo "REFUSE: image is not present in isolated Docker: $image" >&2
  exit 1
fi

if docker -H "$docker_host" inspect "$container" >/dev/null 2>&1; then
  echo "REFUSE: container already exists: $container" >&2
  exit 1
fi

if [[ "$use_opapi_proxy" == 1 ]]; then
  if [[ ! -r "$host_root/runtime/$runtime_version/libopapi.so" ||
        ! -r "$host_root/runtime/$runtime_version/libcust_opapi.so" ||
        ! -r "$host_root/runtime/$runtime_version/real/libopapi_real.so" ||
        ! -r "$host_root/runtime/$runtime_version/real/libcust_opapi_real.so" ]]; then
    echo "REFUSE: scoped opapi/custom-opapi proxy or its real library is missing" >&2
    exit 1
  fi
  runtime_prefix="export LD_LIBRARY_PATH='$runtime_dir:$runtime_dir/real':\$LD_LIBRARY_PATH &&"
  custom_opapi_container="$runtime_dir/real/libcust_opapi_real.so"
  custom_proxy_mount_args=(
    -v "$host_root/runtime/$runtime_version/libcust_opapi.so:$custom_opapi_image_path:ro"
  )
  if [[ "$preload_opapi_proxy" == 1 ]]; then
    # CANN 8.5.1 may call the split libopapi_nn.so directly.  Putting the
    # typed metadata adapter in LD_PRELOAD establishes symbol precedence for
    # that path; LD_LIBRARY_PATH alone is not sufficient.
    preload="$runtime_dir/libopapi.so:$preload"
  fi
  if [[ "$use_split_opapi_nn_proxy" == 1 ]]; then
    if [[ ! -r "$host_root/runtime/$runtime_version/libopapi_nn.so" ||
          ! -r "$host_root/runtime/$runtime_version/real/libopapi_nn_real.so" ]]; then
      echo "REFUSE: split opapi_nn proxy or real library is missing" >&2
      exit 1
    fi
    split_opapi_mount_args=(
      -v "$host_root/runtime/$runtime_version/libopapi_nn.so:/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libopapi_nn.so:ro"
    )
  fi
fi

if [[ "$use_typed_extension" == 1 ]]; then
  if [[ ! -r "$host_root/source/fault26_readonly_trace_v1/fault26_worker_extension.py" ]]; then
    echo "REFUSE: typed worker extension is missing" >&2
    exit 1
  fi
  pythonpath_prefix="export PYTHONPATH='$container_root/source/fault26_readonly_trace_v1':\$PYTHONPATH &&"
  worker_extension_arg="--worker-extension-cls fault26_worker_extension.Fault26ReadOnlyWorkerExtension"
fi

if [[ "$use_write_gate" == 1 ]]; then
  if [[ ! -r "$host_root/runtime/$runtime_version/libfault26_add_rms_lossless_abi0.so" ||
        ! -x "$host_root/runtime/$runtime_version/fault26_write_gate_ctl" ]]; then
    echo "REFUSE: Fault26 write-gate runtime or controller is missing" >&2
    exit 1
  fi
  preload="$runtime_dir/libfault26_add_rms_lossless_abi0.so:$preload"
fi

if [[ "$use_hccl_trace" == 1 ]]; then
  if [[ ! -r "$host_root/runtime/$runtime_version/libfault26_hccl_trace_abi0.so" ]]; then
    echo "REFUSE: Fault26 HCCL trace runtime is missing" >&2
    exit 1
  fi
  preload="$runtime_dir/libfault26_hccl_trace_abi0.so:$preload"
fi

if [[ "$use_qmm_residual_stomp" == 1 ]]; then
  if [[ ! -r "$host_root/runtime/$runtime_version/libfault26_qmm_residual_stomp_abi0.so" ||
        ! -x "$host_root/runtime/$runtime_version/fault26_qmm_residual_stomp_ctl" ]]; then
    echo "REFUSE: Fault26 QMM residual stomp runtime or controller is missing" >&2
    exit 1
  fi
  preload="$runtime_dir/libfault26_qmm_residual_stomp_abi0.so:$preload"
fi

if [[ "$use_fia_writer_proxy" == 1 ]]; then
  if [[ ! -r "$host_root/runtime/$runtime_version/libfault26_fia_writer_proxy_abi0.so" ]]; then
    echo "REFUSE: Fault26 FIA writer runtime is missing" >&2
    exit 1
  fi
  preload="$runtime_dir/libfault26_fia_writer_proxy_abi0.so:$preload"
fi
if [[ "$use_qmm_underflow" == 1 ]]; then
  if [[ ! -r "$host_root/runtime/$runtime_version/libfault26_qmm_output_underflow_abi0.so" ]]; then
    echo "REFUSE: QMM output underflow runtime is missing" >&2
    exit 1
  fi
  preload="$runtime_dir/libfault26_qmm_output_underflow_abi0.so:$preload"
fi

docker -H "$docker_host" run -d \
  --name "$container" \
  --net=host \
  --shm-size=16g \
  --device /dev/davinci2:/dev/davinci2 \
  --device /dev/davinci5:/dev/davinci5 \
  --device /dev/davinci_manager:/dev/davinci_manager \
  --device /dev/devmm_svm:/dev/devmm_svm \
  --device /dev/hisi_hdc:/dev/hisi_hdc \
  -v "$host_root:$container_root" \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$model_host:$model_container:ro" \
  "${custom_proxy_mount_args[@]}" \
  "${split_opapi_mount_args[@]}" \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e FAULT26_TRACE_ENABLED=1 \
  -e FAULT26_TRACE_ALL_APIS="$trace_all_apis" \
  -e FAULT26_TRACE_FILTER="$trace_filter" \
  -e FAULT26_TRACE_CAPTURE_TENSORS="$capture_tensors" \
  -e FAULT26_TRACE_MAX_EVENTS="$max_events" \
  -e FAULT26_TRACE_LOG_PATH="$container_log" \
  -e FAULT26_TYPED_PROBE_ALLOWED="$use_typed_extension" \
  -e FAULT26_TYPED_LOG_PATH="$container_typed_log" \
  -e FAULT26_WRITE_GATE_ALLOWED="$use_write_gate" \
  -e FAULT26_WRITE_GATE_LOG_PATH="$container_write_gate_log" \
  -e FAULT20_METADATA_LOG_PATH="$container_qmm_log" \
  -e FAULT26_HCCL_TRACE_ALLOWED="$use_hccl_trace" \
  -e FAULT26_HCCL_TRACE_LOG_PATH="$container_hccl_log" \
  -e FAULT26_HCCL_TRACE_MAX_EVENTS=200000 \
  -e FAULT26_QMM_STOMP_ALLOWED="$use_qmm_residual_stomp" \
  -e FAULT26_QMM_STOMP_LOG_PATH="$container_qmm_stomp_log" \
  -e FAULT26_FIA_WRITER_PROXY_ALLOWED="$use_fia_writer_proxy" \
  -e FAULT26_FIA_WRITER_PROXY_ENABLED=0 \
  -e FAULT26_FIA_WRITER_PROXY_SESSION=0 \
  -e FAULT26_FIA_WRITER_PROXY_MODE=lossless_output_substitution \
  -e FAULT26_FIA_WRITER_PROXY_TARGET_RESIDUAL_BYTES=0 \
  -e FAULT26_FIA_WRITER_PROXY_VICTIM_OFFSET_BYTES=0 \
  -e FAULT26_FIA_WRITER_PROXY_MAX_BYTES=4194304 \
  -e FAULT26_FIA_WRITER_LOG_PATH="$container_fia_writer_log" \
  -e FAULT26_WRITER_AUDIT_ALLOWED="$use_writer_audit" \
  -e FAULT26_WRITER_AUDIT_ENABLED=0 \
  -e FAULT26_WRITER_AUDIT_SESSION=0 \
  -e FAULT26_WRITER_AUDIT_MAX_QMM_WINDOWS=4096 \
  -e FAULT26_WRITER_AUDIT_MAX_BYTES=4194304 \
  -e FAULT26_WRITER_TOPOLOGY_ENABLED="$use_writer_topology" \
  -e FAULT26_ALLOCATOR_AUDIT_ALLOWED="$use_allocator_audit" \
  -e FAULT26_QMM_UNDERFLOW_ALLOWED="$use_qmm_underflow" \
  -e FAULT26_QMM_UNDERFLOW_ENABLED=0 \
  -e FAULT26_QMM_UNDERFLOW_LOG_PATH="$container_qmm_underflow_log" \
  -e FAULT26_QMM_UNDERFLOW_TARGET_DEVICE=0 \
  -e FAULT26_QMM_UNDERFLOW_TARGET_OUTPUT_ADDRESS=0 \
  -e FAULT26_QMM_UNDERFLOW_TARGET_OUTPUT_BYTES=0 \
  -e FAULT26_QMM_UNDERFLOW_TARGET_RESIDUAL_ADDRESS=0 \
  -e FAULT26_QMM_UNDERFLOW_TARGET_RESIDUAL_BYTES=0 \
  -e FAULT26_QMM_UNDERFLOW_RESIDUAL_ALLOCATION_END=0 \
  -e FAULT26_QMM_UNDERFLOW_OUTPUT_ALLOCATION_BEGIN=0 \
  -e FAULT26_QMM_UNDERFLOW_SHIFT_BYTES=0 \
  -e FAULT26_QMM_UNDERFLOW_MAX_HITS=1 \
  -e FAULT26_QMM_UNDERFLOW_TARGET_PAIRS= \
  -e FAULT26_QMM_UNDERFLOW_ALL_HITS=0 \
  -e FAULT26_REAL_CUSTOM_OPAPI="$custom_opapi_container" \
  -e LD_PRELOAD="$preload" \
  --entrypoint /bin/bash \
  "$image" \
  -lc "source /usr/local/Ascend/ascend-toolkit/set_env.sh && $runtime_prefix $pythonpath_prefix exec vllm serve '$model_container' \
    --served-model-name qwen3-32b-w8a8-fault26 \
    --host 127.0.0.1 \
    --port '$server_port' \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --distributed-executor-backend mp \
    --trust-remote-code \
    --enforce-eager \
    --quantization ascend \
    --max-model-len 4096 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.90 \
    $worker_extension_arg"

printf 'container=%s\n' "$container"
printf 'inventory_log=%s\n' "$host_log"
if [[ "$use_typed_extension" == 1 ]]; then
  printf 'typed_log=%s\n' "$typed_log"
fi
if [[ "$use_write_gate" == 1 ]]; then
  printf 'write_gate_log=%s\n' "$write_gate_log"
fi
printf 'qmm_log=%s\n' "$qmm_log"
if [[ "$use_hccl_trace" == 1 ]]; then
  printf 'hccl_log=%s\n' "$hccl_log"
fi
if [[ "$use_qmm_residual_stomp" == 1 ]]; then
  printf 'qmm_stomp_log=%s\n' "$qmm_stomp_log"
fi
if [[ "$use_fia_writer_proxy" == 1 ]]; then
  printf 'fia_writer_log=%s\n' "$fia_writer_log"
fi
if [[ "$use_qmm_underflow" == 1 ]]; then
  printf 'qmm_underflow_log=%s\n' "$qmm_underflow_log"
fi
printf 'health_url=http://127.0.0.1:%s/health\n' "$server_port"
