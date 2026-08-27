#!/usr/bin/env bash
set -euo pipefail

if [[ "${FAULT20_IN_MOUNT_NS:-0}" != 1 ]]; then
  exec unshare --mount --propagation private env FAULT20_IN_MOUNT_NS=1 "$0" "$@"
fi

rootfs="${FAULT20_ROOTFS:?FAULT20_ROOTFS is required}"
run_dir="${FAULT20_RUN_DIR:?FAULT20_RUN_DIR is required}"
model_host="${FAULT20_HOST_MODEL_PATH:?FAULT20_HOST_MODEL_PATH is required}"
bundle="${FAULT20_BUNDLE:?FAULT20_BUNDLE is required}"
physical_devices="${FAULT20_PHYSICAL_DEVICES:?FAULT20_PHYSICAL_DEVICES is required}"
logical_devices="${FAULT20_ASCEND_RT_VISIBLE_DEVICES:?FAULT20_ASCEND_RT_VISIBLE_DEVICES is required}"
port="${FAULT20_PORT:-18006}"
model_name="${FAULT20_MODEL_NAME:-qwen3-32b-w8a8}"
min_home_kb="${FAULT20_MIN_HOME_FREE_KB:-8388608}"

service_pid=0
cleanup_started=0

control_reset() {
  VLLM_SERVER_DEV_MODE=1 "$bundle/ctl/fault20_buffer_ctl" disarm >/dev/null 2>&1 || true
  VLLM_SERVER_DEV_MODE=1 "$bundle/ctl/fault20_buffer_ctl" reset >/dev/null 2>&1 || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$cleanup_started" == 1 ]]; then
    exit "$status"
  fi
  cleanup_started=1
  set +e
  control_reset
  if [[ "$service_pid" -gt 1 ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill -TERM -- "-$service_pid" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
      kill -0 "$service_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$service_pid" 2>/dev/null; then
      kill -KILL -- "-$service_pid" 2>/dev/null
    fi
    wait "$service_pid" 2>/dev/null
  fi
  control_reset
  ss -H -ltn "sport = :$port" >"$run_dir/port_after.txt" 2>&1
  npu-smi info >"$run_dir/npu_smi_after.txt" 2>&1
  echo "HARNESS_CLEANUP_COMPLETE status=$status" >>"$run_dir/harness.log"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "$(id -u)" == 0 ]] || {
  echo "PREFLIGHT_FAIL: root privileges are required" >&2
  exit 4
}
for path in "$rootfs" "$run_dir" "$model_host" "$bundle"; do
  [[ "$path" == /home/* ]] || {
    echo "PREFLIGHT_FAIL: all large/runtime paths must be under /home: $path" >&2
    exit 4
  }
done
[[ -d "$rootfs" && -x "$rootfs/bin/bash" ]] || {
  echo "PREFLIGHT_FAIL: invalid rootfs: $rootfs" >&2
  exit 4
}
[[ -f "$rootfs/usr/local/Ascend/cann-8.5.1/opp/version.info" ]] || {
  echo "PREFLIGHT_FAIL: CANN 8.5.1 not found in rootfs" >&2
  exit 4
}
[[ -d "$model_host" && -f "$model_host/config.json" ]] || {
  echo "PREFLIGHT_FAIL: model/config.json missing: $model_host" >&2
  exit 4
}
[[ ! -e "$run_dir" ]] || {
  echo "PREFLIGHT_FAIL: run directory already exists: $run_dir" >&2
  exit 4
}
[[ ${#run_dir} -lt 100 ]] || {
  echo "PREFLIGHT_FAIL: host run path is unnecessarily long" >&2
  exit 4
}
[[ ${#port} -le 5 && "$port" =~ ^[0-9]+$ ]] || {
  echo "PREFLIGHT_FAIL: invalid port: $port" >&2
  exit 4
}
if ss -H -ltn "sport = :$port" | grep -q .; then
  echo "PREFLIGHT_FAIL: port is already listening: $port" >&2
  exit 4
fi

IFS=',' read -r -a physical <<<"$physical_devices"
IFS=',' read -r -a logical <<<"$logical_devices"
[[ "${#physical[@]}" -eq 2 && "${#logical[@]}" -eq 2 ]] || {
  echo "PREFLIGHT_FAIL: exactly two physical and two logical devices are required" >&2
  exit 4
}
for device in "${physical[@]}"; do
  [[ "$device" =~ ^[0-9]+$ && -c "/dev/davinci$device" ]] || {
    echo "PREFLIGHT_FAIL: physical device node missing: /dev/davinci$device" >&2
    exit 4
  }
done
for device in "${logical[@]}"; do
  [[ "$device" =~ ^[0-9]+$ ]] || {
    echo "PREFLIGHT_FAIL: invalid logical device: $device" >&2
    exit 4
  }
done

home_free_kb="$(df -Pk /home | awk 'NR==2 {print $4}')"
[[ "$home_free_kb" -ge "$min_home_kb" ]] || {
  echo "PREFLIGHT_FAIL: /home free space ${home_free_kb}KB < ${min_home_kb}KB" >&2
  exit 4
}

FAULT20_CANN_ROOT="$rootfs/usr/local/Ascend/cann-8.5.1" \
  "$bundle/harness/verify_v17_bundle.sh" "$bundle"

install -d -m 0755 "$run_dir" "$run_dir/t" "$run_dir/h"
df -h / /home >"$run_dir/df_before.txt"
npu-smi info >"$run_dir/npu_smi_before.txt"
python3 - "$run_dir/npu_smi_before.txt" "$physical_devices" <<'PY'
import re
import sys
path, requested = sys.argv[1:]
targets = {int(x) for x in requested.split(',')}
in_process_table = False
busy = []
for line in open(path, encoding='utf-8', errors='replace'):
    if 'Process id' in line and 'NPU' in line:
        in_process_table = True
        continue
    if not in_process_table:
        continue
    m = re.match(r'\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|', line)
    if m and int(m.group(1)) in targets:
        busy.append((int(m.group(1)), int(m.group(2))))
if busy:
    raise SystemExit('PREFLIGHT_FAIL: target NPU busy: ' + repr(busy))
print('PREFLIGHT_NPU_IDLE')
PY

internal_tmp="/r/t"
worst_ipc="ipc://${internal_tmp}/00000000-0000-0000-0000-000000000000"
[[ ${#worst_ipc} -lt 100 ]] || {
  echo "PREFLIGHT_FAIL: internal AF_UNIX path budget exceeded: $worst_ipc" >&2
  exit 4
}

python3 - "$run_dir/environment.json" "$rootfs" "$model_host" "$bundle" \
  "$run_dir" "$physical_devices" "$logical_devices" "$port" \
  "${FAULT20_ROOTFS_SOURCE_CONTAINER:-unknown}" <<'PY'
import json
import sys
(_, output, rootfs, model, bundle, run_dir, physical_csv, logical_csv,
 port, source_container) = sys.argv
physical = physical_csv.split(',')
logical = logical_csv.split(',')
mapping = []
for rank, (host, visible) in enumerate(zip(physical, logical)):
    mapping.append({"host_physical_device": int(host),
                    "visible_logical_device": int(visible),
                    "tp_rank": rank})
data = {
    "rootfs": rootfs,
    "rootfs_source_container": source_container,
    "host_model_path": model,
    "bundle": bundle,
    "run_dir": run_dir,
    "internal_paths": {"bundle": "/f20b", "model": "/m", "run": "/r",
                       "tmp": "/r/t", "hf_home": "/r/h"},
    "port": int(port),
    "developer_mode": 1,
    "device_mapping": mapping,
}
with open(output, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
PY

exec >>"$run_dir/harness.log" 2>&1
echo "PREFLIGHT_ROOT_DF=$(df -Pk / | awk 'NR==2 {print $5}')"
echo "PREFLIGHT_HOME_FREE_KB=$home_free_kb"
echo "PREFLIGHT_PORT=$port"
echo "PREFLIGHT_PHYSICAL_DEVICES=$physical_devices"
echo "PREFLIGHT_LOGICAL_DEVICES=$logical_devices"
echo "PREFLIGHT_INTERNAL_IPC_PATH_LENGTH=${#worst_ipc}"

control_reset
VLLM_SERVER_DEV_MODE=1 "$bundle/ctl/fault20_buffer_ctl" status \
  >"$run_dir/control_preflight.json"

mount --make-rprivate /
for path in proc sys dev m f20b r usr/local/Ascend/driver usr/local/dcmi usr/local/bin; do
  install -d -m 0755 "$rootfs/$path"
done
[[ -e "$rootfs/usr/local/bin/npu-smi" ]] || install -m 0755 /dev/null "$rootfs/usr/local/bin/npu-smi"
[[ -e "$rootfs/etc/ascend_install.info" ]] || install -m 0644 /dev/null "$rootfs/etc/ascend_install.info"

mount -t proc proc "$rootfs/proc"
mount --rbind /sys "$rootfs/sys"
mount --make-rslave "$rootfs/sys"
mount --rbind /dev "$rootfs/dev"
mount --make-rslave "$rootfs/dev"
mount --rbind /usr/local/Ascend/driver "$rootfs/usr/local/Ascend/driver"
mount --make-rslave "$rootfs/usr/local/Ascend/driver"
mount --rbind /usr/local/dcmi "$rootfs/usr/local/dcmi"
mount --make-rslave "$rootfs/usr/local/dcmi"
mount --bind /usr/local/bin/npu-smi "$rootfs/usr/local/bin/npu-smi"
mount --bind /etc/ascend_install.info "$rootfs/etc/ascend_install.info"
mount --bind "$model_host" "$rootfs/m"
mount -o remount,bind,ro "$rootfs/m"
mount --bind "$bundle" "$rootfs/f20b"
mount -o remount,bind,ro "$rootfs/f20b"
mount --bind "$run_dir" "$rootfs/r"

setsid chroot "$rootfs" /usr/bin/env -i \
  PATH=/usr/local/python3.11.14/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  HOME=/root \
  VLLM_SERVER_DEV_MODE=1 \
  ASCEND_RT_VISIBLE_DEVICES="$logical_devices" \
  LD_LIBRARY_PATH=/f20b/proxy:/f20b/proxy/real \
  LD_PRELOAD=/f20b/preload/libfault20_buffer_substitute_abi0.so \
  FAULT20_METADATA_LOG_PATH=/r/metadata.jsonl \
  FAULT20_BUFFER_LOG_PATH=/r/buffer.jsonl \
  TMPDIR=/r/t \
  HF_HOME=/r/h \
  FAULT20_MODEL_PATH=/m \
  FAULT20_MODEL_NAME="$model_name" \
  FAULT20_PORT="$port" \
  /bin/bash -lc '
    set -euo pipefail
    source /usr/local/Ascend/cann-8.5.1/set_env.sh
    export LD_LIBRARY_PATH=/f20b/proxy:/f20b/proxy/real:$LD_LIBRARY_PATH
    exec vllm serve "$FAULT20_MODEL_PATH" \
      --served-model-name "$FAULT20_MODEL_NAME" \
      --host 127.0.0.1 \
      --port "$FAULT20_PORT" \
      --tensor-parallel-size 2 \
      --data-parallel-size 1 \
      --distributed-executor-backend mp \
      --trust-remote-code \
      --enforce-eager \
      --quantization ascend \
      --max-model-len 4096 \
      --max-num-seqs 2 \
      --gpu-memory-utilization 0.90
  ' >"$run_dir/server.log" 2>&1 &
service_pid=$!
echo "$$" >"$run_dir/harness.pid"
echo "$service_pid" >"$run_dir/service.pgid"
echo "HARNESS_SERVICE_STARTED pid=$service_pid"
wait "$service_pid"
