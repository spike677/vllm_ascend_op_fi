#!/usr/bin/env bash
set -euo pipefail

preload_dir="${FAULT20_PRELOAD_DIR:?FAULT20_PRELOAD_DIR is required}"
proxy_dir="${FAULT20_PROXY_DIR:?FAULT20_PROXY_DIR is required}"
source_root="${FAULT20_SOURCE_ROOT:?FAULT20_SOURCE_ROOT is required}"
bundle="${FAULT20_BUNDLE_DIR:?FAULT20_BUNDLE_DIR is required}"
harness_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ "$bundle" == /home/* ]] || {
  echo "REFUSE: FAULT20_BUNDLE_DIR must be under /home" >&2
  exit 2
}
[[ ! -e "$bundle" ]] || {
  echo "REFUSE: bundle already exists: $bundle" >&2
  exit 2
}

for path in \
  "$preload_dir/libfault20_buffer_substitute_abi0.so" \
  "$preload_dir/fault20_buffer_ctl" \
  "$proxy_dir/libopapi.so" \
  "$proxy_dir/real/libopapi_real.so" \
  "$source_root/l4_buffer_substitute_allhit_v1/fault20_buffer_api.h" \
  "$source_root/l4_buffer_substitute_allhit_v1/buffer_control.cpp" \
  "$source_root/l4_buffer_substitute_allhit_v1/buffer_substitute_preload.cpp" \
  "$source_root/l4_metadata_lifecycle_proxy/libopapi_metadata_proxy.cpp" \
  "$harness_dir/verify_v17_bundle.sh" \
  "$harness_dir/run_v17_bundle_rootfs.sh"; do
  [[ -f "$path" ]] || {
    echo "REFUSE: required input missing: $path" >&2
    exit 2
  }
done

staging="${bundle}.staging.$$"
[[ ! -e "$staging" ]] || {
  echo "REFUSE: staging path already exists: $staging" >&2
  exit 2
}
install -d -m 0755 "$staging/preload" "$staging/ctl" "$staging/proxy/real" \
  "$staging/harness"
install -m 0555 "$preload_dir/libfault20_buffer_substitute_abi0.so" \
  "$staging/preload/libfault20_buffer_substitute_abi0.so"
install -m 0555 "$preload_dir/fault20_buffer_ctl" \
  "$staging/ctl/fault20_buffer_ctl"
install -m 0555 "$proxy_dir/libopapi.so" "$staging/proxy/libopapi.so"
install -m 0555 "$proxy_dir/real/libopapi_real.so" \
  "$staging/proxy/real/libopapi_real.so"
install -m 0555 "$harness_dir/verify_v17_bundle.sh" \
  "$staging/harness/verify_v17_bundle.sh"
install -m 0555 "$harness_dir/run_v17_bundle_rootfs.sh" \
  "$staging/harness/run_v17_bundle_rootfs.sh"

python3 - "$staging" "$source_root" <<'PY'
import datetime
import hashlib
import json
import os
import sys

bundle, source_root = sys.argv[1:]

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

files = {
    "preload": "preload/libfault20_buffer_substitute_abi0.so",
    "ctl": "ctl/fault20_buffer_ctl",
    "metadata_proxy": "proxy/libopapi.so",
    "metadata_proxy_real": "proxy/real/libopapi_real.so",
    "verifier": "harness/verify_v17_bundle.sh",
    "launcher": "harness/run_v17_bundle_rootfs.sh",
}
source_files = {
    "fault20_buffer_api.h":
        "l4_buffer_substitute_allhit_v1/fault20_buffer_api.h",
    "buffer_control.cpp":
        "l4_buffer_substitute_allhit_v1/buffer_control.cpp",
    "buffer_substitute_preload.cpp":
        "l4_buffer_substitute_allhit_v1/buffer_substitute_preload.cpp",
    "metadata_proxy_source":
        "l4_metadata_lifecycle_proxy/libopapi_metadata_proxy.cpp",
}
manifest = {
    "schema": "fault20-v17-bundle-manifest-1",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "fault20_control_magic": "0x4632304255463137",
    "fault20_control_abi": 17,
    "metadata_abi_magic": "0x4632304d45543137",
    "metadata_abi": 17,
    "metadata_descriptor_size": 32,
    "cann": os.environ.get("FAULT20_CANN_VERSION", "8.5.1"),
    "vllm": os.environ.get("FAULT20_VLLM_VERSION", "0.18.0+empty"),
    "vllm_ascend": os.environ.get("FAULT20_VLLM_ASCEND_VERSION", "0.18.0rc1"),
    "torch_npu": os.environ.get(
        "FAULT20_TORCH_NPU_VERSION", "2.9.0.post1+gitee7ba04"),
    "files": {
        name: {"path": rel, "sha256": digest(os.path.join(bundle, rel))}
        for name, rel in files.items()
    },
    "source_hashes": {
        name: digest(os.path.join(source_root, rel))
        for name, rel in source_files.items()
    },
}
with open(os.path.join(bundle, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
PY

mv "$staging" "$bundle"
chmod -R a-w "$bundle"
"$bundle/harness/verify_v17_bundle.sh" "$bundle"
echo "BUNDLE_CREATED=$bundle"
