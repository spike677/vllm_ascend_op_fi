#!/usr/bin/env bash
set -euo pipefail

bundle="${1:?usage: verify_v17_bundle.sh BUNDLE_DIR}"
manifest="$bundle/manifest.json"
[[ -f "$manifest" ]] || {
  echo "BUNDLE_VERIFY_FAIL: manifest missing: $manifest" >&2
  exit 3
}

python3 - "$bundle" "$manifest" <<'PY'
import hashlib
import json
import os
import sys

bundle, manifest_path = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as f:
    manifest = json.load(f)
expected = {
    "schema": "fault20-v17-bundle-manifest-1",
    "fault20_control_magic": "0x4632304255463137",
    "fault20_control_abi": 17,
    "metadata_abi_magic": "0x4632304d45543137",
    "metadata_abi": 17,
    "metadata_descriptor_size": 32,
    "cann": "8.5.1",
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(
            f"BUNDLE_VERIFY_FAIL: {key}={manifest.get(key)!r}, expected {value!r}")
required = {
    "preload", "ctl", "metadata_proxy", "metadata_proxy_real",
    "verifier", "launcher",
}
if set(manifest.get("files", {})) != required:
    raise SystemExit("BUNDLE_VERIFY_FAIL: manifest file set is incomplete")
for name, item in manifest["files"].items():
    rel = item.get("path", "")
    path = os.path.realpath(os.path.join(bundle, rel))
    root = os.path.realpath(bundle) + os.sep
    if not path.startswith(root) or not os.path.isfile(path):
        raise SystemExit(f"BUNDLE_VERIFY_FAIL: invalid path for {name}: {rel}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != item.get("sha256"):
        raise SystemExit(
            f"BUNDLE_VERIFY_FAIL: sha256 mismatch for {name}: {actual}")
print("BUNDLE_HASHES_OK")
PY

preload="$bundle/preload/libfault20_buffer_substitute_abi0.so"
proxy="$bundle/proxy/libopapi.so"
ctl="$bundle/ctl/fault20_buffer_ctl"

nm -D --defined-only "$preload" | grep -Fq 'fault20_metadata_runtime_abi_v17' || {
  echo 'BUNDLE_VERIFY_FAIL: preload metadata ABI identity missing' >&2
  exit 3
}
for symbol in fault20_qmm_gws_commit_v17 fault20_matmul_gws_commit_v17 \
  fault20_float_matmul_gws_commit_v17; do
  nm -D --defined-only "$preload" | grep -Fq "$symbol" || {
    echo "BUNDLE_VERIFY_FAIL: preload symbol missing: $symbol" >&2
    exit 3
  }
done
nm -D --defined-only "$proxy" | grep -Fq 'fault20_metadata_proxy_abi_v17' || {
  echo 'BUNDLE_VERIFY_FAIL: proxy metadata ABI identity missing' >&2
  exit 3
}

status="$(VLLM_SERVER_DEV_MODE=1 "$ctl" status)"
python3 - "$status" <<'PY'
import json
import sys
d = json.loads(sys.argv[1])
if d.get("magic") != "0x4632304255463137" or d.get("version") != 17:
    raise SystemExit("BUNDLE_VERIFY_FAIL: ctl control ABI mismatch")
print("BUNDLE_CONTROL_ABI_OK")
PY

if [[ -n "${FAULT20_CANN_ROOT:-}" ]]; then
  [[ -d "$FAULT20_CANN_ROOT" ]] || {
    echo "BUNDLE_VERIFY_FAIL: CANN root missing: $FAULT20_CANN_ROOT" >&2
    exit 3
  }
  version_file="$FAULT20_CANN_ROOT/opp/version.info"
  grep -Eq '^Version=8\.5\.1([[:space:]]*)$' "$version_file" || {
    echo "BUNDLE_VERIFY_FAIL: expected CANN 8.5.1 in $version_file" >&2
    exit 3
  }
fi

echo "FAULT20_V17_BUNDLE_VERIFY_PASS"
