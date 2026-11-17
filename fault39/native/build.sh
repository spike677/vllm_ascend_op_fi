#!/usr/bin/env bash
set -euo pipefail
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${FAULT39_OUTPUT_DIR:-$source_dir/build}"
# Each build is immutable; do not leave stale proxies from another toolkit.
[[ ! -e "$output_dir" ]] || { echo 'REFUSE: use a new FAULT39_OUTPUT_DIR' >&2; exit 1; }
mkdir -p "$output_dir"
g++ -std=c++17 -O2 -Wall -Wextra -pedantic "$source_dir/test_core.cpp" -o "$output_dir/test_core"
"$output_dir/test_core"
python3 "$source_dir/test_metadata_ownership.py" --output "$output_dir/metadata_array_cpu_receipt.json"
if [[ "${1:-}" == --cpu-test ]]; then exit 0; fi
[[ "$(uname -m)" == aarch64 ]] || { echo 'REFUSE: native build requires aarch64' >&2; exit 1; }
toolkit="${FAULT39_ASCEND_TOOLKIT:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}"
if [[ -d "$toolkit/aarch64-linux/include" ]]; then toolkit="$toolkit/aarch64-linux"; fi
toolkit="$(realpath "$toolkit")"
[[ -f "$toolkit/ascend_toolkit_install.info" ]] || { echo 'REFUSE: actual toolkit metadata missing' >&2; exit 1; }
real_source="${FAULT39_REAL_LIBOPAPI:-$toolkit/lib64/libopapi.so}"
split_source="$toolkit/lib64/libopapi_nn.so"
# The normal lib64 symlink may resolve into the same installation's opp tree.
python3 - "$toolkit" "$real_source" "$split_source" <<'PY'
from pathlib import Path
import sys
toolkit=Path(sys.argv[1]).resolve(strict=True)
root=toolkit.parent if toolkit.name=='aarch64-linux' else toolkit
for name in [sys.argv[2], str(toolkit/'lib64/libnnopbase.so'), str(toolkit/'lib64/libascendcl.so')]+([sys.argv[3]] if Path(sys.argv[3]).exists() else []):
    path=Path(name).resolve(strict=True)
    if not path.is_relative_to(root): raise SystemExit('REFUSE: mixed toolkit libraries: '+str(path))
PY
flags=(-std=c++17 -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I"$toolkit/include" -I"$toolkit/include/aclnn" -I"$toolkit/pkg_inc" -I"$source_dir" -I"$output_dir")
g++ "${flags[@]}" "$source_dir/abi_probe.cpp" -o "$output_dir/abi_probe"
"$output_dir/abi_probe" > "$output_dir/abi_facts.json"
mkdir "$output_dir/real"
install -m 0755 "$real_source" "$output_dir/real/libopapi_real.so"
"${FAULT39_PATCHELF:-patchelf}" --set-soname libopapi_real.so "$output_dir/real/libopapi_real.so"
profile_args=(--toolkit "$toolkit" --real "$output_dir/real/libopapi_real.so" --source-opapi "$real_source" --output "$output_dir" --abi-facts "$output_dir/abi_facts.json")
if [[ -f "$split_source" ]]; then
 install -m 0755 "$split_source" "$output_dir/real/libopapi_nn_real.so"
 "${FAULT39_PATCHELF:-patchelf}" --set-soname libopapi_nn_real.so "$output_dir/real/libopapi_nn_real.so"
 profile_args+=(--split "$output_dir/real/libopapi_nn_real.so")
fi
python3 "$source_dir/../tools/version_profile.py" "${profile_args[@]}"
g++ "${flags[@]}" -shared "$source_dir/runtime.cpp" -Wl,-soname,libfault39_native_v1.so -ldl -lpthread -o "$output_dir/libfault39_native_v1.so"
g++ "${flags[@]}" -shared "$source_dir/metadata_proxy.cpp" -L"$output_dir/real" -Wl,--no-as-needed -l:libopapi_real.so \
 -Wl,-soname,libopapi.so -Wl,-rpath,'$ORIGIN/real' -Wl,-z,origin -ldl -lpthread -o "$output_dir/libopapi.so"
nm -D "$output_dir/libopapi.so" | grep -E '[[:space:]]fault39_metadata_arrays_v1$'
if [[ -f "$split_source" ]]; then
 g++ "${flags[@]}" -shared -DFAULT39_SPLIT_PROXY '-DFAULT39_REAL_LIBRARY="libopapi_nn_real.so"' "$source_dir/metadata_proxy.cpp" \
  -L"$output_dir/real" -Wl,--no-as-needed -l:libopapi_nn_real.so -Wl,-soname,libopapi_nn.so \
  -Wl,-rpath,'$ORIGIN/real:/native/real' -Wl,-z,origin -ldl -lpthread -o "$output_dir/libopapi_nn.so"
 nm -D "$output_dir/libopapi_nn.so" | grep -E '[[:space:]]fault39_metadata_arrays_v1$'
fi
if readelf -d "$output_dir/libfault39_native_v1.so" | grep -Eq 'libopapi|libnnopbase|libascendcl'; then
 echo 'REFUSE: native runtime unexpectedly links CANN directly' >&2;exit 1
fi
nm -D -C "$output_dir/libfault39_native_v1.so" | grep -F 'op::internal::OpExecCache::Run('
nm -D -C "$output_dir/libfault39_native_v1.so" | grep -F 'op::internal::AddOpExecCache('
nm -D -C "$output_dir/libfault39_native_v1.so" | grep -F 'op::internal::AddParamToBuf('
nm -D "$output_dir/libfault39_native_v1.so" | grep -E 'fault39_(abi_version|build_profile|configure|disable|clean|snapshot|gws_commit)_v1'
nm -D "$output_dir/libfault39_native_v1.so" | grep -E '[[:space:]]fault39_gws_commit_arrays_v1$'
if nm -D "$output_dir/libfault39_native_v1.so" | grep -Eq 'fault(20|26|28)_'; then echo 'REFUSE: unexpected old fault ABI' >&2;exit 1;fi
LD_BIND_NOW=1 python3 "$source_dir/test_control.py" "$output_dir/libfault39_native_v1.so"
python3 "$source_dir/test_identity.py" --output "$output_dir/identity_tests.json"
python3 "$source_dir/../tools/deployment_profile.py" --native "$output_dir" --write-manifest
printf '%s\n' 'BUILD_CHECKS_PASSED: real NPU discovery/lossless and B/F/OFF still required' "$output_dir/version_profile.json"
