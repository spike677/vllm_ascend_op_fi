#!/usr/bin/env bash
set -euo pipefail
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${FAULT39_OUTPUT_DIR:-$source_dir/build}"
mkdir -p "$output_dir"
g++ -std=c++17 -O2 -Wall -Wextra -pedantic "$source_dir/test_core.cpp" -o "$output_dir/test_core"
"$output_dir/test_core"
if [[ "${1:-}" == --cpu-test ]]; then exit 0; fi
toolkit="${FAULT39_ASCEND_TOOLKIT:-/usr/local/Ascend/cann-8.5.1/aarch64-linux}"
case "$toolkit/" in */8.5.1/*|*/cann-8.5.1/*) ;; *) echo 'REFUSE: Fault39 supports real CANN 8.5.1 only' >&2; exit 1;; esac
[[ "$(uname -m)" == aarch64 ]] || { echo 'REFUSE: native build requires aarch64' >&2; exit 1; }
real_source="${FAULT39_REAL_LIBOPAPI:-$toolkit/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so}"
if [[ ! -f "$real_source" ]]; then
 real_source="$(dirname "$toolkit")/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so"
fi
[[ -f "$real_source" ]] || { echo 'REFUSE: supply FAULT39_REAL_LIBOPAPI for the 8.5.1 installation' >&2; exit 1; }
resolved_real="$(realpath "$real_source")"
case "$resolved_real" in */8.5.1/*|*/cann-8.5.1/*) ;; *) echo 'REFUSE: real libopapi is not from 8.5.1' >&2; exit 1;; esac
flags=(-std=c++17 -O2 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=0 -I"$toolkit/include" -I"$toolkit/include/aclnn" -I"$toolkit/pkg_inc")
g++ "${flags[@]}" "$source_dir/runtime.cpp" -Wl,-soname,libfault39_native_v1.so -ldl -lpthread -o "$output_dir/libfault39_native_v1.so"
mkdir -p "$output_dir/real"
install -m 0755 "$real_source" "$output_dir/real/libopapi_real.so"
patchelf --set-soname libopapi_real.so "$output_dir/real/libopapi_real.so"
g++ "${flags[@]}" "$source_dir/metadata_proxy.cpp" -L"$output_dir/real" -Wl,--no-as-needed -l:libopapi_real.so \
 -Wl,-soname,libopapi.so -Wl,-rpath,'$ORIGIN/real' -Wl,-z,origin -ldl -lpthread -o "$output_dir/libopapi.so"
# The installed torch-npu 2.9 resolves aclnnMatmul through libopapi_nn.so.
# This inherits the historical 8.5.1 split-library proxy loading pattern.
split_source="$toolkit/lib64/libopapi_nn.so"
[[ -f "$split_source" ]] || { echo 'REFUSE: matching 8.5.1 split opapi missing' >&2; exit 1; }
install -m 0755 "$split_source" "$output_dir/real/libopapi_nn_real.so"
patchelf --set-soname libopapi_nn_real.so "$output_dir/real/libopapi_nn_real.so"
g++ "${flags[@]}" '-DFAULT39_REAL_LIBRARY="libopapi_nn_real.so"' "$source_dir/metadata_proxy.cpp" \
 -L"$output_dir/real" -Wl,--no-as-needed -l:libopapi_nn_real.so -Wl,-soname,libopapi_nn.so \
 -Wl,-rpath,'$ORIGIN/real:/native/real' -Wl,-z,origin -ldl -lpthread -o "$output_dir/libopapi_nn.so"
if readelf -d "$output_dir/libfault39_native_v1.so" | grep -Eq 'libopapi|libnnopbase|libascendcl'; then
 echo 'REFUSE: native runtime unexpectedly links CANN directly' >&2;exit 1
fi
nm -D -C "$output_dir/libfault39_native_v1.so" | grep -F 'op::internal::OpExecCache::Run('
nm -D -C "$output_dir/libfault39_native_v1.so" | grep -F 'op::internal::AddOpExecCache('
nm -D -C "$output_dir/libfault39_native_v1.so" | grep -F 'op::internal::AddParamToBuf('
nm -D "$output_dir/libfault39_native_v1.so" | grep -E 'fault39_(abi_version|configure|disable|clean|snapshot|gws_commit)_v1'
if nm -D "$output_dir/libfault39_native_v1.so" | grep -Eq 'fault(20|26|28)_'; then echo 'REFUSE: unexpected old fault ABI' >&2;exit 1;fi
LD_BIND_NOW=1 python3 - "$output_dir/libfault39_native_v1.so" <<'PY'
import ctypes,sys
lib=ctypes.CDLL(sys.argv[1]); assert lib.fault39_abi_version_v1()==1
PY
python3 "$source_dir/test_control.py" "$output_dir/libfault39_native_v1.so"
printf '%s\n' "$output_dir/libfault39_native_v1.so" "$output_dir/libopapi.so"
