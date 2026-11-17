#!/usr/bin/env bash
# Shared immutable build procedure; QMM and FIA keep distinct recipes/receipts.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
toolkit="${DCAT_BUILD_TOOLKIT:?actual inspected toolkit required}"
out="${DCAT26_BUILD:?fresh build directory required}"
custom="${DCAT_BUILD_CUSTOM_LIBRARY:?installed custom opapi required}"
header="${DCAT_BUILD_CUSTOM_HEADER:?matching custom API header required}"
[[ ! -e "$out" ]] || exit 1
grep -Eq '^version=8\.5\.1$' "$toolkit/ascend_toolkit_install.info"
[[ "$(uname -m)" == aarch64 ]]
mkdir -p "$out"
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -fPIC -D_GLIBCXX_USE_CXX11_ABI=0
 -I"$toolkit/include" -I"$toolkit/include/aclnn" -I"$toolkit/pkg_inc" -I"$(dirname "$header")")
g++ "${flags[@]}" "$root/fault_injection/native_build/fault26_abi_probe.cpp" -o "$out/abi_probe"
"$out/abi_probe" > "$out/abi_facts.json"
export FAULT26_ASCEND_TOOLKIT="$toolkit" FAULT26_TRACE_OUTPUT_DIR="$out"
export FAULT26_REAL_LIBOPAPI="$toolkit/lib64/libopapi.so" FAULT26_REAL_LIBOPAPI_NN="$toolkit/lib64/libopapi_nn.so"
export FAULT26_REAL_CUSTOM_OPAPI="$custom" FAULT26_USE_TYPED_QMM_PROXY_BUILD=1
export FAULT26_EXPECTED_LIBRARY_PATH_FRAGMENT="${toolkit%/aarch64-linux}/"
bash "$root/fault20_fast_validation/source/fault26_readonly_trace_v1/build.sh"
g++ "${flags[@]}" -shared -DDEMONCAT_NATIVE_PROVIDER_BRIDGE \
 "$root/fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_qmm_output_underflow.cpp" \
 -Wl,-soname,libfault26_qmm_output_underflow_abi0.so -ldl -lpthread -o "$out/libfault26_qmm_output_underflow_abi0.so"
g++ "${flags[@]}" -shared -DDEMONCAT_NATIVE_PROVIDER_BRIDGE \
 "$root/fault20_fast_validation/source/fault26_fia_writer_proxy_v1/fault26_fia_writer_lossless.cpp" \
 -Wl,-soname,libfault26_fia_writer_proxy_abi0.so -ldl -lpthread -o "$out/libfault26_fia_writer_proxy_abi0.so"
