#!/usr/bin/env bash
set -euo pipefail

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${FAULT20_METADATA_PROXY_OUTPUT_DIR:-/fault20/runtime/l4_metadata_lifecycle_proxy}"
real_dir="$output_dir/real"
toolkit="${FAULT20_ASCEND_TOOLKIT:-/usr/local/Ascend/ascend-toolkit/8.3.RC2}"
real_source="${FAULT20_REAL_LIBOPAPI:-$toolkit/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so}"

install -d -m 0755 "$output_dir" "$real_dir"
install -m 0755 "$real_source" "$real_dir/libopapi_real.so"
patchelf --set-soname libopapi_real.so "$real_dir/libopapi_real.so"

g++ -std=c++17 -O2 -fPIC -shared \
  -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I"$toolkit/include" \
  -I"$toolkit/include/aclnn" \
  -I"$toolkit/pkg_inc" \
  "$source_dir/libopapi_metadata_proxy.cpp" \
  -L"$real_dir" \
  -Wl,--no-as-needed -l:libopapi_real.so \
  -Wl,-soname,libopapi.so \
  -Wl,-rpath,'$ORIGIN/real' \
  -Wl,-z,origin \
  -ldl -lpthread \
  -o "$output_dir/libopapi.so"

readelf -d "$output_dir/libopapi.so" | grep -E 'SONAME|NEEDED|RUNPATH'
nm -D "$output_dir/libopapi.so" | grep -E 'aclnn(QuantMatmulWeightNz|MatmulWeightNz|Matmul|AscendQuantV3)(GetWorkspaceSize)?$'
nm -D "$output_dir/libopapi.so" | grep -F 'fault20_metadata_proxy_abi_v17'
if nm -D -C "$output_dir/libopapi.so" | grep -Fq 'aclOpExecutor::GetOpExecCache()'; then
  echo 'REFUSE: proxy must not depend on aclOpExecutor::GetOpExecCache()' >&2
  exit 1
fi

LD_LIBRARY_PATH="$output_dir:$real_dir:${LD_LIBRARY_PATH:-}" \
  python3 -c 'import ctypes, os, sys; lib=ctypes.CDLL(sys.argv[1], mode=os.RTLD_LOCAL); assert getattr(lib, "aclnnQuantMatmulWeightNz"); assert getattr(lib, "aclnnMatmulWeightNz"); assert getattr(lib, "aclnnMatmul"); assert getattr(lib, "aclnnAscendQuantV3")' \
  "$output_dir/libopapi.so"

echo "$output_dir/libopapi.so"
echo "real_libopapi=$real_source"
