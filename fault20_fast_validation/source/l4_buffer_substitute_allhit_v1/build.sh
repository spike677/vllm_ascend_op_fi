#!/usr/bin/env bash
set -euo pipefail

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${FAULT20_BUFFER_OUTPUT_DIR:-/fault20/runtime/l4_buffer_substitute}"
toolkit="${FAULT20_ASCEND_TOOLKIT:-/usr/local/Ascend/ascend-toolkit/8.3.RC2}"
expected_library_fragment="${FAULT20_EXPECTED_LIBRARY_PATH_FRAGMENT:-/8.3.RC2/}"
library="$output_dir/libfault20_buffer_substitute_abi0.so"
control="$output_dir/fault20_buffer_ctl"

install -d -m 0755 "$output_dir"

g++ -std=c++17 -O2 -fPIC -shared \
  -D_GLIBCXX_USE_CXX11_ABI=0 \
  -DFAULT20_EXPECTED_LIBRARY_PATH_FRAGMENT="\"$expected_library_fragment\"" \
  -I"$toolkit/include" \
  -I"$toolkit/include/aclnn" \
  -I"$toolkit/pkg_inc" \
  "$source_dir/buffer_substitute_preload.cpp" \
  -Wl,-soname,libfault20_buffer_substitute_abi0.so \
  -ldl -lpthread -lrt \
  -o "$library"

g++ -std=c++17 -O2 \
  "$source_dir/buffer_control.cpp" \
  -lpthread -lrt \
  -o "$control"

readelf -d "$library" | grep -E 'SONAME|NEEDED'
if readelf -d "$library" | grep -Eq 'libopapi|libnnopbase|libascendcl'; then
  echo 'REFUSE: interposer unexpectedly has a direct CANN DT_NEEDED dependency' >&2
  exit 1
fi
nm -D -C "$library" | grep -F 'op::internal::OpExecCache::Run('
nm -D -C "$library" | grep -F 'op::internal::AddOpExecCache('
nm -D -C "$library" | grep -F 'op::internal::AddParamToBuf(aclTensor const*)'
nm -D -C "$library" | grep -F 'op::internal::AddParamToBuf(aclTensor*)'
nm -D "$library" | grep -E 'fault20_qmm_gws_(begin|commit|abort)'
nm -D "$library" | grep -E 'fault20_(float_)?matmul_gws_(begin|commit|abort)'
LD_BIND_NOW=1 python3 -c 'import ctypes, os, sys; ctypes.CDLL(sys.argv[1], mode=os.RTLD_LOCAL)' "$library"
"$control" reset

echo "$library"
echo "$control"
echo "expected_library_fragment=$expected_library_fragment"
