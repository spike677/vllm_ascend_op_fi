#!/usr/bin/env bash
set -euo pipefail

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${FAULT26_TRACE_OUTPUT_DIR:-/fault39/runtime/fault26_qmm_output_underflow_v1}"
toolkit="${FAULT26_ASCEND_TOOLKIT:-/usr/local/Ascend/cann-8.5.1}"
runtime="$output_dir/libfault26_qmm_output_underflow_abi0.so"

install -d -m 0755 "$output_dir"

g++ -std=c++17 -O2 -fPIC -shared \
  -Wall -Wextra -Werror \
  -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I"$toolkit/include" \
  -I"$toolkit/include/aclnn" \
  -I"$toolkit/pkg_inc" \
  "$source_dir/fault26_qmm_output_underflow.cpp" \
  -Wl,-soname,libfault26_qmm_output_underflow_abi0.so \
  -ldl -lpthread \
  -o "$runtime"

symbols="$(nm -D "$runtime")"
for symbol in fault26_qmm_underflow_prepare \
              fault26_qmm_underflow_complete \
              fault26_qmm_underflow_consumer_complete; do
  if ! grep -Eq "[[:space:]]${symbol}$" <<<"$symbols"; then
    echo "REFUSE: missing underflow callback: $symbol" >&2
    exit 1
  fi
done

if grep -Eq 'aclrt(Malloc|MallocHost|Free|FreeHost|Memcpy|MemcpyAsync)' \
    <<<"$symbols"; then
  echo 'REFUSE: runtime must resolve ACL APIs dynamically' >&2
  exit 1
fi

readelf -d "$runtime" | grep -E 'SONAME|NEEDED'
LD_BIND_NOW=1 python3 -c \
  'import ctypes, os, sys; ctypes.CDLL(sys.argv[1], mode=os.RTLD_LOCAL)' \
  "$runtime"

printf '%s\n' "$runtime"
