#!/usr/bin/env bash
set -euo pipefail

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly_dir="$(cd "$source_dir/../fault26_readonly_trace_v1" && pwd)"
output_dir="${FAULT26_FIA_WRITER_OUTPUT_DIR:-/fault39/runtime/fault26_fia_writer_proxy_v1}"
toolkit="${FAULT26_ASCEND_TOOLKIT:-/usr/local/Ascend/cann-8.5.1}"
writer="$output_dir/libfault26_fia_writer_proxy_abi0.so"

install -d -m 0755 "$output_dir"

FAULT26_TRACE_OUTPUT_DIR="$output_dir" \
FAULT26_ASCEND_TOOLKIT="$toolkit" \
bash "$readonly_dir/build.sh"

g++ -std=c++17 -O2 -fPIC -shared \
  -Wall -Wextra -Werror \
  -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I"$toolkit/include" \
  -I"$toolkit/include/aclnn" \
  -I"$toolkit/pkg_inc" \
  "$source_dir/fault26_fia_writer_lossless.cpp" \
  -Wl,-soname,libfault26_fia_writer_proxy_abi0.so \
  -ldl -lpthread \
  -o "$writer"

nm -D "$writer" | grep -E \
  'fault26_fia_writer_(prepare_output|gws_complete|before_execute|after_execute)$'
nm -D "$writer" | grep -E \
  'fault26_write_gate_(add_rms_norm_bias_gws|before_add_rms_norm_bias|after_add_rms_norm_bias)$'
nm -D "$output_dir/libopapi.so" | \
  grep -E 'aclnnFusedInferAttentionScoreV3(GetWorkspaceSize)?$'

for symbol in aclrtMalloc aclrtFree aclrtMallocHost aclrtFreeHost aclrtMemcpy \
              aclrtSynchronizeStream aclCreateTensor aclDestroyTensor; do
  if ! grep -Fq "\"${symbol}\"" "$source_dir/fault26_fia_writer_lossless.cpp"; then
    echo "REFUSE: FIA writer lossless source does not resolve $symbol" >&2
    exit 1
  fi
done

for required in wrong_destination_residual_stomp \
                FAULT26_FIA_WRITER_PROXY_TARGET_RESIDUAL_BYTES \
                fault26_write_gate_after_add_rms_norm_bias; do
  if ! grep -Fq "$required" "$source_dir/fault26_fia_writer_lossless.cpp"; then
    echo "REFUSE: FIA writer proxy is missing $required" >&2
    exit 1
  fi
done

readelf -d "$writer" | grep -E 'SONAME|NEEDED'
echo "$writer"
echo "$output_dir/libopapi.so"
