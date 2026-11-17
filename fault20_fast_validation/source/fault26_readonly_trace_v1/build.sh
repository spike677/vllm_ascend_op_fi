#!/usr/bin/env bash
set -euo pipefail

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "$source_dir/../../.." && pwd)"
output_dir="${FAULT26_TRACE_OUTPUT_DIR:-/fault20/runtime/fault26_readonly_trace_v1}"
real_dir="$output_dir/real"
toolkit="${FAULT26_ASCEND_TOOLKIT:-/usr/local/Ascend/cann-8.5.1}"
real_source="${FAULT26_REAL_LIBOPAPI:-$toolkit/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so}"
real_split_source="${FAULT26_REAL_LIBOPAPI_NN:-$toolkit/aarch64-linux/lib64/libopapi_nn.so}"
real_custom_source="${FAULT26_REAL_CUSTOM_OPAPI:-/vllm-workspace/vllm-ascend/csrc/build/libcust_opapi.so}"
expected_fragment="${FAULT26_EXPECTED_LIBRARY_PATH_FRAGMENT:-/cann-8.5.1/}"
use_typed_qmm_proxy="${FAULT26_USE_TYPED_QMM_PROXY_BUILD:-0}"
runtime="$output_dir/libfault26_readonly_trace_abi0.so"
proxy="$output_dir/libopapi.so"
custom_proxy="$output_dir/libcust_opapi.so"
split_proxy="$output_dir/libopapi_nn.so"
fia_preload_proxy="$output_dir/libfault26_fia_opapi_proxy.so"
toolkit_root="$(dirname "$toolkit")"
smoke_library_path="$output_dir:$real_dir:$toolkit/lib64:$toolkit/devlib:$toolkit_root/lib64:${LD_LIBRARY_PATH:-}"

install -d -m 0755 "$output_dir" "$real_dir"
python3 -B "$source_root/fault39/native/test_metadata_ownership.py" --writer26 \
  --output "$output_dir/metadata-array-cpu.json"

if [[ "$use_typed_qmm_proxy" != 0 && "$use_typed_qmm_proxy" != 1 ]]; then
  echo "REFUSE: FAULT26_USE_TYPED_QMM_PROXY_BUILD must be 0 or 1" >&2
  exit 1
fi

real_symbols="$(nm -D "$real_source")"
for symbol in aclnnFusedInferAttentionScoreV3 \
              aclnnFusedInferAttentionScoreV3GetWorkspaceSize aclnnReduceSum; do
  # Do not combine `grep -q` with an upstream `nm` while pipefail is enabled:
  # grep's early exit can SIGPIPE nm and make a successful match look like a
  # failed pipeline.
  if ! grep -Eq "[[:space:]]${symbol}$" <<<"$real_symbols"; then
    echo "REFUSE: real libopapi does not export $symbol" >&2
    exit 1
  fi
done
if [[ "$use_typed_qmm_proxy" == 1 ]] &&
   ! grep -Eq '[[:space:]]aclnnQuantMatmulWeightNz(GetWorkspaceSize)?$' <<<"$real_symbols"; then
  echo 'REFUSE: real libopapi does not export the typed QMM API' >&2
  exit 1
fi
if [[ "$use_typed_qmm_proxy" == 1 && ! -r "$real_split_source" ]]; then
  echo "REFUSE: split libopapi_nn is missing: $real_split_source" >&2
  exit 1
fi

if [[ ! -r "$real_custom_source" ]]; then
  echo "REFUSE: custom opapi is missing: $real_custom_source" >&2
  exit 1
fi
custom_symbols="$(nm -D "$real_custom_source")"
for symbol in aclnnAddRmsNormBiasGetWorkspaceSize aclnnAddRmsNormBias; do
  if ! grep -Eq "[[:space:]]${symbol}$" <<<"$custom_symbols"; then
    echo "REFUSE: custom opapi does not export $symbol" >&2
    exit 1
  fi
done

install -m 0755 "$real_source" "$real_dir/libopapi_real.so"
patchelf --set-soname libopapi_real.so "$real_dir/libopapi_real.so"
if [[ "$use_typed_qmm_proxy" == 1 ]]; then
  install -m 0755 "$real_split_source" "$real_dir/libopapi_nn_real.so"
  patchelf --set-soname libopapi_nn_real.so "$real_dir/libopapi_nn_real.so"
fi
install -m 0755 "$real_custom_source" "$real_dir/libcust_opapi_real.so"
patchelf --set-soname libcust_opapi_real.so "$real_dir/libcust_opapi_real.so"

if grep -Eq 'ACL_MEMCPY_(HOST_TO_DEVICE|DEVICE_TO_DEVICE|HOST_TO_HOST)' \
    "$source_dir/fault26_trace_preload.cpp"; then
  echo 'REFUSE: writer audit source contains a non-D2H memcpy direction' >&2
  exit 1
fi
if ! grep -Fq 'ACL_MEMCPY_DEVICE_TO_HOST' \
    "$source_dir/fault26_trace_preload.cpp"; then
  echo 'REFUSE: writer audit D2H direction is missing' >&2
  exit 1
fi

g++ -std=c++17 -O2 -fPIC -shared \
  -Wall -Wextra -Werror \
  -D_GLIBCXX_USE_CXX11_ABI=0 \
  -DFAULT26_EXPECTED_LIBRARY_PATH_FRAGMENT="\"$expected_fragment\"" \
  -I"$toolkit/include" \
  -I"$toolkit/include/aclnn" \
  -I"$toolkit/pkg_inc" \
  "$source_dir/fault26_trace_preload.cpp" \
  -Wl,-soname,libfault26_readonly_trace_abi0.so \
  -ldl -lpthread \
  -o "$runtime"

if [[ "$use_typed_qmm_proxy" == 1 ]]; then
  g++ -std=c++17 -O2 -fPIC -shared \
    -Wall -Wextra -Werror \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DFAULT26_MAIN_PROXY_ONLY=1 \
    -I"$toolkit/include" \
    -I"$toolkit/include/aclnn" \
    -I"$toolkit/pkg_inc" \
    "$source_dir/../l4_metadata_lifecycle_proxy/libopapi_metadata_proxy.cpp" \
    "$source_dir/fault26_opapi_trace_proxy.cpp" \
    -L"$real_dir" \
    -Wl,--no-as-needed -l:libopapi_real.so \
    -Wl,-soname,libopapi.so \
    -Wl,-rpath,'$ORIGIN/real' \
    -Wl,-z,origin \
    -ldl -lpthread \
    -o "$proxy"
  g++ -std=c++17 -O2 -fPIC -shared \
    -Wall -Wextra -Werror \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I"$toolkit/include" \
    -I"$toolkit/include/aclnn" \
    -I"$toolkit/pkg_inc" \
    "$source_dir/../l4_metadata_lifecycle_proxy/libopapi_metadata_proxy.cpp" \
    -L"$real_dir" \
    -Wl,--no-as-needed -l:libopapi_nn_real.so \
    -Wl,-soname,libopapi_nn.so \
    -Wl,-rpath,'$ORIGIN/real' \
    -Wl,-z,origin \
    -ldl -lpthread \
    -o "$split_proxy"
  g++ -std=c++17 -O2 -fPIC -shared \
    -Wall -Wextra -Werror \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DFAULT26_MAIN_PROXY_ONLY=1 \
    -I"$toolkit/include" \
    -I"$toolkit/include/aclnn" \
    -I"$toolkit/pkg_inc" \
    "$source_dir/fault26_opapi_trace_proxy.cpp" \
    -L"$real_dir" \
    -Wl,--no-as-needed -l:libopapi_real.so \
    -Wl,-soname,libfault26_fia_opapi_proxy.so \
    -Wl,-rpath,'$ORIGIN/real' \
    -Wl,-z,origin \
    -ldl -lpthread \
    -o "$fia_preload_proxy"
else
  g++ -std=c++17 -O2 -fPIC -shared \
    -Wall -Wextra -Werror \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DFAULT26_MAIN_PROXY_ONLY=1 \
    -I"$toolkit/include" \
    -I"$toolkit/include/aclnn" \
    -I"$toolkit/pkg_inc" \
    "$source_dir/fault26_opapi_trace_proxy.cpp" \
    -L"$real_dir" \
    -Wl,--no-as-needed -l:libopapi_real.so \
    -Wl,-soname,libopapi.so \
    -Wl,-rpath,'$ORIGIN/real' \
    -Wl,-z,origin \
    -ldl -lpthread \
    -o "$proxy"
fi

g++ -std=c++17 -O2 -fPIC -shared \
  -Wall -Wextra -Werror \
  -D_GLIBCXX_USE_CXX11_ABI=0 \
  -DFAULT26_CUSTOM_PROXY_ONLY=1 \
  -I"$toolkit/include" \
  -I"$toolkit/include/aclnn" \
  -I"$toolkit/pkg_inc" \
  "$source_dir/fault26_opapi_trace_proxy.cpp" \
  -L"$real_dir" \
  -Wl,--no-as-needed -l:libcust_opapi_real.so \
  -Wl,-soname,libcust_opapi.so \
  -Wl,-rpath,'$ORIGIN/real' \
  -Wl,-z,origin \
  -ldl -lpthread \
  -o "$custom_proxy"

nm -D -C "$runtime" | grep -F 'op::internal::OpExecCache::Run('
nm -D -C "$runtime" | grep -F 'op::internal::AddOpExecCache('
nm -D -C "$runtime" | grep -F 'op::internal::AddParamToBuf(aclTensor const*)'
nm -D -C "$runtime" | grep -F 'op::internal::AddParamToBuf(aclTensor*)'
nm -D "$runtime" | grep -F 'fault26_trace_add_rms_consume'
if [[ "$use_typed_qmm_proxy" == 1 ]]; then
  nm -D "$proxy" | grep -E 'aclnn(QuantMatmulWeightNz|AscendQuantV3|Matmul|InplaceCopy|FusedInferAttentionScoreV3)(GetWorkspaceSize)?$'
  nm -D "$split_proxy" | grep -E 'aclnn(QuantMatmulWeightNz|AscendQuantV3|Matmul|InplaceCopy)(GetWorkspaceSize)?$'
  nm -D "$fia_preload_proxy" | grep -E 'aclnnFusedInferAttentionScoreV3(GetWorkspaceSize)?$'
else
  nm -D "$proxy" | grep -E 'aclnn(FusedInferAttentionScoreV3(GetWorkspaceSize)?|ReduceSum)$'
  if nm -D "$proxy" | grep -Eq \
    '[[:space:]][TW][[:space:]]aclnn(AddRmsNormBias|AddRmsNormBiasGetWorkspaceSize)$'; then
    echo 'REFUSE: main opapi proxy must not export custom AddRms' >&2
    exit 1
  fi
fi
nm -D "$custom_proxy" | grep -E 'aclnn(AddRmsNormBias|AddRmsNormBiasGetWorkspaceSize)$'
if nm -D "$custom_proxy" | grep -Eq \
  '[[:space:]][TW][[:space:]]aclnn(FusedInferAttentionScoreV3(GetWorkspaceSize)?|ReduceSum)$'; then
  echo 'REFUSE: custom-opapi proxy must not export FIA or ReduceSum' >&2
  exit 1
fi

runtime_symbols="$(nm -D "$runtime")"
runtime_symbols_demangled="$(nm -D -C "$runtime")"

if grep -Eq 'aclrt(Malloc|MallocHost|Memcpy|MemcpyAsync|Memset|MemsetAsync)' <<<"$runtime_symbols"; then
  echo 'REFUSE: read-only tracer unexpectedly references a device write API' >&2
  exit 1
fi

if grep -Fq 'aclOpExecutor::GetOpExecCache()' <<<"$runtime_symbols_demangled"; then
  echo 'REFUSE: tracer must not inspect aclOpExecutor private layout' >&2
  exit 1
fi

readelf -d "$runtime" | grep -E 'SONAME|NEEDED'
readelf -d "$proxy" | grep -E 'SONAME|NEEDED|RUNPATH'
if [[ "$use_typed_qmm_proxy" == 1 ]]; then
  readelf -d "$split_proxy" | grep -E 'SONAME|NEEDED|RUNPATH'
fi
readelf -d "$custom_proxy" | grep -E 'SONAME|NEEDED'

LD_BIND_NOW=1 python3 -c 'import ctypes, os, sys; ctypes.CDLL(sys.argv[1], mode=os.RTLD_LOCAL)' "$runtime"
if [[ "$use_typed_qmm_proxy" == 1 ]]; then
  LD_LIBRARY_PATH="$smoke_library_path" \
    python3 -c 'import ctypes, os, sys; lib=ctypes.CDLL(sys.argv[1], mode=os.RTLD_LOCAL); assert getattr(lib, "aclnnQuantMatmulWeightNzGetWorkspaceSize"); assert getattr(lib, "aclnnQuantMatmulWeightNz")' \
    "$proxy"
  LD_LIBRARY_PATH="$real_dir:$smoke_library_path" \
    python3 -c 'import ctypes, os, sys; lib=ctypes.CDLL(sys.argv[1], mode=os.RTLD_LOCAL); assert getattr(lib, "aclnnQuantMatmulWeightNzGetWorkspaceSize"); assert getattr(lib, "aclnnQuantMatmulWeightNz")' \
    "$split_proxy"
  LD_LIBRARY_PATH="$smoke_library_path" \
    python3 -c 'import ctypes, os, sys; lib=ctypes.CDLL(sys.argv[1], mode=os.RTLD_LOCAL); assert getattr(lib, "aclnnFusedInferAttentionScoreV3GetWorkspaceSize"); assert getattr(lib, "aclnnFusedInferAttentionScoreV3")' \
    "$fia_preload_proxy"
else
  LD_LIBRARY_PATH="$smoke_library_path" \
    python3 -c 'import ctypes, os, sys; lib=ctypes.CDLL(sys.argv[1], mode=os.RTLD_LOCAL); assert getattr(lib, "aclnnFusedInferAttentionScoreV3GetWorkspaceSize"); assert getattr(lib, "aclnnFusedInferAttentionScoreV3"); assert getattr(lib, "aclnnReduceSum")' \
    "$proxy"
fi
LD_LIBRARY_PATH="$smoke_library_path" \
  python3 -c 'import ctypes, os, sys; lib=ctypes.CDLL(sys.argv[1], mode=os.RTLD_LOCAL); assert getattr(lib, "aclnnAddRmsNormBiasGetWorkspaceSize"); assert getattr(lib, "aclnnAddRmsNormBias")' \
  "$custom_proxy"

echo "$runtime"
echo "$proxy"
echo "$custom_proxy"
if [[ "$use_typed_qmm_proxy" == 1 ]]; then
  echo "$split_proxy"
  echo "$fia_preload_proxy"
  echo "real_split_libopapi=$real_split_source"
fi
echo "expected_library_fragment=$expected_fragment"
echo "real_libopapi=$real_source"
echo "real_custom_opapi=$real_custom_source"
if [[ "$use_typed_qmm_proxy" == 1 ]]; then
  python3 -B "$source_root/fault_injection/native_build/metadata26.py" \
    --source "$source_root" --native "$output_dir"
fi
