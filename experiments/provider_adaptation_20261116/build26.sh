#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
toolkit=/usr/local/Ascend/cann-8.5.1/aarch64-linux
out="${DCAT26_BUILD:?new directory required}"
[[ ! -e "$out" ]] || exit 1
export PATH="/work/bin:$PATH"
export FAULT26_ASCEND_TOOLKIT="$toolkit" FAULT26_TRACE_OUTPUT_DIR="$out"
export FAULT26_REAL_LIBOPAPI="$toolkit/lib64/libopapi.so" FAULT26_REAL_LIBOPAPI_NN="$toolkit/lib64/libopapi_nn.so"
export FAULT26_USE_TYPED_QMM_PROXY_BUILD=1
bash "$root/fault20_fast_validation/source/fault26_readonly_trace_v1/build.sh"
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=0
 -DDEMONCAT_NATIVE_PROVIDER_BRIDGE -I"$toolkit/include" -I"$toolkit/include/aclnn" -I"$toolkit/pkg_inc")
g++ "${flags[@]}" "$root/fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_qmm_output_underflow.cpp" \
 -Wl,-soname,libfault26_qmm_output_underflow_abi0.so -ldl -lpthread -o "$out/libfault26_qmm_output_underflow_abi0.so"
g++ "${flags[@]}" "$root/fault20_fast_validation/source/fault26_fia_writer_proxy_v1/fault26_fia_writer_lossless.cpp" \
 -Wl,-soname,libfault26_fia_writer_proxy_abi0.so -ldl -lpthread -o "$out/libfault26_fia_writer_proxy_abi0.so"
python3 - "$out" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);files={f.relative_to(p).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*.so')}
(p/'build.json').write_text(json.dumps(dict(schema='demoncat.native26.deployment.v1',files=files),indent=2))
PY
