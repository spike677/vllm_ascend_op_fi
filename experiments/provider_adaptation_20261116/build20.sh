#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
toolkit="${ASCEND_HOME_PATH:?source the matched set_env first}"
[[ -d "$toolkit/aarch64-linux/include" ]] && toolkit="$toolkit/aarch64-linux"
out="${DCAT20_BUILD:?new output directory required}"
[[ ! -e "$out" ]] || exit 1
mkdir -p "$out/real"
flags=(-std=c++17 -O2 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=0 -I"$toolkit/include" -I"$toolkit/include/aclnn" -I"$toolkit/pkg_inc")
source="$root/fault20_fast_validation/source/l4_buffer_substitute_allhit_v1"
g++ "${flags[@]}" -DDEMONCAT_NATIVE_PROVIDER_BRIDGE '-DFAULT20_EXPECTED_LIBRARY_PATH_FRAGMENT="/cann-8.5.1/"' \
 "$source/buffer_substitute_preload.cpp" -ldl -lpthread -lrt -o "$out/libdcat20.so"
for spec in opapi opapi_nn; do
 install -m0755 "$toolkit/lib64/lib${spec}.so" "$out/real/lib${spec}_real.so"
 /work/bin/patchelf --set-soname "lib${spec}_real.so" "$out/real/lib${spec}_real.so"
 g++ "${flags[@]}" "$root/fault20_fast_validation/source/l4_metadata_lifecycle_proxy/libopapi_metadata_proxy.cpp" \
  -L"$out/real" -Wl,--no-as-needed "-l:lib${spec}_real.so" "-Wl,-soname,lib${spec}.so" \
  -Wl,-rpath,'$ORIGIN/real:/native/real' -Wl,-z,origin -ldl -lpthread -o "$out/lib${spec}.so"
done
python3 - "$out" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); files={f.relative_to(p).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*.so')}
(p/'build.json').write_text(json.dumps({'mechanism':'20.buffer_substitute','variant':'v17_mantissa','files':files},indent=2))
PY
