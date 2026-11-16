#!/usr/bin/env bash
# Optional Fault20 QMM and WeightNz callback build. Bypasses shared v17 control.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
toolkit="${ASCEND_HOME_PATH:?source the matching CANN environment}"
if [[ -d "$toolkit/aarch64-linux/include" ]]; then toolkit="$toolkit/aarch64-linux"; fi
toolkit="$(realpath "$toolkit")"
grep -Eq '^version=8\.5\.1$' "$toolkit/ascend_toolkit_install.info"
[[ "$(uname -m)" == aarch64 ]]
output="${DEMONCAT_QMM_OUTPUT:?use a fresh build directory}"
[[ ! -e "$output" ]]
mkdir -p "$output/real"
source="$root/fault20_fast_validation/source"
native="$root/fault39/native"
flags=(-std=c++17 -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -DDEMONCAT_QMM_AUTO -I"$toolkit/include" -I"$toolkit/include/aclnn" -I"$toolkit/pkg_inc" -I"$output" -I"$native")
g++ "${flags[@]}" "$root/fault_injection/native_adapters/fault20_abi_probe.cpp" -o "$output/abi_probe"
"$output/abi_probe" > "$output/abi_facts.json"
for base in opapi opapi_nn; do
    install -m0755 "$toolkit/lib64/lib${base}.so" "$output/real/lib${base}_real.so"
    "${FAULT39_PATCHELF:-patchelf}" --set-soname "lib${base}_real.so" "$output/real/lib${base}_real.so"
done
python3 "$root/fault39/tools/version_profile.py" --operator fault20-isolated --toolkit "$toolkit" \
  --real "$output/real/libopapi_real.so" --split "$output/real/libopapi_nn_real.so" \
  --output "$output" --abi-facts "$output/abi_facts.json" --source-opapi "$toolkit/lib64/libopapi.so"
g++ "${flags[@]}" -shared "-DFAULT20_EXPECTED_LIBRARY_PATH_FRAGMENT=\"${toolkit%/aarch64-linux}/\"" \
  -o "$output/libdcat20_qmm.so" "$source/l4_buffer_substitute_allhit_v1/buffer_substitute_preload.cpp" \
  -ldl -lpthread -lrt
for base in opapi opapi_nn; do
    split=()
    if [[ "$base" == opapi_nn ]]; then split=(-DDEMONCAT_QMM_SPLIT); fi
    g++ "${flags[@]}" "${split[@]}" -shared "$source/l4_metadata_lifecycle_proxy/libopapi_metadata_proxy.cpp" \
      -L"$output/real" -Wl,--no-as-needed "-l:lib${base}_real.so" "-Wl,-soname,lib${base}.so" \
      -Wl,-rpath,'$ORIGIN/real:/native/real' -ldl -lpthread -o "$output/lib${base}.so"
done
nm -D "$output/libdcat20_qmm.so" | grep -E 'dcat20_qmm_(install|remove|invoke|receipt|profile)_v1'
nm -D "$output/libdcat20_qmm.so" | grep -E 'dcat20_weight_nz_(install|remove|invoke|receipt)_v1'
python3 "$native/test_identity.py" --output "$output/identity_tests.json"
python3 "$root/fault39/tools/deployment_profile.py" --native "$output" --write-manifest
