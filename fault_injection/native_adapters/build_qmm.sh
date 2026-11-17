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
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread \
  "$root/tests/native/test_qmm_current_call.cpp" -o "$output/current_call_cpu_test"
"$output/current_call_cpu_test" > "$output/current_call_cpu_receipt.txt"
for test in test_matmul_auto test_matmul_requests; do
    g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-error=misleading-indentation -pthread \
        "$root/tests/native/${test}.cpp" -ldl -o "$output/${test}"
    "$output/${test}" > "$output/${test}_receipt.txt"
done
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  "$root/tests/native/test_qmm_pta_gate.cpp" -o "$output/pta_gate_cpu_test"
"$output/pta_gate_cpu_test" > "$output/pta_gate_cpu_receipt.txt"
python3 "$root/tests/native/test_qmm_pta_gate_elf.py" --output "$output/pta_gate_elf_cpu_receipt.json"
python3 "$root/tests/native/test_qmm_pta_resolver_elf.py" --output "$output/pta_resolver_elf_cpu_receipt.json"
source="$root/fault20_fast_validation/source"
native="$root/fault39/native"
python3 "$native/test_metadata_ownership.py" --output "$output/metadata_array_cpu_receipt.json"
flags=(-std=c++17 -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -DDEMONCAT_QMM_AUTO -I"$toolkit/include" -I"$toolkit/include/aclnn" -I"$toolkit/pkg_inc" -I"$output" -I"$native")
g++ "${flags[@]}" "$root/fault_injection/native_adapters/fault20_abi_probe.cpp" -o "$output/abi_probe"
"$output/abi_probe" > "$output/abi_facts.json"
for base in opapi opapi_nn opapi_math; do
    install -m0755 "$toolkit/lib64/lib${base}.so" "$output/real/lib${base}_real.so"
    "${FAULT39_PATCHELF:-patchelf}" --set-soname "lib${base}_real.so" "$output/real/lib${base}_real.so"
done
python3 "$root/fault39/tools/version_profile.py" --operator fault20-isolated --toolkit "$toolkit" \
  --real "$output/real/libopapi_real.so" --split "$output/real/libopapi_nn_real.so" \
  --math "$output/real/libopapi_math_real.so" \
  --output "$output" --abi-facts "$output/abi_facts.json" --source-opapi "$toolkit/lib64/libopapi.so"
g++ "${flags[@]}" -shared "-DFAULT20_EXPECTED_LIBRARY_PATH_FRAGMENT=\"${toolkit%/aarch64-linux}/\"" \
  -o "$output/libdcat20_qmm.so" "$source/l4_buffer_substitute_allhit_v1/buffer_substitute_preload.cpp" \
  -ldl -lpthread -lrt
for base in opapi opapi_nn; do
    split=()
    if [[ "$base" == opapi_nn ]]; then split=(-DDEMONCAT_QMM_SPLIT); fi
    g++ "${flags[@]}" "${split[@]}" -shared "$source/l4_metadata_lifecycle_proxy/libopapi_metadata_proxy.cpp" \
      "$root/fault_injection/native_adapters/qmm_pta_gate.cpp" \
      -L"$output/real" -Wl,--no-as-needed "-l:lib${base}_real.so" "-Wl,-soname,lib${base}.so" \
      -Wl,-rpath,'$ORIGIN/real:/native/real' -ldl -lpthread -o "$output/lib${base}.so"
    nm -D "$output/lib${base}.so" | grep -E '[[:space:]]CanUsePTACache$'
    nm -D "$output/lib${base}.so" | grep -E '[[:space:]]dcat20_pta_gate_status_v1$'
    nm -D "$output/lib${base}.so" | grep -E '[[:space:]]dcat20_metadata_arrays_v1$'
done
# torch_npu 2.9.0.post1 probes this feature handle before nn/main. The
# gate-only proxy retains the real math library and exports no replacement
# math operators. dlsym on all other APIs resolves the original dependency.
g++ "${flags[@]}" -DDEMONCAT_QMM_MATH_GATE -shared \
  "$root/fault_injection/native_adapters/qmm_pta_gate.cpp" \
  -L"$output/real" -Wl,--no-as-needed -l:libopapi_math_real.so \
  -Wl,-soname,libopapi_math.so -Wl,-rpath,'$ORIGIN/real:/native/real' \
  -ldl -lpthread -o "$output/libopapi_math.so"
nm -D "$output/libopapi_math.so" | grep -E '[[:space:]]CanUsePTACache$'
nm -D "$output/libopapi_math.so" | grep -E '[[:space:]]dcat20_pta_gate_status_v1$'
python3 - "$output" <<'PY'
import hashlib, json, pathlib, re, subprocess, sys
root = pathlib.Path(sys.argv[1])
profile = json.loads((root / 'version_profile.json').read_text())
images = {}
for name in ('libopapi.so', 'libopapi_nn.so', 'libopapi_math.so'):
    path = root / name
    note = subprocess.check_output(['readelf', '-n', str(path)], text=True)
    build_id = re.search(r'Build ID:\s*([0-9a-f]+)', note)
    images[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       elf_build_id=build_id.group(1) if build_id else None)
with (root / 'pta_gate_identity.json').open('x') as output:
    json.dump(dict(schema='dcat20.pta_gate.build_identity.v1',
                   profile_sha256=profile['profile_sha256'], images=images,
                   source_sha256={key: value for key, value in profile['sources'].items()
                                  if 'qmm_pta_gate' in key or key.endswith('/pta_resolver.py')},
                   actual_npu_calls=0), output, indent=2)
PY
nm -D "$output/libdcat20_qmm.so" | grep -E 'dcat20_qmm_(install|remove|invoke|receipt|profile|state)_v1'
nm -D "$output/libdcat20_qmm.so" | grep -E 'dcat20_weight_nz_(install|remove|invoke|receipt|state)_v1'
for prefix in dcat20_qmm dcat20_weight_nz; do
    nm -D "$output/libdcat20_qmm.so" | grep -E "[[:space:]]${prefix}_state_v1$"
    for action in install pause resume remove state; do
        nm -D "$output/libdcat20_qmm.so" | grep -E "[[:space:]]${prefix}_${action}_v2$"
    done
    for action in install_group seal_group invoke_group group_state; do
        nm -D "$output/libdcat20_qmm.so" | grep -E "[[:space:]]${prefix}_${action}_v3$"
    done
done
nm -D "$output/libdcat20_qmm.so" | grep -E '[[:space:]]dcat20_independent_run_inventory_v1$'
for symbol in gws_begin gws_commit gws_abort execute_begin execute_end diagnostics; do
    nm -D "$output/libdcat20_qmm.so" | grep -E "[[:space:]]dcat20_auto_${symbol}_v1$"
done
python3 "$native/test_identity.py" --output "$output/identity_tests.json"
python3 "$root/fault39/tools/deployment_profile.py" --native "$output" --write-manifest
