#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
toolkit="${DCAT_BUILD_TOOLKIT:?actual inspected toolkit required}"
out="${DCAT_POSTWRITE_BUILD:?fresh build directory required}"
[[ ! -e "$out" ]] || exit 1
mkdir -p "$out"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread \
    -I"$toolkit/include" "$root/fault_injection/native_adapters/postwrite/test_bound_addresses.cpp" \
    -o "$out/test_bound_addresses"
"$out/test_bound_addresses" > "$out/bound-addresses-cpu.json"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -fPIC -shared -pthread \
    -I"$toolkit/include" "$root/fault_injection/native_adapters/postwrite/postwrite.cpp" \
    -L"$toolkit/lib64" -lascendcl -Wl,-soname,libdcat_postwrite.so \
    -o "$out/libdcat_postwrite.so"
nm -D --defined-only "$out/libdcat_postwrite.so" > "$out/exported-symbols.txt"
grep -Eq '[[:space:]]T dcat_pw_linked_address_v1$' "$out/exported-symbols.txt"
