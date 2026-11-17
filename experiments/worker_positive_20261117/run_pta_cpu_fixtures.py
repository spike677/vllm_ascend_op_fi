"""Owned device-free CPU compilation; no vendor library or framework import."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile
import uuid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--resolver-only', action='store_true')
parser.add_argument('--request-only', action='store_true')
args = parser.parse_args()
args.work.mkdir(exist_ok=False)
with tarfile.open(args.archive) as archive:
    for member in archive.getmembers():
        destination = (args.work / member.name).resolve()
        if not destination.is_relative_to(args.work.resolve()) or not member.isfile():
            raise ValueError('UNSAFE_CPU_TEST_ARCHIVE')
    archive.extractall(args.work)
nonce = uuid.uuid4().hex[:12]
name = 'worker-pta-cpu-' + nonce
script = '''set -euo pipefail
g++ -std=c++17 -O2 -Wall -Wextra -Werror /owned/tests/native/test_qmm_pta_gate.cpp -o /tmp/pta-route-test
/tmp/pta-route-test > /owned/pta-route-receipt.txt
python3 /owned/tests/native/test_qmm_pta_gate_elf.py --output /owned/pta-elf-receipt.json
python3 /owned/tests/native/test_qmm_pta_resolver_elf.py --output /owned/pta-resolver-receipt.json
'''
if args.resolver_only:
    script = 'python3 /owned/tests/native/test_qmm_pta_resolver_elf.py --output /owned/pta-resolver-receipt.json'
if args.request_only:
    script = '''set -euo pipefail
cd /owned
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread tests/native/test_qmm_current_call.cpp -o /tmp/current-call-test
/tmp/current-call-test
for test in test_matmul_auto test_matmul_requests; do
    g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-error=misleading-indentation -pthread tests/native/${test}.cpp -ldl -o /tmp/${test}
    /tmp/${test}
done
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-error=misleading-indentation -pthread -shared -fPIC -DDCAT_TEST_SHARED tests/native/test_matmul_auto.cpp -ldl -o /tmp/libhost_fixture.so
LD_PRELOAD=/tmp/libhost_fixture.so python3 -B tests/native/check_matmul_ctypes.py
LD_PRELOAD=/tmp/libhost_fixture.so python3 -B tests/native/check_matmul_request_ctypes.py
'''
command = ['docker', 'run', '--rm', '--name', name, '--label', 'demoncat.owner=worker-positive-pta-cpu',
           '--label', 'demoncat.nonce=' + nonce, '--runtime', 'runc', '--network', 'none', '--read-only',
           '--cpus', '2', '--memory', '2g', '--pids-limit', '128', '--env', 'ASCEND_VISIBLE_DEVICES=void',
           '--env', 'LD_PRELOAD=', '--mount', 'type=bind,src=/dev/null,dst=/etc/ld.so.preload,readonly',
           '--env', 'TORCH_DEVICE_BACKEND_AUTOLOAD=0', '--env', 'DEMONCAT_F20_REQUIRE_TYPED_GWS=0',
           '--mount', 'type=bind,src=' + str(args.work.resolve()) + ',dst=/owned',
           '--tmpfs', '/tmp:rw,exec,nosuid,size=256m', '--entrypoint', 'bash',
           'quay.io/ascend/vllm-ascend:v0.18.0', '-lc', script]
result = subprocess.run(command, text=True, capture_output=True, timeout=180)
(args.work / 'container-test-receipt.json').write_text(json.dumps({
    'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
    'name': name, 'runtime': 'runc', 'devices': [], 'actual_npu_calls': 0,
    'network': 'none', 'cpu_fixture_only': True}, indent=2))
print(result.stdout)
print(result.stderr[-5000:])
raise SystemExit(result.returncode)
