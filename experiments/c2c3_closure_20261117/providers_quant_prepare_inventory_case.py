"""Reviewable new V2 native inventory; only root may acquire the NPU queue."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    for name in ('installed', 'quant', 'probe', 'source-commit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--model', default='/home/haoran/models/DeepSeek-V2-Lite-W8A8')
    parser.add_argument('--card', type=int, choices=(2, 5), default=2)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (not re.fullmatch('[0-9a-f]{7,40}', args.source_commit)
            or any(not value.startswith('/') or any(c in value for c in ('\n', '\r', '\0', ','))
                   for value in (args.installed, args.quant, args.probe, args.model))):
        parser.error('Actual immutable source commit and absolute inputs required')
    root = Path(args.quant)
    manifest = root / 'runtime/build.json'
    build = json.loads(manifest.read_text())
    if build.get('schema') != 'demoncat.quant.v2_portable.v1' or build.get('actual_npu_calls') != 0:
        raise ValueError('Fresh V2 no-device build required')
    for filename, key in (('native_backend.cpp', 'source_sha256'), ('backend_control.py', 'control_sha256'),
        ('batch_scale_inventory.h', 'inventory_source_sha256'), ('runtime/vendor/op_api/lib/libcust_opapi.so', 'library_sha256')):
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != build[key]:
            raise ValueError('Actual quant Native bundle changed:' + filename)
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    name = 'providers-quant-deepseek-native-inventory'
    case = dict(id=name, timeout=1800, image='quay.io/ascend/vllm-ascend:v0.18.0',
        mounts=[[args.installed, '/installed', 'ro'], [args.model, '/model', 'ro'],
            [args.probe, '/probe.py', 'ro'], [args.quant, '/quant', 'ro']],
        env=dict(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',
            VLLM_ALLOW_INSECURE_SERIALIZATION='1', TASK_QUEUE_ENABLE='0', OMP_NUM_THREADS='4',
            VLLM_HOST_IP='127.0.0.1', GLOO_SOCKET_IFNAME='lo', HCCL_CONNECT_TIMEOUT='120', HCCL_EXEC_TIMEOUT='120',
            DCAT_DQ_V2_ENABLE='1', DCAT_REAL_OPAPI='/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so'),
        command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
            'export ASCEND_CUSTOM_OPP_PATH=/quant/runtime/vendor:/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend\n'
            'export LD_PRELOAD=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so\n'
            'export PYTHONPATH=/installed:/msprobe_packages:$PYTHONPATH\n'
            'exec python3 /probe.py --build-sha256 ' + sha,
        review=dict(experiment_id=name, provider='quant.batch_shared_scale / all native V2 calls before admission, read-only',
            model=args.model, cann='8.5.1', soc='Ascend910B4/A2; recheck Worker', physical_cards=[args.card], ranks=[0],
            worker='standard TP1 eager BF16, two identical B8 generation batches; observe only actual first pure decode; fixed64 KVblocks',
            build_fingerprint=sha, source_commit=args.source_commit, expected_hits=0,
            resource_budget=dict(timeout_seconds=1800, memory_gib=80, cpus=16, native_events=32768, public_calls=4096),
            recovery_plan='No arm function is called. Native descriptor arithmetic and C1 remain unchanged; independent metadata '
                'records every typed V2 prepare and execute before owner selection. Bind actual input/q/scale/current executor '
                'registration to public call, retain unowned/missing calls and all request-row gaps. Require repeated full B8 '
                'prefill/decode logits equality, zero native owner/pending/executing, remove owned method/API/model hooks, then '
                'same-process healthy generation. No C1/C2/C3 qualification is claimed. Root retains failures and removes only '
                'its owned container; restart is never OFF evidence.'))
    with args.out.open('x') as stream:
        json.dump([case], stream, indent=2)


if __name__ == '__main__':
    main()
