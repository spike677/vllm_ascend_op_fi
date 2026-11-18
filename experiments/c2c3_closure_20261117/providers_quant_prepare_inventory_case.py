"""Reviewable new V2 native inventory; only root may acquire the NPU queue."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    for name in ('installed', 'quant', 'probe', 'source-commit', 'image-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--model', default='/home/haoran/models/DeepSeek-V2-Lite-W8A8')
    parser.add_argument('--card', type=int, choices=(2, 5), default=2)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--group-abi', action='store_true')
    parser.add_argument('--single-site-c1', action='store_true')
    args = parser.parse_args()
    if args.single_site_c1 and not args.group_abi:
        parser.error('The isolated group ABI is required for this new single-site C1 adapter')
    if (not re.fullmatch('[0-9a-f]{7,40}', args.source_commit) or not re.fullmatch(r'sha256:[0-9a-f]{64}', args.image_id)
            or any(not value.startswith('/') or any(c in value for c in ('\n', '\r', '\0', ','))
                   for value in (args.installed, args.quant, args.probe, args.model))):
        parser.error('Actual immutable source commit and absolute inputs required')
    root = Path(args.quant)
    manifest = root / ('build.json' if args.group_abi else 'runtime/build.json')
    build = json.loads(manifest.read_text())
    expected_schema = 'demoncat.quant.group_deployment.v1' if args.group_abi else 'demoncat.quant.v2_portable.v1'
    if build.get('schema') != expected_schema or build.get('actual_npu_calls') != 0:
        raise ValueError('Fresh V2 no-device build required')
    files = (build['files'].items() if args.group_abi else ((filename, build[key]) for filename, key in (
        ('native_backend.cpp', 'source_sha256'), ('backend_control.py', 'control_sha256'),
        ('batch_scale_inventory.h', 'inventory_source_sha256'), ('runtime/vendor/op_api/lib/libcust_opapi.so', 'library_sha256'))))
    if args.group_abi and (build.get('actual_image_id') != args.image_id or build.get('legacy_c1_library_present') is not False):
        raise ValueError('Actual reviewed group image and no legacy library required')
    for filename, expected in files:
        path = (root / filename).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Actual quant Native bundle changed:' + filename)
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    name = 'providers-quant-group-deepseek-native-inventory' if args.group_abi else 'providers-quant-deepseek-native-inventory'
    case = dict(id=name, timeout=1800, image=args.image_id,
        mounts=[[args.installed, '/installed', 'ro'], [args.model, '/model', 'ro'],
            [args.probe, '/probe.py', 'ro'], [args.quant, '/quant', 'ro']],
        env=dict(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',
            VLLM_ALLOW_INSECURE_SERIALIZATION='1', TASK_QUEUE_ENABLE='0', OMP_NUM_THREADS='4',
            VLLM_HOST_IP='127.0.0.1', GLOO_SOCKET_IFNAME='lo', HCCL_CONNECT_TIMEOUT='120', HCCL_EXEC_TIMEOUT='120',
            DCAT_DQ_V2_ENABLE='0' if args.group_abi else '1', DCAT_DQ_GROUP_V1_ALLOWED='1' if args.group_abi else '0',
            VLLM_SERVER_DEV_MODE='1', DCAT_REAL_OPAPI='/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so'),
        command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
            'export ASCEND_CUSTOM_OPP_PATH=/quant/runtime/vendor:/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend\n'
            'export LD_PRELOAD=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so\n'
            'export PYTHONPATH=/installed:/msprobe_packages:$PYTHONPATH\n'
            'exec python3 /probe.py --build-sha256 ' + sha + (' --group-abi' if args.group_abi else ''),
        review=dict(experiment_id=name, provider='quant.batch_shared_scale / all native V2 calls before admission, read-only',
            model=args.model, cann='8.5.1', soc='Ascend910B4/A2; recheck Worker', physical_cards=[args.card], ranks=[0],
            worker='standard TP1 eager BF16, two identical B8 generation batches; observe only actual first pure decode; fixed64 KVblocks',
            build_fingerprint=sha, source_commit=args.source_commit, expected_hits=0, actual_runtime_image_id=args.image_id,
            resource_budget=dict(timeout_seconds=1800, memory_gib=80, cpus=16, native_events=32768, public_calls=4096,
                row_operator_calls_per_forward=16384),
            recovery_plan='No arm function is called and the group variant never installs an owner. The group variant loads only '
                'the separate group DSO; the legacy variant keeps its original C1 DSO. Native arithmetic remains unchanged; independent metadata '
                'records every typed V2 prepare and execute before owner selection. Bind actual input/q/scale/current executor '
                'registration to public call, retain unowned/missing calls and all request-row gaps. Observe bounded actual '
                'ATen row provenance; unknown row data/Attention/MoE/indexing never inherit rows from matching shape or bytes. '
                'Mutations invalidate every overlapping physical storage alias, including distinct Storage wrappers and '
                'inference tensors without a version counter. Require repeated full B8 '
                'prefill/decode logits equality, zero native owner/pending/executing, remove owned method/API/model hooks, then '
                'same-process healthy generation. Remove only the owned dispatcher mode; a foreign mode prevents cleanup. '
                'No C1/C2/C3 qualification is claimed. Root retains failures and removes only '
                'its owned container; restart is never OFF evidence.'))
    if args.single_site_c1:
        name = 'providers-quant-group-deepseek-single-site-c1'
        case['id'] = name
        case['review'].update(experiment_id=name,
            provider='quant.batch_shared_scale / public Worker Core V1 / one automatically proven site, not all eligible',
            worker='standard TP1 eager BF16; seven fresh B8 first-decode phases; fixed64 KVblocks; no user-selected layer',
            expected_hits=1,
            recovery_plan='Two complete independent Native/public healthy censuses precede automatic single-site selection. '
                'Preserve every structural, unowned and unknown-row candidate in discovery.json. The actual selected site '
                'must retain module/method/weight/process/Worker/input row identity. Run reference, lossless and qualification OFF '
                'under the new Native owner, seal only the explicit one-site scope, then admit one Fault. Record one model parent '
                'and BOTH real correct-domain reference/helper and flattened-domain effect GWS/execute pairs; actual input stays '
                'unchanged, actual owned Q/scale bytes must differ before exact publication. Stop, synchronize every device stream, '
                'drain the target, recover/release, remove callbacks, then observe identical full healthy B8 OFF on the SAME Worker. '
                'Remove all own profile/dispatch/module/method/API hooks before finish/unregister/detach. Busy or failed drains '
                'retain ownership; unknown output lifetimes are never restored after return. An installed zero-target owner uses '
                'the reviewed cancel_empty stream barrier. Keep functional effect separate from E5 (NOT_ASSESSED here); no C2/C3 '
                'or complete 135-target qualification. Root owns launch/card gate/owned-container cleanup; restart is never OFF.')
        case['review']['resource_budget'].update(native_qualified_targets=1, admitted_model_calls=4,
            fault_model_calls=1, fault_native_helpers=1, max_owned_device_bytes=64*1024*1024, max_host_reference_bytes=128*1024*1024)
    with args.out.open('x') as stream:
        json.dump([case], stream, indent=2)


if __name__ == '__main__':
    main()
