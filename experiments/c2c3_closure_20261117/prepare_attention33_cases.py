"""Prepare natural stock/B or bridge-gated F manifests; never start devices."""
import argparse
import json
from pathlib import Path
import shlex
import sys

from attention33_contract import execution_contract, read_json, require, sha, verify_bridge_header


def prepare(args):
    sys.path[:0] = [str(args.source), str(args.source/'src')]
    from demoncat.runtime.vllm_ascend.worker_extension.paged33_tnd import verify_bundle
    from fault_injection.native_adapters.attention28.launch import bundle_kernel_abi
    from fault_injection.native_adapters.attention28.fia.binary_registration import validate_metadata
    require(args.stage in ('baseline', 'fault') and args.card in (2, 5)
        and 1 <= args.timeout <= 1800, 'TND33_BOUNDED_STAGE_REQUIRED')
    source_manifest = read_json(args.source/'source_manifest.json')
    require(source_manifest.get('schema') == 'demoncat.c2c3.source.v1', 'TND33_FROZEN_SOURCE_REQUIRED')
    source_files = {row['path']: row for row in source_manifest['files']}
    for name in ('attention33_worker_probe.py', 'attention33_contract.py', 'attention33_bridge.py',
                 'prepare_attention33_cases.py'):
        relative = 'experiments/c2c3_closure_20261117/' + name
        require(relative in source_files and source_files[relative]['sha256'] == sha(args.source/relative),
                'TND33_DRIVER_SOURCE_NOT_FROZEN:' + name)
    completion = args.build_root/'evidence/completion.json'
    vendor = args.build_root/'stage/vendors/dcat_f33_fai_fd_transformer'
    bundle = verify_bundle(completion, sha(completion), vendor)
    metadata, _ = bundle_kernel_abi(vendor, bundle['artifacts'], bundle['completion']['kernel_binary_sha256'])
    binary = next(vendor/name for name in bundle['artifacts'] if name.endswith('.o'))
    validate_metadata(metadata, binary, bundle['completion']['variant'])
    controller = read_json(args.build_root.parents[1]/'controller.json')
    require(controller.get('status') == 'BUILD_COMMAND_COMPLETED' and controller.get('actual_npu_calls') == 0
        and controller.get('devices', []) == [] and controller.get('image_id') ==
        'sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0',
        'TND33_EXACT_CPU_BUILD_IMAGE_REQUIRED')
    observer = args.observer/'attention-preflight/result.json';receipt = read_json(observer)
    require(receipt.get('observer_compiled') is True and receipt.get('actual_npu_calls') == 0
        and receipt.get('observer_memory_selftest', {}).get('exit_code') == 0
        and receipt['observer_source']['sha256'] == sha(args.source/'fault_injection/native_adapters/attention28/launch_observer.cpp'),
        'TND33_EXACT_851_OBSERVER_REQUIRED')
    library = args.observer/'attention-preflight'/Path(receipt['observer_library']['path']).name
    require(sha(library) == receipt['observer_library']['sha256'], 'TND33_OBSERVER_BINARY_DIFFERS')
    env = dict(PYTHONPATH='/installed:/msprobe_packages', PYTHONDONTWRITEBYTECODE='1',
        VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_ASCEND_ENABLE_NZ='0', VLLM_ALLOW_INSECURE_SERIALIZATION='1', OMP_NUM_THREADS='4',
        VLLM_HOST_IP='127.0.0.1', GLOO_SOCKET_IFNAME='lo', HCCL_CONNECT_TIMEOUT='120', HCCL_EXEC_TIMEOUT='120',
        ACLNN_CACHE_LIMIT='0', TASK_QUEUE_ENABLE='0', DCAT33_REVIEWED_PHYSICAL_CARD=str(args.card))
    folder = args.source/'experiments/c2c3_closure_20261117'
    contract = execution_contract(args.model, args.installed, folder/'attention33_worker_probe.py', .35, env)
    bridge = None
    if args.stage == 'fault':
        require(args.bridge and args.stock_run and args.baseline_run, 'TND33_PRIOR_STOCK_AND_BASELINE_REQUIRED')
        from attention33_bridge import review_bridge
        # Re-read every raw payload and native argument file now. A copied PASS
        # JSON alone cannot release a new F queue.
        reviewed = review_bridge(args.stock_run, args.baseline_run, args.build_root)
        bridge = verify_bridge_header(args.bridge, sha(args.bridge), contract, sha(completion), folder/'attention33_bridge.py')
        require(reviewed == bridge, 'TND33_SAVED_BRIDGE_DIFFERS_FROM_CURRENT_RAW_EVIDENCE')
    cases = []
    for phase in (('stock', 'baseline') if args.stage == 'baseline' else ('fault',)):
        label = 'attention33-fd-natural-' + phase
        argv = ['python3', '-B', '/repo/experiments/c2c3_closure_20261117/attention33_worker_probe.py',
            '--phase', phase, '--model', '/model', '--out', '/evidence/run', '--experimental-qualification']
        mounts = [[str(args.source), '/repo', 'ro'], [str(args.installed), '/installed', 'ro'],
                  [str(args.model), '/model', 'ro']]
        if phase != 'stock':
            argv += ['--statistics', '--completion', '/native/evidence/completion.json', '--completion-sha256', sha(completion),
                '--vendor', '/native/stage/vendors/dcat_f33_fai_fd_transformer',
                '--public-schema', '/native/evidence/fai_schema/fai_abi_schema.json',
                '--schema-sha256', bundle['completion']['fai_public_schema_sha256'],
                '--observer-receipt', '/build/attention-preflight/result.json', '--observer-receipt-sha256', sha(observer)]
            mounts += [[str(args.build_root), '/native', 'ro'],
                       [str(args.observer/'attention-preflight'), '/build/attention-preflight', 'ro']]
        if phase == 'fault':
            argv += ['--bridge', '/bridge.json', '--bridge-sha256', sha(args.bridge)]
            mounts.append([str(args.bridge), '/bridge.json', 'ro'])
        cases.append(dict(id=label, image=controller['image_id'], timeout=args.timeout, mounts=mounts, env=env,
            command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec ' + shlex.join(argv),
            review=dict(experiment_id=label, provider='33.paged_decode / native TND FAI flash decode KV512-to-256',
                model=str(args.model), cann='8.5.1', soc='Ascend910B4/A2; actual aic must be observed',
                physical_cards=[args.card], ranks=[0], worker='Unmodified standard Qwen TP1 eager; natural 1024-token prefill and KV1025 decode',
                build_fingerprint=dict(completion=sha(completion), schema=bundle['completion']['fai_public_schema_sha256'],
                    kernel=bundle['completion']['kernel_binary_sha256'], observer=sha(observer)),
                source_commit=source_manifest['base_commit'], source_manifest_sha256=sha(args.source/'source_manifest.json'),
                execution_contract=contract, expected_hits='28 natural measured boundaries; 0 changed for stock/B, 28 changed for F; phases counted separately',
                resource_budget=dict(timeout_seconds=args.timeout, memory_gib=48, cpus=16),
                recovery_plan='Same-Worker B/repeat/measured/OFF/healthy; exact full hidden/logits/all28 outputs/valid logical KV; owned cleanup only.',
                prerequisite='Baseline also records actual aic/FD/basekey; F requires independent raw stock/private bridge and 112 original-key launches.',
                bridge_sha256=sha(args.bridge) if bridge else None, public_coverage_qualified=False, c2=False, c3=False)))
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'installed', 'build-root', 'observer', 'model', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stage', choices=('baseline', 'fault'), default='baseline')
    p.add_argument('--card', type=int, choices=(2, 5), default=2)
    p.add_argument('--timeout', type=int, default=1800)
    for name in ('bridge', 'stock-run', 'baseline-run'):p.add_argument('--' + name, type=Path)
    args = p.parse_args();cases = prepare(args)
    with args.out.open('x', encoding='utf-8') as stream:json.dump(cases, stream, indent=2, allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='ROOT_QUEUE_MANIFEST_ONLY', cases=len(cases), output=str(args.out), actual_npu_calls=0)))


if __name__ == '__main__':main()
