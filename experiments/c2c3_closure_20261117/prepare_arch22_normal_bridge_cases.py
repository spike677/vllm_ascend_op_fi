"""Prepare two fresh normal-only stock/private cases; never run a device.

This entry has no fault stage, bridge certificate input or statistics mode.
Both arms use the same frozen driver, installed capture code and fresh observer.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys

import arch22_normal_bridge_evidence as evidence


def case_environment(card):
    return dict(PYTHONPATH='/installed:/msprobe_packages', PYTHONDONTWRITEBYTECODE='1',
        VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_ASCEND_ENABLE_NZ='0', VLLM_ALLOW_INSECURE_SERIALIZATION='1', OMP_NUM_THREADS='4',
        VLLM_HOST_IP='127.0.0.1', GLOO_SOCKET_IFNAME='lo', HCCL_CONNECT_TIMEOUT='120',
        HCCL_EXEC_TIMEOUT='120', ACLNN_CACHE_LIMIT='0', TASK_QUEUE_ENABLE='0',
        DCAT33_REVIEWED_PHYSICAL_CARD=str(card))


def make_cases(args, source, bundle, observer, contract, installed_pins):
    """Pure manifest composition after all real input bytes were checked."""
    evidence.need(args.card in (2, 5) and type(args.timeout) is int and 1 <= args.timeout <= 1800,
                  'NORMAL_BRIDGE_CASE_BUDGET_INVALID')
    cases = []
    s2=evidence.variant(source)==evidence.S2_VARIANT
    normal_all=evidence.variant(source)==evidence.ALL_VARIANT
    for arm in ('stock', 'baseline'):
        label = ('attention33-arch22-all-normal-' if normal_all else 'attention33-arch22-s2-normal-' if s2 else 'attention33-arch22-normal-')+arm
        argv = ['python3', '-B', '/repo/'+evidence.EXPERIMENT+'attention33_worker_probe.py',
            '--phase', arm, '--model', '/model', '--out', '/evidence/run',
            '--experimental-qualification', '--first-boundary-diagnostic',
            '--arch22-all-normal-bridge' if normal_all else '--arch22-s2-normal-bridge' if s2 else '--arch22-normal-bridge',
            '--observer-receipt', '/build/attention-preflight/result.json',
            '--observer-receipt-sha256', observer['sha256']]
        mounts = [[str(args.source), '/repo', 'ro'], [str(args.installed), '/installed', 'ro'],
                  [str(args.model), '/model', 'ro'], [str(observer['folder']), '/build/attention-preflight', 'ro']]
        if observer['alias'] != '/build/attention-preflight':
            mounts.append([str(observer['folder']), observer['alias'], 'ro'])
        if arm == 'baseline':
            argv += ['--completion', '/native/evidence/completion.json',
                '--completion-sha256', bundle['completion_sha256'],
                '--vendor', '/native/stage/vendors/'+evidence.vendor_name(source),
                '--public-schema', '/native/evidence/arch22_schema/fia_abi_schema.json',
                '--schema-sha256', bundle['completion']['arch22_public_schema_sha256']]
            mounts.append([str(args.build_root), '/native', 'ro'])
        evidence.need(len({row[1] for row in mounts}) == len(mounts), 'CASE_MOUNT_ALIAS_COLLISION')
        cases.append(dict(id=label, image=evidence.IMAGE, timeout=args.timeout, mounts=mounts,
            env=case_environment(args.card), command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '+shlex.join(argv),
            review=dict(experiment_id=label, provider='33.paged_decode / arch22 normal bridge candidate only',
                phase=arm, source_commit=source['manifest']['base_commit'],
                source_manifest_sha256=source['manifest_sha256'], execution_contract=contract,
                installed_required_sources_sha256=evidence.digest(installed_pins),
                build_fingerprint=bundle['summary'], observer_receipt_sha256=observer['sha256'],
                observer_library_sha256=observer['receipt']['observer_library']['sha256'],
                model=str(args.model), cann='8.5.1', soc='Ascend910B4/A2', physical_cards=[args.card], ranks=[0],
                worker='Standard Qwen TP1 eager; actual 1024-token prefill and first KV1025 decode',
                expected_hits=('0 faults; all 28 actual decode boundaries, full 56 calls per phase and 5 x 87 payloads; ' if normal_all else '0 faults; 5 first-boundary captures and 5 x 87 complete payloads per arm; ')
                              + ('private first 4 phases each observe only layer0 Native; 27 other calls are unscoped'
                                 if s2 else 'private first 4 phases additionally observe all 28 native calls'),
                recovery_plan='Five normal phases plus ordinary post-clean request; exact repeated outputs and zero owned scope/hooks/wrappers.',
                resource_budget=dict(timeout_seconds=args.timeout, memory_gib=48, cpus=16),
                diagnostic_only=True, statistics=False, fault_authorized=False,
                stock_private_exact_proven=False, public_coverage_qualified=False, c1=False, c2=False, c3=False,
                E5='NOT_ASSESSED', acceptance='Independent raw review required; this manifest is not a fault certificate.',
                limits=['Stock selected ACLNN owner is not independently observed.',
                    'Mask and read-before logical KV are captured hashes; only later matching KV bins are saved.'])))
    return cases


def prepare(args, *, build_variant=evidence.VARIANT):
    for name in ('source','installed','build_root','observer','model'):
        setattr(args,name,getattr(args,name).resolve(strict=True))
    source = evidence.frozen_source(args.source, args.source_manifest_sha256,build_variant=build_variant)
    evidence.need(evidence.sha(Path(__file__)) == source['pins'][evidence.EXPERIMENT+Path(__file__).name],
                  'PREPARER_NOT_THE_FROZEN_SOURCE')
    helper = evidence.helpers(source)
    installed_pins = evidence.installed_sources(source, args.installed)
    bundle = evidence.build_bundle(args.build_root, source, helper)
    observer = evidence.observer_bundle(args.observer, source)
    contract = helper.contract.execution_contract(args.model, args.installed,
        args.source/evidence.EXPERIMENT/'attention33_worker_probe.py', .35, case_environment(args.card))
    return make_cases(args, source, bundle, observer, contract, installed_pins)


def main(*, build_variant=evidence.VARIANT):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'installed', 'build-root', 'observer', 'model', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--card', type=int, choices=(2, 5), default=2)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.addaudithook(evidence.prohibit_runtime)
    evidence.need(not args.out.exists(), 'FRESH_CASE_MANIFEST_REQUIRED')
    try:
        cases = prepare(args,build_variant=build_variant)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(cases, stream, indent=2, allow_nan=False); stream.write('\n')
    except Exception as error:
        print(json.dumps(dict(status='NORMAL_BRIDGE_PREPARATION_REFUSED', error=repr(error), actual_npu_calls=0)))
        return 2
    print(json.dumps(dict(status='NORMAL_ONLY_ROOT_QUEUE_MANIFEST_PREPARED', cases=2,
        path=str(args.out), sha256=evidence.sha(args.out), actual_npu_calls=0, fault_authorized=False)))
    return 0


if __name__ == '__main__': raise SystemExit(main())
