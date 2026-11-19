"""Prepare one root-queued ordinary ROPE22 acceptance; read files, never execute."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex

from experiments.worker_core_v11_20261118 import rope22_task_probe as probe
from experiments.c2c3_closure_20261117.queue_contract import validate_cases

MODEL = '/home/gyl/models/Qwen3-0.6B'


def make_case(*, source, installed, source_manifest_sha256, card=5, timeout=1200, model=MODEL):
    probe.require(isinstance(source_manifest_sha256, str) and re.fullmatch('[0-9a-f]{64}', source_manifest_sha256),
                  'EXACT_SOURCE_MANIFEST_HASH_REQUIRED')
    probe.require(card in (2, 5) and type(timeout) is int and 240 <= timeout <= 1800,
                  'BOUNDED_ROOT_SCHEDULED_SINGLE_CARD_REQUIRED')
    probe.require(model == MODEL, 'FIXED_ACCEPTED_ROPE_DEPLOYMENT_REQUIRED')
    name = 'rope22-single-task-four-requests'
    argv = ['python3', '-B', '-u', '/source/experiments/worker_core_v11_20261118/rope22_task_probe.py',
        '--model', '/model', '--source-manifest-sha256', source_manifest_sha256,
        '--out', '/evidence/probe', '--memory', '.65', '--deadline-seconds', str(min(timeout-120, 1500))]
    row = dict(id=name, image=probe.IMAGE, timeout=timeout,
        mounts=[[str(source), '/source', 'ro'], [str(installed), '/installed', 'ro'], [model, '/model', 'ro']],
        env=dict(PYTHONPATH='/installed:/source:/source/src:/source/fault39/src',
            VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',
            VLLM_ALLOW_INSECURE_SERIALIZATION='1', VLLM_ASCEND_ENABLE_NZ='0',
            OMP_NUM_THREADS='4', VLLM_HOST_IP='127.0.0.1', GLOO_SOCKET_IFNAME='lo',
            HCCL_CONNECT_TIMEOUT='120', HCCL_EXEC_TIMEOUT='120', TASK_QUEUE_ENABLE='0'),
        command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '+shlex.join(argv),
        review=dict(experiment_id=name, provider='public HostAdapter/WorkerFaultClient 22.wrong_row ordinary task',
            model=model, cann='8.5.1', soc='Ascend910B4/A2', physical_cards=[card], ranks=[0],
            worker='standard TP1 serial eager Qwen3-0.6B BF16; two independent tasks, each four actual normal/F/OFF/health requests',
            source_manifest_sha256=source_manifest_sha256,
            build_fingerprint=dict(kind='PYTHON_RUNTIME_SOURCE_ONLY_NO_NEW_NATIVE_BUILD',
                source_manifest_sha256=source_manifest_sha256),
            expected_hits='One actual legal fused position replacement per F request; automatic first source-bound RoPE with final reader/consumer; no lookup fallback',
            resource_budget=dict(timeout_seconds=timeout, soft_seconds=min(timeout-120, 1500), recovery_seconds=120,
                memory_gib=64, cpus=8, nominal_actual_requests=8, max_actual_requests_with_recovery=12, independent_tasks=2,
                target_limit=1, fault_request_limit=1, normal_return_observation_limit=1),
            recovery_plan='Preserve first failure; abort only owned pending request; public recover may enqueue OFF and '
                'health once. Never retry F or reset attempted. Keep owner on unconfirmed cleanup. Full hidden/logits '
                'OFF and health hashes must equal the actual normal request.',
            execution_scope='BLUE_ZONE_PUBLIC_HOST_ACCEPTANCE_NOT_INNER_MAIN_PROJECT',
            statistics=False, dump=False, full_census=False, C1=False, C2=False, C3=False, E5=False,
            existing_queue_required='experiments/c2c3_closure_20261117/run_queue_124.py',
            raw_evidence=['/evidence/probe/result.json', '/evidence/probe/host_receipt.json', '/evidence/process.log']))
    return validate_cases([row], [card])


def prepare(args):
    source = probe.source_inputs(args.source, args.source_manifest_sha256)
    probe.require(probe.sha(__file__) == source['pins']['experiments/worker_core_v11_20261118/prepare_rope22_single_task.py'],
                  'ACTUAL_PREPARER_DIFFERS_FROM_FROZEN_SOURCE')
    installed = {}
    for relative, digest in source['pins'].items():
        target = (relative.removeprefix('src/') if relative.startswith('src/demoncat/')
                  else relative if relative.startswith('fault_injection/') else None)
        if target is not None:
            path = (args.installed/target).resolve(strict=True)
            probe.require(path.is_relative_to(args.installed.resolve()) and probe.sha(path) == digest,
                          'INSTALLED_TASK_MODULE_DIFFERS:'+target)
            installed[target] = digest
    rows = make_case(source=args.source, installed=args.installed, source_manifest_sha256=args.source_manifest_sha256,
        model=str(args.model), card=args.card, timeout=args.timeout)
    rows[0]['review'].update(source_commit=source['commit'], required_sources=source['pins'],
        installed_module_count=len(installed), installed_module_manifest_sha256=hashlib.sha256(
            json.dumps(installed, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'installed', 'out'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--model', type=Path, default=Path(MODEL))
    parser.add_argument('--card', type=int, choices=(2, 5), default=5)
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args(argv)
    probe.require(not args.out.exists(), 'FRESH_CASE_MANIFEST_REQUIRED')
    rows = prepare(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    probe.write(args.out, rows, fresh=True)
    print(json.dumps(dict(status='ONE_CASE_PREPARED_NOT_EXECUTED', case_manifest=str(args.out),
        sha256=probe.sha(args.out), npu_calls=0, qualification_granted=False)))


if __name__ == '__main__': main()
