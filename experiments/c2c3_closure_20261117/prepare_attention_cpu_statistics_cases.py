"""Prepare four fresh CPU-FP64 B/F cases for root's existing NPU queue."""
import argparse
import json
from pathlib import Path
import shlex

from attention_cpu_statistics import BACKEND, require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'registered-legal', 'registered-replay', 'schema', 'observer', 'cpu-preflight', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--card', type=int, choices=(2,5), default=5)
    args = parser.parse_args()
    cpu = json.loads(args.cpu_preflight.read_bytes())
    require(cpu.get('status') == 'CPU_COLLECTOR_AND_WRITER_VERIFIED'
            and cpu.get('actual_npu_tensor_operations') == 0
            and cpu.get('backend',{}).get('backend') == BACKEND
            and cpu['backend']['adapter_sha256'] == sha(args.source/'experiments/c2c3_closure_20261117/attention_cpu_statistics.py')
            and cpu['backend']['original_method_restored'] is True, 'MATCHING_CPU_COLLECTOR_PREFLIGHT_REQUIRED')
    observer = args.observer/'attention-preflight/result.json'
    schema = args.schema/'fia_abi_schema.json'
    cases = []
    for label, variant, registered, vendor in (
            ('legal', 'legal_tiling', args.registered_legal, 'dcat_f28_legal_transformer'),
            ('replay', 'denominator_replay', args.registered_replay, 'dcat_f28_replay_transformer')):
        completion = registered/'completion.json'
        metadata = json.loads(completion.read_bytes())
        require(metadata['variant'] == variant
                and metadata['status'] == 'CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION',
                'MATCHING_NEW_BUILD_RECEIPT_REQUIRED')
        for phase in ('baseline', 'fault'):
            case_id = 'fia-'+label+'-cpu-fp64-'+phase
            argv = ['python3', '/repo/experiments/c2c3_closure_20261117/attention28_launch_probe.py',
                '--completion', '/registered/completion.json', '--completion-sha256', sha(completion),
                '--vendor', '/registered/vendors/'+vendor,
                '--public-schema', '/schema/fia_abi_schema.json', '--schema-sha256', sha(schema),
                '--observer-receipt', '/build/attention-preflight/result.json',
                '--observer-receipt-sha256', sha(observer), '--phase', phase, '--batch', '8',
                '--out', '/evidence/run', '--experimental-qualification', '--statistics',
                '--statistics-backend', 'cpu-fp64-v1']
            cases.append(dict(id=case_id, image='cann:9.1.0-910b-torch2.10.0_npu-atk26.7.8', timeout=900,
                mounts=[[str(args.source),'/repo','ro'],[str(registered),'/registered','ro'],
                        [str(args.schema),'/schema','ro'],[str(args.observer),'/build','ro']],
                env=dict(PYTHONPATH='/repo:/repo/src:/repo/fault39/src:/msprobe_packages',
                         OMP_NUM_THREADS='4', ACLNN_CACHE_LIMIT='0'),
                command='source /usr/local/Ascend/cann-9.1.0/set_env.sh\nexec '+shlex.join(argv),
                review=dict(experiment_id=case_id,provider='28.fia.'+variant,
                    model='standalone fixed FIA input; no model qualification',cann='9.1.0',
                    soc='Ascend910B4/A2',physical_cards=[args.card],ranks=1,
                    worker='standalone synchronous API',build_fingerprint=sha(completion),expected_hits=2,
                    resource_budget=dict(timeout_seconds=900,memory_gib=32,cpus=8),
                    recovery_plan='Two synchronous scoped calls; drain, restore only owned statistics method; no public C2/C3 grant.',
                    statistics_backend=BACKEND,statistics_cpu_preflight_sha256=sha(args.cpu_preflight),
                    independent_e5='New B/F raw dump+stack+construct+config only; rtol=1e-5, atol=1e-8; preserve null/NaN/mismatch.')))
    with args.out.open('x') as stream:
        json.dump(cases, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
