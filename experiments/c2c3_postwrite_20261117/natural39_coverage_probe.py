"""Root-queued natural HF C1 -> all-target C2 -> bounded C3 probe."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


SOURCES = ('fault_injection/deployments/postwrite.py', 'fault_injection/deployments/postwrite_boundaries.py',
    'fault_injection/deployments/postwrite_inventory.py', 'fault_injection/deployments/postwrite_evidence.py',
    'fault_injection/deployments/postwrite_structure.py',
    'fault_injection/deployments/postwrite_coverage.py', 'fault_injection/coverage.py',
    'fault_injection/coverage_session.py', 'fault_injection/runtime_discovery.py',
    'fault_injection/native_adapters/postwrite/postwrite.cpp', 'fault_injection/native_adapters/postwrite/postwrite.h',
    'experiments/c2c3_postwrite_20261117/natural39_coverage_probe.py',
    'experiments/c2c3_postwrite_20261117/model_identity.py')


def collector_window(folder, steps):
    import msprobe
    import torch
    from msprobe.pytorch import PrecisionDebugger
    from experiments.engineering_20261117.model_capture import statistics_model_call
    folder.mkdir(parents=True, exist_ok=False)
    config = dict(task='statistics', level='mix', precision='high', rank=[], step=list(range(steps)),
        dump_path=str(folder/'observation'), async_dump=False,
        statistics=dict(scope=[], list=[], data_mode=['all'], summary_mode='statistics', tensor_list=[]))
    (folder/'config.json').write_text(json.dumps(config, indent=2))
    package = Path(msprobe.__file__).parent
    from experiments.c2c3_postwrite_20261117.model_identity import file_sha
    identity = dict(package=str(package), files={path.relative_to(package).as_posix():file_sha(path)
                    for path in package.rglob('*.py')})
    (folder/'collector_identity.json').write_text(json.dumps(identity, indent=2))
    collector = PrecisionDebugger(config_path=str(folder/'config.json'))
    @contextmanager
    def window(actual):
        with statistics_model_call(collector, actual, torch.npu.synchronize):yield
        collector.step()
    return window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--build-fingerprint', required=True)
    parser.add_argument('--arm', choices=('baseline', 'fault'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--statistics', action='store_true')
    parser.add_argument('--try-c3', action='store_true')
    parser.add_argument('--max-live-leases', type=int, default=1,
                        help='Explicit simultaneous output-lease limit for every-match only')
    parser.add_argument('--max-seconds', type=int, default=1200)
    parser.add_argument('--prompt', default='Explain what a checksum is in one sentence.')
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 1500:parser.error('--max-seconds must be between 1 and 1500')
    if not 1 <= args.max_live_leases <= 256:parser.error('--max-live-leases must be between 1 and 256')
    if args.max_live_leases > 1 and (not args.try_c3 or args.arm != 'fault'):
        parser.error('--max-live-leases above one requires --arm fault and --try-c3')
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(schema='demoncat.postwrite39.natural_coverage_probe.v1', status='STARTED', arm=args.arm,
        mechanism='39.output_proxy', evidence_kind='natural_pretrained_model', worker_extension=False,
        public_qualification_granted=False, c1={}, build_fingerprint=args.build_fingerprint)
    provider = group = None
    def save():
        (args.out/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    try:
        import torch
        import torch_npu
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from fault_injection.coverage_session import CoverageScope
        from fault_injection.deployments.postwrite import NativePostwriteLibrary, Fault39OutputProxyProvider
        from fault_injection.deployments.postwrite_coverage import ProjectionCoverage
        from experiments.c2c3_postwrite_20261117.model_identity import (file_sha, digest,
            safetensors_identity, model_source_identity)
        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = False
        report['weights'] = safetensors_identity(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        network = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16,
            attn_implementation='eager').eval().npu()
        if safetensors_identity(args.model) != report['weights']:
            raise RuntimeError('MODEL_ARTIFACTS_CHANGED_DURING_LOAD')
        class Prefill(torch.nn.Module):
            def __init__(self, actual):
                super().__init__(); self.actual = actual
            def forward(self, ids, mask):
                return self.actual(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
        model = Prefill(network).eval()
        report['model_source'] = model_source_identity(model)
        report['source_sha256'] = {name:file_sha(ROOT/name) for name in SOURCES}
        native = NativePostwriteLibrary(args.library, args.sha256)
        if native._snapshot is None:raise RuntimeError('FRESH_POSTWRITE_READ_SPAN_BUILD_REQUIRED')
        encoded = tokenizer(args.prompt, return_tensors='pt', truncation=True, max_length=32)
        inputs = (encoded['input_ids'].npu(), encoded['attention_mask'].npu())
        environment = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
            device=torch.npu.get_device_name(0), native_library_sha256=native.sha256,
            build_fingerprint=args.build_fingerprint, loaded_library_path=native.path,
            transport_lease_capacity=native.lease_capacity,
            transport_lease_byte_capacity=native.lease_byte_capacity)
        report['environment'] = environment
        provider = Fault39OutputProxyProvider(model, native)
        report['c1']['qualification'] = provider.qualify(*inputs)
        _, report['c1']['lossless'] = provider.run('baseline', *inputs)
        _, report['c1']['fault'] = provider.run('fault', *inputs)
        _, report['c1']['off'] = provider.run('off', *inputs)
        save()
        scope = CoverageScope(provider.mechanism, digest(dict(prompt=args.prompt,
            input_sha256=report['c1']['qualification']['input_sha256'], max_length=32, use_cache=False)),
            digest(environment), digest(report['source_sha256']), evidence_kind='natural_pretrained_model',
            model_weights_sha256=report['weights']['weights_sha256'],
            model_source_sha256=report['model_source']['source_sha256'])
        group = ProjectionCoverage(provider, scope, max_seconds=args.max_seconds,
                                   max_live_leases=args.max_live_leases)
        report['discovery'] = dict(eligible_targets=len(group.eligible), projection_calls=len(group.call_order),
            structural_modules=len(group.module_universe),structural_noncall_inventory=group.structural_inventory,
            max_fault_forwards=group.max_forwards, requested_max_live_leases=group.max_live_leases, scope=scope.__dict__)
        save()
        window = collector_window(args.out/'capture', group.max_forwards) if args.statistics else None
        report['c2'] = group.run('once', *inputs, baseline_control=args.arm == 'baseline', model_observation=window)
        save()
        result = report['c2']
        if not result['recovery']['complete'] or provider.query()['leases']:
            raise RuntimeError('C2_GROUP_RECOVERY_NOT_CONFIRMED')
        if args.arm == 'baseline':
            if (not result['all_targets_controlled_once'] or result['hits'] != 0
                    or result['provider_failure'] is not None
                    or any(item['changed_elements'] for item in result['execution_artifacts'])):
                raise RuntimeError('C2_ALL_TARGET_BASELINE_CONTROL_FAILED')
            report['status'] = 'NATURAL_C2_BASELINE_CONTROL_RECOVERED'
        else:
            if not result['contract_satisfied']:raise RuntimeError('NATURAL_C2_CONTRACT_NOT_SATISFIED')
            report['status'] = 'NATURAL_C2_CONTRACT_RECORDED_PENDING_ROOT_REVIEW'
            if args.try_c3:
                report['c3'] = group.run('every_match', *inputs, fault_repeats=2)
                if not report['c3']['recovery']['complete']:
                    raise RuntimeError('C3_ATTEMPT_RECOVERY_NOT_CONFIRMED')
                report['c3_status'] = ('CONTRACT_RECORDED_PENDING_ROOT_REVIEW' if report['c3']['contract_satisfied']
                                       else 'REFUSED_WITH_EXPLICIT_COVERAGE_AND_RECOVERY')
        _, report['final_off'] = provider.run('off', *inputs)
        report['live_leases'] = provider.query()['leases']
    except BaseException as error:
        report.update(status='FAILED_WITH_EVIDENCE', error=repr(error), traceback=traceback.format_exc())
    finally:
        if provider is not None:
            try:
                if group is not None and group.session is not None and not group.session.recovery_report()['complete']:
                    report['recovery_retry'] = group.retry_recovery(*inputs)
                    if not report['recovery_retry']['complete']:
                        raise RuntimeError('POSTWRITE_GROUP_RECOVERY_STILL_PENDING')
                else:
                    provider.recover()
            except BaseException as error:report['recovery_error'] = repr(error)
            report['live_leases'] = provider.query()['leases']
        if group is not None and group.session is not None and 'c2' not in report:
            report['incomplete_group'] = group.session.report()
        if group is not None and group.discovery_diagnostics is not None:
            report['independent_discovery']=group.discovery_diagnostics
        save()
    print(json.dumps({key:report[key] for key in ('status','arm','error','live_leases') if key in report}))
    return 0 if report['status'].startswith('NATURAL_C2_') and not report.get('recovery_error') else 2


if __name__ == '__main__':raise SystemExit(main())
