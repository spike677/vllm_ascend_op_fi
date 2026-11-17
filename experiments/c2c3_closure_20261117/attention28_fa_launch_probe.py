"""Root-queued fresh FA v3 B/Brepeat/F/unscoped-OFF argument experiment.

New fixed synthetic inputs, actual public FA POD, new source-built vendor and
independent registered-ELF launch evidence. No Worker or historical certificate
is admitted. This script starts no container; only root owns device scheduling.
"""
import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import struct
import sys
import threading
import traceback
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments/engineering_20261117'))
from fault_injection.native_adapters.attention28.launch import LaunchObserver, require, sha
from fault_injection.native_adapters.attention28.fa.launch import verify_bundle, verify_runtime_sdk, verify_fa_launch

EFFECTIVE_SCALE = struct.unpack('<f', struct.pack('<f', 128**-.5))[0]
ENV_KEYS = ('NES_FA_PLAN_JSON', 'NES_FA_LOG_PATH')


def save(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False);stream.write('\n')


def choose_factors(native, target):
    require(set(native) == {'m', 'n_ratio'} and native['m'] in (64, 128) and native['n_ratio'] in (4, 8),
            'FA_NEW_NATIVE_FACTORS_OUTSIDE_SOURCE_CANDIDATES')
    require(target in ('query_tile', 'key_group', 'joint'), 'FA_UNKNOWN_SELECTION_TARGET')
    selected = dict(native)
    if target in ('query_tile', 'joint'):selected['m'] = 192 - native['m']
    if target in ('key_group', 'joint'):selected['n_ratio'] = 12 - native['n_ratio']
    return selected


def typed_plan(bundle, *, batch, call_id, native, selected, experiment_sha):
    data = bundle['completion']
    return dict(schema='nes.fa.typed_plan.v1', operator='FlashAttentionScore',
        mode='controlled' if selected is not None else 'native',
        scope=dict(pid=os.getpid(), native_tid=threading.get_native_id(), token=call_id,
                   phase='operator_execution', request_count=batch),
        context=dict(query_shape=[batch, 16, 2048, 128], key_shape=[batch, 8, 2048, 128],
            value_shape=[batch, 8, 2048, 128], mask_shape=[2048, 2048], dtype='float16', mask_dtype='bool',
            layout='BNSD', scale=EFFECTIVE_SCALE, keep_prob=1, pre_tokens=2147483647, next_tokens=0,
            heads=16, inner_precise=0, sparse_mode=2, optional_inputs_absent=True),
        version=dict(source_commit=data['source_commit'], cann_build='V100R001C11SPC001B243',
                     bridge_abi='nes.fa.cann9.1.s1s2.v1'), requested=dict(native=native, selected=selected),
        certificate=dict(id='fresh-fa-experimental-build-only', source_manifest_sha256=data['patch_receipt_sha256'],
            build_manifest_sha256=bundle['completion_sha256'], validation_sha256=experiment_sha,
            validation_state='experimental'))


def tensor_payload(torch, output, folder, budget):
    """All original API result leaves; readback happens after collector stop."""
    result = {}
    def visit(value, name, depth=0):
        require(depth <= 3, 'FA_RESULT_CONTAINER_DEPTH_BOUND')
        if isinstance(value, torch.Tensor):
            size = value.numel() * value.element_size();budget[0] += size
            require(size <= 128 * 1024**2 and budget[0] <= 512 * 1024**2, 'FA_COMPLETE_PAYLOAD_BYTE_BOUND')
            cpu = value.detach().contiguous().cpu()
            raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
            path = folder / (name + '.bin');path.write_bytes(raw)
            result[name] = dict(kind='tensor', shape=list(cpu.shape), dtype=str(cpu.dtype), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(), finite=bool(torch.isfinite(cpu).all()), path=str(path))
        elif isinstance(value, (list, tuple)):
            require(len(value) <= 16, 'FA_RESULT_CONTAINER_LENGTH_BOUND')
            result[name] = dict(kind=type(value).__name__, length=len(value))
            for index, child in enumerate(value):visit(child, name + '-' + str(index), depth + 1)
        else:
            require(value is None or type(value) in (int, float, bool, str), 'FA_UNKNOWN_RESULT_LEAF')
            result[name] = dict(kind=type(value).__name__, value=value)
    visit(output, 'output')
    return result


def payload_identity(payload):
    return {name: {key: value for key, value in row.items() if key != 'path'} for name, row in payload.items()}


def make_collector(folder, backend):
    import msprobe
    from msprobe.pytorch import PrecisionDebugger
    config = dict(task='statistics', level='mix', precision='high', rank=[], step=[0],
        dump_path=str(folder / 'observation'), async_dump=False,
        statistics=dict(scope=[], list=[], data_mode=['all'], summary_mode='statistics', tensor_list=[]))
    save(folder / 'config.json', config)
    package = Path(msprobe.__file__).resolve().parent
    support = Path(inspect.getsourcefile(PrecisionDebugger)).parents[1] / 'api_dump/support_wrap_ops.yaml'
    require(support.is_file() and '- npu_fusion_attention_v3' in support.read_text(), 'FA_ORIGINAL_COLLECTOR_API_SUPPORT_REQUIRED')
    save(folder / 'collector_identity.json', dict(package=str(package), edited_by_experiment=False,
        files={p.relative_to(package).as_posix(): sha(p) for p in package.rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.yaml')},
        step_meaning={'0': 'second identical measured standalone FA call after fresh native discovery'},
        original_api_support_sha256=sha(support),
        statistics_backend=backend.identity() if backend is not None else None))
    return PrecisionDebugger(config_path=str(folder / 'config.json'))


def execute(args, bundle, receipt):
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='demoncat.fa.new_launch_probe.v1', status='STARTED', mode=args.mode, batch=args.batch,
        selection_target=args.selection_target, calls=[], failures=[], native_launch_verified=False,
        public_worker_qualified=False, c2=False, c3=False, historical_qualification_inherited=False,
        statistical_first_difference='SEPARATE_RAW_COMPARATOR_REQUIRED', actual_npu_api_calls=0,
        completion_sha256=bundle['completion_sha256'], statistics_backend=args.statistics_backend if args.statistics else None,
        limitation='Standalone synthetic QKV; no stock-package bridge or natural Worker qualification')
    def checkpoint():(args.out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    checkpoint();backend = None
    try:
        require(os.environ.get('ACLNN_CACHE_LIMIT') == '0' and os.environ.get('TASK_QUEUE_ENABLE') == '0',
                'FA_UNCACHED_SYNCHRONOUS_PREIMPORT_ENV_REQUIRED')
        require(os.environ.get('ASCEND_CUSTOM_OPP_PATH') == bundle['vendor'], 'FA_EXACT_FRESH_VENDOR_ENV_REQUIRED')
        report['sdk'] = verify_runtime_sdk(bundle, receipt['sdk'])
        import torch
        import torch_npu
        from model_capture import statistics_model_call
        from fresh_fia_qualification import mapped_vendor
        require(callable(getattr(torch_npu, 'npu_fusion_attention_v3', None)), 'FA_REAL_V3_API_REQUIRED')
        torch.use_deterministic_algorithms(True)
        torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.set_device_limit(0, cube_num=16, vector_num=32)
        require(torch.npu.get_device_limit(0) == dict(cube_core_num=16, vector_core_num=32), 'FA_ACTUAL_CORE_BUDGET_DIFFERS')
        report['execution'] = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
            actual_core_limit=torch.npu.get_device_limit(0), jit_compile=False, api='torch_npu.npu_fusion_attention_v3')
        cpu = []
        for heads, factor, offset in ((16, 17, 3), (8, 29, 5), (8, 43, 7)):
            value = ((torch.arange(heads * 2048 * 128, dtype=torch.int32) * factor + offset) % 257 - 128).to(torch.float16) / 128
            cpu.append(value.reshape(1, heads, 2048, 128).repeat(args.batch, 1, 1, 1).contiguous())
        cpu.append(torch.ones((2048, 2048), dtype=torch.bool).triu(1))
        digest = lambda tensor: hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        original_hashes = [digest(tensor) for tensor in cpu]
        inputs = tuple(tensor.to('npu:0') for tensor in cpu);torch.npu.synchronize()
        save(args.out / 'inputs.json', dict(rule='((i*factor+offset)%257-128)/128; identical request copies',
            qkv_parameters=[[16,17,3],[8,29,5],[8,43,7]], batch=args.batch, old_capture_reused=False,
            tensors=[dict(shape=list(t.shape), dtype=str(t.dtype), sha256=h) for t, h in zip(cpu, original_hashes)]))
        experiment = dict(schema='demoncat.fa.experimental_scope.v1', completion_sha256=bundle['completion_sha256'],
            input_manifest_sha256=sha(args.out / 'inputs.json'), native_discovery_required=True,
            selection_target=args.selection_target, compiled_key_is_not_launch_evidence=True,
            historical_certificate_used=False, runtime_qualification_granted=False)
        save(args.out / 'experiment_scope.json', experiment)
        experiment_sha = sha(args.out / 'experiment_scope.json')
        observer = LaunchObserver(receipt['observer_library']['path'], receipt['observer_library']['sha256'])
        active = {}
        class FaProbe(torch.nn.Module):
            def forward(self, query, key, value, mask):
                with observer.scope(active['call_id']):
                    report['actual_npu_api_calls'] += 1
                    result = torch_npu.npu_fusion_attention_v3(query, key, value, head_num=16, input_layout='BNSD',
                        pse=None, padding_mask=None, atten_mask=mask, prefix=None, actual_seq_qlen=None,
                        actual_seq_kvlen=None, sink=None, scale=128**-.5, keep_prob=1.0,
                        pre_tockens=2147483647, next_tockens=0, inner_precise=0, sparse_mode=2,
                        gen_mask_parallel=True, sync=False, softmax_layout='')
                    torch.npu.synchronize();observer.synchronize()
                return result
        model = FaProbe().eval()
        phases = ['baseline', 'baseline_repeat', 'fault', 'off'] if args.mode == 'sequence' else ['native_discovery', args.mode, args.mode + '_repeat']
        native, selected, budget = None, None, [0]
        if args.statistics and args.statistics_backend == 'cpu-fp64-v1':
            from attention_cpu_statistics import CpuFp64Statistics
            backend = CpuFp64Statistics();backend.install()
        for index, phase in enumerate(phases):
            folder = args.out / f'{index}-{phase}';folder.mkdir()
            call_id = uuid.uuid4().hex;active['call_id'] = call_id
            controlled = phase.startswith('fault');off = phase == 'off'
            if controlled:
                require(native is not None, 'FA_NEW_NATIVE_DISCOVERY_REQUIRED')
                selected = choose_factors(native, args.selection_target)
            plan = None if off else typed_plan(bundle, batch=args.batch, call_id=call_id, native=native,
                selected=selected if controlled else None, experiment_sha=experiment_sha)
            require(not any(name in os.environ for name in ENV_KEYS), 'FA_FOREIGN_PLAN_SCOPE')
            environment = {} if off else dict(NES_FA_PLAN_JSON=json.dumps(plan, separators=(',', ':'), allow_nan=False),
                                               NES_FA_LOG_PATH=str(folder / 'host.jsonl'))
            save(folder / 'requested_plan.json', dict(plan=plan, unscoped_off=off))
            report['active_call'] = dict(index=index, phase=phase, call_id=call_id, folder=str(folder));checkpoint()
            debugger = make_collector(folder, backend) if args.statistics and index == 2 else None
            os.environ.update(environment)
            try:
                scope = statistics_model_call(debugger, model, torch.npu.synchronize) if debugger is not None else nullcontext()
                with scope, torch.inference_mode():output = model(*inputs)
                torch.npu.synchronize()
            finally:
                changed = [name for name, value in environment.items() if os.environ.get(name) != value]
                for name, value in environment.items():
                    if os.environ.get(name) == value:del os.environ[name]
                if changed:report['failures'].append(dict(stage='scope_cleanup', changed_owners=changed))
            if debugger is not None:debugger.step()
            host = None
            if not off:
                hosts = [json.loads(line) for line in (folder / 'host.jsonl').read_text().splitlines()]
                require(len(hosts) == 1 and hosts[0]['token'] == call_id and hosts[0]['plan'] == plan, 'FA_ONE_ACTUAL_MATCHED_HOST_CALLBACK_REQUIRED')
                host = hosts[0]
                require(host['mode'] == ('controlled' if controlled else 'native')
                        and host['selection_applied'] is controlled, 'FA_ACTUAL_SELECTOR_MODE_DIFFERS')
            require([digest(tensor) for tensor in inputs] == original_hashes, 'FA_COMPLETE_INPUT_BYTES_CHANGED')
            payload = tensor_payload(torch, output, folder, budget)
            require(all(row.get('finite', True) for row in payload.values()), 'FA_NONFINITE_COMPLETE_OUTPUT')
            launch = verify_fa_launch(os.environ['DCAT_ATTN_OBSERVER_DIR'], call_id, bundle, host,
                                     folder / 'final-tiling.bin', unscoped_off=off)
            pod = launch['typed'];actual = dict(m=pod['coreParams.s1BaseSize'], n_ratio=pod['coreParams.nRatio'])
            require(pod['inputParams.bSize'] == args.batch and pod['inputParams.n2Size'] == 8
                    and pod['inputParams.gSize'] == 2 and pod['inputParams.s1Size'] == pod['inputParams.s2Size'] == 2048
                    and pod['inputParams.dSize'] == 128 and pod['inputParams.keepProb'] == 1
                    and pod['inputParams.scaleValue'] == EFFECTIVE_SCALE and pod['coreParams.s2BaseSize'] == 128,
                    'FA_FINAL_PUBLIC_WORKLOAD_OR_BASE_TILE_DIFFERS')
            if native is None:native = actual;choose_factors(native, args.selection_target)
            require(actual == (selected if controlled else native), 'FA_FINAL_FACTORS_DIFFER_FROM_THIS_RUN_REQUEST')
            if host is not None:
                require(host['native_factors'] == native and host['selected_factors'] == actual, 'FA_RECOMPUTED_NATIVE_FACTORS_CHANGED')
            row = dict(index=index, phase=phase, call_id=call_id, actual_factors=actual, launch=launch, payload=payload,
                full_inputs_unchanged=True, original_api_return_preserved=True, statistics_step=0 if debugger is not None else None)
            report['calls'].append(row);save(folder / 'call.json', row);checkpoint()
            report['active_call'] = None
        report['mapped_fresh_vendor'] = mapped_vendor(dict(vendor_root=bundle['vendor'], files=bundle['files']))
        calls = report['calls'];reference = calls[0]
        if args.mode == 'sequence':
            report['complete_payload_exact_vs_baseline'] = {row['phase']: payload_identity(row['payload']) == payload_identity(reference['payload']) for row in calls}
            require(all(report['complete_payload_exact_vs_baseline'][name] for name in ('baseline_repeat', 'off')), 'FA_COMPLETE_BREPEAT_OR_OFF_DIFFERS')
            require(all(row['launch']['semantic_pod'] == reference['launch']['semantic_pod']
                        and row['launch']['tiling_key'] == reference['launch']['tiling_key']
                        and row['launch']['block_dim'] == reference['launch']['block_dim']
                        and row['launch']['schedule_mode'] == reference['launch']['schedule_mode']
                        for row in (calls[1], calls[3])), 'FA_FINAL_BREPEAT_OR_UNSCOPED_OFF_GEOMETRY_DIFFERS')
            report['fault_complete_payload_changed'] = not report['complete_payload_exact_vs_baseline']['fault']
        else:
            require(payload_identity(calls[1]['payload']) == payload_identity(calls[2]['payload']), 'FA_COMPLETE_MEASURED_REPEAT_DIFFERS')
        require(not report['failures'] and not any(name in os.environ for name in ENV_KEYS), 'FA_SCOPE_RECOVERY_INCOMPLETE')
        report.update(status='FRESH_FA_ARGUMENTS_AND_SEQUENCE_VERIFIED_REQUIRES_REVIEW', native_launch_verified=True,
            scope='FIXED_STANDALONE_FA_ONLY', stock_original_package_bridge_verified=False)
    except BaseException as error:
        report.update(status='FAILED_WITH_EVIDENCE', execution_error=repr(error), traceback=traceback.format_exc())
    finally:
        if backend is not None:
            try:
                if backend.installed:backend.close()
                report['statistics_backend_report'] = backend.report()
            except BaseException as error:
                report['failures'].append(dict(stage='statistics_cleanup', error=repr(error)))
            if report.get('statistics_backend_report', {}).get('status') != 'NEW_COLLECTION_COMPLETE':
                report['status'] = 'FAILED_WITH_EVIDENCE';report['native_launch_verified'] = False
        if report['failures']:report['status'] = 'FAILED_WITH_EVIDENCE';report['native_launch_verified'] = False
        checkpoint()
    print(json.dumps({k: report[k] for k in ('status', 'mode', 'native_launch_verified', 'c2', 'c3')}))
    return 2 if report['status'] == 'FAILED_WITH_EVIDENCE' else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-root', type=Path, required=True)
    p.add_argument('--completion-sha256', required=True)
    p.add_argument('--observer-receipt', type=Path, required=True)
    p.add_argument('--observer-receipt-sha256', required=True)
    p.add_argument('--mode', choices=('sequence', 'baseline', 'fault'), default='sequence')
    p.add_argument('--batch', type=int, choices=(1, 8), default=8)
    p.add_argument('--selection-target', choices=('query_tile', 'key_group', 'joint'), default='joint')
    p.add_argument('--statistics', action='store_true')
    p.add_argument('--statistics-backend', choices=('original', 'cpu-fp64-v1'), default='original')
    p.add_argument('--out', type=Path, default=Path('/evidence/run'))
    p.add_argument('--experimental-qualification', action='store_true')
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    require(a.experimental_qualification and (not a.statistics or a.mode in ('baseline', 'fault')),
            'FA_EXPLICIT_NEW_EXPERIMENT_AND_SEPARATE_BF_STATISTICS_REQUIRED')
    require(a.out.resolve().is_relative_to('/evidence') and not a.out.exists(), 'FA_FRESH_PRIVATE_OUTPUT_REQUIRED')
    bundle = verify_bundle(a.build_root, a.completion_sha256)
    require(sha(a.observer_receipt) == a.observer_receipt_sha256, 'FA_OBSERVER_RECEIPT_DIFFERS')
    receipt = json.loads(a.observer_receipt.read_bytes())
    require(receipt.get('observer_compiled') is True and receipt.get('actual_npu_calls') == 0
            and all(sha(row['path']) == row['sha256'] for row in receipt['headers']), 'FA_MATCHED_OBSERVER_SDK_REQUIRED')
    lib = receipt['observer_library'];require(sha(lib['path']) == lib['sha256'], 'FA_OBSERVER_LIBRARY_DIFFERS')
    if not a.child:
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('ASCEND_CUSTOM_OPP_PATH')
                and not any(name.startswith(('NES_FA_', 'FAULT28_', 'DCAT33_TND_', 'DCAT_ATTN_')) for name in os.environ),
                'FA_CLEAN_PREIMPORT_PROCESS_REQUIRED')
        runtime = a.out.parent / 'runtime';runtime.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, LD_PRELOAD=lib['path'], DCAT_ATTN_OBSERVER_DIR=str(runtime), ASCEND_CUSTOM_OPP_PATH=bundle['vendor'])
        # This isolated single-operator process has no stock vLLM custom APIs.
        # The setting is not a Worker coexistence/loading contract.
        env['LD_LIBRARY_PATH'] = str(Path(bundle['vendor']) / 'op_api/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
        os.execve(sys.executable, [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--child'], env)
    return execute(a, bundle, receipt)


if __name__ == '__main__':raise SystemExit(main())
