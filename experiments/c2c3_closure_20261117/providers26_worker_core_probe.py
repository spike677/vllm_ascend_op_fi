"""New-source public WorkerFaultClient C1; forwards are ordinary enqueues."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Keep reviewed /installed distributions first.
from experiments.c2c3_closure_20261117.providers26_group_probe import PROMPT, actual, single_site_scope
from experiments.c2c3_postwrite_20261117.model_identity import file_sha, model_source_identity, vllm_safetensors_identity
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import read_enqueued_identity


PHASES = ('inventory_1', 'inventory_2', 'single_site_reference', 'single_site_lossless',
          'single_site_qualification_off', 'single_site_fault', 'single_site_recovery_off', 'health')
SUCCESS = 'STANDARD_WORKER_NATIVE26_SINGLE_SITE_C1_EVIDENCE_READY_FOR_REVIEW_NOT_C2'
STATISTICS_REFUSAL = 'CORE26_STATISTICS_BOUNDARY_UNADAPTED: complete diagnostic byte/hash reads are not model statistics'
MODULES = tuple(('fault_injection.deployments.'+name, 'fault_injection/deployments/'+name+'.py') for name in
    ('writer26_loader', 'writer26_group', 'writer26_inventory', 'writer26_experiment', 'writer26_public',
     'writer26_fia_typed', 'residual26')) + (
    ('fault_injection.metadata_arrays', 'fault_injection/metadata_arrays.py'),
    ('fault_injection.providers', 'fault_injection/providers.py')) + tuple(
    ('demoncat.runtime.vllm_ascend.worker_extension.'+name,
     'src/demoncat/runtime/vllm_ascend/worker_extension/'+name.replace('.', '/')+'.py') for name in
    ('serving26_core', 'providers.fault26_core', 'providers.fault26', 'providers._scope', 'serving26_group',
     'serving26_inventory', 'serving', 'serving_context', 'call_binding', 'host', 'config', 'runtime', 'identity',
     'extension', 'registry')) + (('demoncat.runtime.vllm_ascend.worker_extension.providers',
        'src/demoncat/runtime/vllm_ascend/worker_extension/providers/__init__.py'),)
SOURCES = tuple(path for _, path in MODULES) + (
    'experiments/c2c3_closure_20261117/providers26_worker_core_probe.py',
    'experiments/c2c3_closure_20261117/providers26_prepare_group_cases.py',
    'experiments/c2c3_closure_20261117/providers26_group_probe.py',
    'experiments/c2c3_postwrite_20261117/natural41_probe.py',
    'experiments/c2c3_postwrite_20261117/natural41_coverage_probe.py',
    'experiments/c2c3_postwrite_20261117/model_identity.py')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def output(value):
    return dict(tokens=list(value.outputs[0].token_ids), text=value.outputs[0].text)


class RecordingExecutor:
    """Persist every original RPC once, retaining the ordinary Host protocol."""
    def __init__(self, executor, folder):
        self.executor, self.folder, self.receipts = executor, folder, []
        folder.mkdir(exist_ok=False)

    def collective_rpc(self, method, timeout=None, args=()):
        name = method if isinstance(method, str) else method.__module__+'.'+method.__qualname__
        row = dict(method=name, args=args)
        try:
            result = self.executor.collective_rpc(method, timeout=timeout, args=args)
            row['responses'] = result
            return result
        except BaseException as error:
            row['error'] = repr(error)
            raise
        finally:
            path = self.folder/(str(len(self.receipts)).zfill(3)+'.json')
            write_json(path, row)
            self.receipts.append(dict(method=name, path=path.name, sha256=file_sha(path)))


def setup(worker, expected_identity, mechanism, tokens, manifest_sha256, sources):
    worker = actual(worker)
    current = worker.demoncat_worker_identity()
    if current.get('worker') != expected_identity or current.get('quarantined') is not False:
        raise RuntimeError('ACTUAL_DEPLOYMENT_WORKER_IDENTITY_CHANGED')
    loaded = {}
    for name, relative in MODULES:
        path = Path(importlib.import_module(name).__file__).resolve(strict=True)
        sha = file_sha(path)
        if not path.is_relative_to(Path('/installed')) or sha != sources[relative]:
            raise RuntimeError('REVIEWED_INSTALLED_WRITER26_SOURCE_CHANGED:'+relative)
        loaded[relative] = dict(path=str(path), sha256=sha)
    from demoncat.runtime.vllm_ascend.worker_extension.serving26_core import install_writer26_core_binding
    bound = install_writer26_core_binding(worker, mechanism=mechanism, prompt_token_ids=tokens,
        evidence_dir='/evidence/worker-core26-runtime', manifest_sha256=manifest_sha256)
    return dict(worker=expected_identity, binding=bound, loaded_sources=loaded,
                model_source=model_source_identity(worker.model_runner.model))


def remove(worker, expected_identity, mechanism):
    worker = actual(worker)
    if worker.demoncat_worker_identity().get('worker') != expected_identity:
        raise RuntimeError('CLEANUP_WORKER_CHANGED')
    from demoncat.runtime.vllm_ascend.worker_extension.serving26_core import remove_writer26_core_binding
    return remove_writer26_core_binding(worker, mechanism)


def drive(llm, client, *, executor, prompt, params, out, report):
    seen, pending, last = [], [], None
    def save():
        report['rpc_receipts'] = list(executor.receipts)
        write_json(out/'result.json', report)
    def action(name, *args, expected='ACTION_COMPLETED'):
        nonlocal last
        last = getattr(client, name)(*args)
        save()
        if last['status'] != expected:
            raise RuntimeError('PUBLIC_CORE_ACTION_FAILED:'+name+':'+last['status'])
        return last['response'].get('provider_state') or {}
    def enqueue():
        nonlocal pending
        if llm.llm_engine.has_unfinished_requests():
            raise RuntimeError('UNRELATED_PENDING_REQUEST')
        if llm.reset_prefix_cache(reset_running_requests=False) is not True:
            raise RuntimeError('FRESH_PREFILL_RESET_UNCONFIRMED')
        pending = list(llm.enqueue([prompt], params, use_tqdm=False))
        mapping = read_enqueued_identity(llm, pending, seen_internal_request_ids=seen)
        if len(pending) != 1:
            raise RuntimeError('ONE_ACTUAL_ENQUEUE_REQUIRED')
        seen.extend(pending)
        return mapping
    def complete(mapping):
        replies = llm.wait_for_completion(use_tqdm=False)
        if len(replies) != 1 or replies[0].request_id != mapping['external_request_id']:
            raise RuntimeError('ACTUAL_REQUEST_MAPPING_CHANGED')
        return output(replies[0])
    try:
        try:
            for name in ('prepare', 'register', 'check', 'enable'):
                action(name)
            for index, phase in enumerate(PHASES):
                mapping = enqueue()
                state = action('advance', pending[0])
                if state['evidence']['phase'] != phase:
                    raise RuntimeError('EXISTING_SINGLE_SITE_PHASE_CHANGED')
                row = dict(phase=phase, mapping=mapping)
                report['requests'].append(row)
                save()
                row['output'] = complete(mapping)
                state = action('query', expected='QUERY_OK')
                evidence = state['evidence']
                observed = evidence['request_index'][-1]
                if (len(evidence['request_index']) != index+1 or observed['phase'] != phase
                        or observed['request_id'] != pending[0] or evidence['failure'] is not None
                        or state['leases'] or state['inflight'] or not state['serving']['needs_collect']):
                    raise RuntimeError('ACTUAL_WRITER_CONSUMER_OR_REQUEST_READBACK_INCOMPLETE')
                row['pre_collect_query'] = executor.receipts[-1]
                state = action('collect_replay')
                originals = state['evidence']['original_observations']
                if originals['status'] != 'CURRENT_COLLECTED' or state['serving']['needs_collect']:
                    raise RuntimeError('EXPLICIT_ORIGINAL_COLLECTION_UNCONFIRMED')
                row['original_observations'] = originals['originals'][-1]
                pending = []
                save()
            state = action('finish')
            evidence = state['evidence']
            if (not evidence['single_site_c1']['evidence_complete'] or evidence['c2'] or evidence['c3']
                    or state['native_registered'] or state['leases'] or state['inflight']
                    or state['serving']['owned_hooks'] or evidence['owned_hooks']):
                raise RuntimeError('SINGLE_SITE_EFFECT_OFF_HEALTH_AND_ZERO_RESOURCES_UNPROVEN')
            report['c1'] = evidence
            scope = last['worker']
            action('disable')
            action('unregister')
            mapping = enqueue()
            healthy = complete(mapping)
            if healthy != report['healthy_before'] or client.host.route(client.config) != scope:
                raise RuntimeError('POST_UNREGISTER_SAME_WORKER_HEALTH_CHANGED')
            report['post_unregister_health'] = dict(mapping=mapping, output=healthy, worker=scope)
            pending = []
            action('detach', expected='DETACHED')
            report['status'] = SUCCESS
        except BaseException as error:
            report.update(status='REFUSED_OR_FAILED', error=repr(error), traceback=traceback.format_exc())
            if pending:
                try:
                    llm.llm_engine.abort_request(pending, internal=True)
                    report['owned_request_abort'] = list(pending)
                    pending = []
                except BaseException as abort_error:
                    report['abort_error'] = repr(abort_error)
            # One explicit cleanup sequence, no new F or candidate selection.
            if last and last.get('ticket'):
                try:
                    action('recover')
                    action('unregister')
                    action('detach', expected='DETACHED')
                except BaseException as cleanup_error:
                    report['cleanup_error'] = repr(cleanup_error)
    finally:
        final = client.snapshot()  # Only one extra full Host snapshot.
        write_json(out/'host_receipt.json', final)
        report['host_receipt'] = dict(path='host_receipt.json', sha256=file_sha(out/'host_receipt.json'))
        report['ownership_retained'], report['worker'] = final['ownership_retained'], final['worker']
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mechanism', choices=('26.residual_stomp', '26.fia_destination'), required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('c1_single_effects',), default='c1_single_effects')
    parser.add_argument('--max-targets', type=int, default=1)
    parser.add_argument('--memory', type=float, default=.55)
    parser.add_argument('--statistics', action='store_true')
    args = parser.parse_args(argv)
    if args.statistics:
        parser.error(STATISTICS_REFUSAL)
    if args.max_targets != 1 or not .1 <= args.memory <= .8 or len(args.manifest_sha256) != 64 or any(
            c not in '0123456789abcdef' for c in args.manifest_sha256):
        parser.error('Reviewed manifest, bounded memory and exactly one single-site plan required')
    out = Path('/evidence/probe')
    out.mkdir(exist_ok=False)
    report = dict(schema='demoncat.writer26.worker_core_probe.v1', status='STARTING', mechanism=args.mechanism,
        stage=args.stage, scope=single_site_scope(args.mechanism), c2=False, c3=False,
        qualification_granted=False, previous_direct_entry_qualification_inherited=False,
        E5='UNADAPTED_COLLECTOR_BOUNDARY', manifest_sha256=args.manifest_sha256, requests=[])
    llm = recorder = client = None
    try:
        source = args.source_root.resolve(strict=True)
        manifest = source/'source_manifest.json'
        document = json.loads(manifest.read_text())
        indexed = {row['path']:row['sha256'] for row in document['files']}
        sources = {relative:file_sha(source/relative) for relative in SOURCES}
        if any(indexed.get(path) != sha for path, sha in sources.items()):
            raise RuntimeError('FROZEN_SOURCE_MANIFEST_CHANGED')
        if Path(__file__).resolve() != source/'experiments/c2c3_closure_20261117/providers26_worker_core_probe.py':
            raise RuntimeError('FROZEN_PUBLIC_CORE_ENTRY_REQUIRED')
        report['source_manifest'] = dict(sha256=file_sha(manifest), base_commit=document['base_commit'])
        report['sources'] = sources
        # Existing bootstrap, before any Torch/vLLM model or Native owner.
        from fault_injection.deployments.writer26_loader import StartupDependencies
        startup = StartupDependencies(bundle='/native', manifest_sha256=args.manifest_sha256)
        report['metadata_startup'] = startup.evidence
        report['weights'] = vllm_safetensors_identity('/model')
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core
        llm = LLM(model='/model', dtype='bfloat16', load_format='safetensors', trust_remote_code=False,
            tensor_parallel_size=1, enforce_eager=True, max_model_len=128, max_num_seqs=1,
            max_num_batched_tokens=128, num_gpu_blocks_override=64, enable_prefix_caching=False,
            enable_chunked_prefill=False, async_scheduling=False, gpu_memory_utilization=args.memory, seed=42,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension', disable_log_stats=True)
        require_host_driven_core(llm)
        tokens = llm.get_tokenizer().encode(PROMPT, add_special_tokens=True)
        prompt, params = {'prompt_token_ids':tokens}, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        before, = llm.generate([prompt], params, use_tqdm=False)  # Existing one warmup only.
        report['healthy_before'], report['actual_prompt_token_ids'] = output(before), tokens
        recorder = RecordingExecutor(llm, out/'worker_core_rpc')
        def bind(worker, config):
            replies = recorder.collective_rpc(setup, timeout=300,
                args=(worker, config.mechanism, tokens, args.manifest_sha256, sources))
            if len(replies) != 1:
                raise RuntimeError('ONE_ACTUAL_DEPLOYMENT_WORKER_REQUIRED')
            report['setup'] = replies[0]
            return {}
        client = WorkerFaultClient(HostAdapter(recorder, prepare_binding=bind, timeout=300),
            FaultConfig(26, args.mechanism.split('.', 1)[1]))
        drive(llm, client, executor=recorder, prompt=prompt, params=params, out=out, report=report)
        if not report['ownership_retained'] and report.get('setup'):
            replies = recorder.collective_rpc(remove, timeout=300, args=(report['worker'], args.mechanism))
            if len(replies) != 1:
                raise RuntimeError('ONE_ACTUAL_DEPLOYMENT_REMOVE_REQUIRED')
            report['deployment_removed'] = replies[0]
            report['rpc_receipts'] = recorder.receipts
    except BaseException as error:
        report.update(status='REFUSED_OR_FAILED', error=repr(error), traceback=traceback.format_exc())
        if isinstance(getattr(error, 'evidence', None), dict):
            report['metadata_startup_failure'] = error.evidence
    finally:
        write_json(out/'result.json', report)
    return 0 if report['status'] == SUCCESS else 2


if __name__ == '__main__':
    raise SystemExit(main())
