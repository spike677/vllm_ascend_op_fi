"""Root-scheduled public output39 C1; every forward is an ordinary enqueue."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'fault39/src')]
from experiments.c2c3_postwrite_20261117.natural41_probe import PROMPT, actual_worker, output
from experiments.c2c3_postwrite_20261117.natural41_coverage_probe import actual_vllm_weights
from experiments.c2c3_postwrite_20261117.model_identity import file_sha, model_source_identity
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import read_enqueued_identity


PHASES = ('observe', 'observe_repeat', 'baseline', 'fault', 'off', 'health')
SUCCESS = 'STANDARD_WORKER_OUTPUT39_C1_EVIDENCE_READY_FOR_REVIEW'
STATISTICS_REFUSAL = ('OUTPUT39_STATISTICS_BOUNDARY_UNADAPTED: root/logits complete_tensor diagnostics '
    'would enter mix API statistics; no filtering or diagnostic first-difference is accepted')
MODULES = tuple(('fault_injection.deployments.'+name, 'fault_injection/deployments/'+name+'.py')
    for name in ('postwrite', 'postwrite_boundaries', 'postwrite_inventory', 'postwrite_observed')) + tuple(
    ('demoncat.runtime.vllm_ascend.worker_extension.'+name,
     'src/demoncat/runtime/vllm_ascend/worker_extension/'+name.replace('.', '/')+'.py')
    for name in ('serving39_output', 'providers.fault39_output', 'serving', 'serving41',
                 'serving_context', 'call_binding', 'host', 'config', 'runtime', 'identity', 'registry'))
SOURCES = tuple(path for _, path in MODULES) + (
    'experiments/c2c3_postwrite_20261117/natural39_worker_core_probe.py',
    'experiments/c2c3_postwrite_20261117/prepare39_worker_core.py',
    'experiments/c2c3_postwrite_20261117/natural41_probe.py',
    'experiments/c2c3_postwrite_20261117/natural41_coverage_probe.py',
    'experiments/c2c3_postwrite_20261117/model_identity.py')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def setup(worker, expected_identity, library, sha256, prompt, sources):
    from fault_injection.deployments.postwrite import NativePostwriteLibrary
    from demoncat.runtime.vllm_ascend.worker_extension.serving39_output import install_fault39_output_serving_binding
    worker = actual_worker(worker)
    from vllm.v1.worker.worker_base import WorkerBase
    if not isinstance(worker, WorkerBase):
        raise RuntimeError('ACTUAL_STANDARD_WORKER_REQUIRED')
    current = worker.demoncat_worker_identity()
    if current.get('worker') != expected_identity or current.get('quarantined') is not False:
        raise RuntimeError('ACTUAL_DEPLOYMENT_WORKER_IDENTITY_CHANGED')
    loaded = {}
    for name, relative in MODULES:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve(strict=True)
        sha = file_sha(path)
        if sha != sources[relative]:
            raise RuntimeError('ACTUAL_LOADED_SOURCE_CHANGED:'+relative)
        loaded[relative] = dict(path=str(path), sha256=sha)
    transport = NativePostwriteLibrary(library, sha256)
    result = install_fault39_output_serving_binding(worker, transport=transport,
        prompt_token_ids=prompt, evidence_dir='/evidence/output39-runtime')
    return dict(worker=expected_identity, binding=result, loaded_sources=loaded,
                model_source=model_source_identity(worker.model_runner.model))


def remove_binding(worker, expected_identity):
    from demoncat.runtime.vllm_ascend.worker_extension.serving39_output import remove_fault39_output_serving_binding
    worker = actual_worker(worker)
    if worker.demoncat_worker_identity().get('worker') != expected_identity:
        raise RuntimeError('CLEANUP_WORKER_IDENTITY_CHANGED')
    return remove_fault39_output_serving_binding(worker)


class RecordingExecutor:
    """Persist each original collective RPC once; do not change Host history."""
    def __init__(self, executor, folder):
        self.executor, self.folder = executor, folder
        folder.mkdir(exist_ok=False)
        self.receipts = []

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
            self.receipts.append(dict(method=name, path=str(path.name), sha256=file_sha(path)))


def drive(llm, client, *, executor, prompt, params, out, report):
    seen, pending = [], []
    scope, last = None, None
    def save():
        report['rpc_receipts'] = list(executor.receipts)
        write_json(out/'result.json', report)
    def action(name, *args, expected='ACTION_COMPLETED'):
        nonlocal scope, last
        snapshot = getattr(client, name)(*args)
        scope = snapshot['worker']
        last = snapshot
        save()
        if expected is not None and snapshot['status'] != expected:
            raise RuntimeError('PUBLIC_CORE_ACTION_FAILED:'+name+':'+snapshot['status'])
        return snapshot['response'].get('provider_state') or {}
    def enqueue():
        nonlocal pending
        if llm.llm_engine.has_unfinished_requests():
            raise RuntimeError('UNRELATED_PENDING_REQUEST')
        if llm.reset_prefix_cache(reset_running_requests=False) is not True:
            raise RuntimeError('FRESH_PREFILL_CACHE_RESET_UNCONFIRMED')
        pending = list(llm.enqueue([prompt], params, use_tqdm=False))
        mapping = read_enqueued_identity(llm, pending, seen_internal_request_ids=seen)
        if len(pending) != 1:
            raise RuntimeError('ONE_ACTUAL_ENQUEUE_REQUIRED')
        seen.extend(pending)
        return mapping
    def complete(mapping):
        responses = llm.wait_for_completion(use_tqdm=False)
        if len(responses) != 1 or responses[0].request_id != mapping['external_request_id']:
            raise RuntimeError('ACTUAL_REQUEST_MAPPING_CHANGED')
        return output(responses[0])
    def request(phase, *, recovery=False):
        nonlocal pending
        mapping = enqueue()
        state = action('advance', pending[0])
        if state['next_phase'] != phase:
            raise RuntimeError('ACTUAL_C1_PHASE_NOT_SELECTED')
        row = dict(phase=phase, mapping=mapping)
        report.setdefault('recovery_requests' if recovery else 'requests', []).append(row)
        save()
        row['output'] = complete(mapping)
        state = action('query', expected='QUERY_OK')
        actual = state['evidence']['actual_requests'][-1]
        if (actual['actual_internal_request_id'] != pending[0] or actual['phase'] != phase
                or state['failure'] is not None or state['leases'] != 0):
            raise RuntimeError('ACTUAL_C1_REQUEST_OR_NATIVE_READBACK_INCOMPLETE')
        row['worker_observation_receipt'] = executor.receipts[-1]
        pending = []
        save()
    try:
        try:
            for name in ('prepare', 'register', 'check', 'enable'):
                action(name)
            for phase in PHASES:
                request(phase)
            state = action('finish')
            if not state['evidence']['c1_evidence_complete'] or state['owned_hooks'] or state['leases']:
                raise RuntimeError('C1_EFFECT_OFF_HEALTH_OR_CLEANUP_UNPROVEN')
            report['c1'] = state['evidence']
            action('disable'); action('unregister')
            mapping = enqueue()
            healthy = complete(mapping)
            if any(healthy[key] != report['healthy_before'][key] for key in ('tokens','text')):
                raise RuntimeError('POST_UNREGISTER_HEALTH_CHANGED')
            if client.host.route(client.config) != scope:
                raise RuntimeError('POST_UNREGISTER_WORKER_CHANGED')
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
        # Recovery executes outside the original exception context; never F retry.
        if report['status'] == 'REFUSED_OR_FAILED' and last and last.get('ticket'):
            try:
                state = action('recover')
                if state['replay_pending'] and state['next_phase'] == 'off':
                    request('off', recovery=True); request('health', recovery=True)
                action('unregister'); action('detach', expected='DETACHED')
            except BaseException as recovery_error:
                report['recovery_error'] = repr(recovery_error)
    finally:
        # Exactly one extra snapshot, only for the final complete Host receipt.
        final = client.snapshot()
        write_json(out/'host_receipt.json', final)
        report['host_receipt'] = dict(path='host_receipt.json', sha256=file_sha(out/'host_receipt.json'))
        report['ownership_retained'] = final['ownership_retained']
        report['worker'] = final['worker']
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'library', 'sha256', 'build-fingerprint'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--memory', type=float, default=.35)
    parser.add_argument('--statistics', action='store_true')
    args = parser.parse_args(argv)
    if args.statistics:
        parser.error(STATISTICS_REFUSAL)  # Before importing vLLM or creating a model.
    if not .1 <= args.memory <= .8 or any(len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
            for value in (args.sha256, args.build_fingerprint)):
        parser.error('Bounded memory and actual reviewed Native identities required')
    if not args.out.is_absolute() or not args.out.resolve().is_relative_to(Path('/evidence')):
        parser.error('Private complete observations must stay under /evidence')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='demoncat.output39.worker_core_probe.v1', status='STARTED', mechanism='39.output_proxy',
        standard_worker=True, c2=False, c3=False, E5='UNADAPTED_COLLECTOR_BOUNDARY',
        public_qualification_granted=False, model_local_qualification_inherited=False,
        native_sha256=args.sha256, build_fingerprint=args.build_fingerprint, requests=[])
    try:
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core
        report['weights'] = actual_vllm_weights(args.model)
        report['sources'] = {path:file_sha(ROOT/path) for path in SOURCES}
        llm = LLM(model=args.model, dtype='bfloat16', load_format='safetensors', trust_remote_code=False,
            tensor_parallel_size=1, enforce_eager=True, max_model_len=128, max_num_seqs=1,
            max_num_batched_tokens=128, num_gpu_blocks_override=64, enable_prefix_caching=False,
            enable_chunked_prefill=False, async_scheduling=False, gpu_memory_utilization=args.memory,
            seed=42, disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        if actual_vllm_weights(args.model) != report['weights']:
            raise RuntimeError('MODEL_ARTIFACTS_CHANGED_DURING_LOAD')
        tokens = llm.get_tokenizer().encode(PROMPT, add_special_tokens=True)
        prompt, params = dict(prompt_token_ids=tokens), SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        healthy, = llm.generate([prompt], params, use_tqdm=False)
        report['healthy_before'] = output(healthy)
        executor = RecordingExecutor(llm, args.out/'worker_core_rpc')
        def binding(identity, config):
            replies = executor.collective_rpc(setup, args=(identity, args.library, args.sha256, tokens, report['sources']), timeout=300)
            if len(replies) != 1 or replies[0]['worker'] != identity:
                raise RuntimeError('TP1_ACTUAL_DEPLOYMENT_ROUTE_REQUIRED')
            report['binding'] = replies[0]
            return {}
        client = WorkerFaultClient(HostAdapter(executor, prepare_binding=binding, timeout=300), FaultConfig(39, 'output_proxy'))
        drive(llm, client, executor=executor, prompt=prompt, params=params, out=args.out, report=report)
        if not report['ownership_retained'] and report.get('binding'):
            report['binding_removal'] = executor.collective_rpc(remove_binding, args=(report['worker'],), timeout=300)
            report['rpc_receipts'] = list(executor.receipts)
    except BaseException as error:
        report.update(status='REFUSED_OR_FAILED', error=repr(error), traceback=traceback.format_exc())
    finally:
        write_json(args.out/'result.json', report)
    print(json.dumps({key:report[key] for key in ('status','error') if key in report}))
    return 0 if report['status'] == SUCCESS else 2


if __name__ == '__main__':
    raise SystemExit(main())
