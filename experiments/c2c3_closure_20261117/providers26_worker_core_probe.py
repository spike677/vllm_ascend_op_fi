"""Source-bound public WorkerFaultClient C1/C2; ordinary enqueues own forwards."""
import argparse
import importlib
import json
import math
from pathlib import Path
import sys
import time
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
C2_PHASES = ('inventory_1', 'inventory_2', 'qualification_reference_all', 'qualification_lossless_all',
             'qualification_off_all', 'group_fault', 'group_off', 'health')
C2_SUCCESS = 'STANDARD_WORKER_NATIVE26_COMPLETE_SET_C2_EVIDENCE_READY_FOR_REVIEW'
C2_BLOCKED = 'STANDARD_WORKER_NATIVE26_COMPLETE_SET_C2_QUALIFICATION_BLOCKED'
C2_CPU_CONTRACT = 'CPU_CORE26_COMPLETE_SET_CONTRACT_COMPLETE_NO_QUALIFICATION'
C3_PHASES = C2_PHASES[:6]+('group_fault_repeat',)+C2_PHASES[6:]
C3_SUCCESS = 'STANDARD_WORKER_NATIVE26_COMPLETE_SET_C2_C3_EVIDENCE_READY_FOR_REVIEW'
C3_BLOCKED = 'STANDARD_WORKER_NATIVE26_COMPLETE_SET_C3_QUALIFICATION_BLOCKED'
C3_CPU_CONTRACT = 'CPU_CORE26_C2_C3_CONTRACT_COMPLETE_NO_QUALIFICATION'
STATISTICS_REFUSAL = 'CORE26_STATISTICS_BOUNDARY_UNADAPTED: complete diagnostic byte/hash reads are not model statistics'
MODULES = tuple(('fault_injection.deployments.'+name, 'fault_injection/deployments/'+name+'.py') for name in
    ('writer26_loader', 'writer26_group', 'writer26_inventory', 'writer26_experiment', 'writer26_coverage', 'writer26_public',
     'writer26_fia_typed', 'residual26')) + (
    ('fault_injection.metadata_arrays', 'fault_injection/metadata_arrays.py'),
    ('fault_injection.environment', 'fault_injection/environment.py'),
    ('fault_injection.providers', 'fault_injection/providers.py'),
    ('fault_injection.coverage', 'fault_injection/coverage.py'),
    ('fault_injection.coverage_session', 'fault_injection/coverage_session.py')) + tuple(
    ('demoncat.runtime.vllm_ascend.worker_extension.'+name,
     'src/demoncat/runtime/vllm_ascend/worker_extension/'+name.replace('.', '/')+'.py') for name in
    ('serving26_core', 'serving26_isolated', 'providers.fault26_core', 'providers.fault26', 'providers._scope', 'serving26_group',
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


def setup(worker, expected_identity, mechanism, tokens, manifest_sha256, sources,
          stage='c1_single_effects', max_targets=1):
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
        evidence_dir='/evidence/worker-core26-runtime', manifest_sha256=manifest_sha256,
        stage=stage, max_targets=max_targets)
    return dict(worker=expected_identity, binding=bound, loaded_sources=loaded,
                model_source=model_source_identity(worker.model_runner.model))


def remove(worker, expected_identity, mechanism):
    worker = actual(worker)
    if worker.demoncat_worker_identity().get('worker') != expected_identity:
        raise RuntimeError('CLEANUP_WORKER_CHANGED')
    from demoncat.runtime.vllm_ascend.worker_extension.serving26_core import remove_writer26_core_binding
    return remove_writer26_core_binding(worker, mechanism)


def isolated_budget_preflight(requests, plan, *, elapsed_seconds):
    """A new run must earn its own pre-F budget; source77 is not a timing pass."""
    phases = C2_PHASES[2:5]
    samples = [row['complete_cycle_seconds'] for row in requests if row['phase'] in phases]
    if len(samples) != 3 or not all(type(value) in (int, float) and value > 0 for value in samples):
        raise RuntimeError('THREE_ACTUAL_COMPLETE_QUALIFICATION_CYCLES_REQUIRED_FOR_BUDGET')
    count = plan.get('full_native_writer_count')
    if type(count) is not int or count <= 0:
        raise RuntimeError('FULL_NATIVE_DENOMINATOR_REQUIRED_FOR_BUDGET')
    remaining = plan['group_requests']-5
    # The measured full 64-request CPU driver took 279.922s although its first
    # three qualification cycles were <0.3s. Account separately for growing
    # evidence/Host-copy overhead; never extrapolate those early cycles alone.
    control_reserve = math.ceil(360*(plan['group_requests']/64)**3*max(1, count/30))
    forecast = 2*max(samples)*remaining+120+control_reserve
    reasons = []
    if forecast > 1200:
        reasons.append('ISOLATED_SESSION_1200_SECOND_BUDGET_UNPROVEN')
    if elapsed_seconds+forecast+120 > 1800:
        reasons.append('ISOLATED_ROOT_CASE_1800_SECOND_BUDGET_UNPROVEN')
    return dict(schema='demoncat.writer26.isolated_budget_preflight.v1',
        status='REFUSED_BEFORE_F' if reasons else 'CONSERVATIVE_ESTIMATE_WITHIN_EXISTING_LIMITS',
        complete_qualification_cycle_seconds=samples, remaining_requests=remaining,
        per_request_safety_multiplier=2, cleanup_reserve_seconds=120, outside_probe_startup_margin_seconds=120,
        python_control_overhead_reserve_seconds=control_reserve,
        cpu_control_reference=dict(group_requests=64, full_native_writer_count=30, elapsed_seconds=279.922,
            total_evidence_bytes=957530550, actual_npu_calls=0, qualification_granted=False),
        forecast_remaining_seconds=forecast, actual_probe_elapsed_seconds=elapsed_seconds,
        existing_session_limit_seconds=1200, existing_root_case_limit_seconds=1800,
        source77_total_evidence_bytes=536999909, source77_requests=17,
        estimated_evidence_reservation_bytes=math.ceil(536999909*plan['group_requests']*2/17*max(1, count/30)),
        evidence_bytes_are_estimate_not_measured_bound=True, qualification_granted=False, refusal_reasons=reasons)


def drive(llm, client, *, executor, prompt, params, out, report, case_started=None):
    case_started = time.monotonic() if case_started is None else case_started
    seen, pending, last = [], [], None
    isolated = report.get('stage') == 'c2_isolated_once'
    complete_set = report.get('stage') in ('c2_once', 'c2_c3', 'c2_isolated_once')
    every_match = report.get('stage') == 'c2_c3'
    phases = [('c2' if complete_set else 'c1', phase) for phase in (C2_PHASES if complete_set else PHASES)]
    if every_match:
        phases.extend(('c3', phase) for phase in C3_PHASES)
    if isolated:
        phases = [('c2', phase) for phase in C2_PHASES[:2]]
    run_counts = dict(c1=0, c2=0, c3=0)
    blocked_run = None
    blocked = False
    simulation = False
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
    def verify_c2(first):
        coverage = first.get('coverage')
        qualified = (first.get('evidence_kind') == 'cpu_contract' and first['c2'] is False) \
            if first['public_core']['simulation'] else first['c2'] is True
        if (not coverage or not coverage['contract_satisfied'] or not qualified
                or first['c3'] or len(coverage['live_c2_proof']['issued']) != 1
                or coverage['live_c2_proof']['consumed'] is not None):
            raise RuntimeError('COMPLETE_SET_C2_OR_ORIGINAL_PROOF_UNPROVEN')
    try:
        try:
            for name in ('prepare', 'register', 'check', 'enable'):
                action(name)
            index = 0
            while index < len(phases):
                run, phase = phases[index]
                if isolated and phase == 'group_fault' and 'isolated_budget_preflight' not in report:
                    budget = isolated_budget_preflight(report['requests'], report['isolated_plan'],
                        elapsed_seconds=time.monotonic()-case_started)
                    report['isolated_budget_preflight'] = budget
                    save()
                    if budget['refusal_reasons']:
                        raise RuntimeError(';'.join(budget['refusal_reasons']))
                cycle_started = time.monotonic()
                mapping = enqueue()
                state = action('advance', pending[0])
                if state['evidence']['phase'] != phase or state['evidence'].get('current_run', run) != run:
                    raise RuntimeError('EXISTING_WRITER_PLAN_PHASE_CHANGED')
                row = dict(run=run, phase=phase, mapping=mapping)
                report['requests'].append(row)
                save()
                row['output'] = complete(mapping)
                state = action('query', expected='QUERY_OK')
                evidence = state['evidence']
                observed = evidence['request_index'][-1]
                if (len(evidence['request_index']) != run_counts[run]+1 or observed['phase'] != phase
                        or observed['request_id'] != pending[0] or evidence['failure'] is not None
                        or state['leases'] or state['inflight'] or not state['serving']['needs_collect']):
                    raise RuntimeError('ACTUAL_WRITER_CONSUMER_OR_REQUEST_READBACK_INCOMPLETE')
                row['pre_collect_query'] = executor.receipts[-1]
                state = action('collect_replay')
                collected = state['evidence']
                if collected.get('current_run', run) != run:
                    if not (every_match and run == 'c2' and phase == 'health'
                            and collected['current_run'] == 'c3' and collected.get('prior_c2')):
                        raise RuntimeError('EXACT_COMPLETED_C2_BEFORE_C3_TRANSITION_REQUIRED')
                    collected = collected['prior_c2']
                    verify_c2(collected)
                    report['c2_attempt'], report['c2'] = collected, bool(collected['c2'])
                originals = collected['original_observations']
                if originals['status'] != 'CURRENT_COLLECTED' or state['serving']['needs_collect']:
                    raise RuntimeError('EXPLICIT_ORIGINAL_COLLECTION_UNCONFIRMED')
                row['original_observations'] = originals['originals'][-1]
                pending = []
                if complete_set and run_counts[run] == 1 and collected['blockers']:
                    # Keep the entire failed census. Run only its already
                    # planned Native-release health; never pick a subset.
                    blocked = True
                    blocked_run = run
                    phases = phases[:index+1]+[(run, 'health')]
                    report['complete_set_blockers'] = collected['blockers']
                    report['blocked_run'] = run
                elif isolated and run_counts[run] == 1:
                    plan = collected['isolated_plan']
                    count = collected['resolved_candidate_count']
                    if (not 0 < count <= 128 or len(set(plan['target_order'])) != count
                            or plan['group_requests'] != 6+2*count or plan['c3_requested']
                            or not collected['independent_per_target_replay']):
                        raise RuntimeError('EXACT_COMPLETE_ISOLATED_PLAN_REQUIRED')
                    phases.extend(('c2', phase) for phase in
                        C2_PHASES[2:5]+('group_fault', 'group_off')*count+('health',))
                    report['isolated_plan'] = plan
                run_counts[run] += 1
                index += 1
                row['complete_cycle_seconds'] = time.monotonic()-cycle_started
                save()
            if not blocked:
                state = action('finish')
            evidence = state['evidence']
            if complete_set:
                simulation = evidence['public_core']['simulation']
                first = evidence.get('prior_c2', evidence)
                second = evidence if every_match and evidence.get('current_run') == 'c3' else None
                coverage = first.get('coverage')
                if blocked_run != 'c2':
                    verify_c2(first)
                report['c2_attempt'] = first
                report['c2'] = bool(blocked_run != 'c2' and first['c2'])
                if second is not None:
                    report['c3_attempt'] = second
                if every_match and not blocked:
                    current = second.get('coverage') if second else None
                    qualified = (second.get('evidence_kind') == 'cpu_contract' and second['c3'] is False) \
                        if simulation and second else bool(second and second['c3'] is True)
                    if (not current or not current['contract_satisfied'] or not qualified or second['c2']
                            or current['live_c2_proof']['issued'] or not current['live_c2_proof']['consumed']):
                        raise RuntimeError('COMPLETE_SET_C3_OR_LIVE_CONSUMPTION_UNPROVEN')
                    issued, = coverage['live_c2_proof']['issued']
                    consumed = current['live_c2_proof']['consumed']
                    if (not consumed['exact_issued_object_verified']
                            or issued['object_binding'] != consumed['object_binding']
                            or issued['proof_sha256'] != consumed['proof_sha256']
                            or current['owner'] != coverage['owner'] or current['generation'] != coverage['generation']+1
                            or current['scope'] != coverage['scope']):
                        raise RuntimeError('EXACT_SAME_WORKER_LIVE_C2_TO_C3_RECEIPTS_REQUIRED')
                    report['c3'] = bool(second['c3'])
            else:
                if not evidence['single_site_c1']['evidence_complete'] or evidence['c2'] or evidence['c3']:
                    raise RuntimeError('SINGLE_SITE_EFFECT_OFF_HEALTH_UNPROVEN')
                report['c1'] = evidence
            if not blocked and (state['native_registered'] or state['leases'] or state['inflight']
                    or state['serving']['owned_hooks'] or evidence['owned_hooks']):
                raise RuntimeError('WRITER_ALL_OWNED_RESOURCES_MUST_BE_ZERO')
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
            if not complete_set:
                report['status'] = SUCCESS
            elif blocked:
                report['status'] = C3_BLOCKED if blocked_run == 'c3' else C2_BLOCKED
            elif every_match:
                report['status'] = C3_CPU_CONTRACT if simulation else C3_SUCCESS
            else:
                report['status'] = C2_CPU_CONTRACT if simulation else C2_SUCCESS
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
    case_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mechanism', choices=('26.residual_stomp', '26.fia_destination'), required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('c1_single_effects', 'c2_once', 'c2_c3', 'c2_isolated_once'), default='c1_single_effects')
    parser.add_argument('--max-targets', type=int, default=1)
    parser.add_argument('--memory', type=float, default=.55)
    parser.add_argument('--statistics', action='store_true')
    args = parser.parse_args(argv)
    if args.statistics:
        parser.error(STATISTICS_REFUSAL)
    if (not 1 <= args.max_targets <= 128 or args.stage == 'c1_single_effects' and args.max_targets != 1
            or not .1 <= args.memory <= .8 or len(args.manifest_sha256) != 64 or any(
            c not in '0123456789abcdef' for c in args.manifest_sha256)):
        parser.error('Reviewed manifest, bounded memory and explicit complete-set target budget required')
    out = Path('/evidence/probe')
    out.mkdir(exist_ok=False)
    report = dict(schema='demoncat.writer26.worker_core_probe.v1', status='STARTING', mechanism=args.mechanism,
        stage=args.stage, scope=('ALL_STABLE_ELIGIBLE_TARGETS_ISOLATED_F_OFF_FRESH_REQUEST_PAIRS'
            if args.stage == 'c2_isolated_once' else 'SAME_REQUEST_CUMULATIVE_ALL_STABLE_ELIGIBLE_TARGETS_FIXED_FRESH_PREFILL'
            if args.stage in ('c2_once', 'c2_c3') else single_site_scope(args.mechanism)), c2=False, c3=False,
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
                args=(worker, config.mechanism, tokens, args.manifest_sha256, sources, args.stage, args.max_targets))
            if len(replies) != 1:
                raise RuntimeError('ONE_ACTUAL_DEPLOYMENT_WORKER_REQUIRED')
            report['setup'] = replies[0]
            return {}
        client = WorkerFaultClient(HostAdapter(recorder, prepare_binding=bind, timeout=300),
            FaultConfig(26, args.mechanism.split('.', 1)[1]))
        drive(llm, client, executor=recorder, prompt=prompt, params=params, out=out, report=report,
              case_started=case_started)
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
    return 0 if report['status'] in (SUCCESS, C2_SUCCESS, C3_SUCCESS) else 2


if __name__ == '__main__':
    raise SystemExit(main())
