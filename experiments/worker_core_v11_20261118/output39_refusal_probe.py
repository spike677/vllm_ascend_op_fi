"""One bounded fixed-C3 diagnostic, composing the unchanged public driver.

Helpers are ordinary experiment callable RPCs on an already initialized Worker;
they add no Core action, provider, registry entry or Worker lifecycle authority.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'fault39/src')]
from experiments.c2c3_postwrite_20261117 import natural39_worker_core_probe as original
from experiments.worker_core_v11_20261118.output39_refusal_diagnostic import DiagnosticPlan, RefusalObserver, PIN_FILES, require

ATTRIBUTE = '_dcat39_refusal_diagnostic_experiment'
SELF = 'experiments/worker_core_v11_20261118/output39_refusal_probe.py'
SOURCES = tuple(dict.fromkeys(original.SOURCES+PIN_FILES+(SELF,
    'experiments/worker_core_v11_20261118/review39_refusal_diagnostic.py',
    'experiments/worker_core_v11_20261118/prepare_output39_refusal.py')))


def phase_budget(actual_plan, eligible):
    """Describe/validate the unchanged provider plan; never execute this list."""
    actual = list(actual_plan)
    expected = list(original.PHASES)+['c2_lossless']+['c2_fault']*eligible+['c2_off','c2_health'] \
        + ['c3_lossless','c3_fault','c3_fault','c3_off','c3_health']
    require(actual == expected, 'ORIGINAL_PUBLIC_PHASE_SCHEDULE_CHANGED')
    stop = actual.index('c3_fault')+1
    return dict(derivation='ACTUAL_PROVIDER_PLAN_AFTER_ORIGINAL_PREFIX_COLLECTION',
        original_full_group_schedule=actual, diagnostic_normal_prefix=actual[:stop],
        diagnostic_normal_requests=stop, bounded_recovery_schedule=['group_recovery_off','group_recovery_health'],
        initial_ordinary_health=1, post_unregister_ordinary_health=1,
        maximum_user_requests=stop+2+1+1, original_schedule_modified=False)


def install_diagnostic(worker, identity, ticket, source_root, pins, output, plan_dict=None, cpu_contract=False):
    worker = original.actual_worker(worker)
    require(not hasattr(worker, ATTRIBUTE), 'DIAGNOSTIC_WORKER_ATTRIBUTE_ALREADY_OWNED')
    require(worker.demoncat_worker_identity().get('worker') == identity, 'DIAGNOSTIC_WORKER_CHANGED')
    lease = worker._demoncat_worker_fault_runtime._leases['39.output_proxy']
    core = lease.provider
    require(lease.ticket() == ticket and tuple(core._ticket) == (lease.native_owner,lease.generation),
            'DIAGNOSTIC_CORE_OWNER_CHANGED')
    require(core.phase == 'c2_lossless' and core.collected_requests == len(core.rows) == 6,
            'SIX_ORIGINAL_REQUESTS_AND_COLLECTIONS_REQUIRED')
    require(bool(core.binding.simulation) == cpu_contract, 'DIAGNOSTIC_CPU_SCOPE_CHANGED')
    path = Path(output)
    require(path.is_absolute() and (cpu_contract or path.resolve().is_relative_to(Path('/evidence'))),
            'PRIVATE_DIAGNOSTIC_EVIDENCE_DIRECTORY_REQUIRED')
    require(not path.exists(), 'DIAGNOSTIC_ORIGINAL_ALREADY_EXISTS')
    require(Path(__file__).resolve() == Path(source_root).resolve()/SELF
        and original.file_sha(Path(__file__)) == pins.get(SELF), 'ACTUAL_HELPER_SOURCE_NOT_PINNED')
    values = dict(plan_dict or {})
    if 'refused_shape' in values:
        values['refused_shape'] = tuple(values['refused_shape'])
    plan = DiagnosticPlan(**values)
    require(cpu_contract or plan == DiagnosticPlan(), 'PRODUCTION_DIAGNOSTIC_PLAN_IS_FROZEN')
    budget = phase_budget(core.plan,plan.expected_eligible)
    require(budget['maximum_user_requests'] <= 43, 'ORIGINAL_SCHEDULE_EXCEEDS_FIXED_REQUEST_BUDGET')
    observer = RefusalObserver(core, source_root, pins, plan=plan, explicit_cpu_contract=cpu_contract)
    observer.__enter__()
    setattr(worker, ATTRIBUTE, dict(observer=observer, identity=deepcopy(identity), ticket=deepcopy(ticket),
        provider_ticket=list(core._ticket), output=path, thread=threading.get_ident(), source_sha256=pins[SELF]))
    return dict(status='DIAGNOSTIC_INSTALLED_AFTER_ORIGINAL_PREFIX', thread=threading.get_ident(),
                actual_completed_requests=6, plan=plan.__dict__, source_sha256=pins[SELF], phase_budget=budget)


def finish_diagnostic(worker, identity, ticket):
    worker = original.actual_worker(worker)
    entry = getattr(worker, ATTRIBUTE)
    require(entry['identity'] == identity and entry['ticket'] == ticket
            and entry['thread'] == threading.get_ident(), 'DIAGNOSTIC_FINISH_OWNER_OR_THREAD_CHANGED')
    require(worker.demoncat_worker_identity().get('worker') == identity, 'DIAGNOSTIC_WORKER_CHANGED')
    observer = entry['observer']
    observer.close()
    report = observer.report()
    report['helper'] = dict(source_sha256=entry['source_sha256'], thread=entry['thread'],
        actual_public_rpc_ticket=entry['ticket'], actual_provider_ticket=entry['provider_ticket'],
        public_core_modified=False)
    path = entry['output']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    result = dict(status='DIAGNOSTIC_CLOSED', path=str(path), sha256=original.file_sha(path),
        observations=len(report['records']), errors=report['errors'], native_qualification=False,
        trace_removed=sys.gettrace() is None, original_core_profile_managed_by_original_recovery=True)
    delattr(worker, ATTRIBUTE)
    return result


class BoundedDiagnosticClient:
    """Compose original Client actions; never synthesize a successful reply."""
    def __init__(self, client, executor, source_root, pins, output, report, *,
                 deadline_seconds=1500, plan=DiagnosticPlan(), cpu_contract=False, clock=time.monotonic,
                 cancel_deadline=None):
        self.client, self.executor, self.source_root, self.pins = client, executor, source_root, pins
        self.output, self.report, self.plan, self.cpu = output, report, plan, cpu_contract
        self.clock, self.deadline = clock, clock()+deadline_seconds
        self.cancel_deadline = cancel_deadline
        self.installed = self.closed = self.recovering = self.cumulative_started = False
        self.normal_requests = self.recovery_requests = 0
        self.next_phase = 'observe'
        self.identity = self.ticket = None

    def __getattr__(self, name):
        return getattr(self.client, name)

    def _check_budget(self):
        require(self.clock() <= self.deadline, 'DIAGNOSTIC_DEADLINE_EXCEEDED')

    def _rpc(self, function, *args):
        replies = self.executor.collective_rpc(function, args=args, timeout=300)
        require(len(replies) == 1, 'ONE_ACTUAL_DIAGNOSTIC_WORKER_REPLY_REQUIRED')
        return replies[0]

    def advance(self, request_id):
        if self.recovering:
            require(self.recovery_requests < 2 and self.next_phase in ('group_recovery_off','group_recovery_health'),
                    'BOUNDED_ORIGINAL_RECOVERY_ONLY')
            self.recovery_requests += 1
        else:
            self._check_budget()
            require(self.normal_requests < 11+self.plan.expected_eligible, 'BOUNDED_ORIGINAL_REQUEST_LIMIT')
            if self.next_phase == 'c3_fault':
                require(not self.cumulative_started, 'ONLY_ONE_CUMULATIVE_DIAGNOSTIC_REQUEST')
                self.cumulative_started = True
            self.normal_requests += 1
        return self.client.advance(request_id)

    def collect_replay(self):
        result = self.client.collect_replay()
        state = result.get('response',{}).get('provider_state',{})
        self.next_phase = state.get('next_phase',self.next_phase)
        if result['status'] == 'ACTION_COMPLETED' and self.next_phase == 'c2_lossless' and not self.installed:
            self._check_budget()
            self.identity, self.ticket = deepcopy(result['worker']), deepcopy(result['ticket'])
            installed = self._rpc(install_diagnostic, self.identity, self.ticket, str(self.source_root),
                self.pins, str(self.output), self.plan.__dict__, self.cpu)
            self.report['diagnostic_install'] = installed
            self.installed = True
        return result

    def _finish(self):
        if self.installed and not self.closed:
            self.report['diagnostic_original'] = self._rpc(finish_diagnostic, self.identity, self.ticket)
            self.closed = True

    def recover(self):
        # Never let a failed observation write prevent original Native recovery.
        self.recovering = True
        if self.cancel_deadline is not None:
            self.cancel_deadline()
        try:
            self._finish()
        except BaseException as error:
            self.report['diagnostic_close_error'] = repr(error)
        result = self.client.recover()
        state = result.get('response',{}).get('provider_state',{})
        self.next_phase = state.get('next_phase', self.next_phase)
        return result

    def unregister(self):
        try:
            self._finish()
        except BaseException as error:
            self.report['diagnostic_close_error'] = repr(error)
        return self.client.unregister()

    def counts(self):
        return dict(normal_requests=self.normal_requests, recovery_requests=self.recovery_requests,
                    maximum_normal_requests=11+self.plan.expected_eligible,
                    maximum_recovery_requests=2, cumulative_started=self.cumulative_started,
                    installed=self.installed, closed=self.closed)


def drive_once(llm, client, *, executor, prompt, params, out, report, source_root, pins,
               deadline_seconds=1500, plan=DiagnosticPlan(), cpu_contract=False, cancel_deadline=None):
    bounded = BoundedDiagnosticClient(client, executor, source_root, pins, out/'refusal-diagnostic.json',
        report, deadline_seconds=deadline_seconds, plan=plan, cpu_contract=cpu_contract,
        cancel_deadline=cancel_deadline)
    try:
        # Original driver includes actual enqueues, cache reset, RPC collection,
        # abort on failure, OFF/health recovery and detach. No copied protocol.
        original.drive(llm, bounded, executor=executor, prompt=prompt, params=params, out=out, report=report)
    finally:
        try:
            bounded._finish()
        except BaseException as error:
            report['diagnostic_close_error'] = repr(error)
        report['diagnostic_budget'] = bounded.counts()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','library','sha256','build-fingerprint'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--memory', type=float, default=.35)
    parser.add_argument('--deadline-seconds', type=int, default=1500)
    args = parser.parse_args(argv)
    if not .1 <= args.memory <= .8 or not 60 <= args.deadline_seconds <= 1500:
        parser.error('Fixed memory/deadline bounds required')
    if any(len(value)!=64 or any(c not in '0123456789abcdef' for c in value)
           for value in (args.sha256,args.build_fingerprint)):
        parser.error('Actual reviewed Native/build SHA256 required')
    if not args.out.is_absolute() or not args.out.resolve().is_relative_to(Path('/evidence')):
        parser.error('Private evidence must stay under /evidence')
    if not hasattr(signal,'setitimer') or threading.current_thread() is not threading.main_thread():
        parser.error('Linux main-thread soft deadline required')
    if signal.getitimer(signal.ITIMER_REAL) != (0.0,0.0):
        parser.error('Existing timer must not be replaced')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='demoncat.output39.fixed_c3_refusal_probe.v1', status='STARTED',
        mechanism='39.output_proxy', stage='c2_c3', c2=False, c3=False, e5=False,
        qualification_granted=False, requests=[], native_sha256=args.sha256,
        build_fingerprint=args.build_fingerprint, plan=DiagnosticPlan().__dict__)
    old_handler = signal.getsignal(signal.SIGALRM)
    def timeout_handler(*unused):
        raise TimeoutError('DIAGNOSTIC_SOFT_DEADLINE_EXCEEDED')
    try:
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core
        report['weights'] = original.actual_vllm_weights(args.model)
        report['sources'] = {name:original.file_sha(ROOT/name) for name in SOURCES}
        llm = LLM(model=args.model, dtype='bfloat16', load_format='safetensors', trust_remote_code=False,
            tensor_parallel_size=1, enforce_eager=True, max_model_len=128, max_num_seqs=1,
            max_num_batched_tokens=128, num_gpu_blocks_override=64, enable_prefix_caching=False,
            enable_chunked_prefill=False, async_scheduling=False, gpu_memory_utilization=args.memory,
            seed=42, disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        require(original.actual_vllm_weights(args.model) == report['weights'], 'MODEL_ARTIFACTS_CHANGED_DURING_LOAD')
        tokens = llm.get_tokenizer().encode(original.PROMPT, add_special_tokens=True)
        prompt, params = dict(prompt_token_ids=tokens), SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        healthy, = llm.generate([prompt], params, use_tqdm=False)
        report['healthy_before'] = original.output(healthy)
        executor = original.RecordingExecutor(llm, args.out/'worker_core_rpc')
        def binding(identity, config):
            replies = executor.collective_rpc(original.setup, args=(identity,args.library,args.sha256,tokens,
                report['sources'],'c2_c3',128), timeout=300)
            require(len(replies)==1 and replies[0]['worker']==identity, 'ONE_ORIGINAL_STANDARD_WORKER_REQUIRED')
            report['binding'] = replies[0]
            return {}
        client = original.WorkerFaultClient(original.HostAdapter(executor,prepare_binding=binding,timeout=300),
                                           original.FaultConfig(39,'output_proxy'))
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, args.deadline_seconds)
        drive_once(llm, client, executor=executor,prompt=prompt,params=params,out=args.out,report=report,
                   source_root=ROOT,pins=report['sources'],deadline_seconds=args.deadline_seconds,
                   cancel_deadline=lambda:signal.setitimer(signal.ITIMER_REAL,0))
        signal.setitimer(signal.ITIMER_REAL,0)
        if not report['ownership_retained'] and report.get('binding'):
            require(not llm.llm_engine.has_unfinished_requests(), 'PENDING_REQUEST_AFTER_ORIGINAL_RECOVERY')
            require(llm.reset_prefix_cache(reset_running_requests=False) is True, 'HEALTH_CACHE_RESET_UNCONFIRMED')
            pending = list(llm.enqueue([prompt],params,use_tqdm=False))
            seen = [row['mapping']['internal_request_id'] for row in
                    report.get('requests',[])+report.get('recovery_requests',[])]
            mapping = original.read_enqueued_identity(llm,pending,seen_internal_request_ids=seen)
            responses = llm.wait_for_completion(use_tqdm=False)
            require(len(responses)==1 and responses[0].request_id==mapping['external_request_id'], 'POST_UNREGISTER_REQUEST_MAPPING')
            result = original.output(responses[0])
            require(all(result[key] == report['healthy_before'][key] for key in ('tokens','text')),
                    'POST_UNREGISTER_HEALTH_CHANGED')
            require(client.host.route(client.config) == report['worker'], 'POST_UNREGISTER_WORKER_CHANGED')
            report['post_unregister_health'] = dict(mapping=mapping,output=result,worker=report['worker'])
            report['binding_removal'] = executor.collective_rpc(original.remove_binding,args=(report['worker'],),timeout=300)
            report['rpc_receipts'] = list(executor.receipts)
    except BaseException as error:
        # Preserve the original model/Native failure if it already exists.
        key = 'runner_error' if report.get('error') else 'error'
        report[key] = repr(error)
        report['runner_traceback'] = traceback.format_exc()
        report['status'] = 'REFUSED_OR_FAILED'
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_handler)
        original.write_json(args.out/'result.json',report)
    print(json.dumps({key:report[key] for key in ('status','error','runner_error') if key in report}))
    # The original cumulative request is deliberately incomplete/refused.
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
