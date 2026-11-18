"""Root-queued, zero-F public Attention discovery on an ordinary loaded Worker.

Two actual enqueued requests, complete public receipts, owned cleanup, then
one ordinary enqueue after unregister/detach. No Native loader or F is exposed
by this probe. Unsupported calls stay visible; completion grants no C1/C2/C3.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'fault39/src')]
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.attention_native import MECHANISMS
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import read_enqueued_identity

SUCCESS = 'PUBLIC_ATTENTION_ZERO_F_CENSUS_AND_CLEANUP_RECORDED'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2), b''):value.update(chunk)
    return value.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, allow_nan=False, indent=2)+'\n', encoding='utf-8')


def actual(worker):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    return worker.worker if isinstance(worker, WorkerWrapperBase) else worker


def setup(worker, expected, mechanism, prompt, provenance):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import install_attention_serving_binding
    worker = actual(worker)
    observed = worker.demoncat_worker_identity()
    if observed.get('status') != 'WORKER_IDENTIFIED' or observed.get('quarantined') or observed['worker'] != expected:
        raise RuntimeError('ATTENTION_DEPLOYMENT_WORKER_IDENTITY_CHANGED')
    install_attention_serving_binding(worker, mechanism, prompt_token_ids=prompt, provenance=provenance)
    return dict(worker=observed['worker'], mechanism=mechanism, native_loaded_by_probe=False,
                mode='ZERO_F_ACTUAL_CALL_CENSUS', qualification_granted=False)


def remove(worker, expected, mechanism):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import remove_attention_serving_binding
    worker = actual(worker)
    if worker.demoncat_worker_identity().get('worker') != expected:
        raise RuntimeError('ATTENTION_REMOVAL_WORKER_IDENTITY_CHANGED')
    return remove_attention_serving_binding(worker, mechanism)


def drive(llm, client, *, prompt, params, out, report):
    folder = out/'worker_core_rpc';folder.mkdir(exist_ok=False)
    core = report.setdefault('worker_core', dict(receipts=[], qualification_granted=False))
    history_count, seen, pending = 0, [], []
    def action(name, *args, expected='ACTION_COMPLETED'):
        nonlocal history_count
        reply = getattr(client, name)(*args)
        for item in reply['history'][history_count:]:
            path = folder/(str(len(core['receipts'])).zfill(3)+'-'+item['action']+'.json')
            save(path, item)
            core['receipts'].append(dict(action=item['action'], path=str(path.relative_to(out)), sha256=sha(path)))
        history_count = len(reply['history'])
        state = reply.get('response', {}).get('provider_state')
        if state is not None:report['provider_state'] = deepcopy(state)
        core['last_scope'] = {key: deepcopy(value) for key, value in reply.items()
                              if key not in ('history','response','capability_enumeration')}
        save(out/'result.json', report)
        if expected is not None and reply['status'] != expected:
            raise RuntimeError('PUBLIC_ATTENTION_ACTION_FAILED:'+name+':'+reply['status'])
        return reply, state
    def enqueue():
        if llm.llm_engine.has_unfinished_requests():raise RuntimeError('UNRELATED_PENDING_REQUEST')
        if llm.reset_prefix_cache(reset_running_requests=False) is not True:
            raise RuntimeError('FRESH_REQUEST_CACHE_RESET_UNCONFIRMED')
        ids = list(llm.enqueue([prompt], params, use_tqdm=False))
        if len(ids) != 1:raise RuntimeError('ONE_ACTUAL_REQUEST_REQUIRED')
        mapping = read_enqueued_identity(llm, ids, seen_internal_request_ids=seen)
        seen.extend(ids)
        return ids, mapping
    try:
        for name in ('prepare','register','check','enable'):action(name)
        report['requests'] = []
        for phase in ('observe','observe_repeat'):
            pending, mapping = enqueue()
            _, before = action('advance', pending[0])
            if before['serving']['next_phase'] != phase:raise RuntimeError('ATTENTION_PHASE_CHANGED')
            results = llm.wait_for_completion(use_tqdm=False)
            if len(results) != 1 or results[0].request_id != mapping['external_request_id']:
                raise RuntimeError('ACTUAL_DRIVER_WORKER_REQUEST_MAPPING_CHANGED')
            _, queried = action('query', expected='QUERY_OK')
            if queried['serving']['pending_request'] != [pending[0], phase] or queried['evidence']['requests'] != before['evidence']['requests']:
                raise RuntimeError('READ_ONLY_QUERY_ADVANCED_REQUEST')
            _, collected = action('collect_replay')
            row = collected['evidence']['requests'][-1]
            if row['request_id'] != pending[0] or row['phase'] != phase:
                raise RuntimeError('ATTENTION_ACTUAL_PHASE_BINDING_CHANGED')
            report['requests'].append(dict(phase=phase, mapping=mapping, worker=row,
                output_tokens=list(results[0].outputs[0].token_ids)))
            pending = []
        _, state = action('finish')
        if (state['owned_hooks'] or state['owned_api_wrappers'] or state['inflight'] or state['leases']
                or state['native_loaded'] or state['C1'] or state['C2'] or state['C3']):
            raise RuntimeError('ZERO_F_DISCOVERY_CLEANUP_OR_QUALIFICATION_CHANGED')
        report['final_cleanup'] = state
        action('disable');action('unregister');action('detach', expected='DETACHED')
        # Same ordinary host path after all owned observation resources ended.
        pending, mapping = enqueue()
        results = llm.wait_for_completion(use_tqdm=False)
        if len(results) != 1 or results[0].request_id != mapping['external_request_id']:
            raise RuntimeError('POST_UNREGISTER_REQUEST_MAPPING_CHANGED')
        tokens = list(results[0].outputs[0].token_ids)
        report['post_unregister_health'] = dict(mapping=mapping, output_tokens=tokens,
            scope='TOKENS_ONLY_NO_POST_UNREGISTER_LOGITS_OBSERVER')
        if tokens != report['healthy_before'] or any(x['output_tokens'] != tokens for x in report['requests']):
            raise RuntimeError('SAME_WORKER_TOKEN_HEALTH_CHANGED')
        pending = []
        report['status'] = SUCCESS
    except BaseException as error:
        report.update(status='REFUSED_OR_FAILED', error=repr(error), traceback=traceback.format_exc())
        if pending:
            try:llm.llm_engine.abort_request(pending, internal=True)
            except BaseException as abort_error:report['abort_error'] = repr(abort_error)
    finally:
        if client.snapshot()['ownership_retained']:
            for name in ('disable','unregister','detach'):
                try:action(name, expected=None)
                except BaseException as cleanup_error:report.setdefault('cleanup_errors', []).append(repr(cleanup_error))
        host = out/'host_receipt.json';save(host, client.snapshot())
        report['host_receipt'] = dict(path=host.name, sha256=sha(host), bytes=host.stat().st_size)
        core['ownership_retained'] = client.snapshot()['ownership_retained']
        save(out/'result.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mechanism', choices=MECHANISMS, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--memory', type=float, default=.25)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to('/evidence') or not 0 < args.memory <= .5:
        parser.error('Private /evidence output and bounded memory required')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='STARTED', mechanism=args.mechanism, probe_sha256=sha(__file__),
        public_qualification_granted=False, C1=False, C2=False, C3=False, E5='NOT_ASSESSED',
        mode='ZERO_F_ACTUAL_CALL_CENSUS', native_loaded_by_probe=False)
    client = None
    try:
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core
        weights = {path.name: dict(sha256=sha(path), bytes=path.stat().st_size)
            for path in sorted({*args.model.glob('*.safetensors'), *args.model.glob('pytorch_model*.bin')})}
        if not weights:raise RuntimeError('ACTUAL_PRETRAINED_WEIGHTS_REQUIRED')
        report['model_identity'] = dict(weights=weights, config_sha256=sha(args.model/'config.json'))
        llm = LLM(model=str(args.model), dtype='float16', tensor_parallel_size=1, enforce_eager=True,
            max_model_len=2304, max_num_seqs=1, max_num_batched_tokens=2304, block_size=16,
            num_gpu_blocks_override=160, enable_prefix_caching=False, enable_chunked_prefill=False,
            async_scheduling=False, gpu_memory_utilization=args.memory, seed=42, disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        words = llm.get_tokenizer().encode('Explain how attention reads a sequence of tokens. ', add_special_tokens=True)
        length = 1024 if args.mechanism == '33.paged_decode' else 2048
        tokens = (words*((length+len(words)-1)//len(words)))[:length]
        prompt = dict(prompt_token_ids=tokens)
        params = SamplingParams(temperature=0, max_tokens=2 if args.mechanism == '33.paged_decode' else 1, ignore_eos=True)
        healthy, = llm.generate([prompt], params, use_tqdm=False)
        report['healthy_before'] = list(healthy.outputs[0].token_ids)
        provenance = dict(model_identity=report['model_identity'], prompt_token_ids=tokens,
            standard_worker=True, constructed_replay=False, probe_sha256=report['probe_sha256'])
        def deploy(expected, config):
            rows = llm.collective_rpc(setup, args=(expected, config.mechanism, tokens, provenance), timeout=300)
            if len(rows) != 1 or rows[0]['worker'] != expected:raise RuntimeError('TP1_DEPLOYMENT_IDENTITY_CHANGED')
            report['setup'] = rows[0]
            return {}
        fault, variant = args.mechanism.split('.', 1)
        client = WorkerFaultClient(HostAdapter(llm, prepare_binding=deploy, timeout=300), FaultConfig(int(fault), variant))
        drive(llm, client, prompt=prompt, params=params, out=args.out, report=report)
        if report.get('setup') and not client.snapshot()['ownership_retained']:
            removed = llm.collective_rpc(remove, args=(client.snapshot()['worker'], args.mechanism), timeout=300)
            if len(removed) != 1 or removed[0]['status'] != 'SERVING_DEPLOYMENT_REMOVED':
                raise RuntimeError('ATTENTION_BINDING_REMOVAL_UNCONFIRMED')
            report['binding_removal'] = removed[0]
    except BaseException as error:
        report.update(status='REFUSED_OR_FAILED', error=repr(error), traceback=traceback.format_exc())
    finally:
        save(args.out/'result.json', report)
    print(json.dumps({key: report[key] for key in ('status','error') if key in report}))
    return 0 if report['status'] == SUCCESS else 2


if __name__ == '__main__':raise SystemExit(main())
