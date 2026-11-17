"""Root-queued healthy DeepSeek whole-model quant/QMM inventory; no injection."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import traceback

PROMPT = 'Explain in one sentence why the sky appears blue during the day.'


def worker_actual(worker):
    from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase
    if isinstance(worker, WorkerWrapperBase):
        worker = worker.worker
    if not isinstance(worker, WorkerBase):
        raise RuntimeError('ACTUAL_STANDARD_WORKER_REQUIRED')
    return worker


def setup(worker, tokens):
    worker = worker_actual(worker)
    from demoncat.runtime.vllm_ascend.worker_extension.serving_quant_inventory import install_quant_inventory
    import demoncat.runtime.vllm_ascend.worker_extension.quant_inventory as inventory
    sources = []
    for module in (install_quant_inventory, inventory):
        path = Path(inspect.getfile(module)).resolve()
        if not path.is_relative_to(Path('/installed')):
            raise RuntimeError('FRESH_INSTALLED_OBSERVER_REQUIRED')
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(binding=install_quant_inventory(worker, prompt_token_ids=tokens,
        evidence_dir='/evidence/public-quant-inventory'), sources=sources)


def advance(worker, request_id):
    session = worker_actual(worker)._demoncat_quant_inventory
    session.expect(request_id)
    return session.report()


def query(worker):
    return worker_actual(worker)._demoncat_quant_inventory.report()


def remove(worker):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_quant_inventory import remove_quant_inventory
    return remove_quant_inventory(worker_actual(worker))


def rpc(llm, function, *args):
    replies = llm.collective_rpc(function, args=args, timeout=300)
    if len(replies) != 1:
        raise RuntimeError('ONE_ACTUAL_WORKER_REQUIRED')
    return replies[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/model')
    parser.add_argument('--out', type=Path, default=Path('/evidence/result.json'))
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core, read_enqueued_identity
    report = dict(status='STARTING', scope='NATURAL_WHOLE_WORKER_PUBLIC_QUANT_LINEAGE_ONLY',
                  native_injections=0, c1=False, c2=False, c3=False, phases=[])
    def save():
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    llm = None
    installed = False
    ids_seen = []
    try:
        llm = LLM(model=args.model, dtype='bfloat16', tensor_parallel_size=1, enforce_eager=True,
            max_model_len=128, max_num_seqs=1, max_num_batched_tokens=128, enable_prefix_caching=False,
            enable_chunked_prefill=False, async_scheduling=False, gpu_memory_utilization=.55,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension', disable_log_stats=True)
        require_host_driven_core(llm)
        tokens = llm.get_tokenizer().encode(PROMPT, add_special_tokens=True)
        prompt = {'prompt_token_ids': tokens}
        sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        before, = llm.generate([prompt], sampling, use_tqdm=False)
        report['healthy_before'] = list(before.outputs[0].token_ids)
        report['setup'] = rpc(llm, setup, tokens)
        installed = True
        for index in range(2):
            if llm.llm_engine.has_unfinished_requests() or llm.reset_prefix_cache(reset_running_requests=False) is not True:
                raise RuntimeError('IDLE_FRESH_PREFILL_REQUIRED')
            ids = llm.enqueue([prompt], sampling, use_tqdm=False)
            mapping = read_enqueued_identity(llm, ids, seen_internal_request_ids=ids_seen)
            ids_seen.append(ids[0])
            row = dict(index=index, mapping=mapping, advance=rpc(llm, advance, ids[0]))
            report['phases'].append(row)
            save()
            outputs = llm.wait_for_completion(use_tqdm=False)
            if len(outputs) != 1 or outputs[0].request_id != mapping['external_request_id']:
                raise RuntimeError('ACTUAL_REQUEST_MAPPING_CHANGED')
            row['inventory'] = rpc(llm, query)
            row['generated_tokens'] = list(outputs[0].outputs[0].token_ids)
            save()
        report['remove'] = rpc(llm, remove)
        installed = False
        after, = llm.generate([prompt], sampling, use_tqdm=False)
        report['healthy_after'] = list(after.outputs[0].token_ids)
        final = report['remove']['report']
        if not final['complete_healthy_replays'] or report['healthy_before'] != report['healthy_after']:
            raise RuntimeError('HEALTHY_OBSERVER_REPLAY_OR_RESTORE_NOT_CONFIRMED')
        report['status'] = 'PUBLIC_INVENTORY_OBSERVED_NATIVE_QUALIFICATION_PENDING'
    except BaseException as error:
        report.update(status='FAILED_WITH_EVIDENCE', error=repr(error), traceback=traceback.format_exc())
    finally:
        if installed and llm is not None:
            try:
                report['cleanup'] = rpc(llm, remove)
            except BaseException as error:
                report['cleanup_error'] = repr(error)
        save()
    return 0 if report['status'] == 'PUBLIC_INVENTORY_OBSERVED_NATIVE_QUALIFICATION_PENDING' else 2


if __name__ == '__main__':
    raise SystemExit(main())
