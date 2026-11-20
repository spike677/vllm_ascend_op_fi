"""Private acceptance observer: whole hidden/logits hashes, no fault hooks."""
import threading


class HealthHashes:
    def __init__(self, model, processor):
        self.model, self.processor = model, processor
        self.thread = threading.get_ident()
        self.rows, self.handles = [], []
    def observe(self, kind, module, args, output):
        from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
        if threading.get_ident() != self.thread or len(self.rows) >= 4:
            raise RuntimeError('HEALTH_SERIAL_TWO_FORWARDS_REQUIRED')
        cpu, digest, finite = complete_tensor(output)
        if not finite:
            raise RuntimeError('HEALTH_NONFINITE_OUTPUT')
        self.rows.append(dict(kind=kind, shape=list(cpu.shape), dtype=str(cpu.dtype), sha256=digest, finite=finite))
    def install(self):
        if self.handles or self.rows:
            raise RuntimeError('HEALTH_OBSERVER_SINGLE_USE')
        try:
            self.handles.append(self.model.register_forward_hook(lambda m,a,o: self.observe('hidden',m,a,o)))
            self.handles.append(self.processor.register_forward_hook(lambda m,a,o: self.observe('logits',m,a,o)))
        except BaseException:
            self.close()
            raise
    def close(self):
        while self.handles:
            self.handles.pop().remove()
    def report(self):
        if self.handles or [r['kind'] for r in self.rows] != ['hidden','logits','hidden','logits']:
            raise RuntimeError('HEALTH_COMPLETE_PREFILL_DECODE_REQUIRED')
        if self.rows[2]['shape'][0] != 8 or self.rows[3]['shape'][0] != 8:
            raise RuntimeError('HEALTH_B8_DECODE_REQUIRED')
        return dict(rows=list(self.rows), owned_hooks=0, complete=True)


def install(worker):
    from experiments.worker_core_v11_20261118.probe_quant_ordinary import actual
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import discover_logits_boundary, require_serial_eager
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    worker=actual(worker);runner=worker.model_runner
    require_serial_eager(runner,expected_max_seqs=8)
    if hasattr(worker,'_dcat_quant_health_probe'):
        raise RuntimeError('HEALTH_OBSERVER_ALREADY_OWNED')
    boundary=discover_logits_boundary(runner.model,actual_logits_processor_type=LogitsProcessor,
        actual_vocab_size=runner.vllm_config.model_config.hf_config.vocab_size)
    observer=HealthHashes(runner.model,boundary['processor']);observer.install()
    worker._dcat_quant_health_probe=observer
    return dict(installed=True)


def remove(worker, completed):
    from experiments.worker_core_v11_20261118.probe_quant_ordinary import actual
    worker=actual(worker);observer=worker._dcat_quant_health_probe
    observer.close();del worker._dcat_quant_health_probe
    return observer.report() if completed else dict(owned_hooks=0,complete=False)


def generate(llm, prompts, params):
    from experiments.worker_core_v11_20261118.probe_quant_ordinary import rpc
    if llm.llm_engine.has_unfinished_requests() or llm.reset_prefix_cache(reset_running_requests=False) is not True:
        raise RuntimeError('HEALTH_IDLE_FRESH_BATCH_REQUIRED')
    rpc(llm,install)
    completed=False
    try:
        outputs=llm.generate(prompts,params,use_tqdm=False)
        completed=True
    finally:
        observation=rpc(llm,remove,completed)
    return [list(value.outputs[0].token_ids) for value in outputs], observation
