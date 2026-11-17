"""Root-queued natural model census. No injection, layer selector, or format cast.

Run --kind worker for genuine standard vLLM requests, --kind hf for pretrained
HF inference, or --kind vae for the existing pinned HunyuanVideo VAE component.
"""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import traceback
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
PROMPT = 'Explain what a checksum is in one sentence.'
VAE_FILES = {
    'autoencoder_kl_causal_3d.py': 'e2596d5cafa8ef4b454daea672fb7da38d8470d96be1cdd20d7d9916866b268e',
    'vae.py': '0c25afa6a33180232e04f3b518c953915bff4ee5d61751052f46a790c756b5d9',
    'unet_causal_3d_blocks.py': '181b187d3660e13eb370c0cb132fb3392f62b7ec96887c57ef36ca963cc42a58'}


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(data)
    return value.hexdigest()


def output_identity(value):
    import torch
    from fault_injection.tensor_observer import tensors
    result = []
    for tensor in tensors(value):
        cpu = tensor.detach().cpu().contiguous()
        result.append(dict(shape=list(cpu.shape), dtype=str(cpu.dtype),
            sha256=hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
            finite=bool(torch.isfinite(cpu).all()) if cpu.is_floating_point() else True))
    if not result:
        raise RuntimeError('ACTUAL_MODEL_TENSOR_OUTPUT_REQUIRED')
    return result


def load_local(args):
    import torch
    if args.kind == 'hf':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        network = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation='eager').eval().npu()
        class Prefill(torch.nn.Module):
            def __init__(self, actual):
                super().__init__(); self.actual = actual
            def forward(self, ids, mask):
                return self.actual(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
        model = Prefill(network).eval()
        encoded = tokenizer(PROMPT, return_tensors='pt', max_length=32, truncation=True)
        return model, (encoded['input_ids'].npu(), encoded['attention_mask'].npu()), dict(
            scope='NATURAL_PRETRAINED_HF_PREFILL', standard_vllm_worker=False,
            model_class=type(network).__module__+'.'+type(network).__name__)
    if args.vae_source is None or args.latent is None:
        raise ValueError('Pinned VAE source and existing fixed latent paths are required')
    for name, expected in VAE_FILES.items():
        if sha_file(args.vae_source / name) != expected:
            raise RuntimeError('PINNED_VAE_SOURCE_MISMATCH:' + name)
    if sha_file(args.model / 'pytorch_model.pt') != '95d1fc707c1421ccd88ea542838ab4c5d45a5babb48205bac9ce0985525f9818':
        raise RuntimeError('PINNED_PRETRAINED_WEIGHTS_MISMATCH')
    if sha_file(args.model / 'config.json') != '3e77776ccbc0e366f2987c6304c37b00696e4d8b2b1f0487cba8cbc93dc6abc4':
        raise RuntimeError('PINNED_VAE_CONFIG_MISMATCH')
    package = types.ModuleType('hv_vae')
    package.__path__ = [str(args.vae_source.resolve())]
    if 'hv_vae' in sys.modules:
        raise RuntimeError('FOREIGN_VAE_PACKAGE_ALREADY_LOADED')
    sys.modules['hv_vae'] = package
    cls = importlib.import_module('hv_vae.autoencoder_kl_causal_3d').AutoencoderKLCausal3D
    vae = cls.from_config(json.loads((args.model / 'config.json').read_text()))
    checkpoint = torch.load(args.model / 'pytorch_model.pt', map_location='cpu', weights_only=True)
    state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    if any(k.startswith('vae.') for k in state):
        state = {k.replace('vae.', ''): value for k, value in state.items() if k.startswith('vae.')}
    vae.load_state_dict(state, strict=True)
    del state, checkpoint
    vae.requires_grad_(False).eval().to(dtype=torch.bfloat16, device='npu')
    vae.enable_tiling()
    if (vae.tile_latent_min_size, vae.tile_sample_min_size) != (32, 256):
        raise RuntimeError('UNREVIEWED_VAE_TILE_GEOMETRY')
    from safetensors.torch import load_file
    payload = load_file(str(args.latent), device='cpu')
    latent = payload['latent']
    if set(payload) != {'latent'} or tuple(latent.shape) != (1, 16, 2, 33, 33) or latent.dtype != torch.bfloat16:
        raise RuntimeError('PINNED_LATENT_GEOMETRY_REQUIRED')
    class Decode(torch.nn.Module):
        def __init__(self, actual):
            super().__init__(); self.actual = actual
        def forward(self, latent):
            return self.actual.decode(latent).sample
    return Decode(vae).eval(), (latent.npu(),), dict(scope='NATURAL_PRETRAINED_VAE_COMPONENT',
        standard_vllm_worker=False, model_class=type(vae).__module__+'.'+type(vae).__name__,
        source_hashes=VAE_FILES, weight_sha256=sha_file(args.model / 'pytorch_model.pt'),
        latent_sha256=sha_file(args.latent), tiling_geometry_modified=False)


def _worker(worker):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    return worker.worker if isinstance(worker, WorkerWrapperBase) else worker


def worker_install(worker, prompt):
    worker = _worker(worker)
    from fault_injection.tensor_observer import NaturalTensorObserver
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import snapshot_fresh_prefill
    from vllm.forward_context import get_forward_context
    model = worker.model_runner.model
    if hasattr(worker, '_tensor_census'):
        raise RuntimeError('FOREIGN_CENSUS_BINDING')
    state = dict(expected=None, seen=[], rows=[], handles=[], prompt=prompt)
    signature = inspect.signature(model.forward)
    def pre(module, args, kwargs):
        if state['expected'] is None:
            raise RuntimeError('UNSCHEDULED_CENSUS_REQUEST')
        bound = signature.bind(*args, **kwargs); bound.apply_defaults()
        scheduler = snapshot_fresh_prefill(worker.model_runner, model, bound.arguments,
            get_forward_context(), expected_prompt_token_ids=prompt,
            expected_internal_request_id=state['expected'], seen_internal_request_ids=state['seen'])
        state['rows'].append(dict(request_id=state['expected'], scheduler=scheduler))
    state['handles'].append(model.register_forward_pre_hook(pre, with_kwargs=True))
    state['observer'] = NaturalTensorObserver(model).__enter__()
    def post(module, args, kwargs, output):
        state['rows'][-1]['complete_output'] = output_identity(output)
        state['seen'].append(state['expected']); state['expected'] = None
    state['handles'].append(model.register_forward_hook(post, with_kwargs=True))
    worker._tensor_census = state
    return dict(model_class=type(model).__module__+'.'+type(model).__name__,
                scope='NATURAL_STANDARD_VLLM_WORKER', observer_installed=True)


def worker_advance(worker, request_id):
    state = _worker(worker)._tensor_census
    if state['expected'] is not None or request_id in state['seen']:
        raise RuntimeError('CENSUS_REQUEST_LIFETIME_COLLISION')
    state['expected'] = request_id
    return dict(request_id=request_id)


def worker_close(worker):
    worker = _worker(worker); state = worker._tensor_census
    state['observer'].close()
    for handle in reversed(state['handles']):
        handle.remove()
    report = dict(census=state['observer'].report(), phases=state['rows'],
                  root_context_hooks=0, pending_request=state['expected'])
    del worker._tensor_census
    return report


def run_worker(args, report):
    from vllm import LLM, SamplingParams
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core, read_enqueued_identity
    llm = LLM(model=str(args.model), dtype='bfloat16', tensor_parallel_size=1, enforce_eager=True,
        max_model_len=128, max_num_seqs=1, max_num_batched_tokens=128, num_gpu_blocks_override=64,
        enable_prefix_caching=False, enable_chunked_prefill=False, async_scheduling=False,
        gpu_memory_utilization=args.memory, seed=42, disable_log_stats=True)
    require_host_driven_core(llm)
    def rpc(function, *values):
        rows = llm.collective_rpc(function, args=values, timeout=300)
        if len(rows) != 1:
            raise RuntimeError('TP1_CENSUS_REQUIRED')
        return rows[0]
    tokens = llm.get_tokenizer().encode(PROMPT, add_special_tokens=True)
    prompt = dict(prompt_token_ids=tokens)
    params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    llm.generate([prompt], params, use_tqdm=False)
    report['setup'] = rpc(worker_install, tokens)
    report['requests'] = []; seen = []
    try:
        for phase in ('baseline', 'baseline_repeat'):
            if llm.reset_prefix_cache(reset_running_requests=False) is not True:
                raise RuntimeError('CACHE_RESET_UNCONFIRMED')
            ids = llm.enqueue([prompt], params, use_tqdm=False)
            mapping = read_enqueued_identity(llm, ids, seen_internal_request_ids=seen)
            rpc(worker_advance, ids[0]); seen += ids
            responses = llm.wait_for_completion(use_tqdm=False)
            report['requests'].append(dict(phase=phase, mapping=mapping,
                output_tokens=list(responses[0].outputs[0].token_ids)))
    finally:
        report.update(rpc(worker_close))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('hf', 'vae', 'worker'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--vae-source', type=Path)
    parser.add_argument('--latent', type=Path)
    parser.add_argument('--memory', type=float, default=.35)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='STARTED', model=str(args.model), kind=args.kind,
        fault_injections=0, qualification_granted=False, c2=False, c3=False)
    try:
        import torch
        import torch_npu
        torch.npu.set_device(0)
        report['runtime'] = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
            device=torch.npu.get_device_name(0))
        if args.kind == 'worker':
            run_worker(args, report)
            outputs = [row['complete_output'] for row in report['phases']]
        else:
            from fault_injection.tensor_observer import NaturalTensorObserver
            model, inputs, provenance = load_local(args)
            report['setup'] = provenance
            outputs = []
            with NaturalTensorObserver(model) as observer, torch.inference_mode():
                for phase in ('baseline', 'baseline_repeat'):
                    output = model(*inputs)
                    torch.npu.synchronize()
                    outputs.append(output_identity(output))
            report['census'] = observer.report()
        report['outputs'] = outputs
        report['baseline_repeat_exact'] = len(outputs) == 2 and outputs[0] == outputs[1]
        report['status'] = ('NATURAL_CENSUS_COMPLETE' if report['census']['complete']
            and report['baseline_repeat_exact'] else 'CENSUS_INCOMPLETE_OR_NONDETERMINISTIC')
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
    report['probe_sha256'] = sha_file(__file__)
    (args.out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('status', 'kind', 'error') if key in report}))
    return 0 if report['status'] == 'NATURAL_CENSUS_COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
