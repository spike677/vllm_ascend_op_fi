#!/usr/bin/env python3
"""Real vLLM 0.18 validation runner. No fake model or native fallback exists.

Run in a fresh, explicitly provisioned NPU container with both native libraries
already on LD_PRELOAD/library search paths. The default stage only discovers.
This module is importable without torch/vLLM for its pure RPC contract tests.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import uuid
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demoncat_fault39 import Fault39Config, Fault39OutputInspector, NativeBackend, WorkerEndpoint
from demoncat_fault39.schema import INTEGRATION_STATUS, PATTERNS, REQUEST_MAPPING
from demoncat_fault39.target import TargetError

WORKER_CLASS = "demoncat_fault39.vllm_worker.Fault39WorkerExtension"
DEFAULT_PROMPT = "Explain what a checksum is and why it is useful in software systems. Answer in one concise paragraph."


class RpcContractError(RuntimeError):
    pass


def _identity(result):
    if not isinstance(result, dict):
        raise RpcContractError("NON_OBJECT_WORKER_REPLY")
    rank = result.get("rank", result.get("identity", {}).get("rank"))
    return rank


class WorkerRpc:
    def __init__(self, model_executor, expected_workers, timeout=60.0):
        if not callable(getattr(model_executor, "collective_rpc", None)):
            raise RpcContractError("REAL_MODEL_EXECUTOR_COLLECTIVE_RPC_REQUIRED")
        self.executor, self.expected_workers, self.timeout = model_executor, expected_workers, timeout

    def call(self, action, payload=None, rank=None):
        payload = dict(payload or {})
        if rank is not None:
            payload["dispatch_rank"] = rank
        replies = self.executor.collective_rpc("fault39_rpc", timeout=self.timeout, args=(action, payload))
        if not isinstance(replies, (list, tuple)) or len(replies) != self.expected_workers:
            raise RpcContractError("WORKER_REPLY_CARDINALITY_MISMATCH")
        if rank is None:
            identities = [_identity(item) for item in replies]
            if any(type(value) is not int or value < 0 for value in identities) or len(set(identities)) != len(identities):
                raise RpcContractError("ACTUAL_WORKER_IDENTITY_REQUIRED")
            return {identity: item for identity, item in zip(identities, replies)}
        selected, skipped = [], []
        for reply in replies:
            if not isinstance(reply, dict):
                raise RpcContractError("NON_OBJECT_WORKER_REPLY")
            if reply.get("status") == "RPC_NOT_TARGETED":
                skipped.append(_identity(reply))
            else:
                selected.append(reply)
        if len(selected) != 1 or rank in skipped or len(set(skipped)) != len(skipped) or any(type(v) is not int for v in skipped):
            raise RpcContractError("RANK_DISPATCH_NOT_UNIQUE")
        identity = _identity(selected[0])
        if identity is not None and identity != rank:
            raise RpcContractError("WRONG_WORKER_REPLY")
        return selected[0]


class RemoteProvider:
    """The exact NativeProvider protocol transported to one actual worker.

    Diagnostic mode is an explicit runner-only selection: the backend's enable
    action is translated to lossless during P3, and to inject during P4.
    """
    def __init__(self, rpc, rank, device):
        self.rpc, self.rank, self.device = rpc, rank, device
        self.run_mode, self.input_mask = "inject", 3

    def prepare(self):
        reply = self.rpc.call("prepare", rank=self.rank)
        if reply.get("rank") != self.rank or reply.get("logical_device") != self.device:
            raise RpcContractError("ACTUAL_WORKER_DEVICE_CHANGED")
        return reply["provider"]

    def configure(self, owner, config, rank, device, mode="inject", target_signature="", input_mask=3):
        if rank != self.rank or device != self.device:
            raise RpcContractError("CONTROLLER_WORKER_IDENTITY_MISMATCH")
        if mode == "inject":
            mode, input_mask = self.run_mode, self.input_mask
        reply = self.rpc.call("configure", dict(owner=owner, config=config.as_dict(), rank=rank, device=device,
            mode=mode, target_signature=target_signature, input_mask=input_mask), rank=self.rank)
        generation = reply.get("generation")
        if type(generation) is not int or generation <= 0:
            raise RpcContractError("INVALID_NATIVE_GENERATION")
        return generation

    def snapshot(self):
        return self.evidence()["native"]

    def evidence(self):
        result = self.rpc.call("snapshot", rank=self.rank)
        if result.get("identity", {}).get("rank") != self.rank:
            raise RpcContractError("SNAPSHOT_IDENTITY_MISMATCH")
        return result

    def disable(self, owner, generation, timeout_ms):
        return self.rpc.call("disable", dict(owner=owner, generation=generation, timeout_ms=timeout_ms), rank=self.rank)

    def clean(self, owner, generation, timeout_ms):
        return self.rpc.call("clean", dict(owner=owner, generation=generation, timeout_ms=timeout_ms), rank=self.rank)


def create_backend(executor, tp, ranks, config, timeout=60.0):
    rpc = WorkerRpc(executor, tp, timeout)
    identities = rpc.call("prepare")
    if set(identities) != set(range(tp)):
        raise RpcContractError("TP_ACTUAL_RANK_SET_MISMATCH")
    providers, endpoints = {}, {}
    for rank, identity in identities.items():
        device = identity.get("logical_device")
        if type(device) is not int or device < 0:
            raise RpcContractError("ACTUAL_DEVICE_REQUIRED")
        providers[rank] = RemoteProvider(rpc, rank, device)
        endpoints[rank] = WorkerEndpoint(rank, device, providers[rank])
    if tuple(ranks) != config.ranks:
        raise ValueError("Controller rank/config mismatch")
    backend = NativeBackend(endpoints)
    backend.prepare()
    backend.register(config)
    return backend, providers, identities


def output_record(output):
    request_id = getattr(output, "request_id", None)
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError("ACTUAL_VLLM_REQUEST_ID_UNAVAILABLE")
    choices = [{"text": item.text, "token_ids": list(item.token_ids),
                "finish_reason": item.finish_reason, "stop_reason": getattr(item, "stop_reason", None)}
               for item in output.outputs]
    return dict(request_id=request_id, prompt=getattr(output, "prompt", None),
        prompt_token_ids=list(output.prompt_token_ids or []), choices=choices,
        usage=dict(prompt_tokens=len(output.prompt_token_ids or []), completion_tokens=sum(len(c["token_ids"]) for c in choices)))


def response_key(response):
    comparable = {key: response[key] for key in ("choices", "prompt_token_ids", "usage")}
    return hashlib.sha256(json.dumps(comparable, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _last_context_index(envelope):
    contexts = envelope.get("contexts", [])
    return max((c["index"] for c in contexts), default=-1)


def request_evidence(before, after, response_id):
    start = _last_context_index(before) + 1
    end = _last_context_index(after) + 1
    contexts = [c for c in after.get("contexts", []) if start <= c["index"] < end]
    complete = end > start and [c["index"] for c in contexts] == list(range(start, end))
    instrumentation_ok = complete and all(c.get("completed") is True and c.get("instrumentation_complete") is True
        and not c.get("instrumentation_errors") for c in contexts)
    # A returned client ID is matched only against the actual scheduler IDs.
    # No counter, timestamp or fault hit is allowed to fabricate that mapping.
    mapped = instrumentation_ok and all(c.get("request_ids") == [response_id] for c in contexts)
    return {"request_mapping": "SINGLE_REQUEST_SCHEDULER_MAPPING_VERIFIED" if mapped else REQUEST_MAPPING,
            "instrumentation_complete": instrumentation_ok, "context_range": [start, end],
            "contexts_dropped": after.get("contexts_dropped", 0), "contexts": contexts,
            "capture_complete": instrumentation_ok and all(c.get("capture_started") is True for c in contexts)}


@contextmanager
def observe_request_ids(processor):
    """Observe vLLM's own mapping; never infer it by stripping a suffix."""
    original = processor.assign_request_id
    had_override = 'assign_request_id' in vars(processor)
    mappings = []
    def observed(request):
        result = original(request)
        mappings.append({'external_req_id': getattr(request, 'external_req_id', None),
                         'request_id': getattr(request, 'request_id', None)})
        return result
    processor.assign_request_id = observed
    try:
        yield mappings
    finally:
        if processor.assign_request_id is not observed:
            raise RuntimeError('REQUEST_MAPPING_WRAPPER_OWNERSHIP_CHANGED')
        if had_override:
            processor.assign_request_id = original
        else:
            del processor.assign_request_id


def generate_one(llm, sampling_factory, prompt, providers, label):
    before = {rank: provider.evidence() for rank, provider in providers.items()}
    started = time.time()
    with observe_request_ids(llm.llm_engine.input_processor) as actual_mappings:
        outputs = llm.generate([prompt], sampling_factory(), use_tqdm=False)
    if len(outputs) != 1:
        raise RuntimeError("ONE_REAL_REQUEST_EXPECTED")
    response = output_record(outputs[0])
    matching = [m for m in actual_mappings if m['external_req_id'] == response['request_id']
                and isinstance(m['request_id'], str) and m['request_id']]
    internal_id = matching[0]['request_id'] if len(matching) == 1 else None
    after = {rank: provider.evidence() for rank, provider in providers.items()}
    return {"label": label, "response": response, "response_sha256": response_key(response),
            "elapsed_seconds": time.time() - started, 'actual_engine_request_mappings': actual_mappings,
            'request_mapping_source': 'EngineCoreRequest fields immediately after real InputProcessor.assign_request_id',
            "workers": {rank: {"before_native": before[rank]["native"], "after_native": after[rank]["native"],
                **request_evidence(before[rank], after[rank], internal_id)} for rank in providers}}


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hits(snapshot):
    return snapshot.get("hits", snapshot.get("counters", {}).get("hit_count", 0))


def _native_lossless_verified(snapshot, expected_mask):
    observation = snapshot.get("last_observation", {})
    proof = observation.get("proof", {})
    if proof.get("input_mask") != expected_mask or observation.get("changed") != [0, 0]:
        return False
    result = Fault39OutputInspector().inspect(snapshot)
    return (result["native_execution_verified"] and result["finite_inputs_verified"]
        and result["resources_released"] and result["verdict"] == "REAL_OPERATOR_HIT_VERIFIED")


def _gate_lossless(receipt_path, report):
    if not receipt_path:
        raise RuntimeError("LOSSLESS_RECEIPT_REQUIRED_BEFORE_PATTERN")
    profiles=report.get('native_build_profiles')
    if (not isinstance(profiles,dict) or not profiles or set(profiles)!={str(rank) for rank in report.get('ranks',[])} or
            any(not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value) for value in profiles.values())):
        raise RuntimeError('NATIVE_BUILD_PROFILE_REQUIRED')
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if receipt.get("status") != "LOSSLESS_VALIDATED" or receipt.get("stage") != "lossless":
        raise RuntimeError("LOSSLESS_GATE_NOT_PASSED")
    for key in ("model", "tp", "ranks", "environment", "native_build_profiles"):
        if receipt.get(key) != report.get(key):
            raise RuntimeError("LOSSLESS_GATE_ENVIRONMENT_MISMATCH: " + key)
    if receipt.get("targets") != report.get("targets"):
        raise RuntimeError("LOSSLESS_GATE_TARGET_MISMATCH")
    report["lossless_receipt_sha256"] = hashlib.sha256(Path(receipt_path).read_bytes()).hexdigest()


def finish_group_capture(selected, owner):
    """Attempt every selected worker even if one stop/RPC fails."""
    workers = {}
    for rank, provider in selected.items():
        try:
            result = provider.rpc.call('finish_capture', {'owner': owner}, rank=rank)
            if (not isinstance(result, dict) or result.get('status') != 'CAPTURE_FINISHED'
                    or result.get('rank') != rank or result.get('owner') != owner
                    or result.get('capture_active') is not False):
                result = {'status': 'CAPTURE_FINISH_FAILED', 'reply': result}
        except Exception as error:
            result = {'status': 'CAPTURE_FINISH_FAILED', 'error_type': type(error).__name__}
        workers[rank] = result
    return {'status': 'CAPTURE_FINISHED' if workers and all(
        item['status'] == 'CAPTURE_FINISHED' for item in workers.values()) else 'CAPTURE_FINISH_FAILED',
        'workers': workers}


def validate(args, llm, sampling_factory, environment):
    config = Fault39Config(pattern=args.pattern, amplitude_policy=args.amplitude_policy,
        ranks=tuple(args.ranks), target_policy=args.target_policy, max_hits=1, max_bytes=args.max_bytes)
    executor = getattr(llm.llm_engine, "model_executor", None)
    backend, providers, identities = create_backend(executor, args.tp, args.ranks, config, args.rpc_timeout)
    report = {"schema_version": 1, "stage": args.stage, "status": "STARTED", "model": str(Path(args.model).resolve()),
        "tp": args.tp, "ranks": list(args.ranks), "config": config.as_dict(), "environment": environment,
        "worker_identities": identities, "phases": [], "inner_framework_integration": INTEGRATION_STATUS,
        "request_mapping": REQUEST_MAPPING, "msprobe_first_difference": "NOT_ANALYZED", "tensor_payloads_uploaded": False}
    # Version adaptation invalidates an old lossless receipt even if model and
    # package version strings are unchanged. Bind all selected worker profiles.
    report['native_build_profiles']={str(rank):providers[rank].snapshot().get('build_profile_sha256') for rank in args.ranks}
    if any(not isinstance(value,str) or len(value)!=64 for value in report['native_build_profiles'].values()):
        raise RuntimeError('NATIVE_BUILD_PROFILE_REQUIRED')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report_file = output / "validation.json"
    _write_json(report_file, report)
    selected = {rank: providers[rank] for rank in args.ranks}
    try:
        if args.capture_steps and not args.capture_group:
            if not output.resolve().is_relative_to(Path("/evidence")):
                raise RuntimeError("MSPROBE_CAPTURE_REQUIRES_OUTPUT_UNDER_EVIDENCE")
            report["capture"] = {rank: provider.rpc.call("capture", dict(path=str(output / f"rank{rank}"),
                steps=list(range(args.capture_steps))), rank=rank) for rank, provider in selected.items()}
        backend.discover()
        baselines = []
        for round_number in range(2):
            group = []
            for index, prompt in enumerate(args.prompts):
                phase = generate_one(llm, sampling_factory, prompt, providers, f"baseline{round_number + 1}/prompt{index}")
                report["phases"].append(phase)
                group.append(phase)
                _write_json(report_file, report)
            baselines.append(group)
            report[f"inventory_round_{round_number + 1}"] = backend.observe_inventory()
        report["baseline_stable"] = all(a["response_sha256"] == b["response_sha256"] for a, b in zip(*baselines))
        report["targets"] = {str(rank): signature for rank, signature in backend.targets.items()}
        report["discovery_disable"] = backend.end_discovery(args.drain_timeout_ms)
        if not report["baseline_stable"]:
            raise RuntimeError("BASELINE_UNSTABLE")
        if report["discovery_disable"]["status"] != "DISABLED_CONFIRMED":
            raise RuntimeError(report["discovery_disable"]["status"])
        report["discovery_snapshot"] = backend.query()
        if args.stage == "discovery":
            report["status"] = "STABLE_UNIQUE_TARGET_DISCOVERED"
            return report
        if args.stage == "pattern":
            _gate_lossless(args.lossless_receipt, report)
            if args.capture_group:
                if not args.capture_steps:
                    raise RuntimeError('CAPTURE_STEPS_REQUIRED')
                report['capture'] = {rank: provider.rpc.call('capture', dict(path=str(output / args.capture_group / f'rank{rank}'),
                    steps=list(range(args.capture_steps))), rank=rank) for rank, provider in selected.items()}
        masks = (1, 2, 3) if args.stage == "lossless" else (3,)
        validated = []
        for mask in masks:
            for provider in selected.values():
                provider.run_mode = "lossless" if args.stage == "lossless" else 'observer' if args.capture_group == 'baseline' else "inject"
                provider.input_mask = mask
            for index, prompt in enumerate(args.prompts):
                backend.enable()
                try:
                    phase = generate_one(llm, sampling_factory, prompt, providers, f"{args.stage}/mask{mask}/prompt{index}")
                finally:
                    if args.capture_group:
                        report['capture_finish'] = finish_group_capture(selected, backend.owner)
                        _write_json(report_file, report)
                if args.capture_group and report['capture_finish']['status'] != 'CAPTURE_FINISHED':
                    raise RuntimeError('CAPTURE_FINISH_FAILED: OFF workload was not started')
                report["phases"].append(phase)
                phase["disable"] = backend.disable(args.drain_timeout_ms)
                if phase["disable"]["status"] != "DISABLED_CONFIRMED":
                    raise RuntimeError(phase["disable"]["status"])
                inspector = {rank: Fault39OutputInspector().inspect(phase["workers"][rank]["after_native"])
                             for rank in args.ranks}
                phase["native_inspection"] = inspector
                before_off = {rank: provider.snapshot() for rank, provider in providers.items()}
                recovery = generate_one(llm, sampling_factory, prompt, providers, f"off/mask{mask}/prompt{index}")
                report["phases"].append(recovery)
                recovery_ok = (recovery["response_sha256"] == baselines[0][index]["response_sha256"]
                    and all(_hits(before_off[rank]) == _hits(recovery["workers"][rank]["after_native"]) for rank in providers)
                    and all(not recovery["workers"][rank]["after_native"].get("enabled")
                        and recovery["workers"][rank]["after_native"].get("inflight") == 0
                        and recovery["workers"][rank]["after_native"].get("leases") == 0 for rank in args.ranks))
                phase["recovery_verified"] = recovery_ok
                unmapped = any(phase["workers"][rank]["request_mapping"] == REQUEST_MAPPING for rank in args.ranks)
                unselected_zero = all(_hits(phase["workers"][rank]["after_native"]) == 0 for rank in providers if rank not in args.ranks)
                if args.stage == "lossless":
                    native_ok = all(_native_lossless_verified(phase["workers"][rank]["after_native"], mask) for rank in args.ranks)
                    native_ok = native_ok and phase["response_sha256"] == baselines[0][index]["response_sha256"]
                elif args.capture_group == 'baseline':
                    native_ok = phase['response_sha256'] == baselines[0][index]['response_sha256']
                    for rank in args.ranks:
                        snap = phase['workers'][rank]['after_native']
                        obs = snap.get('last_observation', {})
                        native_ok = native_ok and snap.get('mode') == 4 and snap.get('hits') == 1 and snap.get('inflight') == 0 and snap.get('leases') == 0
                        native_ok = native_ok and all(obs.get(k) == 0 for k in ('pre_sync_status','copy_status','run_status','post_sync_status','free_status'))
                else:
                    native_ok = all(value["verdict"] in ("REAL_OPERATOR_HIT_VERIFIED", "MATMUL_FINITE_INPUT_NONFINITE_REPRODUCED",
                        "MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED") for value in inspector.values())
                phase["validated"] = native_ok and recovery_ok and not unmapped and unselected_zero
                validated.append(phase["validated"])
                _write_json(report_file, report)
                if not phase["validated"]:
                    raise RuntimeError(REQUEST_MAPPING if unmapped else "NATIVE_OR_RECOVERY_GATE_FAILED")
        report["status"] = "LOSSLESS_VALIDATED" if args.stage == "lossless" else "PATTERN_NATIVE_VALIDATED_MSPROBE_ANALYSIS_PENDING"
        if args.capture_group == 'baseline':
            report['status'] = 'BASELINE_MODEL_CAPTURED'
        return report
    except Exception as error:
        report["status"] = str(error) if isinstance(error, (TargetError, RuntimeError)) else "VALIDATION_FAILED"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        return report
    finally:
        report["cleanup"] = backend.clean(args.drain_timeout_ms)
        if report["cleanup"]["status"] != "DISABLED_CONFIRMED":
            report["status"] = report["cleanup"]["status"]
        else:
            try:
                report["detached"] = {rank: provider.rpc.call("detach", rank=rank) for rank, provider in providers.items()}
            except Exception as error:
                report["detach_error"] = type(error).__name__ + ": " + str(error)
                report["status"] = "DETACH_FAILED"
        _write_json(report_file, report)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True)
    result.add_argument("--tp", type=int, choices=(1, 2, 4), default=1)
    result.add_argument("--ranks", type=int, nargs="+", default=[0])
    result.add_argument("--output", required=True)
    result.add_argument("--stage", choices=("discovery", "lossless", "pattern"), default="discovery")
    result.add_argument("--pattern", choices=PATTERNS, default="half")
    result.add_argument("--amplitude-policy", choices=("historical58", "auto"), default="auto")
    result.add_argument('--target-policy', choices=('unique', 'max_k_min_m_unique'), default='unique')
    result.add_argument("--lossless-receipt")
    result.add_argument("--prompt", dest="prompts", action="append")
    result.add_argument("--max-tokens", type=int, default=16)
    result.add_argument("--max-model-len", type=int, default=512)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--capture-steps", type=int, default=0)
    result.add_argument('--capture-group', choices=('baseline', 'fault'))
    result.add_argument("--memory-utilization", type=float, default=0.6)
    result.add_argument("--max-bytes", type=int, default=512 * 1024 * 1024)
    result.add_argument("--rpc-timeout", type=float, default=60.0)
    result.add_argument("--drain-timeout-ms", type=int, default=10000)
    return result


def main():
    args = parser().parse_args()
    args.prompts = args.prompts or [DEFAULT_PROMPT]
    if not Path(args.model).is_dir():
        raise SystemExit("A locally mounted model directory is required; downloads are not supported")
    if Path(args.output).exists():
        raise SystemExit("A new output directory is required; prior evidence is never overwritten")
    if args.max_tokens <= 0 or not 0 <= args.capture_steps <= 256 or not 0 < args.memory_utilization < 1:
        raise SystemExit("Invalid token/capture/memory bounds")
    if args.capture_group and args.stage != 'pattern':
        raise SystemExit('Common B/F capture callsite requires --stage pattern')
    if args.capture_group and len(args.prompts) != 1:
        raise SystemExit('A B/F capture group requires exactly one selected request per fresh worker')
    if any(rank not in range(args.tp) for rank in args.ranks) or len(set(args.ranks)) != len(args.ranks):
        raise SystemExit("Invalid selected ranks")
    if not os.environ.get("LD_PRELOAD") or os.environ.get("VLLM_SERVER_DEV_MODE") != "1":
        raise SystemExit("Worker preload and VLLM_SERVER_DEV_MODE=1 must be established before Python starts")
    if os.environ.get("TASK_QUEUE_ENABLE") != "0":
        raise SystemExit("This TLS request-scope validation requires TASK_QUEUE_ENABLE=0 before Python starts")
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "0") != "0":
        raise SystemExit("Direct model_executor access requires VLLM_ENABLE_V1_MULTIPROCESSING=0")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from vllm import LLM, SamplingParams
    versions = {name: importlib.metadata.version(name) for name in ("vllm", "vllm-ascend", "torch", "torch-npu")}
    if not versions["vllm"].startswith(("0.18.", "0.23.")):
        raise SystemExit("Runner adapters cover vLLM 0.18/0.23; actual worker/scheduler contracts are checked at attachment")
    preload_files = [Path(value) for value in os.environ["LD_PRELOAD"].replace(":", " ").split()]
    preload_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in preload_files if path.is_file()}
    environment = {"packages": versions, "preload_files": preload_hashes, "worker_extension": WORKER_CLASS,
                   "enforce_eager": True, "dtype": "bfloat16", "max_model_len": args.max_model_len,
                   "task_queue_enable": "0", "async_scheduling": False,
                   "seed": args.seed, "max_tokens": args.max_tokens}
    config_path = Path(args.model) / 'config.json'
    model_config = json.loads(config_path.read_text())
    weights = sorted(Path(args.model).glob('*.safetensors')) or sorted(Path(args.model).glob('pytorch_model*.bin'))
    if not weights:
        raise SystemExit('Actual local model weights required')
    identity = {'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                'model_type': model_config.get('model_type'), 'architectures': model_config.get('architectures'), 'weights': {}}
    for weight in weights:
        digest = hashlib.sha256()
        with weight.open('rb') as stream:
            for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                digest.update(block)
        identity['weights'][weight.name] = {'bytes': weight.stat().st_size, 'sha256': digest.hexdigest()}
    environment['model_identity'] = identity
    extra = ({'additional_config': {'enable_flashcomm1': False}, 'enable_chunked_prefill': False,
              'block_size': 128}
             if versions['vllm'].startswith('0.23.') else {})
    environment['runner_options'] = extra
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16", enforce_eager=True,
        worker_extension_cls=WORKER_CLASS, max_model_len=args.max_model_len, max_num_seqs=1,
        gpu_memory_utilization=args.memory_utilization, seed=args.seed, trust_remote_code=False,
        enable_prefix_caching=False, async_scheduling=False, **extra)
    report = validate(args, llm, lambda: SamplingParams(temperature=0, seed=args.seed, max_tokens=args.max_tokens), environment)
    print(json.dumps({"status": report["status"], "report": str(Path(args.output) / "validation.json")}, ensure_ascii=False))
    return 0 if report["status"] in ("STABLE_UNIQUE_TARGET_DISCOVERED", "LOSSLESS_VALIDATED", 'BASELINE_MODEL_CAPTURED', "PATTERN_NATIVE_VALIDATED_MSPROBE_ANALYSIS_PENDING") else 2


if __name__ == "__main__":
    raise SystemExit(main())
