#!/usr/bin/env python3
"""One real torch_npu operator case per process; never launches other cases.

The caller must preload the native runtime and select the isolated proxy before
Python starts. No tensor payload is written. This is not a model request probe.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demoncat_fault39.inspector import Fault39OutputInspector, validate_statistics
from demoncat_fault39.provider import CtypesProvider, NativeError
from demoncat_fault39.schema import Fault39Config, PATTERNS
from demoncat_fault39.target import select_stable_target


def verify_dynamic_slot_proof(snapshot, addresses_before, addresses_after, input_mask, *, observer_only=False):
    """CPU-checkable relation to live torch storage, beyond a private ABI assumption."""
    if (len(addresses_before) != 3 or addresses_before != addresses_after
            or any(type(address) is not int or address <= 0 for address in addresses_before)
            or len(set(addresses_before)) != 3):
        raise RuntimeError("LIVE_TORCH_STORAGE_ADDRESSES_INVALID_OR_CHANGED")
    observation = snapshot.get("last_observation", {})
    proof = observation.get("proof", {})
    tensors = proof.get("tensors", [])
    if (proof.get("api") != "aclnnMatmul" or proof.get("slots") != [0, 1, 2]
            or proof.get("formal_order_proven") is not True or len(tensors) != 3
            or proof.get("input_mask") != input_mask
            or observation.get("generation") != snapshot.get("generation")
            or proof.get("generation") != snapshot.get("generation")
            or not snapshot.get("owner") or observation.get("owner") != snapshot.get("owner")):
        raise RuntimeError("DYNAMIC_SLOT_PROOF_INCOMPLETE")
    effective_mask = 0 if observer_only else input_mask
    for slot, (tensor, expected) in enumerate(zip(tensors, addresses_before)):
        if tensor.get("original_address") != expected:
            raise RuntimeError("DYNAMIC_SLOT_ADDRESS_MISMATCH: slot=" + str(slot))
        address = tensor.get("run_address")
        if type(address) is not int or address <= 0:
            raise RuntimeError("INVALID_RUN_ADDRESS: slot=" + str(slot))
        should_replace = slot < 2 and bool(effective_mask & (1 << slot))
        if (address != expected) != should_replace:
            raise RuntimeError("DYNAMIC_SLOT_REPLACEMENT_MISMATCH: slot=" + str(slot))
    return {"status": "PASS", "observer_only": observer_only, "buffer_substitution": not observer_only,
            "slots": [0, 1, 2], "torch_storage_addresses": list(addresses_before),
            "run_addresses": [tensor["run_address"] for tensor in tensors],
            "effective_input_mask": effective_mask, "original_addresses_match_live_torch": True,
            "output_storage_unchanged": True}


def verify_active_owner_isolation(provider, owner, generation, config, rank, device, target_signature):
    """Exercise rejected ABI operations while the original observer stays armed.

    No owner is released, and no successful configure/stop is expected here.
    The caller must subsequently execute its already planned observer Run.
    """
    before = provider.snapshot()
    if (before.get('owner') != owner or before.get('generation') != generation
            or before.get('enabled') is not True or before.get('mode') != 4
            or before.get('inflight') != 0 or before.get('leases') != 0
            or type(generation) is not int or generation <= 1):
        raise RuntimeError('OWNER_ISOLATION_REQUIRES_ACTIVE_IDLE_OBSERVER_AND_PRIOR_GENERATION')
    foreign = 'fault39-isolation-foreign'
    if owner == foreign:
        foreign = 'fault39-isolation-other'
    report = {'status': 'OWNER_REJECTIONS_VERIFIED_PENDING_RUN', 'owner': owner,
              'generation': generation, 'foreign_owner': foreign, 'stale_generation': generation - 1,
              'scope': 'Fault39 native ABI active owner; not internal cross-fault integration',
              'before': before, 'operations': [], 'owner_execution_verified': False}
    operations = (
        ('foreign_configure', -4, lambda: provider.configure(foreign, config, rank, device,
            mode='observer', target_signature=target_signature, input_mask=3)),
        ('foreign_disable', -4, lambda: provider.disable(foreign, generation, 0)),
        ('foreign_clean', -4, lambda: provider.clean(foreign, generation, 0)),
        ('stale_generation_disable', -5, lambda: provider.disable(owner, generation - 1, 0)),
    )
    for name, expected, action in operations:
        try:
            action()
        except NativeError as error:
            if error.code != expected:
                raise RuntimeError(f'OWNER_ISOLATION_WRONG_REJECTION: {name}: {error.code}, expected {expected}') from error
            code = error.code
        else:
            raise RuntimeError('OWNER_ISOLATION_OPERATION_NOT_REJECTED: ' + name)
        after = provider.snapshot()
        if after != before:
            raise RuntimeError('OWNER_ISOLATION_REJECTION_CHANGED_STATE: ' + name)
        report['operations'].append({'operation': name, 'native_code': code,
                                     'expected_native_code': expected, 'snapshot_unchanged': True})
    report['after'] = after
    report['snapshot_unchanged'] = True
    return report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("discovery", "lossless", "pattern"), default="discovery")
    p.add_argument("--pattern", choices=PATTERNS, default="half")
    p.add_argument("--input-mask", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--amplitude-policy", choices=("historical58", "auto"), default="historical58")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--k", type=int, default=12800)
    p.add_argument("--n", type=int, default=5120)
    p.add_argument("--device", type=int, default=0, help="Logical NPU device in the isolated process")
    p.add_argument("--max-storage-bytes", type=int, default=256 * 1024**2)
    p.add_argument("--max-working-set-bytes", type=int, default=1024**3)
    p.add_argument("--drain-timeout-ms", type=int, default=10000)
    p.add_argument('--capture-group', choices=('baseline', 'fault'))
    p.add_argument('--check-owner-isolation', action='store_true',
                   help='Check rejected foreign owner/stale generation operations before the planned observer Run')
    args = p.parse_args()
    if min(args.m, args.k, args.n) <= 0 or args.k % 2:
        p.error("M/K/N must be positive and K must be even")
    if args.device < 0 or args.max_storage_bytes <= 0 or args.max_working_set_bytes <= 0:
        p.error("Invalid device or resource cap")
    if not 0 <= args.drain_timeout_ms < 2**32:
        p.error("Invalid drain timeout")
    if args.stage != "lossless" and args.input_mask != 3:
        p.error("Only --stage lossless accepts --input-mask 1 or 2")
    if args.capture_group and args.stage != 'pattern':
        p.error('B/F capture requires the common pattern-stage callsite')
    sizes = [2 * args.m * args.k, 2 * args.k * args.n, 2 * args.m * args.n]
    # Includes originals, native copies, host verification and CPU output copies.
    estimate = 5 * sum(sizes[:2]) + 4 * sizes[2] + 64 * 1024**2
    if max(sizes) > args.max_storage_bytes or estimate > args.max_working_set_bytes:
        p.error("Requested shape exceeds explicit storage/working-set limits")
    args.storage_bytes, args.working_set_estimate = sizes, estimate
    return args


def run_probe(args, result):
    # Import only after resource checks. The root controller alone launches this.
    import torch
    import torch_npu

    provider = CtypesProvider.attach_preloaded()
    owner = "fault39-operator-probe-" + str(os.getpid())
    generation = None
    device = None
    result.update(owner=owner, request_scope_kind="SINGLE_PROCESS_OPERATOR_PROBE",
                  model_request_mapping="NOT_APPLICABLE_OPERATOR_PROBE", evidence_kind="REAL_NPU_OPERATOR")

    def checked_stop():
        ack = provider.disable(owner, generation, args.drain_timeout_ms)
        if ack["status"] != "DISABLED_CONFIRMED":
            raise RuntimeError("DRAIN_NOT_CONFIRMED: " + ack["status"])
        return ack

    def digest_cpu(tensor):
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        return hashlib.sha256(memoryview(raw)).hexdigest()

    def stats_cpu(tensor):
        flat = tensor.detach().reshape(-1)
        stats = dict.fromkeys(("total", "finite", "zero", "positive", "negative", "nan", "posinf", "neginf"), 0)
        low = high = maximum_abs = None
        sum_abs = 0.0
        # Bound scratch memory independently of the historical 65M-element weight.
        for offset in range(0, flat.numel(), 1024 * 1024):
            values = flat[offset:offset + 1024 * 1024].float()
            finite = torch.isfinite(values)
            selected = values[finite]
            stats["total"] += values.numel()
            stats["finite"] += int(finite.sum().item())
            stats["zero"] += int((selected == 0).sum().item())
            stats["positive"] += int((selected > 0).sum().item())
            stats["negative"] += int((selected < 0).sum().item())
            stats["nan"] += int(torch.isnan(values).sum().item())
            stats["posinf"] += int(torch.isposinf(values).sum().item())
            stats["neginf"] += int(torch.isneginf(values).sum().item())
            if selected.numel():
                chunk_low, chunk_high = float(selected.min().item()), float(selected.max().item())
                chunk_abs = float(selected.abs().max().item())
                low = chunk_low if low is None else min(low, chunk_low)
                high = chunk_high if high is None else max(high, chunk_high)
                maximum_abs = chunk_abs if maximum_abs is None else max(maximum_abs, chunk_abs)
                sum_abs += float(selected.double().abs().sum().item())
        stats.update(min=low, max=high, max_abs=maximum_abs,
                     mean_abs=sum_abs / stats["finite"] if stats["finite"] else None)
        return validate_statistics(stats)

    def exact(a, b):
        return bool(torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)))

    try:
        # There is no distributed model worker here: actual singleton operator rank is zero.
        provider.bind_worker_rank(0)
        torch.set_num_threads(1)
        torch.npu.set_device(args.device)
        device = int(torch.npu.current_device())
        torch.npu.set_compile_mode(jit_compile=False)
        result["versions"] = {"torch": torch.__version__, "torch_npu": getattr(torch_npu, "__version__", "unknown")}
        result["boot_identity"] = {"rank": 0, "source": "singleton_operator_process", "logical_device": device}
        before = provider.prepare()
        result["initial_native"] = before
        if before.get("generation") != 0 or before.get("owner") or before.get("enabled"):
            raise RuntimeError("FRESH_PROCESS_REQUIRED")

        # Integer multiples of 1/16 are exactly representable finite BF16 values.
        indices = torch.arange(args.k, dtype=torch.int64)
        row_a = (((indices % 13) - 6).float() / 16).to(torch.bfloat16)
        row_b = (((indices % 17) - 8).float() / 16).to(torch.bfloat16)
        self_cpu = row_a.expand(args.m, -1).contiguous()
        backing_cpu = row_b.expand(args.n, -1).contiguous()
        self_npu = torch_npu.npu_format_cast(self_cpu.to("npu:" + str(device)), 2)
        backing_npu = torch_npu.npu_format_cast(backing_cpu.to("npu:" + str(device)), 2)
        mat2_npu = backing_npu.transpose(0, 1)
        output = torch_npu.npu_format_cast(torch.empty((args.m, args.n), dtype=torch.bfloat16,
                                                       device="npu:" + str(device)), 2)
        torch.npu.synchronize()
        for tensor, expected in ((self_npu, args.storage_bytes[0]), (backing_npu, args.storage_bytes[1])):
            if tensor.storage_offset() != 0 or not tensor.is_contiguous() or tensor.untyped_storage().nbytes() != expected:
                raise RuntimeError("UNPROVEN_FULL_STORAGE_HASH: backing is not the complete dense ND storage")

        def input_hashes():
            torch.npu.synchronize()
            return {"self_full_storage_sha256": digest_cpu(self_npu.cpu()),
                    "mat2_backing_full_storage_sha256": digest_cpu(backing_npu.cpu())}

        initial_hashes = input_hashes()
        expected_hashes = {"self_full_storage_sha256": digest_cpu(self_cpu),
                           "mat2_backing_full_storage_sha256": digest_cpu(backing_cpu)}
        if initial_hashes != expected_hashes:
            raise RuntimeError("INPUT_TRANSFER_NOT_EXACT")
        result["inputs"] = {"hashes_before": initial_hashes, "hash_scope": "complete_contiguous_backing_storage",
                            "self_shape": list(self_npu.shape), "self_strides": list(self_npu.stride()),
                            "mat2_shape": list(mat2_npu.shape), "mat2_strides": list(mat2_npu.stride()),
                            "self_statistics": stats_cpu(self_cpu), "mat2_statistics": stats_cpu(backing_cpu)}
        del backing_cpu, self_cpu, row_a, row_b, indices

        class MatmulProbe(torch.nn.Module):
            def forward(self, left, right, destination):
                return torch.matmul(left, right, out=destination)
        probe_module = MatmulProbe()
        captured = False

        def execute(case):
            nonlocal captured
            scope = "operator-probe:" + str(os.getpid()) + ":" + case
            debugger = None
            capture_this = bool(args.capture_group and case.startswith('pattern-'))
            if capture_this:
                if captured:
                    raise RuntimeError('CAPTURE_ALREADY_COMPLETED')
                from msprobe.pytorch import PrecisionDebugger
                import msprobe
                capture_dir = args.output.parent / args.capture_group
                capture_dir.mkdir(exist_ok=False)
                config = dict(task='statistics', level='mix', precision='high', rank=[], step=[0],
                    dump_path=str(capture_dir / 'observation'), async_dump=False,
                    statistics=dict(scope=[], list=[], data_mode=['all'], summary_mode='statistics', tensor_list=[]))
                config_path = capture_dir / 'config.json'
                config_path.write_text(json.dumps(config, indent=2))
                package = Path(msprobe.__file__).resolve().parent
                identity = dict(package=str(package), edited_by_fault39=False,
                    files={p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in package.rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.json')})
                (capture_dir / 'collector_identity.json').write_text(json.dumps(identity, indent=2))
                debugger = PrecisionDebugger(config_path=str(config_path))
            addresses_before = [int(t.untyped_storage().data_ptr()) for t in (self_npu, backing_npu, output)]
            with provider.context(owner, [scope]):
                try:
                    if debugger is not None:
                        debugger.start(model=probe_module, rank_id=0)
                    probe_module(self_npu, mat2_npu, output)
                    torch.npu.synchronize()
                finally:
                    if debugger is not None:
                        debugger.stop()
                        debugger.step()
                        captured = True
            addresses_after = [int(t.untyped_storage().data_ptr()) for t in (self_npu, backing_npu, output)]
            cpu = output.cpu().clone()
            return cpu, {"operator_scope_id": scope, "output_sha256": digest_cpu(cpu),
                         "output_statistics": stats_cpu(cpu), "native": provider.snapshot(),
                         "live_storage_addresses_before": addresses_before,
                         "live_storage_addresses_after": addresses_after}

        config = Fault39Config(pattern=args.pattern, amplitude_policy=args.amplitude_policy,
                               ranks=(0,), devices=(device,), max_hits=1, max_bytes=args.max_storage_bytes)
        # The first ACLNN execution constructs the executor; subsequent calls
        # enter OpExecCache::Run. Warm up while admission is still disabled.
        _, result['disabled_warmup'] = execute('disabled-warmup')
        generation = provider.configure(owner, config, 0, device, mode="discover")
        baseline0, first = execute("discover-1")
        baseline, second = execute("discover-2")
        result["discovery"] = {"first": first, "second": second,
                               "baseline_exact_repeat": exact(baseline0, baseline)}
        signature = select_stable_target(second["native"].get("candidates", []), first["native"].get("candidates", []))
        if not exact(baseline0, baseline):
            raise RuntimeError("BASELINE_NOT_BITWISE_STABLE")
        if second["output_statistics"]["finite"] != args.m * args.n:
            raise RuntimeError("BASELINE_NOT_FINITE")
        if second["native"].get("interposed_run_calls", 0) <= before.get("interposed_run_calls", 0) or second["native"].get("typed_metadata_calls", 0) <= before.get("typed_metadata_calls", 0):
            raise RuntimeError("REAL_INTERCEPTION_NOT_PROVEN")
        candidates = second["native"]["candidates"]
        if (candidates[0].get("M"), candidates[0].get("K"), candidates[0].get("N")) != (args.m, args.k, args.n):
            raise RuntimeError("DISCOVERED_DIMENSIONS_MISMATCH")
        result["discovery"]["selected_signature"] = signature
        result["discovery"]["disable"] = checked_stop()

        # P2: inspect an actual Run without replacing either input. The true
        # live torch storage must independently prove all three private slots.
        observer_config = Fault39Config(pattern=args.pattern, amplitude_policy="historical58",
                                       ranks=(0,), devices=(device,), max_hits=1,
                                       max_bytes=args.max_storage_bytes)
        generation = provider.configure(owner, observer_config, 0, device, mode="observer",
                                        target_signature=signature, input_mask=3)
        if args.check_owner_isolation:
            result['owner_isolation'] = verify_active_owner_isolation(
                provider, owner, generation, observer_config, 0, device, signature)
        observed, observer = execute("observer-dynamic-slot-proof")
        result["observer_gate"] = observer
        observer["observer_only"] = True
        observer["buffer_substitution"] = False
        observer["baseline_bitwise_equal"] = exact(baseline, observed)
        observer["disable"] = checked_stop()
        observer["dynamic_slot_proof"] = verify_dynamic_slot_proof(
            observer["native"], observer["live_storage_addresses_before"],
            observer["live_storage_addresses_after"], 3, observer_only=True)
        observation = observer["native"].get("last_observation", {})
        observer["request_scope_exact"] = json.loads(observation.get("actual_request_ids_json", "[]")) == [observer["operator_scope_id"]]
        observer["input_hashes_after"] = input_hashes()
        for name in ("original_self", "original_mat2", "temporary_self", "temporary_mat2", "output"):
            validate_statistics(observation.get(name, {}))
        if (observer["native"].get("mode") != 4 or observer["native"].get("hits") != 1
                or any(observation.get(key) != 0 for key in
                       ("pre_sync_status", "copy_status", "run_status", "post_sync_status", "free_status"))
                or observation.get("output_source") != "post_run_device_readback"
                or not observer["baseline_bitwise_equal"] or not observer["request_scope_exact"]
                or observer["input_hashes_after"] != initial_hashes):
            raise RuntimeError("OBSERVER_DYNAMIC_SLOT_GATE_FAILED")
        observer["status"] = "PASS"
        if args.check_owner_isolation:
            result['owner_isolation'].update(status='PASS', owner_execution_verified=True,
                post_check_run={'generation': observer['native']['generation'],
                                'hits': observer['native']['hits'], 'mode': observer['native']['mode'],
                                'baseline_bitwise_equal': observer['baseline_bitwise_equal'],
                                'request_scope_exact': observer['request_scope_exact'],
                                'input_storage_hashes_unchanged': observer['input_hashes_after'] == initial_hashes})

        def selected_case(mode, mask, label):
            nonlocal generation
            generation = provider.configure(owner, config, 0, device, mode=mode,
                                            target_signature=signature, input_mask=mask)
            actual, evidence = execute(label)
            evidence["baseline_bitwise_equal"] = exact(baseline, actual)
            evidence["inspector"] = Fault39OutputInspector().inspect(evidence["native"])
            evidence["disable"] = checked_stop()
            evidence.update(mode=mode, input_mask=mask, case=label)
            result.setdefault("attempts", []).append(evidence)
            evidence["dynamic_slot_proof"] = verify_dynamic_slot_proof(
                evidence["native"], evidence["live_storage_addresses_before"],
                evidence["live_storage_addresses_after"], mask, observer_only=mode == 'observer')
            # Snapshot before clean preserves the generation-bound native evidence.
            if mode == 'observer':
                observation = evidence['native'].get('last_observation', {})
                if (not evidence['baseline_bitwise_equal'] or any(observation.get(k) != 0 for k in
                        ('pre_sync_status', 'copy_status', 'run_status', 'post_sync_status', 'free_status'))):
                    raise RuntimeError('OBSERVER_BASELINE_FAILED')
            elif not evidence["inspector"]["native_execution_verified"] or not evidence["inspector"]["resources_released"]:
                raise RuntimeError("SELECTED_RUN_NOT_PROVEN: " + label)
            if evidence["native"].get("hits") != 1:
                raise RuntimeError("EXPECTED_EXACTLY_ONE_SELECTED_RUN")
            return evidence

        if args.stage in ("lossless", "pattern"):
            result["lossless"] = []
            masks = (1, 2, 3) if args.stage == "pattern" else (args.input_mask,)
            for mask in masks:
                evidence = selected_case("lossless", mask, "lossless-mask" + str(mask))
                evidence["input_mask"] = mask
                result["lossless"].append(evidence)
                if not evidence["baseline_bitwise_equal"] or input_hashes() != initial_hashes:
                    raise RuntimeError("LOSSLESS_GATE_FAILED: mask=" + str(mask))
        if args.stage == "pattern":
            selected_mode = 'observer' if args.capture_group == 'baseline' else 'inject'
            result["pattern"] = selected_case(selected_mode, 3, "pattern-" + args.pattern)
            if selected_mode == 'inject' and not result["pattern"]["inspector"]["exact_cancellation_verified"]:
                raise RuntimeError("EXACT_CANCELLATION_NOT_PROVEN")

        hits_before_off = provider.snapshot().get("hits")
        off, off_evidence = execute("off-recovery")
        hashes_after = input_hashes()
        result["off_recovery"] = off_evidence
        result["off_recovery"].update(baseline_bitwise_equal=exact(baseline, off),
                                      hits_unchanged=off_evidence["native"].get("hits") == hits_before_off)
        result["inputs"]["hashes_after"] = hashes_after
        result["inputs"]["full_storage_bitwise_unchanged"] = hashes_after == initial_hashes
        if not exact(baseline, off) or hashes_after != initial_hashes or off_evidence["native"].get("hits") != hits_before_off:
            raise RuntimeError("OFF_RECOVERY_FAILED")
        result["status"] = "PASS"
        result["verdict"] = (result["pattern"]["inspector"]["verdict"] if args.stage == "pattern"
                             else "LOSSLESS_REAL_OPERATOR_VERIFIED" if args.stage == "lossless"
                             else "STABLE_UNIQUE_REAL_TARGET_DISCOVERED")
        if args.capture_group == 'baseline':
            result['verdict'] = 'REAL_OPERATOR_OBSERVER_BASELINE'
    finally:
        result['loaded_native_paths'] = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
            if any(name in line for name in ('libopapi', 'libnnopbase', 'libfault39'))})
        if generation is not None:
            try:
                result["final_disable"] = provider.disable(owner, generation, args.drain_timeout_ms)
                result["final_clean"] = provider.clean(owner, generation, args.drain_timeout_ms)
                if result["final_clean"]["status"] != "DISABLED_CONFIRMED":
                    result["status"] = "FAIL"
                    result["cleanup_failure"] = result["final_clean"]["status"]
            except Exception as exc:
                result["status"] = "FAIL"
                result["cleanup_failure"] = type(exc).__name__ + ": " + str(exc)


def main():
    args = parse_args()
    result = {"schema_version": 1, "stage": args.stage, "pattern_name": args.pattern,
              "input_mask": args.input_mask, "amplitude_policy": args.amplitude_policy,
              "shape": {"M": args.m, "K": args.k, "N": args.n}, "tensor_payloads": False,
              "storage_bytes": args.storage_bytes, "working_set_estimate_bytes": args.working_set_estimate,
              "status": "FAIL", "started_unix": time.time()}
    try:
        run_probe(args, result)
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
    result["finished_unix"] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": result["status"], "verdict": result.get("verdict"),
                      "error": result.get("error"), "output": str(args.output)}, allow_nan=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
