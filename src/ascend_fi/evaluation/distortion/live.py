"""Thin adapter consumer for an EXISTING fault lifecycle owner.

The adapter owns NPU lock, endpoint authentication, frozen R1 install/arm/query,
per-request try/finally disarm, labelled padding and final service restoration.
This module only schedules fixed probes. It contains no RPC/controller binding.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from .contracts import FaultConfigurationRecord, content_hash
from .mechanism import cleanup_gate


def run_existing_lifecycle(args, suite):
    if args.cycles < 3:
        raise ValueError("live validation requires >=3 FULL cycles")
    config = FaultConfigurationRecord.from_dict(json.loads(Path(args.fault_config).read_text(encoding="utf-8-sig")))
    if not config.extra.get("frozen_arm_config") or not config.config_hash:
        raise ValueError("exact previously validated R1 config and source hash required")
    module, separator, factory_name = args.live_adapter.partition(":")
    if not separator:
        raise ValueError("adapter must be an explicit existing module:factory")
    factory = getattr(importlib.import_module(module), factory_name)
    return collect_cycles(factory(args), suite, config, args.cycles)


def collect_cycles(owner, suite, config, count):
    """Owner is a context manager; never acquire/guess a second NPU lock."""
    cycles = []
    config_before = content_hash(config.to_dict())
    provenance = {"mode": "LIVE_EXISTING_LIFECYCLE", "suite_hash": suite["hash"],
                  "fault_config_hash": config_before, "controller_implemented_here": False}
    with owner as lifecycle:
        preflight = lifecycle.preflight()
        if preflight.get("status") != "PASS" or preflight.get("exclusive_lock_owned") is not True:
            raise RuntimeError("LIVE_VALIDATION_BLOCKED: existing owner did not prove safe resources/lock")
        provenance["preflight"] = preflight
        try:
            for index in range(1, count+1):
                before, before_health = lifecycle.clean_query(), lifecycle.service_healthy()
                if cleanup_gate(before, before_health, config.world_size)["status"] != "PASS":
                    raise RuntimeError("clean baseline precondition failed")
                pairs = {p["probe_id"]: {} for p in suite["probes"]}
                for probe in suite["probes"]:
                    pairs[probe["probe_id"]]["baseline"] = lifecycle.request(probe, "baseline", index)
                # Existing lifecycle must preserve original strength, target/rank
                # policy and request re-arm semantics. It persists raw evidence.
                try:
                    lifecycle.install(config)
                    for probe in suite["probes"]:
                        pairs[probe["probe_id"]]["fault"] = lifecycle.fault_request(probe, config, index)
                finally:
                    lifecycle.disarm_and_remove()
                after, health = lifecycle.clean_query(), lifecycle.service_healthy()
                if cleanup_gate(after, health, config.world_size)["status"] != "PASS":
                    raise RuntimeError("RECOVERY_ENVIRONMENT_BLOCKED")
                for probe in suite["probes"]:
                    pairs[probe["probe_id"]]["recovery"] = lifecycle.request(probe, "recovery", index)
                cycles.append({"cycle_id": f"cycle_{index:02d}", "pairs": pairs,
                               "before_baseline": before, "service_healthy_before": before_health,
                               "after_remove": after, "service_healthy": lifecycle.service_healthy()})
                if content_hash(config.to_dict()) != config_before:
                    raise RuntimeError("frozen configuration mutated by adapter")
        finally:
            lifecycle.disarm_and_remove()
            final = cleanup_gate(lifecycle.clean_query(), lifecycle.service_healthy(), config.world_size)
            provenance["final_cleanup"] = final
            if final["status"] != "PASS":
                raise RuntimeError("RECOVERY_ENVIRONMENT_BLOCKED: final clean state not proven")
    provenance["raw_evidence_refs"] = owner.evidence_refs()
    return suite, config, cycles, {"status": "NOT_REQUESTED"}, provenance
