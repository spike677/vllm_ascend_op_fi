"""Fail-closed validation of recorded FakeRequant RPC evidence, not summaries.

No arm/query/disarm implementation lives here. Historical measurements below the
transaction budget (short requests followed by padding) are explicitly separated.
"""

from __future__ import annotations

from .contracts import FaultConfigurationRecord


def _integer(value):
    return type(value) is int and value >= 0


def rpc_workers(envelope):
    if not isinstance(envelope, dict) or envelope.get("http_status") != 200:
        return None
    workers = envelope.get("results")
    return workers if isinstance(workers, list) and workers and all(isinstance(w, dict) for w in workers) else None


def cleanup_gate(envelope, service_healthy, world_size):
    workers = rpc_workers(envelope)
    if workers is None or world_size is None or service_healthy is None:
        return {"status": "UNKNOWN", "reason": "missing raw cleanup/health/world-size evidence"}
    okay = (service_healthy is True and len(workers) == world_size
            and len({w.get("pid") for w in workers}) == world_size
            and all(type(w.get("pid")) is int and w.get("enabled") is False
                    and w.get("installed") is False for w in workers))
    return {"status": "PASS" if okay else "FAIL", "reason": "raw worker query after remove + recorded health",
            "pids": [w.get("pid") for w in workers]}


def validate_mechanism(row, config: FaultConfigurationRecord, service_healthy=None):
    transaction = row.get("fault_transaction", {})
    workers = rpc_workers(transaction.get("measurement_query"))
    failures, unknowns, ranks = [], [], []
    if not config.actual_module_names or not config.targeted_layers or not config.targeted_ranks or not config.world_size:
        unknowns.append("frozen target/module/layer/rank configuration incomplete")
    if row.get("http_status") is None or service_healthy is None:
        unknowns.append("missing request/service health")
    elif row["http_status"] != 200 or service_healthy is not True or row.get("error"):
        failures.append("request/service unhealthy")
    if workers is None:
        unknowns.append("missing successful raw measurement_query")
        workers = []
    elif len(workers) != config.world_size:
        failures.append("world size mismatch")
    arm_workers = rpc_workers(transaction.get("arm"))
    if arm_workers is None:
        unknowns.append("missing successful raw arm")
    elif len(arm_workers) != config.world_size or not all(w.get("ok") is True for w in arm_workers):
        failures.append("arm not confirmed on every worker")
    expected_modules = set(config.actual_module_names)
    expected_layers = set(config.targeted_layers)
    seen_ranks, seen_pids = set(), set()
    for worker in workers:
        modules = worker.get("matched_modules")
        if not isinstance(modules, list) or not modules:
            unknowns.append("worker target binding absent")
            continue
        rank_set = {m.get("tp_rank") for m in modules}
        if len(rank_set) != 1 or not all(type(x) is int for x in rank_set):
            failures.append("ambiguous actual rank; never infer from result order")
            continue
        rank = next(iter(rank_set))
        pid = worker.get("pid")
        if rank in seen_ranks or pid in seen_pids or type(pid) is not int:
            failures.append("duplicate/invalid worker identity")
        seen_ranks.add(rank)
        seen_pids.add(pid)
        if (set(m.get("module") for m in modules) != expected_modules
                or set(m.get("layer") for m in modules) != expected_layers
                or any(m.get("tp_size") != config.world_size for m in modules)):
            failures.append(f"rank{rank}: target/module/layer/topology mismatch")
        if worker.get("installed") is not True or worker.get("developer_mode") is not True:
            failures.append(f"rank{rank}: hook/developer-mode not confirmed")
        armed = worker.get("armed_config")
        if not isinstance(armed, dict):
            unknowns.append(f"rank{rank}: raw armed_config missing")
        else:
            frozen = config.extra.get("frozen_arm_config", {})
            for key, value in frozen.items():
                # Generation identifies the current request, not fault strength.
                if key == "request_generation":
                    continue
                actual = armed.get("scale_value" if key == "fixed_scale" else key)
                if key not in armed and not (key == "fixed_scale" and "scale_value" in armed):
                    unknowns.append(f"rank{rank}: armed config missing {key}")
                elif actual != value:
                    failures.append(f"rank{rank}: frozen config changed: {key}")
        hits = worker.get("hit_count")
        history = worker.get("tensor_effect_history")
        reject = worker.get("reject_reasons")
        changed = None
        if not _integer(hits):
            unknowns.append(f"rank{rank}: hit_count missing/invalid")
        elif rank in config.targeted_ranks:
            if hits == 0:
                failures.append(f"rank{rank}: zero actual measurement hits")
            if not isinstance(history, list) or len(history) != hits:
                unknowns.append(f"rank{rank}: incomplete raw tensor_effect_history")
            else:
                changes = [h.get("changed_elements") for h in history]
                if not all(_integer(x) for x in changes):
                    unknowns.append(f"rank{rank}: changed_elements absent")
                else:
                    changed = sum(changes)
                    if changed <= 0:
                        failures.append(f"rank{rank}: no numerical change")
                    if any(not _integer(h.get("total_elements")) for h in history):
                        unknowns.append(f"rank{rank}: total element bounds missing")
                    elif any(h["changed_elements"] > h["total_elements"] for h in history):
                        failures.append(f"rank{rank}: impossible changed-element count")
                if {h.get("hit_index") for h in history} != set(range(1, hits+1)):
                    failures.append(f"rank{rank}: raw hit indices incomplete/duplicate")
                for h in history:
                    bound_layer = {m.get("module"): m.get("layer") for m in modules}.get(h.get("module"))
                    if h.get("rank") != rank or h.get("module") not in expected_modules or h.get("layer") not in expected_layers or h.get("layer") != bound_layer:
                        failures.append(f"rank{rank}: effect target mismatch")
                        break
                for key in ("mode", "scale_provider", "scale_multiplier"):
                    value = config.extra.get("frozen_arm_config", {}).get(key)
                    if value is not None and any(h.get(key) != value for h in history):
                        failures.append(f"rank{rank}: effect strength mismatch: {key}")
            budget = config.hit_policy.get("max_hits")
            if _integer(budget) and hits > budget:
                failures.append(f"rank{rank}: measurement exceeds budget")
        else:
            if hits != 0:
                failures.append(f"rank{rank}: unexpected rank injected")
            if not isinstance(reject, dict) or not _integer(reject.get("rank_filter")):
                unknowns.append(f"rank{rank}: raw rank_filter rejection absent")
            elif reject["rank_filter"] <= 0:
                failures.append(f"rank{rank}: rank_filter not observed")
        ranks.append({"rank": rank, "pid": pid, "hit_count": hits, "changed_elements": changed,
                      "reject_reasons": reject, "worker_responsive": True,
                      "hit_layers": sorted({h["layer"] for h in history or [] if type(h.get("layer")) is int}),
                      "health_basis": "successful post-request RPC response with worker pid"})
    if config.world_size and seen_ranks != set(range(config.world_size)):
        failures.append("all expected ranks not independently identified")
    if arm_workers and {w.get("pid") for w in arm_workers} != seen_pids:
        failures.append("arm/measurement worker identity mismatch")
    # Same request generation on arm and measurement; generation is not list order.
    generations = {w.get("armed_config", {}).get("request_generation") for w in workers}
    arm_generations = {w.get("armed_config", {}).get("request_generation") for w in arm_workers or []}
    if None in generations or not generations or None in arm_generations or not arm_generations:
        unknowns.append("request generation missing")
    elif len(generations) != 1 or generations != arm_generations:
        failures.append("arm/measurement request generation mismatch")
    disarm = transaction.get("transaction_query", {}).get("disarm", row.get("fault_disarm"))
    disarm_workers = rpc_workers(disarm)
    if disarm_workers is None:
        unknowns.append("raw per-request disarm missing")
    elif len(disarm_workers) != config.world_size or not all(w.get("ok") is True and w.get("disarmed") is True for w in disarm_workers):
        failures.append("per-request disarm failed")
    elif {w.get("pid") for w in disarm_workers} != seen_pids:
        failures.append("disarm/measurement worker identity mismatch")
    complete = rpc_workers(transaction.get("transaction_query"))
    budget = config.hit_policy.get("max_hits")
    if complete is None:
        unknowns.append("transaction query missing (padding cannot substitute measurement)")
    elif _integer(budget):
        final_counts = []
        for w in complete:
            binding = w.get("matched_modules") or []
            r = {m.get("tp_rank") for m in binding}
            if len(r) != 1:
                failures.append("transaction rank ambiguous")
                continue
            rank = next(iter(r))
            final_counts.append({"rank": rank, "hit_count": w.get("hit_count")})
            expected = budget if rank in config.targeted_ranks else 0
            if w.get("hit_count") != expected:
                failures.append(f"rank{rank}: complete transaction budget mismatch")
        if {r["rank"] for r in final_counts} != seen_ranks or len(final_counts) != len(seen_ranks):
            failures.append("complete transaction ranks missing/duplicate")
        if {w.get("pid") for w in complete} != seen_pids:
            failures.append("complete transaction worker mismatch")
    status = "INVALID" if failures else "UNKNOWN" if unknowns else "PASS"
    return {"status": status, "classification": "VALID_FAULT_RUN" if status == "PASS" else "INVALID_FAULT_RUN",
            "failures": failures, "unknowns": unknowns, "ranks": ranks,
            "request_generation": next(iter(generations)) if len(generations) == 1 else None,
            "measurement_hit_count": sum(r["hit_count"] for r in ranks if _integer(r["hit_count"])),
            "changed_elements": sum(r["changed_elements"] for r in ranks if r["changed_elements"] is not None) if not unknowns else None,
            "source": "raw fault_transaction.measurement_query; raw tensor_effect_history",
            "transaction_query_budget_required": budget,
            "padding_is_not_scored_response": True}
