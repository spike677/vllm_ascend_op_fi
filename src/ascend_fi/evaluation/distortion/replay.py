"""Lossless-source, read-only replay adapter for historical FakeRequant results."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

from .contracts import FaultConfigurationRecord, content_hash
from .mechanism import rpc_workers


def load_live_snapshot(path, frozen_suite):
    """Read completed cycle checkpoints without touching the live collector.

    Partial request directories are never counted as complete cycles. A new
    output directory per snapshot preserves every earlier scorer result.
    """
    root = Path(path)
    suite = json.loads((root / 'suite.json').read_text(encoding='utf-8'))
    if suite != frozen_suite:
        raise ValueError('frozen live suite differs; never retune during replay')
    config = FaultConfigurationRecord.from_dict(json.loads((root / 'fault_configuration.json').read_text(encoding='utf-8')))
    files = sorted((root / 'cycles').glob('cycle_*/cycle.json'))
    if not files:
        raise ValueError('no COMPLETE cycle checkpoint; ongoing requests are not cycles')
    cycles = [json.loads(p.read_text(encoding='utf-8')) for p in files]
    provenance = {'mode': 'READONLY_COMPLETED_CYCLE_SNAPSHOT', 'source_root': str(root.resolve()),
                  'files_read': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
                  'collector_modified': False, 'no_npu_used': True,
                  'interpretation': 'scorer revision differs from frozen collector; compare revised replays, not equality to old scorer labels'}
    return suite, config, cycles, {'status': 'NOT_REQUESTED'}, provenance


class RawSource:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.archive = zipfile.ZipFile(self.path) if self.path.is_file() else None
        self.names = self.archive.namelist() if self.archive else [p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()]
        self.inventory = []
        self.export = json.loads(self.archive.read("export_manifest.json")) if self.archive and "export_manifest.json" in self.names else {}
        self.root = self.export.get("source_root", str(self.path))
        self.expected_hashes = {f["path"]: f.get("sha256") for f in self.export.get("files", [])}

    def close(self):
        if self.archive:
            self.archive.close()

    def read(self, name):
        if name not in self.names:
            self.inventory.append({"path": name, "status": "MISSING"})
            return None
        data = self.archive.read(name) if self.archive else (self.path / name).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if self.expected_hashes.get(name) not in (None, sha):
            raise ValueError(f"raw export checksum mismatch: {name}")
        self.inventory.append({"path": name, "sha256": sha, "bytes": len(data)})
        return data

    def json(self, name):
        data = self.read(name)
        return json.loads(data) if data is not None else {}

    def rows(self, name):
        data = self.read(name)
        rows = {}
        for number, line in enumerate((data or b"").decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pid = row.get("sample_id")
            if not isinstance(pid, str) or pid in rows:
                raise ValueError(f"missing/duplicate sample id: {name}:{number}")
            row["source_ref"] = f"{self.root}/{name}:{number}"
            # The ZIP retains full original JSONL. In-memory projection preserves
            # every effect record used by the gate, not derived gate summaries.
            row.pop("raw_response", None)
            tx = row.get("fault_transaction", {})
            for envelope in (tx.get("measurement_query"), tx.get("transaction_query")):
                for worker in rpc_workers(envelope) or []:
                    if envelope is tx.get("measurement_query") and isinstance(worker.get("tensor_effect_history"), list):
                        keys = ("changed_elements", "total_elements", "hit_index", "module", "layer", "rank", "mode", "scale_provider", "scale_multiplier")
                        worker["tensor_effect_history"] = [{k: h[k] for k in keys if k in h} for h in worker["tensor_effect_history"]]
                    else:
                        worker.pop("tensor_effect_history", None)
            tx.pop("padding_response", None)
            rows[pid] = row
        return rows


def configuration_from_raw(manifest, worker):
    arm = manifest.get("r1_arm_config", {})
    claimed = manifest.get("r1_arm_config_sha256")
    if claimed is not None and claimed != "sha256:" + content_hash(arm):
        raise ValueError("frozen R1 manifest config hash mismatch")
    bindings = worker.get("matched_modules", []) if worker else []
    modules = sorted({m["module"] for m in bindings if isinstance(m.get("module"), str)})
    layers = sorted({m["layer"] for m in bindings if type(m.get("layer")) is int})
    worlds = {m.get("tp_size") for m in bindings}
    world = next(iter(worlds)) if len(worlds) == 1 else None
    family = ["MLP_DOWN"] if modules and all(re.search(r"\.mlp\.down_proj$", m) for m in modules) else []
    return FaultConfigurationRecord.from_dict({
        "capability": "python.fake_requant.activation", "strategy": arm.get("mode"),
        "module_families": family, "actual_module_names": modules, "targeted_layers": layers,
        "available_layers": len(layers) if arm.get("layer_filter") is None and layers else None,
        "targeted_ranks": arm.get("rank_filter", []), "world_size": world,
        "strength": {k: arm[k] for k in ("scale_multiplier", "scale_provider", "eps", "fixed_scale") if k in arm},
        "hit_policy": {"max_hits": arm.get("max_hits"), "interval": arm.get("interval"),
                       "scope": "PER_REQUEST_REARM", "short_request_padding": manifest.get("fault_padding_policy")},
        "persistence": {"window": "one arm transaction, including separately labelled padding"},
        "runtime_hash": manifest.get("runtime_hash"), "config_hash": manifest.get("r1_arm_config_sha256"),
        "model_fingerprint": {"served_model_name": manifest.get("model"), "weights_hash": None},
        "topology": {"source": "raw matched_modules tp_rank/tp_size", "device_ids": None,
                     "available_layers_source": "all matching down_proj bindings; no layer filter"},
        "frozen_arm_config": arm, "raw_config_content_sha256": content_hash(arm),
        "unknown_fields": ["runtime_hash", "weights_hash", "physical_device_ids"],
        "capability_status": worker.get("research_status") if worker else None,
    })


def _legacy_probe(pid, category, row):
    req = row.get("request") or {}
    expected = {}
    detectors = ["garbage", "crash"]
    if category == "gsm8k":
        if row.get("gold_answer") is not None:
            expected = {"answer": row["gold_answer"], "answer_type": "number"}
        detectors.insert(0, "task_answer")
    elif category == "short":
        answer = row.get("expected")
        if isinstance(answer, str):
            if "\n" in answer:
                expected = {"structure": "line_list", "items": answer.splitlines()}
                detectors.insert(0, "structure")
            else:
                expected = {"answer": answer, "answer_type": "number" if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer) else "text"}
                detectors.insert(0, "task_answer")
    elif category == "long":
        detectors = ["repetition", "lexical_collapse", "punctuation", "length", "garbage", "crash"]
    return {"probe_id": pid, "category": category, "prompt": req.get("prompt", ""),
            "generation_config": {k: req[k] for k in ("temperature", "seed", "max_tokens", "stop") if k in req},
            "expected": expected, "applicable_detectors": detectors,
            "definition_source": "existing raw baseline request/expected, not redesigned for new suite"}


def load_legacy(path, frozen_suite):
    source = RawSource(path)
    try:
        manifest = source.json("manifest.json")
        cycle_ids = sorted({n.split("/")[1] for n in source.names if re.match(r"cycles/cycle_[^/]+/", n)})
        if not cycle_ids:
            raise ValueError("no recorded cycles")
        config, probes, cycles, analyses = None, {}, [], []
        for cid in cycle_ids:
            summary = source.json(f"cycles/{cid}/cycle_summary.json")
            pairs, auxiliary = {}, []
            for category in ("gsm8k", "short", "long", "wikitext"):
                phases = {phase: source.rows(f"cycles/{cid}/{phase}/{category}_responses.jsonl")
                          for phase in ("baseline", "fault", "recovery")}
                for pid in sorted(set().union(*(set(v) for v in phases.values()))):
                    if config is None:
                        workers = rpc_workers(phases["fault"].get(pid, {}).get("fault_transaction", {}).get("measurement_query"))
                        if workers:
                            config = configuration_from_raw(manifest, workers[0])
                    if category == "wikitext":
                        if pid in phases["fault"]:
                            auxiliary.append(phases["fault"][pid])
                        continue
                    if pid in pairs:
                        raise ValueError(f"duplicate probe across categories: {pid}")
                    pair = {phase: rows.get(pid, {}) for phase, rows in phases.items()}
                    pairs[pid] = pair
                    definition = _legacy_probe(pid, category, pair["baseline"])
                    if pid not in probes:
                        probes[pid] = definition
                    elif probes[pid] != definition:
                        raise ValueError(f"historical prompt/expected drift between cycles: {pid}")
            clean = summary.get("clean_precondition")
            baseline_http = [p["baseline"].get("http_status") for p in pairs.values()]
            cycles.append({"cycle_id": cid, "pairs": pairs, "auxiliary_fault_requests": auxiliary,
                           "before_baseline": clean, "after_remove": summary.get("after_remove"),
                           "service_healthy": summary.get("service_health"),
                           "service_healthy_before": (all(x == 200 for x in baseline_http) if baseline_http and all(x is not None for x in baseline_http) else None),
                           "pre_health_source": "recorded baseline HTTP responses and raw clean query"})
            analyses.append({"cycle_id": cid, "source": f"{source.root}/cycles/{cid}/cycle_summary.json",
                             "probability": summary.get("probability"),
                             "gsm8k": summary.get("gsm8k"), "classification": "LEGACY_ANALYSIS_NOT_DISTORTION_GATE"})
        suite = {"version": "LEGACY_FAKE_REQUANT_EVAL_SET", "probe_count": len(probes),
                 "probes": list(probes.values()), "thresholds": frozen_suite["thresholds"],
                 "robust_gate": frozen_suite["robust_gate"], "detector_policy_source_hash": frozen_suite["hash"]}
        suite["hash"] = content_hash(suite)
        provenance = {"mode": "OFFLINE_READ_ONLY", "source_root": source.root, "local_input": str(source.path),
                      "export_manifest": source.export, "files_read": source.inventory,
                      "probe_suite": suite["version"], "frozen_detector_policy_hash": frozen_suite["hash"],
                      "original_evidence_modified": False, "no_npu_used": True,
                      "evidence_projection": "complete original files retained in raw ZIP; every measurement tensor effect revalidated",
                      "missing_evidence_policy": "UNKNOWN; never inferred from old summary"}
        provenance["historical_final_cleanup"] = source.json("final_cleanup.json")
        # Full cleanup queries may retain tensor histories from prior requests.
        # Retain only state/identity, never treat those retained hits as active.
        for section in ("query", "remove", "disarm"):
            for worker in rpc_workers(provenance["historical_final_cleanup"].get(section)) or []:
                worker.pop("tensor_effect_history", None)
                worker.pop("last_tensor_effect", None)
        return suite, config or configuration_from_raw(manifest, {}), cycles, {"cycles": analyses, "legacy_thresholds": manifest.get("long_thresholds")}, provenance
    finally:
        source.close()
