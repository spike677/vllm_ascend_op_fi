"""Read-only C0 discovery on the live Worker; never replay, arm or load a DSO."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from .registry import worker_core_manifest

def discover_worker(worker, model, identity):
    from fault_injection.model_discovery import inspect_structure
    from fault_injection.providers import registered_runtimes
    runner = worker.model_runner
    if runner.model is not model:
        raise RuntimeError('DISCOVERY_MODEL_LIFETIME_CHANGED')
    structure = inspect_structure(model, runner=runner)
    if runner.model is not model or structure['object_identity'] != id(model):
        raise RuntimeError('DISCOVERY_MODEL_LIFETIME_CHANGED')
    # Retain every alias, unresolved role and ambiguous candidate. A structural
    # snapshot cannot authorize a target or supply its actual call denominator.
    candidates = [dict(candidate_id=hashlib.sha256(json.dumps(
        [identity, row['object_identity'], row['aliases']], sort_keys=True).encode()).hexdigest(),
        status='STRUCTURAL_UNVERIFIED', structural=row,
        actual_call_observed=False, target_qualification_granted=False) for row in structure['modules']]
    return dict(schema='demoncat.worker_core.discovery.v1', worker=deepcopy(identity),
        model=structure, candidates=candidates, all_structural_candidates_returned=True,
        alias_paths_complete=structure['alias_paths_complete'],
        actual_call_census=dict(status='NOT_EXECUTED', eligible_denominator_complete=False,
            required='Source-bound observation during a naturally scheduled request; provider semantics still apply.'),
        process_registry_names=registered_runtimes(), process_registry_is_model_binding_proof=False,
        capabilities=worker_core_manifest(), read_only=True, forward_calls=0, native_loads=0,
        scope='one current Worker identity and model; snapshot cannot authorize a later call',
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence=dict(c0='STRUCTURE_OBSERVED', c1='NOT_ASSESSED', c2='NOT_ASSESSED', c3='NOT_ASSESSED',
            recovery='NO_RESOURCES_ACQUIRED', e5='NOT_ASSESSED'), qualification_granted=False)
