"""Append-only experiment evidence and separate execution/analysis outcomes.

These files are audit receipts, not admission certificates. Runtime qualification
remains with each Provider. Raw statistics stay at their original location.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


PHASES = ('baseline', 'lossless', 'fault', 'off')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', value):
        raise ValueError('INVALID_EVIDENCE_NAME')
    return value


class EvidenceManager:
    """Create a fresh attempt, binding every receipt to one actual run scope."""
    def __init__(self, root, scope):
        from .catalog import resolve
        resolve(scope['mechanism'])
        for key in ('provider_version', 'model_instance_id', 'worker_start_identity',
                    'process_start_identity', 'execution_boundary', 'operator', 'owner'):
            if not isinstance(scope.get(key), str) or not scope[key]:
                raise ValueError('MISSING_SCOPE_' + key)
        for key in ('pid', 'rank', 'logical_device', 'generation'):
            if type(scope.get(key)) is not int or scope[key] < (1 if key in ('pid', 'generation') else 0):
                raise ValueError('INVALID_SCOPE_' + key)
        for key in ('source_sha256', 'environment_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', scope.get(key, '')):
                raise ValueError('MISSING_SCOPE_' + key)
        if scope['execution_boundary'] != 'python_hook':
            if not re.fullmatch('[0-9a-f]{64}', scope.get('native_build_sha256', '')):
                raise ValueError('MISSING_NATIVE_BUILD_IDENTITY')
        self.scope = json.loads(json.dumps(scope, allow_nan=False))
        self.scope_sha256 = canonical_sha(self.scope)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self._closed = False
        self.write('scope', self.scope)

    def write(self, name, payload):
        if self._closed:
            raise ValueError('EVIDENCE_ALREADY_FINALIZED')
        target = self.root / (_name(name) + '.json')
        record = dict(schema='demoncat.evidence.v1', scope_sha256=self.scope_sha256,
                      recorded_at=datetime.now(timezone.utc).isoformat(), payload=payload)
        encoded = (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
        with target.open('xb') as stream:
            stream.write(encoded)
        return target

    def reference(self, name, path):
        """Hash original bytes without rewriting them or copying tensor payloads."""
        path = Path(path).resolve(strict=True)
        return self.write(name, dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path)))

    def statistics(self, phase, step, rank, directory, *, config, collector):
        if phase not in PHASES or any(type(v) is not int or v < 0 for v in (step, rank)):
            raise ValueError('INVALID_STATISTICS_PHASE_OR_STEP')
        paths = {name: Path(directory) / name for name in ('dump.json', 'stack.json', 'construct.json')}
        paths.update(config=Path(config), collector=Path(collector))
        records = {name: dict(path=str(path.resolve(strict=True)), bytes=path.stat().st_size,
                              sha256=digest(path)) for name, path in paths.items()}
        return self.write(f'{phase}.step{step}.rank{rank}.statistics',
                          dict(phase=phase, step=step, rank=rank, files=records))

    def finalize(self):
        manifest = {path.name: digest(path) for path in sorted(self.root.glob('*.json'))}
        path = self.write('checksums', manifest)
        self._closed = True
        return path


def _noncomparable(records):
    nulls, nonfinite = [], []
    for record in records:
        for name, value in record['statistics'].items():
            item = dict(identity=record['identity'], field=name)
            if value is None:
                nulls.append(item)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value):
                    nonfinite.append(dict(item, kind='NaN' if math.isnan(value) else 'Inf'))
            elif isinstance(value, str) and value.lower().strip() in ('nan', '+nan', '-nan', 'inf', '+inf', '-inf'):
                nonfinite.append(dict(item, kind='NaN' if 'nan' in value.lower() else 'Inf'))
    return dict(null_fields=nulls, nonfinite_fields=nonfinite)


def independent_statistics_pair(*, comparator, baseline, fault, baseline_config, fault_config,
                                baseline_collector, fault_collector):
    """Only statistics and collector metadata enter this independent stage.

    First difference is computed before execution receipts are associated. Fixed
    B-reference tolerances and the existing raw-schema comparator are reused.
    Null/nonfinite inventories cover the entire dump, including after the first
    reported difference. A failed self-compare never turns into a clean result.
    """
    comparator = Path(comparator).resolve(strict=True)
    spec = importlib.util.spec_from_file_location('_dcat_raw_statistics', comparator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module.ComparisonPolicy(1e-8, 1e-5)
    pair = module.compare(baseline, fault, policy, baseline_config=baseline_config,
                          fault_config=fault_config, baseline_collector=baseline_collector,
                          fault_collector=fault_collector)
    self_pair = module.compare(baseline, baseline, policy, baseline_config=baseline_config,
                               fault_config=baseline_config, baseline_collector=baseline_collector,
                               fault_collector=baseline_collector)
    inventory = {name: _noncomparable(module.load_raw(path)['records'])
                 for name, path in (('baseline', baseline), ('fault', fault))}
    self_status = self_pair.get('comparison', {}).get('status')
    self_ok = self_status in ('NO_STATISTICS_DIFFERENCE', 'NO_OBSERVABLE_STATISTICS_DIFFERENCE')
    status = pair.get('comparison', {}).get('status', pair.get('status', 'UNAVAILABLE'))
    comparable_statuses = ('FIRST_STATISTICS_DIFFERENCE', 'FIRST_OBSERVABLE_STATISTICS_DIFFERENCE',
                           'NO_STATISTICS_DIFFERENCE', 'NO_OBSERVABLE_STATISTICS_DIFFERENCE')
    nan_fields = [dict(group=group, **item) for group, row in inventory.items()
                  for item in row['nonfinite_fields'] if item['kind'] == 'NaN']
    return dict(schema='demoncat.independent_statistics.v1', rtol=1e-5, atol=1e-8,
                reference='baseline', native_logs_read=False, private_tensors_read=False,
                comparator_sha256=digest(comparator), pair=pair, baseline_self=self_pair,
                full_dump_inventory=inventory, baseline_self_comparable=self_ok,
                first_difference_status=status,
                whole_dump_comparable=self_ok and not nan_fields and status in comparable_statuses)


def assess_execution(record):
    """Classify saved observations; no Native or model admission is granted.

    Values remain independent, so a valid restoration or unchanged top-1 cannot
    hide an unobserved effect or a statistics alignment failure.
    """
    genuine = record.get('execution_kind') == 'npu'
    stages = record.get('observations', {})
    calls = stages.get('original_calls')
    e1 = genuine and type(calls) is int and calls > 0 and stages.get('original_completed') is True
    hits = stages.get('actual_interventions')
    e2 = e1 and type(hits) is int and hits > 0 and stages.get('mechanism_verified') is True
    e3 = e1 and record.get('pretrained_model') is True and stages.get('complete_model_output_compared') is True
    e4 = all(stages.get(k) is True for k in
             ('off_no_new_hits', 'drained', 'device_completed', 'state_restored', 'leases_released', 'off_exact'))
    e4 = e1 and e4
    analysis = record.get('independent_statistics', {})
    e5 = (analysis.get('schema') == 'demoncat.independent_statistics.v1'
          and analysis.get('native_logs_read') is False and analysis.get('private_tensors_read') is False
          and analysis.get('rtol') == 1e-5 and analysis.get('atol') == 1e-8
          and analysis.get('reference') == 'baseline' and analysis.get('whole_dump_comparable') is True)
    return dict(E1_execution=e1, E2_intervention=e2, E3_model_comparison=e3, E4_recovery=e4,
                E5_independent_statistics=e5, fault_effect=stages.get('fault_effect', 'UNOBSERVED'),
                generalized=False, all_eligible=False, every_match=False,
                grants_runtime_admission=False)
