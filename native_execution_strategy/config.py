"""Strict task configuration. Optional YAML uses PyYAML's safe loader only."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .models import BuildFingerprints, integer, json_value, number, require


def fields(value: Any, allowed: set[str], required: set[str], label: str) -> dict:
    require(type(value) is dict, f'{label} must be an object')
    require(not (set(value) - allowed), f'{label} has unknown keys: {sorted(set(value) - allowed)}')
    require(required <= set(value), f'{label} missing keys: {sorted(required - set(value))}')
    return value


def string_list(value: Any, label: str) -> tuple[str, ...]:
    require(type(value) in (list, tuple) and all(isinstance(v, str) and v for v in value), f'{label} must be a string list')
    require(len(set(value)) == len(value), f'{label} contains duplicates')
    return tuple(value)


@dataclass(frozen=True)
class Predicate:
    field: str
    op: str
    value: Any

    def __post_init__(self) -> None:
        valid = self.field in ('operator', 'request_count', 'token_count', 'dtype', 'layout', 'seq_lens',
                                'seq_lens.min', 'seq_lens.max', 'seq_lens.sum', 'seq_lens.count')
        valid |= isinstance(self.field, str) and re.fullmatch(r'(matrix_dims|attributes)\.[A-Za-z_][A-Za-z0-9_]*', self.field) is not None
        require(valid, f'Unsupported semantic context field: {self.field}')
        require(self.op in ('eq', 'ne', 'lt', 'lte', 'gt', 'gte', 'in', 'not_in', 'exists'), 'Unsupported predicate operator')
        object.__setattr__(self, 'value', json_value(self.value, 'predicate.value'))
        if self.op == 'exists':
            require(type(self.value) is bool, 'exists value must be bool')
        if self.op in ('in', 'not_in'):
            require(type(self.value) is list and bool(self.value), 'membership predicate requires a nonempty list')
        if self.op in ('lt', 'lte', 'gt', 'gte'):
            require(type(self.value) in (int, float), 'Ordering predicate requires a finite numeric value')


def predicates(value: Any) -> tuple[Predicate, ...]:
    require(type(value) in (list, tuple), 'predicates must be a list')
    result = []
    for raw in value:
        x = fields(raw, {'field', 'op', 'value'}, {'field', 'op', 'value'}, 'predicate')
        result.append(Predicate(**x))
    return tuple(result)


@dataclass(frozen=True)
class AdaptiveRule:
    predicates: tuple[Predicate, ...]
    preferred_strategy_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require(all(isinstance(v, Predicate) for v in self.predicates), 'Invalid adaptive predicates')
        object.__setattr__(self, 'predicates', tuple(self.predicates))
        object.__setattr__(self, 'preferred_strategy_ids', string_list(self.preferred_strategy_ids, 'preferred_strategy_ids'))
        require(bool(self.preferred_strategy_ids), 'Adaptive rule needs a nonempty preference list')


@dataclass(frozen=True)
class PolicyConfig:
    mode: str = 'every_match'
    selection: str = 'legal_alternative'
    predicates: tuple[Predicate, ...] = ()
    preferred_strategy_ids: tuple[str, ...] = ()
    adaptive_rules: tuple[AdaptiveRule, ...] = ()
    enabled: bool = True
    target: str | None = None

    def __post_init__(self) -> None:
        require(self.mode in ('every_match', 'adaptive'), 'mode must be every_match or adaptive')
        require(self.selection == 'legal_alternative', 'Only adapter-certified legal_alternative selection is supported')
        require(type(self.enabled) is bool, 'enabled must be bool')
        require(self.target is None or (isinstance(self.target, str) and bool(self.target)), 'target must be a nonempty adapter target or None')
        require(all(isinstance(v, Predicate) for v in self.predicates), 'Invalid policy predicates')
        require(all(isinstance(v, AdaptiveRule) for v in self.adaptive_rules), 'Invalid adaptive rules')
        require(self.mode == 'adaptive' or not self.adaptive_rules, 'adaptive_rules require adaptive mode')
        object.__setattr__(self, 'predicates', tuple(self.predicates))
        object.__setattr__(self, 'adaptive_rules', tuple(self.adaptive_rules))
        object.__setattr__(self, 'preferred_strategy_ids', string_list(self.preferred_strategy_ids, 'preferred_strategy_ids'))


@dataclass(frozen=True)
class DiscoveryBudget:
    max_cases: int = 32
    max_elapsed_seconds: float = 300.0
    max_device_seconds: float | None = None
    max_memory_bytes: int | None = None
    workload_axis: str = 'request_count'

    def __post_init__(self) -> None:
        integer(self.max_cases, 'max_cases', minimum=1)
        Predicate(self.workload_axis, 'exists', True)
        number(self.max_elapsed_seconds, 'max_elapsed_seconds', minimum=0.000001)
        if self.max_device_seconds is not None:
            number(self.max_device_seconds, 'max_device_seconds', minimum=0.000001)
        if self.max_memory_bytes is not None:
            integer(self.max_memory_bytes, 'max_memory_bytes', minimum=1)


@dataclass(frozen=True)
class AdapterSelector:
    name: str
    version: str = 'validated'
    expected_fingerprints: BuildFingerprints | None = None
    route: str | None = None
    on_unsupported: str = 'reject'

    def __post_init__(self) -> None:
        require(isinstance(self.name, str) and bool(self.name), 'Adapter name is required')
        require(isinstance(self.version, str) and bool(self.version), 'Adapter version is required')
        require(self.expected_fingerprints is None or isinstance(self.expected_fingerprints, BuildFingerprints), 'Invalid expected fingerprints')
        require(self.route is None or (isinstance(self.route, str) and bool(self.route)), 'route must be a nonempty string or None')
        require(self.on_unsupported in ('reject', 'passthrough'), 'on_unsupported must be reject or passthrough')

    @property
    def operator(self) -> str:
        return self.name


@dataclass(frozen=True)
class ObservationConfig:
    verify_native_launch: bool = True
    compare_complete_target: bool = True
    msprobe_task: str = 'statistics'
    msprobe_level: str = 'mix'
    msprobe_summary_mode: str = 'statistics'

    def __post_init__(self) -> None:
        require(type(self.verify_native_launch) is bool and type(self.compare_complete_target) is bool,
                'Observation flags must be bool')
        require(self.msprobe_task == 'statistics' and self.msprobe_level == 'mix' and self.msprobe_summary_mode == 'statistics',
                'This interface supports statistics/mix with statistics summaries only')

    def as_dict(self) -> dict:
        return {'verify_native_launch': self.verify_native_launch, 'compare_complete_target': self.compare_complete_target,
                'msprobe': {'task': self.msprobe_task, 'level': self.msprobe_level, 'summary_mode': self.msprobe_summary_mode}}


@dataclass(frozen=True)
class ExecutionConfig:
    adapter: AdapterSelector
    policy: PolicyConfig
    discovery: DiscoveryBudget
    schema_version: int = 1
    mechanism: str = 'native_execution_strategy'
    observation: ObservationConfig = ObservationConfig()

    def __post_init__(self) -> None:
        require(type(self.schema_version) is int and self.schema_version == 1, 'Unsupported schema_version')
        require(self.mechanism == 'native_execution_strategy', 'Unsupported mechanism')
        require(isinstance(self.adapter, AdapterSelector) and isinstance(self.policy, PolicyConfig)
                and isinstance(self.discovery, DiscoveryBudget) and isinstance(self.observation, ObservationConfig),
                'ExecutionConfig requires typed validated components')

    @property
    def selection_target(self) -> str | None:
        return self.policy.target


def from_dict(raw: Mapping[str, Any]) -> ExecutionConfig:
    x = fields(json_value(raw), {'schema_version', 'mechanism', 'adapter', 'policy', 'selection', 'trigger', 'discovery', 'observation'}, {'adapter'}, 'config')
    require(type(x.get('schema_version', 1)) is int and x.get('schema_version', 1) == 1, 'Unsupported schema_version')
    require(x.get('mechanism', 'native_execution_strategy') == 'native_execution_strategy', 'Unsupported mechanism')
    a = fields(x['adapter'], {'name', 'operator', 'version', 'route', 'expected_fingerprints', 'on_unsupported'}, set(), 'adapter')
    a = dict(a)
    require(('name' in a) != ('operator' in a), 'adapter requires exactly one of operator or legacy name')
    if 'operator' in a:
        a['name'] = a.pop('operator')
    if 'expected_fingerprints' in a:
        f = fields(a['expected_fingerprints'], {'source_sha256', 'build_sha256', 'certificate_sha256'},
                   {'source_sha256', 'build_sha256'}, 'expected_fingerprints')
        a['expected_fingerprints'] = BuildFingerprints(**f)
    require('policy' not in x or not ({'selection', 'trigger'} & set(x)), 'Do not mix legacy policy with selection/trigger')
    if 'selection' in x or 'trigger' in x:
        s = fields(x.get('selection', {}), {'target', 'policy', 'preferred_strategy_ids', 'adaptive_rules'}, {'target'}, 'selection')
        t = fields(x.get('trigger', {}), {'mode', 'condition', 'predicates', 'enabled'}, set(), 'trigger')
        require(t.get('mode', 'every_match') == 'every_match', 'Public trigger mode must be every_match')
        require(t.get('condition', 'adaptive') in ('adaptive', 'always'), 'trigger condition must be adaptive or always')
        p = {'mode': 'adaptive' if t.get('condition', 'adaptive') == 'adaptive' else 'every_match',
             'selection': s.get('policy', 'legal_alternative'), 'target': s['target'],
             'predicates': t.get('predicates', []), 'enabled': t.get('enabled', True),
             'preferred_strategy_ids': s.get('preferred_strategy_ids', []), 'adaptive_rules': s.get('adaptive_rules', [])}
    else:
        p = fields(x.get('policy', {}), {'mode', 'selection', 'predicates', 'preferred_strategy_ids', 'adaptive_rules', 'enabled', 'target'}, set(), 'policy')
    p = dict(p)
    p['predicates'] = predicates(p.get('predicates', []))
    rules = p.get('adaptive_rules', [])
    require(type(rules) is list, 'adaptive_rules must be a list')
    p['adaptive_rules'] = tuple(AdaptiveRule(predicates(r['predicates']), string_list(r['preferred_strategy_ids'], 'preferred_strategy_ids'))
                                for raw_rule in rules
                                for r in [fields(raw_rule, {'predicates', 'preferred_strategy_ids'}, {'predicates', 'preferred_strategy_ids'}, 'adaptive rule')])
    d = fields(x.get('discovery', {}), {'max_cases', 'workload_axis', 'resource_budget', 'max_elapsed_seconds', 'max_device_seconds', 'max_memory_bytes'}, set(), 'discovery')
    d = dict(d)
    if 'resource_budget' in d:
        require(not ({'max_elapsed_seconds', 'max_device_seconds', 'max_memory_bytes'} & set(d)), 'Do not mix nested and flat discovery budgets')
        d.update(fields(d.pop('resource_budget'), {'max_elapsed_seconds', 'max_device_seconds', 'max_memory_bytes'}, set(), 'resource_budget'))
    o = dict(fields(x.get('observation', {}), {'verify_native_launch', 'compare_complete_target', 'msprobe'}, set(), 'observation'))
    m = fields(o.pop('msprobe', {}), {'task', 'level', 'summary_mode'}, set(), 'msprobe')
    o.update({'msprobe_' + k: v for k, v in m.items()})
    return ExecutionConfig(AdapterSelector(**a), PolicyConfig(**p), DiscoveryBudget(**d), observation=ObservationConfig(**o))


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate configuration key: {key}')
        result[key] = value
    return result


def loads(text: str, *, format: str = 'json') -> ExecutionConfig:
    require(len(text.encode('utf-8')) <= 1048576, 'Configuration exceeds 1 MiB')
    if format == 'json':
        def reject_number(value):
            raise ValueError(f'Nonfinite JSON number: {value}')
        data = json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=reject_number)
    elif format in ('yaml', 'yml'):
        try:
            import yaml
        except ImportError as exc:
            raise ValueError('YAML is optional; install PyYAML or provide JSON') from exc
        class StrictSafeLoader(yaml.SafeLoader):
            def compose_node(self, parent, index):
                if self.check_event(yaml.AliasEvent):
                    raise ValueError('YAML aliases are not supported in execution configuration')
                return super().compose_node(parent, index)
        def construct_mapping(loader, node):
            return _unique_pairs((loader.construct_object(k, deep=True), loader.construct_object(v, deep=True)) for k, v in node.value)
        StrictSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
        data = yaml.load(text, Loader=StrictSafeLoader)
    else:
        raise ValueError('Configuration format must be json or yaml')
    return from_dict(data)


def load_config(path: str | Path) -> ExecutionConfig:
    p = Path(path)
    require(p.stat().st_size <= 1048576, 'Configuration exceeds 1 MiB')
    suffix = p.suffix.lower()
    require(suffix in ('.json', '.yaml', '.yml'), 'Configuration extension must be .json/.yaml/.yml')
    return loads(p.read_text(encoding='utf-8'), format=suffix[1:])
