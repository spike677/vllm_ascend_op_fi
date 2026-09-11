"""Bounded, append-only numerical evidence; no tensor mutation or fault policy."""
from collections import deque
import copy
import math

SCHEMA = 'BOUNDED_EFFECT_SUMMARY_V1'


class EffectStats:
    """List-compatible sink for the reference owner's existing append/clear calls.

    Every accepted event contributes to exact counters. Only first/last events
    remain as individual records. Additional zero/L2 diagnostics are sampled;
    their denominator is separate from the full population.
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self.count = 0
        self.per_module = {}
        self.first = []
        self.last = deque(maxlen=16)
        self.errors = []

    def __iter__(self):
        seen = set()
        for event in self.first + list(self.last):
            if event['hit_index'] not in seen:
                seen.add(event['hit_index'])
                yield copy.deepcopy(event)

    def append(self, event):
        self.count += 1
        if event['hit_index'] != self.count and len(self.errors) < 16:
            self.errors.append('NONCONTIGUOUS_HIT_INDEX')
        if len(self.first) < 16:
            self.first.append(dict(event))
        self.last.append(dict(event))
        m = self.per_module.setdefault(event['module'], {
            **{k: event[k] for k in ('module', 'layer', 'rank', 'mode', 'scale_provider', 'scale_multiplier')},
            'hit_count': 0, 'changed_calls': 0, 'changed_elements': 0, 'total_elements': 0,
            'clip_count': 0, 'abs_error_sum': 0., 'max_abs_error': 0.,
            'fault_scale_min': None, 'fault_scale_max': None, 'fault_scale_element_sum': 0.,
            'sampled_calls': 0, 'sampled_elements': 0, 'sampled_nonzero_elements': 0,
            'sampled_new_zero_elements': 0, 'sampled_input_sq_sum': 0., 'sampled_error_sq_sum': 0.,
            'first_hit_index': event['hit_index'], 'first_wall_time_ns': event.get('wall_time_ns'),
        })
        for k in ('layer', 'rank', 'mode', 'scale_provider', 'scale_multiplier'):
            if m[k] != event[k] and len(self.errors) < 16:
                self.errors.append('MIXED_MODULE_IDENTITY:' + k)
        m['hit_count'] += 1
        m['changed_calls'] += event['changed_elements'] > 0
        for k in ('changed_elements', 'total_elements', 'clip_count'):
            m[k] += event[k]
        m['abs_error_sum'] += event['mean_abs_error'] * event['total_elements']
        m['max_abs_error'] = max(m['max_abs_error'], event['max_abs_error'])
        if 'scale_min' in event:
            m['fault_scale_min'] = min(m['fault_scale_min'] if m['fault_scale_min'] is not None else math.inf, event['scale_min'])
            m['fault_scale_max'] = max(m['fault_scale_max'] if m['fault_scale_max'] is not None else -math.inf, event['scale_max'])
            m['fault_scale_element_sum'] += event['scale_mean'] * event['total_elements']
        sample = event.get('dose_sample')
        if sample:
            m['sampled_calls'] += 1
            for src, dst in [('elements', 'sampled_elements'), ('nonzero', 'sampled_nonzero_elements'),
                             ('new_zero', 'sampled_new_zero_elements'), ('input_sq', 'sampled_input_sq_sum'),
                             ('error_sq', 'sampled_error_sq_sum')]:
                m[dst] += sample[src]
        m['last_hit_index'] = event['hit_index']
        m['last_wall_time_ns'] = event.get('wall_time_ns')

    def snapshot(self):
        return {'schema': SCHEMA, 'event_count': self.count, 'contiguous_hit_indices': not self.errors,
                'errors': list(self.errors), 'per_module': copy.deepcopy(self.per_module),
                'individual_events_retained': len(list(self)), 'individual_event_limit': 32,
                'aggregate_population': 'ALL_ACCEPTED_ACTIVATION_CALLS',
                'dose_sampling': 'first and every 64th call per module; not full population',
                'phase': 'UNKNOWN', 'generation_token_positions': 'UNKNOWN'}


def validate_stats(worker, expected_modules, expected_layers, rank, frozen, *, identity=False):
    """Verify the new evidence explicitly, never fabricate a per-hit history."""
    s = worker.get('effect_summary') or {}
    failures = []
    if s.get('schema') != SCHEMA or s.get('event_count') != worker.get('hit_count') or s.get('contiguous_hit_indices') is not True or s.get('errors'):
        failures.append('aggregate event continuity/count mismatch')
    modules = s.get('per_module', {})
    if not isinstance(modules, dict) or set(modules) != set(expected_modules):
        failures.append('aggregate module coverage mismatch')
        modules = modules if isinstance(modules, dict) else {}
    total_hits = total_changed = total_elements = 0
    bound = {m['module']: m['layer'] for m in worker.get('matched_modules', [])}
    for name, m in modules.items():
        if (m.get('module') != name or m.get('rank') != rank or m.get('layer') not in expected_layers
                or m.get('layer') != bound.get(name)):
            failures.append('aggregate target mismatch')
        for k in ('mode', 'scale_provider', 'scale_multiplier'):
            value = 'IDENTITY' if identity and k == 'mode' else frozen.get(k)
            if value is not None and m.get(k) != value:
                failures.append('aggregate strength mismatch:' + k)
        ints = [m.get(k) for k in ('hit_count', 'changed_calls', 'changed_elements', 'total_elements', 'clip_count')]
        if not all(type(v) is int and v >= 0 for v in ints):
            failures.append('invalid aggregate integers'); continue
        hits, changed_calls, changed, elements, clips = ints
        if not hits or not elements or changed > elements or clips > elements or changed_calls > hits:
            failures.append('aggregate count bounds')
        runtime = worker.get('per_module', {}).get(name, {})
        if runtime.get('hit_count') != hits:
            failures.append('aggregate/runtime module count mismatch')
        total_hits += hits; total_changed += changed; total_elements += elements
    if total_hits != worker.get('hit_count'):
        failures.append('aggregate module sum mismatch')
    if identity and total_changed != 0:
        failures.append('identity changed elements')
    if not identity and total_changed <= 0:
        failures.append('no numerical change')
    return {'status': 'PASS' if not failures else 'INVALID', 'failures': failures,
            'changed_elements': total_changed, 'total_elements': total_elements,
            'hit_count': total_hits, 'hit_layers': sorted({m.get('layer') for m in modules.values() if type(m.get('layer')) is int})}
