"""Bounded display from existing CPU readbacks; no dump or device calls.

Arithmetic differences do not prove that two values belong to a matched
request/phase/semantic position. The owning adapter establishes that link.
"""
import math
from copy import deepcopy
import hashlib
import json


def current_completed_row(latest_row, *, pending=False, pending_matches_latest=False):
    """Select display evidence only; pending work cannot borrow an older row.

    The caller compares actual request/phase identities. Some serving paths
    publish a completed row before explicit collection, so they may prove that
    the pending ticket already matches it. This selector executes no work and
    grants no identity, effect or recovery qualification.
    """
    if type(pending) is not bool or type(pending_matches_latest) is not bool:
        raise ValueError('Explicit completed/pending request match flags are required')
    return None if pending and not pending_matches_latest else latest_row


def _scalar(value):
    value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else '+Inf' if value > 0 else '-Inf'
    return value


def numeric_change(before, after, *, sample_limit=4):
    """Compare complete already-read values; return at most eight samples.

    No hidden cpu(), synchronization or tensor persistence. NaN pairs count
    as the same nonfinite class; signed infinities remain distinct. Finite
    subtraction overflow is explicit rather than published as NaN/Infinity.
    """
    import torch
    if (not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor)
            or before.device.type != 'cpu' or after.device.type != 'cpu'
            or before.shape != after.shape or before.dtype != after.dtype
            or before.layout != torch.strided or after.layout != torch.strided
            or before.is_complex() or before.is_quantized):
        raise ValueError('Matched real strided CPU readbacks with identical shape/dtype are required')
    if type(sample_limit) is not int or not 0 <= sample_limit <= 8:
        raise ValueError('sample_limit must be an integer from 0 to 8')
    a, b = before.detach().reshape(-1), after.detach().reshape(-1)
    result = dict(schema='demoncat.effect.numeric_change.v1', elements=a.numel(),
                  shape=list(before.shape), dtype=str(before.dtype), changed_elements=0,
                  finite_pair_elements=0, finite_difference_overflow=0,
                  max_abs_diff_finite=None, finite_to_nan=0, finite_to_positive_inf=0,
                  finite_to_negative_inf=0, nonfinite_to_finite=0,
                  before_nonfinite=0, after_nonfinite=0, samples=[],
                  causal_attribution='REQUIRES_ADAPTER_REQUEST_PHASE_POSITION_BINDING',
                  qualification_granted=False)
    # Integer differences are counted exactly; a floating maximum for int64
    # would lose units above2**53, so use exact Python integers for that case.
    for offset in range(0, a.numel(), 65536):
        x, y = a[offset:offset+65536], b[offset:offset+65536]
        fx, fy = torch.isfinite(x), torch.isfinite(y)
        same = torch.eq(x, y) | (torch.isnan(x) & torch.isnan(y))
        changed = ~same
        result['changed_elements'] += int(changed.sum())
        result['before_nonfinite'] += int((~fx).sum())
        result['after_nonfinite'] += int((~fy).sum())
        result['finite_to_nan'] += int((fx & torch.isnan(y)).sum())
        result['finite_to_positive_inf'] += int((fx & torch.isposinf(y)).sum())
        result['finite_to_negative_inf'] += int((fx & torch.isneginf(y)).sum())
        result['nonfinite_to_finite'] += int((~fx & fy).sum())
        finite = fx & fy
        result['finite_pair_elements'] += int(finite.sum())
        maximum = None
        if bool(finite.any()):
            if not before.is_floating_point():
                maximum = max(abs(int(v)-int(u)) for u,v in zip(x.tolist(),y.tolist()))
            else:
                delta = (y[finite].double()-x[finite].double()).abs()
                valid = torch.isfinite(delta)
                result['finite_difference_overflow'] += int((~valid).sum())
                if bool(valid.any()): maximum = float(delta[valid].max())
        if maximum is not None:
            prior = result['max_abs_diff_finite']
            result['max_abs_diff_finite'] = maximum if prior is None else max(prior, maximum)
        remaining = sample_limit-len(result['samples'])
        if remaining:
            for index in changed.nonzero().flatten()[:remaining].tolist():
                result['samples'].append(dict(flat_index=offset+index, before=_scalar(x[index]), after=_scalar(y[index])))
    return result


def actual_request_effect(row, *, position_limit=8):
    """Display an existing serving row; never compare unrelated requests.

    Only adapter-bound same-return summaries are recognized. No tensors are
    loaded, and a hit without a supported readback remains effect-unverified.
    The current request is explicit: this function does not substitute an old
    fault effect for an OFF/healthy request.
    """
    if type(position_limit) is not int or not 0 <= position_limit <= 8:
        raise ValueError('position_limit must be an integer from 0 to 8')
    report = dict(schema='demoncat.effect.actual_request.v1',
                  status='EFFECT_NOT_VERIFIED', scope=None, matched_effects=0,
                  applied_events=0, omitted_positions=0, positions=[],
                  qualification_granted=False, recovery_proven=False)
    if not row:
        return report
    if not all(row.get(key) for key in ('request_id', 'phase', 'invocation', 'input')):
        return report
    report['scope'] = dict(request_id=row['request_id'], phase=row['phase'],
                          invocation=row['invocation'], input_identity_sha256=hashlib.sha256(
                              json.dumps(row['input'], sort_keys=True, allow_nan=False,
                                         separators=(',', ':')).encode()).hexdigest())
    for event in (row.get('mechanism') or {}).get('events', ()):
        if event.get('applied') is not True:
            continue
        report['applied_events'] += 1
        effect = event.get('effect') or {}
        position = effect.get('semantic_position')
        values = effect.get('values') or {}
        if (effect.get('comparison') != 'SAME_ACTUAL_RETURN_BEFORE_AFTER'
                or position != {key: event.get(key) for key in ('path', 'ordinal', 'forward')}
                or values.get('schema') != 'demoncat.effect.numeric_change.v1'
                or event.get('original_execution_verified') is not True
                or event.get('return_readback_exact') is not True):
            continue
        report['matched_effects'] += 1
        if len(report['positions']) < position_limit:
            report['positions'].append(dict(semantic_position=deepcopy(position),
                actual_consumer_observed=bool(event.get('consumers')),
                values=deepcopy(values)))
    report['omitted_positions'] = report['matched_effects'] - len(report['positions'])
    if report['matched_effects']:
        report['status'] = ('MATCHED_SAME_RETURN_VALUES_OBSERVED' if
            report['matched_effects'] == report['applied_events'] else 'PARTIAL_EFFECT_OBSERVATION')
    return report
