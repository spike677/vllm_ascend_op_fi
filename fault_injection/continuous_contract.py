"""Pure declarations and boundary review for the existing Worker lifecycle.

Not a second task state machine, lease manager, or grant of NPU qualification.
Evidence is supplied by sites; this module cannot establish device completion.
"""
from copy import deepcopy
from .request_plan import FOUR

MECHANISMS = ('18.api_half', '22.wrong_row', '39.output_proxy', '39.matmul_range',
    'quant.batch_shared_scale', '41.tail_column', '17.spatial_padding',
    '8.restore_offset', '28.fia.denominator_replay', '20.buffer_substitute',
    '26.fia_destination', '33.paged_decode')
COUNTERS = ('observed','eligible','attempted','applied','changed','consumed','restored','failed')


def continuous_request_plan(mechanism, *, batch_size=1):
    if mechanism not in MECHANISMS:
        raise ValueError('MECHANISM_OUTSIDE_CONTINUOUS_DELIVERY')
    batch = mechanism == 'quant.batch_shared_scale'
    if type(batch_size) is not int or (batch_size < 2 if batch else batch_size != 1):
        raise ValueError('BATCH_REQUIRES_TWO_OR_MORE_REQUESTS' if batch else 'ONE_REQUEST_REQUIRED')
    return dict(schema='demoncat.continuous.request_plan.v1', mechanism=mechanism,
        phases=[dict(phase=p, request_count=batch_size, max_tokens=4,
                     includes_prefill=True, includes_decode=True) for p in FOUR],
        unit='batch' if batch else 'request', max_iterations=4,
        max_requests=4*batch_size, external_health_required=False,
        advance_argument='request_ids' if batch else 'request_id',
        injection_policy='ALL_RANK_EVERY_ELIGIBLE_CALL', qualification_granted=False,
        completion='next_phase == complete AND all rank recovery checks passed')


def review_rank_coverage(members, receipts):
    """At a shared control boundary only; supports world_size=1 identically.

No minimum local hit count: an EP rank can have no local expert call. Overall
success still requires a real modification somewhere, complete observations,
and every eligible call consumed and restored on every participating rank.
"""
    members = tuple(members)
    if (not members or len(set(members)) != len(members)
            or any(type(r) is not int or r < 0 for r in members)):
        raise ValueError('EXPLICIT_UNIQUE_RANK_MEMBERS_REQUIRED')
    failures = []
    if set(receipts) != set(members):
        failures.append('RANK_RECEIPTS_INCOMPLETE')
    total_applied = 0
    for rank in members:
        r = receipts.get(rank, {})
        c = r.get('counts', {})
        def fail(reason):
            failures.append(f'rank{rank}:{reason}')
        if not all(type(c.get(k)) is int and c[k] >= 0 for k in COUNTERS):
            fail('INVALID_COUNTERS')
            continue
        if not (c['observed'] >= c['eligible'] == c['attempted'] == c['applied']
                == c['consumed'] == c['restored'] >= c['changed'] and c['failed'] == 0):
            fail('ELIGIBLE_CALLS_NOT_FULLY_COMPLETED')
        for flag in ('observation_complete', 'same_stream_verified', 'bytes_restored',
                     'off_exact', 'health_exact'):
            if r.get(flag) is not True:
                fail(flag.upper() + '_NOT_PROVEN')
        if r.get('uncovered_sites') != []:
            fail('SITE_COVERAGE_INCOMPLETE')
        if r.get('resources') != dict(leases=0, hooks=0, inflight=0):
            fail('RESOURCES_NOT_CLEARED')
        if c['observed'] != c['eligible'] and not r.get('inapplicable_reasons'):
            fail('INAPPLICABLE_REASONS_MISSING')
        total_applied += c['applied']
    if total_applied == 0:
        failures.append('NO_ACTUAL_INJECTION')
    return deepcopy(dict(passed=not failures, failures=failures, ranks=receipts,
                         qualification_granted=False))
