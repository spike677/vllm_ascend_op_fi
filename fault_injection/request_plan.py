"""Declarative ordinary request schedules, not permission to run a mechanism.

Both capability query and Worker task receipts use these same immutable tuples.
Returned plans are new objects. Recovery is a separate failed-task path; it
must not silently repeat a fault to fill a successful plan.
"""
FOUR = ('normal', 'fault', 'off', 'health')
KV41 = ('normal', 'normal_repeat', 'lossless', 'fault', 'off', 'health')
BATCH = ('normal', 'reference', 'lossless', 'calibration_off', 'fault', 'off')
_PHASES = {name: FOUR for name in ('8.restore_offset','17.spatial_padding',
    '18.api_half','22.wrong_row','39.output_proxy')}
_PHASES.update({'41.tail_column': KV41, 'quant.batch_shared_scale': BATCH})

def phases(mechanism):
    return _PHASES.get(mechanism, ())

def request_plan(mechanism):
    ordered = phases(mechanism)
    if not ordered: return None
    batch = mechanism == 'quant.batch_shared_scale'
    return dict(schema='demoncat.ordinary.request_plan.v1',mechanism=mechanism,
        phases=[dict(phase=name,request_count=8 if batch else 1,
                     batch_size=8 if batch else None) for name in ordered],
        unit='batch' if batch else 'request',max_iterations=len(ordered),
        max_batches=len(ordered) if batch else None,
        max_requests=8*len(ordered) if batch else len(ordered),
        external_health_required=batch,
        max_total_requests_including_external=64 if batch else len(ordered),
        external_health=[dict(position=position,request_count=8,batch_size=8,phase='health')
                         for position in ('before_plan','after_cleanup')] if batch else [],
        external_health_in_iteration_budget=False,
        advance_argument='request_ids' if batch else 'request_id',
        completion='next_phase == complete AND no current task failure',
        task_close_requires='mechanism recovery, finish, unregister and zero owned resources',
        failure_policy='STOP_PLAN_USE_EXPLICIT_RECOVERY_NEVER_RETRY_FAULT',
        qualification_granted=False)

def task_state(mechanism, state):
    """Pure projection, leaves detailed mechanism observations intact."""
    plan=request_plan(mechanism)
    if plan is None:return state
    serving=state.get('serving') or {}
    next_phase=state.get('next_phase',serving.get('next_phase'))
    failed=bool(state.get('evidence_limit_error') or state.get('replay_failed')
                or state.get('failure') or serving.get('error') or serving.get('failure'))
    complete=next_phase=='complete' and not failed
    return dict(state,request_plan=plan,next_phase=next_phase,complete=complete,
                request_plan_state='failed' if failed else 'complete' if complete else 'pending')
