"""Cost-only source82 originals for a new isolated C2 -> cumulative C3 plan.

No success/qualification flag is used as timing evidence or live authority.
The raw, separately SHA-pinned result and controller remain on the NPU host.
"""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


RESULT_SHA256 = 'f84c1cc357ae5cce58b3b1d52d1e4a977e887de5acca883e8c0a9d08af654643'
CONTROLLER_SHA256 = '77779e19ac4adbaa9443393d81f37eefb26762df1ec2061fea348f4af2c31319'
QUALIFICATION = ('qualification_reference_all', 'qualification_lossless_all', 'qualification_off_all')


def _require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_cost_reference(result_path, controller_path):
    originals, documents = {}, {}
    for name, path, expected in (('result', result_path, RESULT_SHA256), ('controller', controller_path, CONTROLLER_SHA256)):
        _require(path is not None, 'ISOLATED_C3_COST_ORIGINAL_REQUIRED:' + name)
        path = Path(path).resolve(strict=True)
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == expected, 'ISOLATED_C3_COST_ORIGINAL_SHA_CHANGED:' + name)
        originals[name] = dict(path=str(path), sha256=expected, bytes=len(raw))
        documents[name] = json.loads(raw)
    result, controller = documents['result'], documents['controller']
    expected_phases = ['inventory_1', 'inventory_2', *QUALIFICATION] + ['group_fault', 'group_off'] * 29 + ['health']
    requests = result['requests']
    _require(result['stage'] == 'c2_isolated_once' and result['mechanism'] == '26.fia_destination'
        and len(requests) == 64 and [row['phase'] for row in requests] == expected_phases
        and result['isolated_plan']['full_native_writer_count'] == 30
        and all(_positive(row['complete_cycle_seconds']) for row in requests),
        'COMPLETE_SOURCE82_ACTUAL_COST_REQUESTS_REQUIRED')
    cases = [row for row in controller['cases'] if row['id'] == 'providers26-core-fia-smol-c2_isolated_once']
    _require(len(cases) == 1, 'EXACT_SOURCE82_COST_CONTROLLER_CASE_REQUIRED')
    started, ended = (datetime.fromisoformat(cases[0][key]) for key in ('started_utc', 'ended_utc'))
    duration = (ended - started).total_seconds()
    cycles = [dict(phase=row['phase'], seconds=row['complete_cycle_seconds']) for row in requests]
    total = sum(row['seconds'] for row in cycles)
    _require(started.tzinfo is not None and ended.tzinfo is not None and 0 < total <= duration <= 1800,
             'SOURCE82_COST_CLOCK_OR_COMPLETE_CYCLES_INCONSISTENT')
    value = dict(schema='demoncat.writer26.source82_cost_reference.v1', originals=originals,
        actual_controller_started=cases[0]['started_utc'], actual_controller_ended=cases[0]['ended_utc'],
        actual_controller_seconds=duration, actual_complete_cycle_seconds=total,
        outside_complete_cycles_seconds=duration-total, cycles=cycles, full_native_writer_count=30,
        group_requests=64, qualification_granted=False, used_success_flags=False)
    value['cost_sha256'] = _digest(value)
    return value


def transition_budget(requests, plan, reference, *, elapsed_seconds, run='c2'):
    _require(run in ('c2', 'c3') and type(elapsed_seconds) in (int, float)
        and math.isfinite(elapsed_seconds) and elapsed_seconds >= 0, 'ACTUAL_FULL_PROBE_CLOCK_REQUIRED')
    _require(isinstance(reference, dict) and reference.get('cost_sha256') == _digest(
        {key: value for key, value in reference.items() if key != 'cost_sha256'})
        and reference['originals']['result']['sha256'] == RESULT_SHA256
        and reference['originals']['controller']['sha256'] == CONTROLLER_SHA256,
        'PINNED_COST_ORIGINALS_AND_UNCHANGED_DERIVATION_REQUIRED')
    current = [row for row in requests if row.get('run') == run]
    _require([row['phase'] for row in current] == ['inventory_1', 'inventory_2', *QUALIFICATION]
        and all(_positive(row.get('complete_cycle_seconds')) for row in current),
        'FIVE_COMPLETE_CURRENT_GENERATION_HEALTHY_CYCLES_BEFORE_FIRST_FAULT_REQUIRED')
    samples = [row['complete_cycle_seconds'] for row in current if row['phase'] in QUALIFICATION]
    _require(len(samples) == 3 and all(_positive(value) for value in samples),
             'THREE_ACTUAL_COMPLETE_QUALIFICATION_CYCLES_REQUIRED_FOR_BUDGET')
    count, c2_requests = plan['full_native_writer_count'], plan['c2_group_requests']
    _require(type(count) is int and count > 0 and 0 < len(set(plan['target_order'])) == len(plan['target_order']) <= min(128, count)
        and type(c2_requests) is int
        and c2_requests == 6+2*len(plan['target_order']) and plan['c3_group_requests'] == 9
        and plan['combined_group_requests'] == c2_requests+9, 'EXACT_COMPLETE_C2_C3_PLAN_REQUIRED_FOR_BUDGET')
    prior = reference['cycles']
    baseline = [row['seconds'] for row in prior if row['phase'] in QUALIFICATION] if run == 'c2' \
        else [max(row['seconds'] for row in prior)]*3
    # Faster first cycles may not erase the real late-stage 20-25 second
    # evidence/Host cost. C3 is already after the long C2 history, so compare
    # its qualification cycles with the actual late history cost, not the
    # source82 sub-second initial qualification (which would count history
    # overhead twice). Slower comparable cycles amplify every retained cost.
    slowdown = max(1., *(actual / old for actual, old in zip(samples, baseline)))
    denominator_scale = max(1., count / 30)
    c2_remaining = sum(row['seconds'] for row in prior[5:]) * (c2_requests/64)**3 * denominator_scale * slowdown * 1.25
    observed = [row['complete_cycle_seconds'] for row in requests if _positive(row.get('complete_cycle_seconds'))]
    tail = max(max(row['seconds'] for row in prior), max(observed, default=0))
    tail_cycle = tail * 1.25 * ((c2_requests+9)/64)**2 * denominator_scale * slowdown
    c3_remaining = 9 if run == 'c2' else 9-sum(row.get('run') == 'c3' for row in requests)
    _require(c3_remaining in (9, 4), 'C3_BUDGET_CHECK_MUST_PRECEDE_FIRST_FAULT')
    forecast = (c2_remaining if run == 'c2' else 0) + c3_remaining*tail_cycle + 120
    generations = dict(c2=c2_remaining+120 if run == 'c2' else None, c3=4*tail_cycle+120)
    reasons = []
    if any(seconds is not None and seconds > 1200 for seconds in generations.values()):
        reasons.append('ISOLATED_C3_PER_GENERATION_1200_SECOND_BUDGET_UNPROVEN')
    if elapsed_seconds+forecast+120 > 1800:
        reasons.append('ISOLATED_C3_ROOT_CASE_1800_SECOND_BUDGET_UNPROVEN')
    return dict(schema='demoncat.writer26.isolated_transition_budget.v1',
        status='REFUSED_BEFORE_F' if reasons else 'CONSERVATIVE_ESTIMATE_WITHIN_EXISTING_LIMITS',
        checked_before_run=run, cost_reference_sha256=reference['cost_sha256'],
        cost_originals=reference['originals'], complete_qualification_cycle_seconds=samples,
        qualification_cost_baseline_seconds=baseline,
        qualification_cost_baseline='SOURCE82_R_L_OFF' if run == 'c2' else 'SOURCE82_LATE_COMPLETE_CYCLE',
        historical_remaining_c2_cycle_seconds=sum(row['seconds'] for row in prior[5:]),
        historical_controller_seconds=reference['actual_controller_seconds'],
        observed_late_cycle_seconds=tail, actual_cycle_slowdown=slowdown, cost_safety_multiplier=1.25,
        native_denominator_scale=denominator_scale, c3_remaining_requests=c3_remaining,
        generation_forecast_seconds=generations, forecast_remaining_seconds=forecast,
        actual_probe_elapsed_seconds=elapsed_seconds, projected_outer_seconds=elapsed_seconds+forecast+120,
        cleanup_reserve_seconds=120, outside_probe_startup_exit_disk_margin_seconds=120,
        session_start='PER_COVERAGE_GENERATION_AT_FIRST_FAULT_BUILD_GROUP',
        case_clock='ONE_CONTINUOUS_PROBE_CLOCK_INCLUDING_MODEL_INITIALIZATION',
        existing_session_limit_seconds=1200, existing_root_case_limit_seconds=1800,
        qualification_granted=False, refusal_reasons=reasons)
