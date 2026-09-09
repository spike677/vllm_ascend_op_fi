"""Derived metadata and pre-fault calibration for the ONE frozen probe source.

Exported catalog is a materialized view, never another editable prompt registry.
Calibration uses healthy properties, not free-text byte equality.
"""
from .contracts import DistortionProbe, content_hash
from .detectors import (DETECTORS, task_fact, structure_fact, repetition_fact,
                        punctuation_fact, text)

VERSION = 'NORMAL_DIAGNOSTIC_CASES_V1'


def catalog(suite):
    cases = []
    properties = {'task_answer': 'approved answer resolves and is correct',
                  'structure': 'declared structural validator passes',
                  'repetition': 'no pathological repeated run or periodic loop',
                  'lexical_collapse': 'lexical diversity inspected relative to clean output',
                  'punctuation': 'no punctuation run/spam above frozen bounds',
                  'length': 'nonempty generation; length alone does not prove incompleteness',
                  'garbage': 'no replacement/control character corruption',
                  'crash': 'HTTP 200 and responsive service'}
    for p in suite['probes']:
        case = {**p, 'case_id': p['probe_id'], 'suite_version': VERSION,
                'language': 'en', 'context': None,
                'expected_healthy_properties': {d: properties[d] for d in p['applicable_detectors']},
                'reference_validator': p.get('expected', {}),
                'source': 'project-authored fixed engineering smoke cases',
                'source_id': p['probe_id'], 'revision': suite['hash'],
                'license': 'external dataset license: not applicable; repository terms apply',
                'semantic_status': 'VALIDATOR_SCOPED' if p.get('expected') else 'UNVERIFIED',
                'normality_review': 'self-contained ordinary request; no external tools or adversarial instruction'}
        case['hash'] = content_hash(case)
        cases.append(case)
    result = {'kind': 'NORMAL_DIAGNOSTIC_CASES', 'version': VERSION,
              'source_path': 'profiles/evaluation/distortion_probe_suite_v1.json',
              'source_suite_version': suite['version'], 'source_suite_hash': suite['hash'],
              'request_protocol': suite.get('request_protocol', {'endpoint': '/v1/completions'}),
              'case_count': len(cases), 'cases': cases,
              'model_input_boundary': 'only prompt and generation_config; metadata/reference/thresholds are evaluator-only'}
    result['hash'] = content_hash(result)
    return result


def _healthy(probe, row, name, thresholds):
    if row.get('http_status') != 200 or text(row) is None:
        return False
    if name == 'task_answer':
        fact = task_fact(probe, row)
        return fact['correct'] if fact is not None else None
    if name == 'structure':
        fact = structure_fact(probe, row)
        return fact['valid'] if fact is not None else None
    if name == 'repetition':
        f = repetition_fact(row, thresholds)
        return not (f['token_count'] >= thresholds['repetition_min_tokens'] and (
            f['periodic_loop'] or f['same_token_max_run'] >= thresholds['repeated_run_min']
            or f['seq_rep_4'] >= thresholds['seq_rep_4_min']))
    if name == 'punctuation':
        f = punctuation_fact(row)
        return f['ratio'] < thresholds['punctuation_rate_min'] and f['max_run'] < thresholds['punctuation_run_min']
    if name == 'garbage':
        f = DETECTORS[name].evaluate(probe, row, row, {'thresholds': thresholds})
        return f.baseline_value['bad_character_count'] < thresholds['bad_character_min']
    if name in ('length', 'lexical_collapse'):
        return bool(text(row).strip())
    return True


def calibrate(suite, runs):
    if len(runs) != 2:
        raise ValueError('two complete Fault OFF calibrations required')
    ids = {p['probe_id'] for p in suite['probes']}
    if any(set(run) != ids for run in runs):
        raise ValueError('retain every calibration case, including failed requests')
    result = {'version': 'CLEAN_CALIBRATION_V1', 'suite_hash': suite['hash'],
              'frozen_before_fault': True, 'case_count': len(ids), 'matrix': {},
              'policy': 'properties must pass both OFF runs; unknown is not healthy; all cases retained'}
    for p in suite['probes']:
        probe = DistortionProbe.from_dict(p)
        row = {}
        for name in DETECTORS:
            values = [_healthy(probe, run[p['probe_id']], name, suite['thresholds']) for run in runs] if name in probe.applicable_detectors else []
            if not values:
                status = 'NOT_APPLICABLE'
            elif None in values:
                status = 'BASELINE_UNSTABLE'
            elif values == [True, True]:
                status = 'ELIGIBLE'
            elif values == [False, False]:
                status = 'BASELINE_FAILED'
            else:
                status = 'BASELINE_UNSTABLE'
            row[name] = {'status': status, 'healthy_properties': values}
        result['matrix'][p['probe_id']] = row
    result['hash'] = content_hash(result)
    return result
