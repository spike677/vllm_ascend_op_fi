"""Shared read-only deployment/mechanism/live-call prerequisite schema."""
from copy import deepcopy

LAYERS = ('deployment_identity', 'mechanism_compatibility', 'actual_call')
_MISSING = object()


def observed_field(root, path):
    value = root
    for name in path.split('.'):
        value = getattr(value, name, _MISSING)
        if value is _MISSING:
            return _MISSING
    return value


def expectation(layer, field, expected, actual, code):
    if layer not in LAYERS:
        raise ValueError('UNKNOWN_PREREQUISITE_LAYER')
    missing = actual is _MISSING
    matched = not missing and type(actual) is type(expected) and actual == expected
    return dict(layer=layer, field=field, expected=deepcopy(expected),
                actual={'missing': True} if missing else deepcopy(actual),
                status='pending' if missing else 'matched' if matched else 'mismatch', code=code)


def assessment(checks):
    ordered = sorted((deepcopy(row) for row in checks), key=lambda row: LAYERS.index(row['layer']))
    unmet = [row for row in ordered if row['status'] != 'matched']
    return dict(schema='demoncat.prerequisites.v1', checks=ordered, unmet=unmet,
                status='refused' if any(row['status']=='mismatch' for row in unmet)
                       else 'pending' if unmet else 'matched',
                primary=unmet[0] if unmet else None, execution_authorized=False)


def require_assessment(checks,error_type=RuntimeError):
    report=assessment(checks)
    if report['unmet']:
        code=report['primary']['code']
        error=error_type(code)
        error.code=code.split(':',1)[0]
        error.prerequisites=report
        raise error
    return report


def require_identity(condition,component,field,expected,actual,code,error_type=RuntimeError):
    row=expectation('deployment_identity',component+'.'+field,expected,actual,code)
    # Some checks describe a relationship/range, rather than scalar equality.
    row['status']='matched' if condition else 'mismatch'
    return require_assessment([row],error_type)
