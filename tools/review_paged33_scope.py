"""Offline checkpoint association, never a registration or execution grant.

TP1 and TP2 each need their own observed launch, local geometry and checkpoint.
Scope authenticity still requires source/runtime review; self-declaration alone
does not prove it. This helper checks association after that review.
"""
import hashlib
import json


def review(checkpoint, observed_scope, expected_scope, launch_review_bytes):
    def need(ok, reason):
        if not ok:
            raise ValueError(reason)
    fields = {'hardware_name', 'hardware_family', 'tp', 'rank', 'q_heads', 'kv_heads',
              'head_dim', 'kv_length', 'block_size', 'runtime_source_sha256'}
    need(type(observed_scope) is dict and type(expected_scope) is dict
         and set(observed_scope) == set(expected_scope) == fields, 'PAGED33_SCOPE_FIELDS_REQUIRED')
    need(all(type(observed_scope[k]) is type(expected_scope[k]) and observed_scope[k] == expected_scope[k]
             for k in fields), 'PAGED33_SCOPE_IDENTITY_MISMATCH')
    s = observed_scope
    need(type(s['tp']) is int and s['tp'] in (1, 2)
         and type(s['rank']) is int and 0 <= s['rank'] < s['tp'], 'PAGED33_TP_RANK_INVALID')
    need(all(type(s[k]) is int and s[k] > 0 for k in ('q_heads', 'kv_heads', 'head_dim', 'kv_length', 'block_size'))
         and s['q_heads'] % s['kv_heads'] == 0 and s['kv_length'] > 128, 'PAGED33_LOCAL_GEOMETRY_INVALID')
    def valid_sha(v):
        return type(v) is str and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
    need(s['hardware_family'] in ('A2', 'A3') and type(s['hardware_name']) is str
         and bool(s['hardware_name']) and valid_sha(s['runtime_source_sha256']), 'PAGED33_DEVICE_SOURCE_REQUIRED')
    launch = json.loads(launch_review_bytes)
    need(launch.get('schema') == 'dcat.paged33.observed_spans.v1'
         and launch.get('operator') == 'FIA_paged_decode'
         and launch.get('arithmetic') == 'BF16_MM_FP32_OUT_PBF16_LSE_FP32', 'PAGED33_LAUNCH_REVIEW_REQUIRED')
    from tools.paged33_control_checkpoint import validate_spans
    validate_spans(launch.get('spans'), s['kv_length'])
    need(all(valid_sha(launch.get(k)) for k in ('qkv_sha256', 'launch_sha256', 'serializer_sha256')),
         'PAGED33_LAUNCH_HASHES_REQUIRED')
    need(checkpoint.get('schema') == 'dcat.paged33.checkpoint.v1'
         and checkpoint.get('passed') is True and checkpoint.get('device') == s['hardware_name']
         and checkpoint.get('qkv_sha256') == launch['qkv_sha256']
         and checkpoint.get('launch_review_sha256') == hashlib.sha256(launch_review_bytes).hexdigest()
         and checkpoint.get('spans') == launch['spans'], 'PAGED33_CHECKPOINT_JOIN_FAILED')
    count = s['q_heads'] * s['head_dim']
    checks = checkpoint.get('native_checks')
    need(type(checks) is list and len(checks) == 2, 'PAGED33_TWO_CONTROLS_REQUIRED')
    for c in checks:
        need(type(c.get('elements')) is int and c['elements'] == count
             and type(c.get('bitwise_equal')) is int and 0 <= c['bitwise_equal'] <= count
             and 100*c['bitwise_equal'] >= 99*count and c.get('checkpoint_99pct_passed') is True,
             'PAGED33_CONTROL_COUNT_OR_THRESHOLD')
    repeat = checkpoint.get('repeat', {})
    need(type(repeat.get('elements')) is int and type(repeat.get('bitwise_equal')) is int
         and repeat['elements'] == repeat['bitwise_equal'] == count, 'PAGED33_REPEAT_NOT_EXACT')
    return dict(scope=dict(s), checkpoint_scope_consistent=True,
                qualification_granted=False, all_rank_admission=False)
