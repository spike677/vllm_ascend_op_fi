"""Extract native log evidence between synchronized API markers, without ABI decoding.

A marker window is a temporal association, not proof of the native log thread's
identity. Raw tiling framing has not yet been established for this installation;
this calibration parser therefore never reports COMPLETE_PROFILE or acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


MARKER_EVENTS = {'native_api_enter', 'native_api_exit'}
IDENTITY_FIELDS = ('pid', 'thread_id', 'case_id', 'rank', 'call_id',
                   'native_index', 'layer_index', 'layer_name', 'api')
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
TILING = re.compile(r'Tiling\s+Key:\s*(0x[0-9a-fA-F]+|\d+)\s*,\s*'
                    r'len:\s*(\d+)\s*,\s*numBlocks:\s*(\d+)', re.I)
KERNEL = re.compile(r'kernel\s+name|dev_func|function handle by tiling key|Save launch config', re.I)
ROUTE = re.compile(r'(?:fia[_a-z0-9]*tiling[_a-z0-9]*|fused_infer_attention_score[_a-z0-9]*'
                   r'|prompt_flash_attention[_a-z0-9]*|flash_attention_score[_a-z0-9]*)\.(?:cpp|h)|'
                   r'(?:Fia|FIA|FusedInferAttention|PromptFlashAttention)[A-Za-z0-9_:]*Tiling')
RAW = re.compile(r'PrintTilingData|Aligned\s+TilingData\s+Size|tiling\s+data\s*:', re.I)
FIA_NAME = re.compile(r'FusedInferAttention|fused_infer_attention|FiaTiling|FIA|FlashAttentionScore|flash_attention|npu_fusion_attention|scaled_dot_product_attention', re.I)
LOG_PID = re.compile(r'\bOP\((\d+),[^)]*\)')
SELECTED = re.compile(r'\[NnopbaseExecutorDoTiling\]\[(?P<thread>\d+)\]\s+OpName:\[(?P<op>[^\]]+)\]\s+'
                      r'Tiling outputs of (?P<type>\w+) are \[tilingKey:\s*(?P<key>0x[0-9a-f]+|\d+),\s*'
                      r'numBlocks:\s*(?P<blocks>\d+),\s*scheMode:\s*(?P<mode>\d+)', re.I)
LAUNCH = re.compile(r'\[NnopbaseExecutorLaunchKernel\]\[(?P<thread>\d+)\]\s+OpName:\[(?P<op>[^\]]+)\]\s+'
                    r'Launch dynamic kernel (?P<type>\w+) task,\s*tilingKey is (?P<key>0x[0-9a-f]+|\d+),\s*'
                    r'scheMode is (?P<mode>\d+),\s*numBlocks is (?P<blocks>\d+),\s*'
                    r'stream is (?P<stream>0x[0-9a-f]+)', re.I)
REGISTRATION = re.compile(r'\[indv_bininfo\.cpp:\d+\]|\bMergeKernel\b|op type .+ has been registered', re.I)


def selected_fields(match, text):
    pid = LOG_PID.search(text)
    value = match.group('key')
    key = int(value, 16 if value.lower().startswith('0x') else 10)
    return {'log_pid': int(pid.group(1)) if pid else None,
            'log_thread_id': int(match.group('thread')),
            'op_name': match.group('op'), 'op_type': match.group('type'),
            'tiling_key_decimal': str(key), 'num_blocks': int(match.group('blocks')),
            'schedule_mode': int(match.group('mode'))}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_marker(line):
    """Allow a logger prefix, but require a complete JSON object at line end."""
    clean = ANSI.sub('', line).strip()
    position = clean.find('{')
    if position < 0:
        return None
    try:
        value, end = json.JSONDecoder().raw_decode(clean[position:])
    except (json.JSONDecodeError, ValueError):
        return None
    if clean[position + end:].strip() or not isinstance(value, dict):
        return None
    return value if value.get('event') in MARKER_EVENTS else None


def validate_marker(marker, case_id):
    missing = [field for field in IDENTITY_FIELDS if field not in marker]
    if missing:
        return 'marker_missing_fields:' + ','.join(missing)
    if marker['case_id'] != case_id:
        return 'marker_case_id_mismatch'
    if marker['rank'] != 0:
        return 'marker_rank_is_not_zero'
    if any(not isinstance(marker[key], int) or isinstance(marker[key], bool)
           for key in ('pid', 'thread_id', 'native_index', 'layer_index')):
        return 'marker_integer_identity_invalid'
    if marker['call_id'] != f"pid{marker['pid']}:r0:n{marker['native_index']}":
        return 'marker_call_id_inconsistent'
    if not isinstance(marker['api'], str) or not FIA_NAME.search(marker['api']):
        return 'marker_api_is_not_fia'
    if marker['event'] == 'native_api_exit' and marker.get('status') not in ('returned', 'error'):
        return 'marker_exit_status_invalid'
    return None


def parse_lines(lines, case_id, expected_calls=None, candidate_limit=2000):
    calls = {}
    active = {}
    errors = []
    outside_candidates = []
    outside_candidate_count = 0
    ignored_registration_count = 0
    non_utf8_line_count = 0
    non_utf8_lines = []
    total_lines = 0
    for line_number, line in enumerate(lines, 1):
        total_lines = line_number
        if any('\udc80' <= char <= '\udcff' for char in line):
            non_utf8_line_count += 1
            if len(non_utf8_lines) < 100:
                non_utf8_lines.append({'line': line_number,
                    'raw_line_sha256': hashlib.sha256(line.encode('utf-8', errors='surrogateescape')).hexdigest()})
            # Preserve invalid bytes and fail closed if they contaminate a
            # marker or a profile candidate. Unrelated native binary log noise
            # is recorded without discarding the rest of the capture.
            if ('native_api_enter' in line or 'native_api_exit' in line or
                    any(pattern.search(line) for pattern in (TILING, SELECTED, LAUNCH, ROUTE, RAW))):
                errors.append({'line': line_number, 'error': 'non_utf8_profile_or_marker_line'})
                continue
        marker = json_marker(line)
        if marker is not None:
            problem = validate_marker(marker, case_id)
            if problem:
                errors.append({'line': line_number, 'error': problem, 'marker': marker})
                continue
            call_id = marker['call_id']
            if marker['event'] == 'native_api_enter':
                if call_id in calls:
                    errors.append({'line': line_number, 'error': 'duplicate_enter', 'call_id': call_id})
                    continue
                item = {'call_id': call_id, 'enter_line': line_number, 'enter': marker,
                        'exit_line': None, 'exit': None, 'tiling_records': [],
                        'host_route_candidates': [], 'kernel_candidates': [],
                        'selected_tiling_records': [], 'selected_launch_records': [],
                        'ignored_registration_count': 0,
                        'raw_log_candidates': [], 'candidate_count': 0,
                        'candidate_limit_exceeded': False, 'overlap': bool(active),
                        'native_thread_identity_verified': False,
                        'association': 'synchronized_marker_window_only',
                        'raw_tiling_complete': False,
                        'raw_tiling_reason': 'No version-matched raw-byte framing/length decoder has been established.'}
                for previous in active.values():
                    previous['overlap'] = True
                calls[call_id] = item
                active[call_id] = item
            else:
                item = active.get(call_id)
                if item is None:
                    errors.append({'line': line_number, 'error': 'exit_without_active_enter', 'call_id': call_id})
                    continue
                if any(item['enter'][key] != marker[key] for key in IDENTITY_FIELDS):
                    errors.append({'line': line_number, 'error': 'enter_exit_identity_mismatch', 'call_id': call_id})
                item.update(exit_line=line_number, exit=marker)
                active.pop(call_id)
            continue

        text = ANSI.sub('', line).rstrip('\r\n')
        if REGISTRATION.search(text):
            ignored_registration_count += 1
            if len(active) == 1:
                next(iter(active.values()))['ignored_registration_count'] += 1
            continue
        tiling = TILING.search(text)
        selected = SELECTED.search(text)
        launched = LAUNCH.search(text)
        if selected and not FIA_NAME.search(selected.group('type')):
            selected = None
        if launched and not FIA_NAME.search(launched.group('type')):
            launched = None
        route = ROUTE.search(text)
        kernel = KERNEL.search(text)
        raw = RAW.search(text)
        if not (tiling or selected or launched or route or kernel or raw):
            continue
        evidence = {'line': line_number, 'text': text, 'fia_named_in_line': bool(FIA_NAME.search(text))}
        if len(active) != 1:
            outside_candidate_count += 1
            if len(outside_candidates) < candidate_limit:
                outside_candidates.append({**evidence, 'active_call_ids': sorted(active)})
            continue
        item = next(iter(active.values()))
        item['candidate_count'] += 1
        if item['candidate_count'] > candidate_limit:
            item['candidate_limit_exceeded'] = True
            continue
        if tiling:
            key = tiling.group(1)
            item['tiling_records'].append({**evidence, 'tiling_key': int(key, 16 if key.lower().startswith('0x') else 10),
                                           'length_from_log': int(tiling.group(2)),
                                           'num_blocks': int(tiling.group(3)),
                                           'length_unit': 'not_inferred_from_log_label'})
        if selected:
            item['selected_tiling_records'].append({**evidence, **selected_fields(selected, text),
                'evidence_kind': 'NnopbaseExecutorDoTiling_output',
                'length_from_log': None,
                'scope': 'Actual executor tiling selection; inner PFA/FIA route and raw bytes are not inferred.'})
        if launched:
            item['selected_launch_records'].append({**evidence, **selected_fields(launched, text),
                'stream_from_log': launched.group('stream'),
                'evidence_kind': 'NnopbaseExecutorLaunchKernel_dynamic_launch'})
        if route:
            item['host_route_candidates'].append(evidence)
        if kernel:
            item['kernel_candidates'].append(evidence)
        if raw:
            item['raw_log_candidates'].append(evidence)

    for call_id, item in active.items():
        errors.append({'line': item['enter_line'], 'error': 'unterminated_call', 'call_id': call_id})
    if not calls:
        errors.append({'error': 'no_valid_fia_marker_calls'})
    expected_ids = []
    if expected_calls is None:
        errors.append({'error': 'result_native_callers_list_missing'})
    else:
        for entry in expected_calls:
            if not isinstance(entry, dict) or not isinstance(entry.get('call_id'), str):
                errors.append({'error': 'result_native_caller_invalid'})
                continue
            call_id = entry['call_id']
            expected_ids.append(call_id)
            item = calls.get(call_id)
            if item is None:
                errors.append({'error': 'result_call_missing_markers', 'call_id': call_id})
            elif any(key in entry and entry[key] != item['enter'].get(key)
                     for key in IDENTITY_FIELDS):
                errors.append({'error': 'result_caller_identity_mismatch', 'call_id': call_id})
        if len(expected_ids) != len(set(expected_ids)):
            errors.append({'error': 'duplicate_result_native_call_id'})
        if set(calls) - set(expected_ids):
            errors.append({'error': 'marker_calls_missing_from_result',
                           'call_ids': sorted(set(calls) - set(expected_ids))})
    for item in calls.values():
        missing = ['complete_tiling_bytes', 'native_thread_correlation']
        correlations = []
        for selection in item['selected_tiling_records']:
            matches = [launch for launch in item['selected_launch_records']
                if selection['line'] < launch['line']
                and all(selection[key] == launch[key] for key in
                        ('log_pid', 'op_name', 'op_type', 'tiling_key_decimal', 'num_blocks', 'schedule_mode'))]
            marker_pid_matches = selection['log_pid'] == item['enter']['pid']
            selection_thread_matches = selection['log_thread_id'] == item['enter']['thread_id']
            correlations.append({'selection_line': selection['line'],
                'launch_lines': [launch['line'] for launch in matches],
                'tiling_key_decimal': selection['tiling_key_decimal'],
                'op_name': selection['op_name'], 'num_blocks': selection['num_blocks'],
                'schedule_mode': selection['schedule_mode'], 'native_pid': selection['log_pid'],
                'selection_thread': selection['log_thread_id'],
                'launch_threads': [launch['log_thread_id'] for launch in matches],
                'marker_pid_matches': marker_pid_matches,
                'selection_thread_matches_marker': selection_thread_matches,
                'unique_launch_match': len(matches) == 1,
                'correlated': (len(matches) == 1 and marker_pid_matches and selection_thread_matches
                               and not item['overlap']),
                'scope': 'Observed operator name/key/PID/block count/schedule match inside the API window; launch may use a different dispatch thread.'})
        item['selection_launch_correlations'] = correlations
        item['selected_key_launch_correlated'] = (len(correlations) == 1 and correlations[0]['correlated']
                                                  and len(item['selected_launch_records']) == 1)
        if not item['selected_key_launch_correlated']:
            missing.append('unique_selected_tiling_and_launch_correlation')
        if not item['tiling_records']:
            missing.append('tiling_length')
            if not item['selected_tiling_records']:
                missing.append('selected_tiling_key_num_blocks')
        if not item['host_route_candidates']:
            missing.append('host_tiling_route')
        if not item['selected_launch_records']:
            missing.append('native_kernel_evidence')
        if item['exit'] is None or item['exit'].get('status') != 'returned':
            missing.append('successful_native_return')
        if item['overlap']:
            missing.append('nonoverlapping_marker_window')
        if item['candidate_limit_exceeded']:
            missing.append('untruncated_candidates')
        item['missing_evidence'] = missing
        item['status'] = 'INCOMPLETE_PROFILE'
    return {'schema': 'fault28.calibration_log.v2', 'case_id': case_id,
            'status': 'INVALID_MARKER_CORRELATION' if errors else 'INCOMPLETE_PROFILE',
            'mechanism_accepted': False, 'total_lines': total_lines,
            'call_count': len(calls), 'calls': list(calls.values()), 'errors': errors,
            'outside_or_overlapping_candidate_count': outside_candidate_count,
            'ignored_registration_count': ignored_registration_count,
            'non_utf8_line_count': non_utf8_line_count, 'non_utf8_lines': non_utf8_lines,
            'log_decoding': 'UTF-8 surrogateescape; raw bytes preserved, contaminated profile/marker lines invalidate correlation.',
            'outside_or_overlapping_candidates': outside_candidates,
            'scope': 'Preserves native log candidates inside marker windows; no private executor decoding, no claim that temporal order proves thread identity or host-route completeness.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--case-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--callers-key', default='native_calls')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Output must be fresh; original capture files are never rewritten')
    result = json.loads(args.case_result.read_text(encoding='utf-8'))
    case_id = result.get('case_id')
    if not isinstance(case_id, str) or not case_id:
        raise RuntimeError('Probe result has no case_id')
    callers = result.get(args.callers_key)
    if callers is not None and not isinstance(callers, list):
        raise RuntimeError('Native caller record must be a list')
    before_log = sha(args.log)
    before_result = sha(args.case_result)
    with args.log.open(encoding='utf-8', errors='surrogateescape') as stream:
        report = parse_lines(stream, case_id, callers)
    if sha(args.log) != before_log or sha(args.case_result) != before_result:
        raise RuntimeError('Source evidence changed while parsing')
    report['inputs'] = {'log': str(args.log), 'log_sha256': before_log,
                        'case_result': str(args.case_result), 'case_result_sha256': before_result,
                        'probe_status': result.get('status'), 'callers_key': args.callers_key}
    report['parser_sha256'] = sha(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=True, indent=2)
    print(json.dumps({'status': report['status'], 'case_id': case_id,
                      'call_count': report['call_count'], 'mechanism_accepted': False}))
    return 1 if report['status'] == 'INVALID_MARKER_CORRELATION' else 0


if __name__ == '__main__':
    raise SystemExit(main())
