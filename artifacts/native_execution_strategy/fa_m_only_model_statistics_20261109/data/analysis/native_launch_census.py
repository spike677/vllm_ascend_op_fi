"""Independent, whole-case census of original FA/FIA native launch logs.

Reads raw model/SDPA markers and NNOP launch lines. Optional-input eligibility,
host hits, strategy decisions and tensor values are never inputs to this census.
The call sidecars supply only actual SDPA argument metadata and marker identity.
No attention launch in a decode window does NOT establish a math fallback.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SCHEMA = 'nes.model_native_launch_census.v1'
MAX_LOG_BYTES = 16 * 1024 ** 3
PID = re.compile(r'\bOP\((\d+),[^)]*\)')
OP = r'OpName:\[(?P<op>[^\]]+)\]'
LEGACY_TILING = re.compile(r'\[NNOP\]\[Launch\]\[(?P<tid>\d+)\]\s+' + OP +
    r'\s+Tiling Key:\s*(?P<key>0x[0-9a-fA-F]+|\d+),\s*len:\s*(?P<size>\d+),\s*'
    r'numBlocks:\s*(?P<blocks>\d+),\s*Workspace Num:\s*(?P<workspaces>\d+)\.')
LEGACY_DISPATCH = re.compile(r'\[NNOP\]\[LaunchKernel\]\[(?P<tid>\d+)\]\s+' + OP +
    r'\s+Launch kernel engine type:\s*(?P<engine>\d+),\s*numBlocks:\s*(?P<blocks>\d+),'
    r'\s*scheduleMode:\s*(?P<schedule>\d+)')
NNOPBASE_DISPATCH = re.compile(r'\[NNOP\]\[NnopbaseExecutorLaunchKernel\]\[(?P<tid>\d+)\]\s+' + OP +
    r'\s+Launch (?:dynamic|static) kernel (?P<kernel>\S+) task,\s*tilingKey is (?P<key>0x[0-9a-fA-F]+|\d+),'
    r'\s*scheMode is (?P<schedule>\d+),\s*numBlocks is (?P<blocks>\d+)')
ANY_DISPATCH = re.compile(r'\[NNOP\]\[(?:LaunchKernel|NnopbaseExecutorLaunchKernel)\]\[\d+\]\s+' + OP)
# Detect unfamiliar logger tags/TID decorations without inventing their ABI.
# Runtime StreamLaunchKernelV2's duplicate report has no OpName or this payload.
BROAD_DISPATCH = re.compile(r'(?:NnopbaseExecutorLaunchKernel|LaunchKernel).*?' + OP +
    r'.*?(?:Launch (?:dynamic|static) kernel |Launch kernel engine type:)')
ATTENTION_END = re.compile(r'(?:^|_)(FlashAttentionScore|FusedInferAttentionScore|PromptFlashAttention|IncreFlashAttention)(?:V\d+)?(?:_\d+)?$')
PHASE_FIELDS = ('case_id', 'pid', 'thread_id', 'batch_size', 'step', 'run_index', 'mode', 'stage')
CALL_FIELDS = ('case_id', 'call_id', 'api', 'pid', 'thread_id', 'phase', 'step', 'mode',
               'batch_size', 'scope_token', 'layer_index', 'model_stage', 'run_index')
EVENTS = {'model_phase_enter', 'model_phase_exit', 'native_api_enter', 'native_api_exit'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def check_log_size(size):
    need(type(size) is int and 0 < size <= MAX_LOG_BYTES, 'Case log outside bounded16GiB contract')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_marker(text):
    # This accepts a logger prefix but not interleaved/truncated JSON suffixes.
    position = text.find('{')
    if position < 0:
        return None
    try:
        item, end = json.JSONDecoder().raw_decode(text[position:])
    except ValueError:
        need(not any('"' + name + '"' in text for name in EVENTS), 'Malformed selected marker JSON')
        return None
    if not isinstance(item, dict) or item.get('event') not in EVENTS:
        return None
    need(not text[position + end:].strip(), 'Interleaved/trailing selected marker text')
    return item


def family(op):
    normalized = op.removeprefix('aclnn')
    match = ATTENTION_END.search(normalized)
    if match:
        return 'FA' if match[1] == 'FlashAttentionScore' else 'FIA'
    # A transpose that belongs to an attention API is not an attention kernel.
    if 'Attention' in op and op.endswith('_TransposeAiCore'):
        return 'AUXILIARY_TRANSPOSE'
    return 'UNKNOWN_ATTENTION' if 'Attention' in op else None


def sidecars(case_dir):
    entries = {}
    root = Path(case_dir).resolve(strict=True)
    for path in sorted((root / 'native_calls').glob('call_*/call.json')):
        need(not path.is_symlink() and path.resolve().is_relative_to(root), 'Unsafe sidecar path')
        payload = path.read_bytes()
        item = json.loads(payload)
        native, metadata = item.get('native_call'), item.get('sdpa_metadata')
        need(isinstance(native, dict) and isinstance(metadata, dict), 'Sidecar lacks actual call/SDPA metadata')
        identity = native.get('call_id')
        need(isinstance(identity, str) and identity and identity not in entries, 'Repeated/missing sidecar call ID')
        for name in ('query', 'key', 'value'):
            row = metadata.get(name)
            need(isinstance(row, dict) and isinstance(row.get('shape'), list)
                 and len(row['shape']) == 4 and all(type(x) is int and x > 0 for x in row['shape'])
                 and isinstance(row.get('dtype'), str), 'Missing actual rank4 SDPA argument metadata')
        entries[identity] = {'native_identity': {key: native.get(key) for key in CALL_FIELDS},
                            'sdpa_metadata': metadata,
                            'metadata_source': {'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest(),
                                                'bytes': len(payload)}}
    need(entries, 'No actual call sidecar metadata found')
    return entries


def census(log_path, case_dir):
    log_path = Path(log_path)
    need(log_path.is_file() and not log_path.is_symlink(), 'Missing/symlinked original case log')
    before = log_path.stat()
    check_log_size(before.st_size)
    metadata = sidecars(case_dir)
    phases, calls, launches, tilings, auxiliary, unparsed, issues = [], [], [], [], [], [], []
    pending = {}
    active_phase = active_call = None
    ids = set()
    identity = None
    digest = hashlib.sha256()
    offset = 0

    def issue(code, **details):
        issues.append({'code': code, **details})

    def location(number, payload, text):
        return {'line_number': number, 'byte_range': [offset, offset + len(payload)],
                'line_sha256': hashlib.sha256(payload).hexdigest(), 'source_line': text}

    def binding():
        return {'step': None if active_phase is None else active_phase['identity']['step'],
                'call_id': None if active_call is None else active_call['identity']['call_id']}

    with log_path.open('rb') as stream:
        for number, payload in enumerate(stream, 1):
            digest.update(payload)
            text = payload.decode('utf-8', errors='surrogateescape').rstrip('\r\n')
            event = parse_marker(text)
            if event is not None:
                loc = location(number, payload, text)
                need(all(type(event.get(key)) is int and event[key] >= 0
                         for key in ('pid', 'thread_id', 'step', 'run_index', 'batch_size')),
                     'Marker missing exact integer identities')
                need(event['pid'] > 0 and event['thread_id'] > 0 and event['batch_size'] > 0,
                     'Marker has nonpositive process/batch identity')
                case_identity = (event.get('case_id'), event['pid'], event['batch_size'])
                need(isinstance(case_identity[0], str) and case_identity[0], 'Missing case ID')
                if identity is None: identity = case_identity
                need(identity == case_identity, 'Mixed case/process/batch markers in one original log')
                kind = event['event']
                if kind == 'model_phase_enter':
                    need(active_phase is None and active_call is None, 'Nested model phase')
                    need(event.get('stage') in ('prefill', 'decode') and event['step'] == len(phases),
                         'Model phases are not complete sequential prefill/decode steps')
                    need(event['run_index'] == event['step'] // 2 and
                         event['stage'] == ('prefill' if event['step'] % 2 == 0 else 'decode'), 'Stage/step mismatch')
                    active_phase = {'identity': {key: event.get(key) for key in PHASE_FIELDS},
                                    'enter': loc, 'calls': [], 'launches': []}
                elif kind == 'model_phase_exit':
                    need(active_phase is not None and active_call is None, 'Unmatched/early model phase exit')
                    need(event.get('status') == 'returned' and all(event.get(key) == active_phase['identity'][key]
                         for key in PHASE_FIELDS), 'Model phase exit identity/status differs')
                    active_phase['exit'] = loc
                    phases.append(active_phase)
                    active_phase = None
                elif kind == 'native_api_enter':
                    need(active_phase is not None and active_call is None, 'SDPA call outside model phase or nested')
                    need(all(event.get(key) == active_phase['identity'][key] for key in
                         ('case_id', 'pid', 'thread_id', 'batch_size', 'step', 'run_index', 'mode'))
                         and event.get('model_stage') == active_phase['identity']['stage'], 'SDPA/phase mismatch')
                    need(event.get('api') == 'torch.nn.functional.scaled_dot_product_attention'
                         and event.get('phase') == 'operator_execution', 'Unexpected wrapped API identity')
                    call_id = event.get('call_id')
                    need(isinstance(call_id, str) and call_id and call_id not in ids, 'Duplicate/missing SDPA call ID')
                    need(type(event.get('layer_index')) is int and 0 <= event['layer_index'] < 28, 'Invalid layer marker')
                    ids.add(call_id)
                    active_call = {'identity': {key: event.get(key) for key in CALL_FIELDS},
                                   'enter': loc, 'launches': []}
                else:
                    need(active_call is not None and event.get('status') == 'returned', 'Unmatched/failed SDPA exit')
                    need(all(event.get(key) == active_call['identity'][key] for key in CALL_FIELDS), 'SDPA exit identity differs')
                    active_call['exit'] = loc
                    call_id = active_call['identity']['call_id']
                    need(call_id in metadata, 'No actual SDPA metadata for a log-window call')
                    sidecar = metadata[call_id]
                    need(sidecar['native_identity'] == active_call['identity'], 'Sidecar metadata binds a different native call')
                    active_call.update({key: sidecar[key] for key in ('sdpa_metadata', 'metadata_source')})
                    q = sidecar['sdpa_metadata']['query']['shape']
                    k = sidecar['sdpa_metadata']['key']['shape']
                    active_call['query_geometry'] = {'query_length': q[2], 'key_length': k[2],
                        'classification': 'QUERY_LENGTH_1_OUTSIDE_CERTIFIED_2048_PREFILL_SCOPE' if q[2] == 1 else
                        'QUERY_LENGTH_2048_OBSERVED' if q[2] == 2048 else 'OTHER_QUERY_LENGTH_OBSERVED',
                        'basis': 'Original SDPA argument metadata; no host-event or injection-ledger input'}
                    index = len(calls)
                    calls.append(active_call)
                    active_phase['calls'].append(index)
                    active_call = None
            else:
                match = LEGACY_TILING.search(text)
                dispatch = NNOPBASE_DISPATCH.search(text) or LEGACY_DISPATCH.search(text)
                unknown = ANY_DISPATCH.search(text) or BROAD_DISPATCH.search(text)
                selected = match or dispatch or unknown
                if selected and family(selected['op']):
                    pid = PID.search(text)
                    need(pid is not None, 'Native attention line lacks process identity')
                    pid = int(pid[1])
                    loc = location(number, payload, text)
                    role = family(selected['op'])
                    row = {'family': role, 'pid': pid, 'op_name': selected['op'],
                           **binding(), 'source': loc}
                    if role == 'AUXILIARY_TRANSPOSE':
                        auxiliary.append(row)
                    elif not match and not dispatch:
                        unparsed.append(row)
                        if active_phase is not None:
                            issue('UNPARSED_NATIVE_ATTENTION_DISPATCH_IN_MODEL_PHASE', line=number, op=row['op_name'])
                    elif match:
                        row.update(tid=int(match['tid']), tiling_key=str(int(match['key'], 0) if match['key'].startswith('0x') else int(match['key'])),
                                   block_dim=int(match['blocks']), raw_size=int(match['size']), workspace_count=int(match['workspaces']))
                        row['matched_dispatch_index'] = None
                        key = (pid, row['tid'], row['op_name'])
                        pending.setdefault(key, []).append(len(tilings))
                        tilings.append(row)
                    else:
                        row.update(tid=int(dispatch['tid']), block_dim=int(dispatch['blocks']), schedule_mode=int(dispatch['schedule']))
                        row['dispatch_format'] = 'NNOPBASE' if dispatch.re is NNOPBASE_DISPATCH else 'NNOP_LEGACY'
                        row['tiling_key'], row['tiling_metadata_index'] = None, None
                        if dispatch.re is NNOPBASE_DISPATCH:
                            row['tiling_key'] = str(int(dispatch['key'], 0) if dispatch['key'].startswith('0x') else int(dispatch['key']))
                            row['kernel_name'] = dispatch['kernel']
                        else:
                            key = (pid, row['tid'], row['op_name'])
                            candidates = pending.get(key, [])
                            if candidates:
                                i = candidates.pop(0)
                                tiling = tilings[i]
                                row['tiling_metadata_index'], row['tiling_key'] = i, tiling['tiling_key']
                                tiling['matched_dispatch_index'] = len(launches)
                                if tiling['step'] != row['step'] or tiling['call_id'] != row['call_id'] or tiling['block_dim'] != row['block_dim']:
                                    issue('TILING_AND_DISPATCH_SCOPE_OR_BLOCK_MISMATCH', line=number, tiling_index=i)
                        if active_phase is not None:
                            active_phase['launches'].append(len(launches))
                            if pid != active_phase['identity']['pid']:
                                issue('FOREIGN_PID_ATTENTION_INSIDE_MODEL_PHASE', line=number, pid=pid)
                            if active_call is None:
                                issue('UNSCOPED_ATTENTION_LAUNCH_INSIDE_MODEL_PHASE', line=number, op=row['op_name'])
                            else:
                                active_call['launches'].append(len(launches))
                            if row['tiling_key'] is None:
                                issue('ATTENTION_DISPATCH_KEY_UNOBSERVED', line=number, op=row['op_name'])
                            if role == 'UNKNOWN_ATTENTION':
                                issue('UNKNOWN_ATTENTION_OPERATOR_INSIDE_MODEL_PHASE', line=number, op=row['op_name'])
                        launches.append(row)
            offset += len(payload)
    after = log_path.stat()
    need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns) and offset == before.st_size,
         'Original case log changed during analysis')
    need(active_phase is None and active_call is None and phases and len(phases) % 2 == 0, 'Incomplete model/SDPA windows')
    need(set(metadata) == ids, 'Sidecar set and whole-log SDPA call set differ')
    for row in launches:
        if row['step'] is None and row['pid'] == identity[1]:
            issue('SAME_PROCESS_ATTENTION_DISPATCH_OUTSIDE_MODEL_PHASES',
                  line=row['source']['line_number'], op=row['op_name'])
    for row in unparsed:
        if row['step'] is None and row['pid'] == identity[1]:
            issue('SAME_PROCESS_UNPARSED_ATTENTION_DISPATCH_OUTSIDE_MODEL_PHASES',
                  line=row['source']['line_number'], op=row['op_name'])
    for index, row in enumerate(tilings):
        if row['step'] is not None and row['matched_dispatch_index'] is None:
            issue('TILING_METADATA_WITHOUT_ACTUAL_DISPATCH_INSIDE_MODEL_PHASE', tiling_index=index)
    for phase in phases:
        selected = [calls[i] for i in phase['calls']]
        actual = [launches[i] for i in phase['launches']]
        step, stage = phase['identity']['step'], phase['identity']['stage']
        if len(selected) != 28 or [row['identity']['layer_index'] for row in selected] != list(range(28)):
            issue('INCOMPLETE_OR_REPEATED_28_LAYER_SDPA_WINDOWS', step=step, call_count=len(selected))
        if stage == 'prefill':
            if len(actual) != 28 or any(row['family'] != 'FA' for row in actual):
                issue('PREFILL_NOT_EXACTLY_28_ACTUAL_FA_DISPATCHES', step=step,
                      counts=dict(Counter(row['family'] for row in actual)))
            for row in selected:
                own = [launches[i] for i in row['launches']]
                if len(own) != 1 or own[0]['family'] != 'FA':
                    issue('PREFILL_WRAPPER_NOT_EXACTLY_ONE_ACTUAL_FA_DISPATCH', call_id=row['identity']['call_id'])
                if row['query_geometry']['query_length'] != 2048:
                    issue('PREFILL_ACTUAL_QUERY_LENGTH_NOT_2048', call_id=row['identity']['call_id'])
        else:
            for row in selected:
                if row['query_geometry']['query_length'] != 1 or row['query_geometry']['key_length'] != 2049:
                    issue('DECODE_ACTUAL_Q1_K2049_NOT_ESTABLISHED', call_id=row['identity']['call_id'])
        phase['native_dispatch_counts'] = dict(Counter(row['family'] for row in actual))
        phase['original_sdpa_window_count'] = len(selected)
    for call in calls:
        owned = [launches[i] for i in call['launches']]
        call['observed_route'] = '+'.join(row['family'] for row in owned) if owned else 'NO_NATIVE_ATTENTION_DISPATCH_OBSERVED'
    return {'schema': SCHEMA, 'status': 'PASS_NATIVE_LAUNCH_CENSUS' if not issues else 'FAIL_NATIVE_LAUNCH_CENSUS',
            'case_id': identity[0], 'pid': identity[1], 'batch_size': identity[2],
            'source_log': {'path': str(log_path.resolve()), 'sha256': digest.hexdigest(), 'bytes': offset},
            'analysis_source_sha256': sha(Path(__file__)), 'issues': issues,
            'summary': {'phases': len(phases), 'original_sdpa_windows': len(calls),
                'attention_dispatches': len(launches),
                'dispatches_inside_model_phases': sum(row['step'] is not None for row in launches),
                'dispatches_outside_model_phases': sum(row['step'] is None for row in launches),
                'unscoped_dispatches_inside_model_phases': sum(row['step'] is not None and row['call_id'] is None for row in launches),
                'decode_route_counts': dict(Counter(row['observed_route'] for row in calls if row['identity']['model_stage'] == 'decode')),
                'independently_observed_q1_calls': sum(row['query_geometry']['query_length'] == 1 for row in calls)},
            'phases': phases, 'calls': calls, 'attention_dispatches': launches, 'legacy_tiling_metadata': tilings,
            'auxiliary_transpose_lines': auxiliary, 'unparsed_attention_dispatches': unparsed,
            'limits': ['This is an execution-log census, not an injection-effect or numerical acceptance.',
                'A decode window without native attention dispatch does not identify its math implementation.',
                'Runtime StreamLaunchKernel lines are secondary logs of the same launch and are not counted twice.',
                'No host hit, requested profile, strategy verdict, private tensor or preset first-difference is read.',
                'Raw tiling byte/POD validation and independent msprobe/module denominators are separate checks.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), 'Census output must be a new file')
    try:
        report = census(args.log, args.case_dir)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report = {'schema': SCHEMA, 'status': 'FAIL_NATIVE_LAUNCH_CENSUS',
                  'analysis_source_sha256': sha(Path(__file__)), 'error': str(exc),
                  'source_log': {'path': str(args.log), 'sha256': sha(args.log) if args.log.is_file() else None}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': report['status'], 'sha256': sha(args.output), 'summary': report.get('summary'),
                      'error': report.get('error'), 'issue_count': len(report.get('issues', []))}))
    return 0 if report['status'] == 'PASS_NATIVE_LAUNCH_CENSUS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
