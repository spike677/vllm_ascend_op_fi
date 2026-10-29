"""Extract actual legacy NNOP FA launch and logged uint32 tiling payload.

No buffer offsets or tiling schema are guessed. The declared-length prefix is
only a framing candidate until it matches a public-context raw capture from
the rebuilt OFF host. Aligned bytes and every source line remain auditable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

PID = re.compile(r'\bOP\((\d+),[^)]*\)')
OPNAME = r'(?P<op>aclnnFlashAttentionScore_\d+_FlashAttentionScore)'
LAUNCH = re.compile(r'\[NNOP\]\[Launch\]\[(?P<tid>\d+)\]\s+OpName:\[' + OPNAME +
    r'\]\s+Tiling Key:\s*(?P<key>0x[0-9a-fA-F]+|\d+),\s*len:\s*(?P<size>\d+),\s*'
    r'numBlocks:\s*(?P<blocks>\d+),\s*Workspace Num:\s*(?P<workspaces>\d+)\.')
RAW = re.compile(r'\[SplitDataAndPrint\]\[(?P<tid>\d+)\]\s+OpName:\[' + OPNAME +
    r'\]\s+PrintTilingData\[(?P<index>\d+)\]:\s*'
    r'(?:Aligned TilingData Size:\s*(?P<aligned>\d+),\s*data\(uint32_t\):\s*)?(?P<words>[\d\s]+)$')
WORKSPACE = re.compile(r'\[CommonOpExecutorRun\]\[(?P<tid>\d+)\]\s+'
    r'OpName:\[(?P<op>aclnnFlashAttentionScore_\d+)\]\s+Workspace addr: \S+,\s*size:\s*(?P<size>\d+),')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def marker(text):
    position = text.find('{')
    if position < 0:
        return None
    try:
        row, end = json.JSONDecoder().raw_decode(text[position:])
    except ValueError:
        return None
    if text[position + end:].strip() or not isinstance(row, dict):
        return None
    return row if row.get('event') in ('native_api_enter', 'native_api_exit') else None


def extract(lines, case_id, expected_calls):
    active, calls = None, []
    for number, payload in enumerate(lines, 1):
        text = payload.decode('utf-8', errors='surrogateescape').rstrip('\r\n')
        event = marker(text)
        if event:
            require(event['case_id'] == case_id and event['rank'] == 0, 'Marker case/rank differs')
            require(event['call_id'] == f"pid{event['pid']}:r0:n{event['native_index']}", 'Marker identity differs')
            if event['event'] == 'native_api_enter':
                require(active is None, 'Overlapping native calls')
                active = {'enter': event, 'enter_line': number, 'launches': [], 'chunks': [], 'api_workspaces': []}
            else:
                require(active is not None and event.get('status') == 'returned', 'Missing or failed native call')
                require(all(active['enter'][key] == event[key] for key in
                    ('pid', 'thread_id', 'case_id', 'rank', 'call_id', 'native_index', 'layer_index', 'layer_name', 'api')),
                    'Native enter/exit differ')
                active.update(exit=event, exit_line=number)
                calls.append(active)
                active = None
            continue
        match = LAUNCH.search(text) or RAW.search(text) or WORKSPACE.search(text)
        if match is None:
            continue
        require(not any('\udc80' <= char <= '\udcff' for char in text), 'Non-UTF8 FA evidence line')
        require(active is not None, 'FA evidence outside a native marker window')
        pid = PID.search(text)
        require(pid is not None and int(pid[1]) == active['enter']['pid'], 'Native log PID differs')
        row = {'line': number, 'text': text, **match.groupdict()}
        if match.re is LAUNCH:
            active['launches'].append(row)
        elif match.re is RAW:
            active['chunks'].append(row)
        else:
            active['api_workspaces'].append(row)
    require(active is None, 'Unterminated native call')
    require([c['enter']['call_id'] for c in calls] == [c['call_id'] for c in expected_calls], 'Caller ledger mismatch')
    for call, entry in zip(calls, expected_calls):
        require(all(call['enter'][key] == entry[key] for key in
            ('call_id', 'native_index', 'layer_index', 'layer_name', 'api')), 'Caller metadata differs')
        require(len(call['launches']) == 1, 'Expected exactly one actual FA kernel launch')
        launch = call['launches'][0]
        require(call['chunks'] and all(c['op'] == launch['op'] and c['tid'] == launch['tid'] for c in call['chunks']),
                'Raw chunks are not from the launched FA kernel')
        require([int(c['index']) for c in call['chunks']] == list(range(len(call['chunks']))), 'Missing/duplicate raw chunk')
        aligned = int(call['chunks'][0]['aligned'])
        size = int(launch['size'])
        require(0 < size <= aligned <= 1024 * 1024 and aligned % 4 == 0, 'Invalid bounded tiling lengths')
        require(all(c['aligned'] is None for c in call['chunks'][1:]), 'Unexpected raw header repetition')
        words = [int(word) for c in call['chunks'] for word in c['words'].split()]
        require(len(words) * 4 == aligned and all(0 <= word <= 0xffffffff for word in words), 'Raw word count/range differs')
        parent = launch['op'].removesuffix('_FlashAttentionScore')
        workspaces = [w for w in call['api_workspaces'] if w['op'] == parent and w['tid'] == launch['tid']]
        require(len(workspaces) == 1, 'Expected unique enclosing API workspace record')
        call['_aligned_payload'] = struct.pack('<' + str(len(words)) + 'I', *words)
        call.update(tiling_key=str(int(launch['key'], 0) if launch['key'].startswith('0x') else int(launch['key'])),
                    declared_raw_bytes=size, aligned_log_bytes=aligned, block_dim=int(launch['blocks']),
                    kernel_workspace_count=int(launch['workspaces']), api_workspace_bytes=int(workspaces[0]['size']),
                    framing_verified=False, schema_decoded=False,
                    scope='Actual FA launch; raw framing candidate pending public-context byte comparison. API workspace includes helper operators.')
    return calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--case-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output must be fresh')
    before = {'log': sha(args.log), 'result': sha(args.case_result)}
    result = json.loads(args.case_result.read_bytes())
    require(result['status'] == 'CAPTURE_COMPLETED', 'Case capture was not completed')
    with args.log.open('rb') as stream:
        calls = extract(stream, result['case_id'], result['native_calls'])
    require(before == {'log': sha(args.log), 'result': sha(args.case_result)}, 'Source evidence changed')
    args.output.mkdir(parents=True)
    for index, call in enumerate(calls):
        aligned = call.pop('_aligned_payload')
        prefix = aligned[:call['declared_raw_bytes']]
        name = 'call_' + str(index).zfill(3)
        for suffix, data in (('.aligned_log.bin', aligned), ('.declared_prefix.bin', prefix)):
            path = args.output / (name + suffix)
            path.write_bytes(data)
            call[suffix[1:] + '_sha256'] = hashlib.sha256(data).hexdigest()
        call['prefix_file'] = name + '.declared_prefix.bin'
    report = {'schema': 'fault28.fa.native_log_profile.v1', 'status': 'CAPTURED_UNDECODED_FRAMING_CANDIDATE',
              'case_id': result['case_id'], 'source_sha256': before, 'extractor_sha256': sha(Path(__file__)),
              'call_count': len(calls), 'calls': calls, 'mechanism_accepted': False}
    (args.output / 'profile.json').write_text(json.dumps(report, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'case_id': result['case_id'], 'call_count': len(calls),
                      'keys': sorted({c['tiling_key'] for c in calls})}))


if __name__ == '__main__':
    main()
