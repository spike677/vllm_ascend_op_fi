"""Read-only binding of one replay call to public typed/raw and native FA launch.

No desired profile or override flag is used to derive execution geometry.
Private executor memory offsets are never read. The native log prefix becomes
framed evidence only after exact equality with the public TilingContext bytes.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def module(path, name, expected=None):
    if expected is not None:
        require(sha(path) == expected, 'Audited helper SHA mismatch: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def single_native_log(path, result, parser):
    before = sha(path)
    active, calls, loads = None, [], []
    for line_no, payload in enumerate(Path(path).open('rb'), 1):
        text = payload.decode('utf-8', 'surrogateescape').rstrip('\r\n')
        marker = parser.marker(text)
        if marker:
            require(not any('\udc80' <= x <= '\udcff' for x in text), 'Non-UTF8 marker')
            expected = result['native_call']
            for k in ('case_id', 'api', 'pid', 'thread_id', 'scope_token', 'phase', 'batch_size'):
                require(marker[k] == expected[k], 'Caller marker mismatch: '+k)
            if marker['event'] == 'native_api_enter':
                require(active is None and not calls, 'Multiple/overlapping original replay calls')
                active = {'enter': marker, 'enter_line': line_no, 'launches': [], 'chunks': [], 'workspaces': []}
            else:
                require(active is not None and marker['status'] == 'returned', 'Missing/failed original call')
                active.update(exit=marker, exit_line=line_no)
                calls.append(active)
                active = None
            continue
        if 'LoadOppPackage' in text and 'load so, so_name is ' in text:
            loads.append({'line': line_no, 'path': text.split('load so, so_name is ', 1)[1].strip().replace('//','/'),
                          'line_sha256': hashlib.sha256(payload).hexdigest()})
        match = parser.LAUNCH.search(text) or parser.RAW.search(text) or parser.WORKSPACE.search(text)
        if not match:
            continue
        require(active is not None, 'Actual FA evidence outside synchronized replay call')
        pid = parser.PID.search(text)
        require(pid and int(pid[1]) == active['enter']['pid'], 'Native launch/raw PID mismatch')
        require(not any('\udc80' <= x <= '\udcff' for x in text), 'Non-UTF8 native FA evidence')
        row = {'line': line_no, 'line_sha256': hashlib.sha256(payload).hexdigest(), **match.groupdict()}
        active['launches' if match.re is parser.LAUNCH else 'chunks' if match.re is parser.RAW else 'workspaces'].append(row)
    require(active is None and len(calls) == 1, 'Need exactly one completed original native call')
    row = calls[0]
    require(len(row['launches']) == 1, 'Need exactly one actual FA kernel')
    launch = row['launches'][0]
    chunks = row['chunks']
    require(chunks and all(x['op'] == launch['op'] and x['tid'] == launch['tid'] for x in chunks), 'Raw launch identity differs')
    require([int(x['index']) for x in chunks] == list(range(len(chunks))), 'Missing/repeated raw chunk')
    size, aligned = int(launch['size']), int(chunks[0]['aligned'])
    require(size == 1464 and size <= aligned <= 4096 and aligned % 4 == 0, 'Unexpected raw framing sizes')
    require(all(x['aligned'] is None for x in chunks[1:]), 'Repeated raw header')
    words = [int(x) for chunk in chunks for x in chunk['words'].split()]
    require(len(words)*4 == aligned and all(0 <= x <= 0xffffffff for x in words), 'Malformed raw word payload')
    raw = struct.pack('<'+str(len(words))+'I', *words)[:size]
    workspace = [x for x in row['workspaces'] if x['op'] == launch['op'].removesuffix('_FlashAttentionScore') and x['tid'] == launch['tid']]
    require(len(workspace) == 1, 'Missing/ambiguous enclosing API workspace')
    require(sha(path) == before, 'Log changed during read-only analysis')
    return {'source_sha256': before, 'raw': raw, 'tiling_key': str(int(launch['key'], 0) if launch['key'].startswith('0x') else int(launch['key'])),
            'block_dim': int(launch['blocks']), 'workspace_count': int(launch['workspaces']),
            'api_workspace_bytes': int(workspace[0]['size']), 'launch': launch, 'enter_line': row['enter_line'],
            'exit_line': row['exit_line'], 'raw_lines': [x['line'] for x in chunks], 'loads': loads}


FORMATS = {'int64_t': 'q', 'uint64_t': 'Q', 'int32_t': 'i', 'uint32_t': 'I',
           'int16_t': 'h', 'uint16_t': 'H', 'int8_t': 'b', 'uint8_t': 'B', 'float': 'f'}


def decode(raw, schema):
    require(schema['size'] == len(raw) == 1464 and schema['type'] == 'FlashAttentionScoreGeneralTilingData', 'Unapproved FA POD geometry')
    values = {}
    for item in schema['fields']:
        fmt = '<'+str(item['count'])+FORMATS[item['type']]
        require(struct.calcsize(fmt) == item['count']*item['width'], 'POD declared width differs')
        require(item['name'] not in values and 0 <= item['offset'] <= len(raw)-struct.calcsize(fmt), 'Duplicate/out-of-bounds POD field')
        value = struct.unpack_from(fmt, raw, item['offset'])
        values[item['name']] = value[0] if len(value) == 1 else list(value)
    return values


def verified_profile(folder, log, result, schema, util, parser, oracle):
    event_path = folder/'host_events.jsonl'
    events = [json.loads(x) for x in event_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    require(len(events) == 1, 'Exactly one public host callback expected')
    event = events[0]
    require(event['schema'] == 'fault28.causal.typed_host.v1' and event['callback_status'] == 0, 'Wrong or failed public host record')
    call = result['native_call']
    require(event['pid'] == call['pid'] and event['tid'] == call['thread_id'] and event['token'] == call['scope_token'], 'Public callback/caller scope mismatch')
    require(event['phase'] == 'operator_replay' and int(event['scope_batch']) == result['batch_size'], 'Host scope shape/phase differs')
    native = single_native_log(log, result, parser)
    raw = bytes.fromhex(event['raw_hex'])
    require(event['raw_size'] == len(raw) and raw == native['raw'], 'Public raw differs from actual launched tiling')
    for key in ('tiling_key', 'block_dim', 'workspace_count'):
        require(str(event[key]) == str(native[key]), 'Actual launch '+key+' differs')
    require(event['workspace_count'] == len(event['workspace_sizes']) == 1 and event['schedule_mode'] == 0, 'Unapproved native workspace/schedule')
    pod = decode(raw, schema)
    require(not util.typed_matches(event['typed'], pod), 'Public typed fields disagree with raw POD')
    # The expanded SDK projection exposes nested TCubeTiling rather than
    # assigning guessed offsets into a library-owned structure.
    bmm_fields = {}
    for section in ('bmm1', 'bmm2'):
        observed = event['typed'].get(section)
        require(isinstance(observed, dict) and len(observed) == 50, 'Missing public TCubeTiling fields: '+section)
        matches = {}
        for name, value in observed.items():
            candidates = [section+'TilingData.'+name, section+'.'+name]
            keys = [k for k in candidates if k in pod]
            require(len(keys) == 1 and pod[keys[0]] == value, 'BMM typed/raw field differs: '+section+'.'+name)
            matches[name] = value
        bmm_fields[section] = matches
    # The new record retains the old audited geometric fields; adapt schema
    # name only in memory. Neither control intent nor hits reaches the oracle.
    geometric = {k: event[k] for k in ('callback_status','workspace_count','workspace_sizes','raw_size','raw_hex','typed','tiling_key')}
    geometric['schema'] = 'fault28.fa.typed_host.v1'
    trace = oracle.derive(geometric, target_batch=0, target_query=2047)
    root = result['vendor']['actual_custom_opp_path'].rstrip('/')
    loaded = {x['path'] for x in native['loads']}
    require(root+'/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so' in loaded and
            root+'/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so' in loaded, 'No actual approved custom library load evidence')
    return {'event': event, 'pod': pod, 'trace': trace, 'bmm': bmm_fields,
            'raw_sha256': hashlib.sha256(raw).hexdigest(), 'host_events_sha256': sha(event_path),
            'native': {k:v for k,v in native.items() if k != 'raw'},
            'profile_fingerprint': hashlib.sha256(raw+str(event['tiling_key']).encode()+str(event['schedule_mode']).encode()+json.dumps(event['workspace_sizes']).encode()).hexdigest()}
