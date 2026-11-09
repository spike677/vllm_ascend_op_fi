"""Read actual CANN FA launch lines inside one synchronized original API call.

Regexes follow the already observed CANN 9.1 legacy NNOP log format. The raw
prefix is not called verified until compared with the public host callback.
No framework imports, NPU operations or private executor offsets are used.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import struct

PID = re.compile(r'\bOP\((\d+),[^)]*\)')
OPNAME = r'(?P<op>aclnnFlashAttentionScore_\d+_FlashAttentionScore)'
LAUNCH = re.compile(r'\[NNOP\]\[Launch\]\[(?P<tid>\d+)\]\s+OpName:\[' + OPNAME +
    r'\]\s+Tiling Key:\s*(?P<key>0x[0-9a-fA-F]+|\d+),\s*len:\s*(?P<size>\d+),\s*'
    r'numBlocks:\s*(?P<blocks>\d+),\s*Workspace Num:\s*(?P<workspaces>\d+)\.')
RAW_IDENT = re.compile(r'\[SplitDataAndPrint\]\[\d+\]\s+OpName:\[' + OPNAME +
    r'\]\s+PrintTilingData\[\d+\]')
# CANN splits the decimal text by characters, including inside one number.
# Consume exactly the logger's fixed separator, retaining payload whitespace.
# A second leading space can belong to the payload and must not be stripped.
RAW = re.compile(r'\[SplitDataAndPrint\]\[(?P<tid>\d+)\]\s+OpName:\[' + OPNAME +
    r'\]\s+PrintTilingData\[(?P<index>\d+)\]: '
    r'(?:Aligned TilingData Size: (?P<aligned>\d+), data\(uint32_t\): )?(?P<words>[0-9 ]+)$')
WORKSPACE = re.compile(r'\[CommonOpExecutorRun\]\[(?P<tid>\d+)\]\s+'
    r'OpName:\[(?P<op>aclnnFlashAttentionScore_\d+)\]\s+Workspace addr: \S+,\s*size:\s*(?P<size>\d+),')
FORMATS = {'int64_t': 'q', 'uint64_t': 'Q', 'int32_t': 'i', 'uint32_t': 'I',
           'int16_t': 'h', 'uint16_t': 'H', 'int8_t': 'b', 'uint8_t': 'B', 'float': 'f'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def marker(text):
    start = text.find('{')
    if start < 0:
        return None
    try:
        value, end = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    if text[start + end:].strip() or not isinstance(value, dict):
        return None
    return value if value.get('event') in ('native_api_enter', 'native_api_exit') else None


def decode_pod(raw, schema):
    require(schema['size'] == len(raw) and schema['type'] == 'FlashAttentionScoreGeneralTilingData',
            'Actual launch does not match the approved public POD schema')
    result = {}
    for item in schema['fields']:
        fmt = '<' + str(item['count']) + FORMATS[item['type']]
        size = struct.calcsize(fmt)
        require(size == item['count'] * item['width'] and item['name'] not in result
                and 0 <= item['offset'] <= len(raw) - size, 'Invalid public POD field')
        value = struct.unpack_from(fmt, raw, item['offset'])
        result[item['name']] = value[0] if len(value) == 1 else list(value)
    require(len(result) == 205, 'Expected all 205 public fields from the approved decoder build')
    return result


def actual_launch(log_path, call, output):
    """Save one immutable byte window from a live append-only controller log.

    Only bytes through this call's flushed exit marker are examined. Later
    calls may append to the parent log without changing this completed window.
    """
    active, completed = False, False
    chunks, launches, workspaces, window, loads = [], [], [], [], []
    prefix = hashlib.sha256()
    offset, start_offset, end_offset, start_line, end_line = 0, None, None, None, None
    with Path(log_path).open('rb') as stream:
        for number, payload in enumerate(stream, 1):
            prefix.update(payload)
            text = payload.decode('utf-8', 'surrogateescape').rstrip('\r\n')
            event = marker(text)
            if 'LoadOppPackage' in text and 'load so, so_name is ' in text:
                loads.append({'line': number, 'path': text.split('load so, so_name is ', 1)[1].strip().replace('//', '/'),
                              'line_sha256': hashlib.sha256(payload).hexdigest()})
            if event is not None and event.get('call_id') == call['call_id']:
                for key in ('call_id', 'case_id', 'api', 'pid', 'thread_id', 'scope_token', 'phase', 'batch_size', 'step'):
                    require(event.get(key) == call.get(key), 'Marker identity differs: ' + key)
                if event['event'] == 'native_api_enter':
                    require(not active and start_offset is None, 'Duplicate selected call entry')
                    active = True
                    start_offset, start_line = offset, number
                else:
                    require(active and event.get('status') == 'returned', 'Missing or failed selected call')
                    window.append(payload)
                    end_offset, end_line = offset + len(payload), number
                    completed = True
                    break
            if active:
                window.append(payload)
                require(not RAW_IDENT.search(text) or RAW.search(text),
                        'Malformed selected native raw record or fixed delimiter')
                match = LAUNCH.search(text) or RAW.search(text) or WORKSPACE.search(text)
                if match:
                    require(not any('\udc80' <= x <= '\udcff' for x in text), 'Non-UTF8 native FA evidence')
                    pid = PID.search(text)
                    require(pid and int(pid[1]) == call['pid'], 'Native evidence PID differs')
                    row = {'line': number, 'line_sha256': hashlib.sha256(payload).hexdigest(), **match.groupdict()}
                    (launches if match.re is LAUNCH else chunks if match.re is RAW else workspaces).append(row)
            offset += len(payload)
    require(completed and len(launches) == 1, 'Incomplete call window or ambiguous native FA launch')
    launch = launches[0]
    require(chunks and all(x['op'] == launch['op'] and x['tid'] == launch['tid'] for x in chunks), 'Raw launch identity differs')
    # CANN can launch on a worker TID distinct from the Python/host-callback TID.
    # PID and call framing bind this invocation; launch/raw/workspace must agree
    # with each other, while verified_record separately binds host TID to caller.
    require([int(x['index']) for x in chunks] == list(range(len(chunks))), 'Missing/repeated raw chunk')
    size, aligned = int(launch['size']), int(chunks[0]['aligned'])
    require(0 < size <= aligned <= 1024 * 1024 and aligned % 4 == 0, 'Unexpected raw framing sizes')
    require(all(x['aligned'] is None for x in chunks[1:]), 'Repeated raw header')
    # Do not split each chunk: e.g. "102" followed by "4 " is one uint32 1024.
    # Do not insert or strip whitespace: that changes real word boundaries.
    raw_text = ''.join(item['words'] for item in chunks)
    words = [int(word) for word in raw_text.split()]
    require(len(words) * 4 == aligned and all(0 <= x <= 0xffffffff for x in words), 'Malformed raw uint32 payload')
    raw = struct.pack('<' + str(len(words)) + 'I', *words)[:size]
    workspaces = [x for x in workspaces if x['op'] == launch['op'].removesuffix('_FlashAttentionScore') and x['tid'] == launch['tid']]
    require(len(workspaces) == 1, 'Missing/ambiguous enclosing API workspace')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(b''.join(window))
    return {'raw_hex': raw.hex(), 'raw_size': len(raw), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
            'tiling_key': str(int(launch['key'], 0) if launch['key'].startswith('0x') else int(launch['key'])),
            'block_dim': int(launch['blocks']), 'workspace_count': int(launch['workspaces']),
            'api_workspace_bytes': int(workspaces[0]['size']), 'pid': call['pid'], 'tid': int(launch['tid']),
            'op_name': launch['op'], 'source_log': str(log_path), 'source_prefix_sha256': prefix.hexdigest(),
            'raw_reconstruction': 'exact_character_concat_before_uint32_split',
            'raw_chunk_payload_lengths': [len(item['words']) for item in chunks],
            'raw_text_sha256': hashlib.sha256(raw_text.encode('ascii')).hexdigest(),
            'source_prefix_bytes': end_offset, 'window_byte_range': [start_offset, end_offset],
            'window_line_range': [start_line, end_line], 'window_file': str(output), 'window_sha256': sha(output),
            'loaded_libraries_before_exit': loads, 'framing_verified_against_public_context': False}


def bound_resources(metadata_path, call):
    if metadata_path is None:
        return {'device_seconds': None, 'peak_memory_bytes': None}, {'accounting_status': 'RESOURCE_ACCOUNTING_UNKNOWN'}, []
    path = Path(metadata_path)
    metadata = json.loads(path.read_bytes())
    require(metadata['schema'] == 'nes.harness.step_metadata.v1' and metadata['native_call'] == call,
            'Resource measurement does not bind this exact synchronized call')
    measurement = metadata['resource_measurement']
    resources = measurement['resources']
    require(set(resources) == {'device_seconds', 'peak_memory_bytes'}, 'Unexpected resource usage fields')
    seconds, peak = resources['device_seconds'], resources['peak_memory_bytes']
    require(seconds is None or type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
            'Invalid call-window duration')
    require(peak is None or type(peak) is int and peak >= 0, 'Invalid allocator peak')
    if seconds is not None:
        require(call.get('status') == 'returned'
                and seconds == (call['exit_monotonic_ns'] - call['enter_monotonic_ns']) / 1_000_000_000,
                'Resource duration differs from synchronized call markers')
    if peak is not None:
        observations = measurement['allocator_observations']
        require(all(type(observations.get(key)) is int and observations[key] >= 0
                    for key in ('memory_stats_peak_bytes', 'max_memory_allocated_bytes')),
                'Invalid allocator observation counter')
        require(measurement['peak_reset_completed'] is True
                and measurement['reset_outside_msprobe_capture'] is True
                and measurement['sample_outside_msprobe_capture'] is True
                and peak == max(observations['memory_stats_peak_bytes'], observations['max_memory_allocated_bytes']),
                'Allocator peak is not bound to the reset/sample observations')
    reference = {'step_metadata_file': str(path), 'step_metadata_sha256': sha(path),
                 'resource_meter_source_sha256': metadata['resource_meter_source_sha256'],
                 'accounting_status': measurement['accounting_status'],
                 'device_seconds_semantics': measurement['device_seconds_semantics'],
                 'allocator_window_semantics': measurement['allocator_window_semantics']}
    return resources, reference, [{'path': str(path), 'sha256': sha(path)}]


def verified_record(log_path, call, output, host_path, schema_path, expected_schema_sha, resource_metadata_path=None, *, launch_reader=None):
    require(sha(schema_path) == expected_schema_sha, 'Approved POD schema changed')
    reader = actual_launch if launch_reader is None else launch_reader.actual_launch
    native = reader(log_path, call, output / 'native_window.log')
    schema = json.loads(Path(schema_path).read_bytes())
    pod = decode_pod(bytes.fromhex(native['raw_hex']), schema)
    original_host = Path(host_path).read_bytes()
    events = [json.loads(line) for line in original_host.decode('utf-8').splitlines() if line.strip()]
    matched = [event for event in events if event.get('token') == call['scope_token']]
    require(len(matched) == 1, 'Exactly one native host event per scoped original call required')
    event = matched[0]
    require(event['schema'] == 'nes.fa.host_observation.v1' and event['callback_status'] == 0, 'Missing/failed native host callback')
    require(event['pid'] == call['pid'] and event['tid'] == call['thread_id'], 'Native host PID/TID differs')
    require(event['raw_hex'] == native['raw_hex'] and event['raw_size'] == native['raw_size'], 'Host raw differs from actual launched bytes')
    for key in ('tiling_key', 'block_dim', 'workspace_count'):
        require(str(event[key]) == str(native[key]), 'Host/launch identity differs: ' + key)
    native['framing_verified_against_public_context'] = True
    host_snapshot = output / 'host_event.json'
    with host_snapshot.open('x', encoding='utf-8') as stream:
        json.dump(event, stream, indent=2, allow_nan=False)
        stream.write('\n')
    host_original_path = output / 'host_events.original.jsonl'
    with host_original_path.open('xb') as stream:
        stream.write(original_host)
    schema_snapshot = output / 'public_pod_schema.json'
    with schema_snapshot.open('xb') as stream:
        stream.write(Path(schema_path).read_bytes())
    require(sha(schema_snapshot) == expected_schema_sha, 'Schema copy changed')
    resources, resource_reference, resource_evidence = bound_resources(resource_metadata_path, call)
    record = {'host_event': event, 'native_launch': native, 'pod': pod,
              'evidence': [{'path': str(output / 'native_window.log'), 'sha256': sha(output / 'native_window.log')},
                           {'path': str(host_original_path), 'sha256': sha(host_original_path)},
                           {'path': str(schema_snapshot), 'sha256': expected_schema_sha}, *resource_evidence],
              'resources': resources, 'resource_measurement': resource_reference}
    return record
