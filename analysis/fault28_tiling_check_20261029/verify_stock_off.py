"""Read-only stock/private-OFF compatibility gate. No torch, NPU, or hit-led localization.

The stock process has one forward. Each OFF forward is compared against it.
Numerical comparisons are performed before consulting host control events.
Only target batch-slot values are covered; full-batch integrity is a separate
OFF-side check when available. Equal SHA is a recorded bitwise-value witness,
not an independent recomputation of unavailable private tensor bytes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def contained(root, relative):
    p = (root / relative).resolve()
    require(p.is_relative_to(root.resolve()), 'Evidence path escapes case directory')
    return p


RECORD_FIELDS = ('shape', 'stride', 'dtype', 'storage_offset', 'actual_format',
                 'hashed_slice', 'logical_sha256', 'logical_bytes', 'byteorder', 'raw_order')


def record_projection(record):
    require(isinstance(record, dict), 'Missing tensor observation')
    require(all(k in record for k in RECORD_FIELDS), 'Incomplete tensor observation')
    require(record['dtype'] == 'torch.float16', 'Expected FP16 native value')
    require(re.fullmatch('[0-9a-f]{64}', record['logical_sha256']) is not None, 'Invalid logical SHA')
    shape = record['shape']
    require(shape and all(type(x) is int and x > 0 for x in shape), 'Invalid shape')
    target = list(shape)
    if record['hashed_slice'] == 'batch[0:1], all heads/positions':
        target[0] = 1
    else:
        require(record['hashed_slice'] == 'entire logical tensor', 'Unknown hashed slice')
    require(record['logical_bytes'] == math.prod(target) * 2, 'Hash byte count does not cover declared slice')
    require(record['byteorder'] == 'little' and record['raw_order'] == 'C-contiguous logical elements', 'Unknown value serialization')
    return {k: record[k] for k in RECORD_FIELDS}


def integrity(before, after):
    """Within one API call keep storage identity; allow only exact 0 -> 2 format."""
    a, b = record_projection(before), record_projection(after)
    transition = [a.pop('actual_format'), b.pop('actual_format')]
    require(a == b, 'API changed logical input or geometry')
    require(transition[0] == transition[1] or transition == [0, 2], 'Unapproved native format transition')
    for key in ('object_id', 'tensor_ptr', 'storage_offset'):
        require(before.get(key) is not None and before[key] == after.get(key), 'API changed input identity: ' + key)
    return transition


def load_calls(folder, result, evidence):
    rows = []
    require(result['status'] == 'CAPTURE_COMPLETED', 'Runtime capture incomplete')
    require(result['prefill_certified'] is True and result['all_requests_in_one_prefill'] is True, 'Full prefill not certified')
    for index, item in enumerate(result['native_calls']):
        require(item['native_index'] == index, 'Non-contiguous native ledger')
        p = contained(folder, item['file'])
        digest = sha_file(p)
        require(digest == item['sha256'], 'Native call file SHA mismatch')
        evidence[str(p)] = digest
        call = read_json(p)
        for field in ('native_index', 'call_id', 'forward_index', 'layer_index', 'layer_name', 'api', 'status'):
            require(call[field] == item[field], 'Native ledger/call mismatch: ' + field)
        require(call['status'] == 'returned' and call['original_api_calls'] == 1, 'Original SDPA did not return exactly once')
        require(call['target_mapping_certified'] is True and call['rank'] == 0, 'Uncertified target mapping')
        require(call['api'] == 'torch.nn.functional.scaled_dot_product_attention', 'Unexpected API')
        rows.append(call)
    expected = [(f, layer) for f in range(result['forward_count']) for layer in range(28)]
    require([(c['forward_index'], c['layer_index']) for c in rows] == expected, 'Need 28 ordered layers per forward')
    return rows


def compare_calls(stock, off):
    differences, input_transitions = [], {}
    for field in ('source_sha256', 'api', 'layer_index', 'layer_name', 'actual_layout', 'target_mapping', 'scalar_arguments'):
        if stock[field] != off[field]:
            differences.append({'field': field, 'kind': 'metadata'})
    for name in ('query', 'key', 'value'):
        transitions = []
        for label, call in (('stock', stock), ('off', off)):
            transitions.append(integrity(call['target_inputs_before'][name], call['target_inputs_after'][name]))
        input_transitions[name] = transitions
        if transitions[0] != transitions[1]:
            differences.append({'field': name + '.format_transition', 'kind': 'metadata'})
        for phase in ('target_inputs_before', 'target_inputs_after'):
            a, b = record_projection(stock[phase][name]), record_projection(off[phase][name])
            for field in RECORD_FIELDS:
                if a[field] != b[field]:
                    differences.append({'field': phase + '.' + name + '.' + field,
                                        'kind': 'value_hash' if field == 'logical_sha256' else 'metadata'})
    for field in ('target_output', 'attention_mask'):
        a, b = stock[field], off[field]
        if a is None or b is None:
            require(field == 'attention_mask' and a is None and b is None,
                    'Missing required target output or changed attention mask presence')
            continue
        # This calibration's SDPA mask is None; do not silently support another mask dtype.
        a, b = record_projection(a), record_projection(b)
        for key in RECORD_FIELDS:
            if a[key] != b[key]:
                differences.append({'field': field + '.' + key, 'kind': 'value_hash' if key == 'logical_sha256' else 'metadata'})
    full = off.get('selected_layer_full_inputs_before', {})
    full_after = off.get('selected_layer_full_inputs_after', {})
    if off['layer_index'] == 1:
        require(set(full) == set(full_after) == {'query', 'key', 'value', 'attn_mask'}, 'Missing selected full-batch input integrity')
        for name in full:
            if full[name] is None or full_after[name] is None:
                require(full[name] is None and full_after[name] is None, 'Full mask presence changed')
            else:
                require(full[name]['hashed_slice'] == 'entire logical tensor', 'Incomplete full-batch observation')
                integrity(full[name], full_after[name])
    return {'forward_index': off['forward_index'], 'layer_index': off['layer_index'],
            'target_inputs_and_output_equal': not differences, 'differences': differences,
            'input_format_transitions': input_transitions,
            'stock_call_id': stock['call_id'], 'off_call_id': off['call_id']}


TYPE_FORMAT = {'int64_t': 'q', 'uint64_t': 'Q', 'int32_t': 'i', 'uint32_t': 'I',
               'uint16_t': 'H', 'int16_t': 'h', 'uint8_t': 'B', 'int8_t': 'b', 'float': 'f'}


def decode_pod(raw, schema):
    require(len(raw) == schema['size'] == 1464, 'Public POD byte size differs')
    require(schema['type'] == 'FlashAttentionScoreGeneralTilingData', 'Wrong POD type')
    result = {}
    for field in schema['fields']:
        fmt = '<' + str(field['count']) + TYPE_FORMAT[field['type']]
        require(struct.calcsize(fmt) == field['width'] * field['count'], 'POD field width differs')
        value = list(struct.unpack_from(fmt, raw, field['offset']))
        result[field['name']] = value[0] if field['count'] == 1 else value
    require(len(result) == 105, 'Unexpected POD projection completeness')
    return result


TYPED_MAP = {
    'm': 'coreParams.s1BaseSize', 's2_base': 'coreParams.s2BaseSize', 'n_ratio': 'coreParams.nRatio',
    's1_outer': 'coreParams.s1OuterSize', 's2_outer': 'coreParams.s2OuterSize',
    's1_tail': 'coreParams.s1BaseTailSize', 's2_tail': 'coreParams.s2BaseTailSize',
    'core_num': 'multiCoreParams.coreNum', 'total_query_tasks': 'multiCoreParams.totalSize',
    'split_factor': 'multiCoreParams.splitFactorSize', 'split_factor_tail': 'multiCoreParams.splitFactorTailSize',
    'sparse_start_idx': 'multiCoreParams.sparseStartIdx', 'b': 'inputParams.bSize',
    'n2': 'inputParams.n2Size', 'g': 'inputParams.gSize', 's1': 'inputParams.s1Size',
    's2': 'inputParams.s2Size', 'd': 'inputParams.dSize', 'dv': 'coreParams.d2Size',
    'sparse_type': 'inputParams.sparseType', 'impl_mode': 'inputParams.implMode',
    'keep_prob': 'inputParams.keepProb', 'scale': 'inputParams.scaleValue',
    's1_sparse_valid_size': 'coreParams.s1SparseValidSize', 's2_sparse_valid_size': 'coreParams.s2SparseValidSize',
}


def typed_matches(typed, decoded):
    errors = []
    for key, field in TYPED_MAP.items():
        a, b = typed.get(key), decoded[field]
        equal = struct.pack('<f', a) == struct.pack('<f', b) if key in ('keep_prob', 'scale') and a is not None else a == b
        if not equal:
            errors.append(key)
    if typed.get('n1') != typed.get('n2', 0) * typed.get('g', 0):
        errors.append('n1')
    if typed.get('s2_group') != typed.get('s2_base', 0) * typed.get('n_ratio', 0):
        errors.append('s2_group')
    return errors


LAUNCH = re.compile(r'\[NNOP\]\[Launch\]\[(?P<tid>\d+)\]\s+OpName:\[(?P<op>aclnnFlashAttentionScore_\d+_FlashAttentionScore)\]\s+Tiling Key:\s*(?P<key>\d+),\s*len:\s*(?P<size>\d+),\s*numBlocks:\s*(?P<blocks>\d+),\s*Workspace Num:\s*(?P<workspaces>\d+)\.')
PID = re.compile(r'\bOP\((\d+),[^)]*\)')


def scan_native_log(path, calls, approved_root):
    h, active, windows, loaded = hashlib.sha256(), None, [], []
    for line_number, payload in enumerate(path.open('rb'), 1):
        h.update(payload)
        text = payload.decode('utf-8', 'surrogateescape').rstrip('\r\n')
        event = None
        p = text.find('{')
        if p >= 0 and 'native_api_' in text:
            try:
                value = json.loads(text[p:])
                if isinstance(value, dict) and value.get('event') in ('native_api_enter', 'native_api_exit'):
                    event = value
            except ValueError:
                require('"event": "native_api_' not in text, 'Malformed native marker')
        if event:
            require(not any('\udc80' <= c <= '\udcff' for c in text), 'Invalid UTF8 marker')
            if event['event'] == 'native_api_enter':
                require(active is None, 'Overlapping native API windows')
                active = {'enter': event, 'enter_line': line_number, 'launches': []}
            else:
                require(active is not None and event['status'] == 'returned', 'Missing native enter or failed exit')
                for key in ('call_id', 'native_index', 'forward_index', 'layer_index', 'pid', 'thread_id'):
                    require(active['enter'][key] == event[key], 'Native window identity differs')
                active.update(exit_line=line_number)
                windows.append(active)
                active = None
        match = LAUNCH.search(text)
        if match:
            require(active is not None, 'Actual FA launch outside API window')
            pid = PID.search(text)
            require(pid is not None and int(pid[1]) == active['enter']['pid'], 'Launch PID mismatch')
            require(not any('\udc80' <= c <= '\udcff' for c in text), 'Invalid UTF8 launch line')
            active['launches'].append({'line': line_number, 'line_sha256': sha_bytes(payload), 'text': text, **match.groupdict()})
        if 'LoadOppPackage' in text and 'load so, so_name is ' in text:
            lib = text.split('load so, so_name is ', 1)[1].strip().replace('//', '/')
            if lib.startswith(approved_root + '/'):
                loaded.append({'line': line_number, 'line_sha256': sha_bytes(payload), 'library': lib, 'text': text})
    require(active is None, 'Unterminated API window')
    require([w['enter']['call_id'] for w in windows] == [c['call_id'] for c in calls], 'Log/call ledger differs')
    for window, call in zip(windows, calls):
        require(len(window['launches']) == 1, 'Need exactly one actual FA kernel per API call')
        require(window['enter']['layer_index'] == call['layer_index'] and window['enter']['forward_index'] == call['forward_index'], 'Log layer/forward differs')
    require(any(x['library'].endswith('/libcust_opmaster_rt2.0.so') for x in loaded), 'No actual custom host library load evidence')
    return {'sha256': h.hexdigest(), 'windows': windows, 'custom_library_loads': loaded,
            'correlation': 'synchronized API window plus same process; launch uses a different native worker thread'}


def verify_vendor(root, off_folder, off_result, manifest, manifest_path, log):
    launch_path = off_folder.parent / (off_result['case_id'] + '.vendor_launch.json')
    launch = read_json(launch_path)
    private = off_result['private_vendor']
    require(private['launcher_receipt_sha256'] == sha_file(launch_path), 'Launcher receipt SHA mismatch')
    require(launch['status'] == 'VERIFIED_READY_TO_EXEC' and launch['case_id'] == off_result['case_id'], 'Bad vendor launch receipt')
    require(launch['case_sha256'] == off_result['source_case_sha256'], 'Launcher/source case mismatch')
    require(launch['probe_sha256'] == off_result['source_sha256']['hf_runtime_probe.py'], 'Launcher/probe source mismatch')
    verified = launch['verified_vendor']
    require(private['manifest_sha256'] == verified['manifest_sha256'] == sha_file(manifest_path), 'Vendor manifest SHA mismatch')
    require(manifest['same_binary_off_on'] is True and manifest['global_sdk_modified'] is False, 'Vendor scope differs')
    approved = private['approved_custom_opp_path']
    require(approved == verified['vendor_root'] == launch['child_environment']['ASCEND_CUSTOM_OPP_PATH'], 'Approved loader root mismatch')
    require(launch['child_environment']['LD_PRELOAD'] is None and launch['child_environment']['ACLNN_CACHE_LIMIT'] == '0', 'Unexpected preload/cache configuration')
    require(launch['child_environment']['LD_LIBRARY_PATH'].split(':')[0] == approved + '/op_api/lib', 'Custom API library lacks loader priority')
    reported = {x['path']: x for x in verified['artifacts']}
    staged = contained(root, manifest['vendor_root'])
    files = []
    important = ('op_api/lib/libcust_opapi.so',
                 'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so',
                 'op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so')
    original = {x['path']: x for x in manifest['artifacts']}
    for relative in important:
        path = contained(staged, relative)
        digest, size = sha_file(path), path.stat().st_size
        require(digest == original[relative]['sha256'] == reported[relative]['sha256'], 'Loaded library SHA mismatch')
        require(size == original[relative]['bytes'] == reported[relative]['bytes'], 'Loaded library size mismatch')
        files.append({'path': relative, 'sha256': digest, 'bytes': size})
    loaded = {x['library'] for x in log['custom_library_loads']}
    require(approved + '/' + important[1] in loaded and approved + '/' + important[2] in loaded, 'Manifest library paths lack native load evidence')
    return {'status': 'VERIFIED', 'launcher_sha256': sha_file(launch_path), 'manifest_sha256': sha_file(manifest_path),
            'libraries': files, 'load_lines': log['custom_library_loads'],
            'limit': 'Native logs evidence opmaster/proto load; op_api priority and file SHA do not alone prove a specific API symbol resolution.'}


def compare_pair(root, batch, stock_run, off_run, schema, manifest_path, evidence):
    stock_folder = root / 'outputs' / stock_run / ('stock_b' + str(batch))
    off_folder = root / 'outputs' / off_run / ('fa_off_b' + str(batch))
    sr, fr = read_json(stock_folder / 'result.json'), read_json(off_folder / 'result.json')
    for folder in (stock_folder, off_folder):
        evidence[str(folder / 'result.json')] = sha_file(folder / 'result.json')
    require(sr['forward_count'] == 1 and fr['forward_count'] == (1 if batch == 1 else 2), 'Unexpected forward counts')
    for field in ('batch_size', 'prompt_length', 'target_index', 'layer_index', 'checkpoint_config', 'versions', 'environment', 'actual_sdpa_context_flags'):
        require(sr[field] == fr[field], 'Case configuration mismatch: ' + field)
    require(sr['batch_size'] == batch and sr['prompt_length'] == 2048 and sr['target_index'] == 0, 'Unexpected fixed workload')
    require(sr['operator_variant'] == 'stock' and fr['operator_variant'] == 'private_vendor', 'Wrong variants')
    for field in sr['options']:
        if field != 'full_prefill_forward_count':
            require(sr['options'][field] == fr['options'][field], 'Execution option mismatch: ' + field)
    require(sr['environment']['ACLNN_CACHE_LIMIT'] == '0', 'Tiling cache not disabled')
    a, b = load_calls(stock_folder, sr, evidence), load_calls(off_folder, fr, evidence)
    rows = [compare_calls(a[c['layer_index']], c) for c in b]
    logits = []
    sf = sr['forward_results'][0]
    for forward in fr['forward_results']:
        req_a, req_b = sf['requests'][0], forward['requests'][0]
        require(len(sf['requests']) == len(forward['requests']) == batch, 'Request count differs')
        for request_a, request_b in zip(sf['requests'], forward['requests']):
            for key in ('prompt_index', 'batch_slot', 'prompt_token_sha256', 'last_token_position'):
                require(request_a[key] == request_b[key], 'Request/prefix mapping differs: ' + key)
        require(req_a['prompt_token_sha256'] == req_b['prompt_token_sha256'] and req_a['last_token_position'] == req_b['last_token_position'], 'Logits target prefix differs')
        logits.append({'forward_index': forward['forward_index'],
                       'target_logits_equal': record_projection(sf['target_logits']) == record_projection(forward['target_logits']),
                       'argmax_equal': req_a['next_token_argmax'] == req_b['next_token_argmax'],
                       'stock_sha256': sf['target_logits']['logical_sha256'], 'off_sha256': forward['target_logits']['logical_sha256'],
                       'stock_argmax': req_a['next_token_argmax'], 'off_argmax': req_b['next_token_argmax']})
    # Numerical first difference is fixed above, independent of host control/profile.
    numerical_pass = all(x['target_inputs_and_output_equal'] for x in rows) and all(x['target_logits_equal'] and x['argmax_equal'] for x in logits)
    log_path = off_folder.parent / (fr['case_id'] + '.log')
    log = scan_native_log(log_path, b, fr['private_vendor']['approved_custom_opp_path'])
    evidence[str(log_path)] = log['sha256']
    vendor = verify_vendor(root, off_folder, fr, read_json(manifest_path), manifest_path, log)
    profile_folder = stock_folder.parent / (sr['case_id'] + '.fa_log_profile')
    profile_path = profile_folder / 'profile.json'
    profile = read_json(profile_path)
    evidence[str(profile_path)] = sha_file(profile_path)
    require(profile['source_sha256']['result'] == evidence[str(stock_folder / 'result.json')], 'Stock profile/result SHA mismatch')
    require(len(profile['calls']) == 28, 'Stock profile coverage differs')
    selected = profile['calls'][1]
    require(selected['enter']['call_id'] == a[1]['call_id'] and selected['enter']['layer_index'] == 1, 'Wrong stock selected layer')
    raw_path = contained(profile_folder, selected['prefix_file'])
    raw = raw_path.read_bytes()
    evidence[str(raw_path)] = sha_bytes(raw)
    require(sha_bytes(raw) == selected['declared_prefix.bin_sha256'], 'Stock raw prefix SHA mismatch')
    decoded = decode_pod(raw, schema)
    event_path = off_folder / 'fa_host_events.jsonl'
    events = [json.loads(line) for line in event_path.read_text(encoding='utf8').splitlines() if line.strip()]
    evidence[str(event_path)] = sha_file(event_path)
    require(len(events) == fr['forward_count'] and len({e['token'] for e in events}) == len(events), 'Host event count/uniqueness differs')
    host = []
    for forward, event in enumerate(events):
        call = b[forward * 28 + 1]
        scope = call['host_scope']
        entry = log['windows'][forward * 28 + 1]['enter']
        require(event['schema'] == 'fault28.fa.typed_host.v1', 'Unknown public host event schema')
        require(event['token'] == scope['token'] and event['pid'] == scope['pid'] and event['tid'] == scope['thread_id'], 'Public host callback not bound to actual selected API scope')
        require(event['pid'] == entry['pid'] and event['tid'] == entry['thread_id'], 'Public callback not on selected API caller thread')
        require(event['forward'] == str(forward) and event['layer'] == 1 and event['phase'] == 'prefill', 'Host callback scope mismatch')
        require(scope['environment_restored'] is True, 'Scoped host environment not restored')
        payload = bytes.fromhex(event['raw_hex'])
        require(event['raw_size'] == len(payload), 'Public raw length mismatch')
        pod = decode_pod(payload, schema)
        typed_errors = typed_matches(event['typed'], pod)
        launch = log['windows'][forward * 28 + 1]['launches'][0]
        raw_diff = [i for i, (x, y) in enumerate(zip(raw, payload)) if x != y]
        errors = []
        typed_route = {'template': 'FlashAttentionScoreS1s2Bn2gs1', 'dtype_enum': 1,
                       'layout': 'BNSD', 'calc_bytes': 4, 'high_precision': True,
                       'sparse_mode': 2, 'has_mask': 1, 'has_dropout': 0, 'has_pse': 0,
                       'enable_l1_reuse': True}
        for ok, message in (
            (event['mode'] == 'off' and event['policy_applied'] is False, 'OFF policy was applied or wrong mode'),
            (event['callback_status'] == 0, 'Host callback failed'),
            (event['eligible'] is (batch == 8), 'Expected B8 eligible and B1 ineligible'),
            (not raw_diff and len(payload) == len(raw), 'Public raw differs from stock declared prefix'),
            (not typed_errors and pod == decoded, 'Typed public POD differs'),
            (event['tiling_key'] == selected['tiling_key'] == launch['key'], 'Tiling key differs'),
            (event['block_dim'] == selected['block_dim'] == int(launch['blocks']), 'Block dimension differs'),
            (event['workspace_count'] == selected['kernel_workspace_count'] == int(launch['workspaces']), 'Workspace count differs'),
            (len(event['workspace_sizes']) == event['workspace_count'], 'Public workspace vector incomplete'),
            (all(event['typed'].get(k) == v for k, v in typed_route.items()), 'Unexpected source-bound host route/precision'),
        ):
            if not ok:
                errors.append(message)
        host.append({'forward_index': forward, 'errors': errors, 'eligible_observed': event['eligible'],
                     'eligible_required': batch == 8, 'policy_applied': event['policy_applied'],
                     'callback_status': event['callback_status'], 'raw_equal': not raw_diff and payload == raw,
                     'raw_different_byte_offsets': raw_diff, 'raw_sha256': sha_bytes(payload),
                     'typed_mismatch_fields': typed_errors, 'tiling_key': event['tiling_key'],
                     'block_dim': event['block_dim'], 'schedule_mode_off': event['schedule_mode'],
                     'schedule_mode_stock': None, 'schedule_comparison': 'UNOBSERVABLE_IN_STOCK_LOG_PROFILE',
                     'workspace_sizes_off': event['workspace_sizes'],
                     'workspace_sizes_stock': None, 'workspace_comparison': 'Stock API workspace includes helpers; not comparable to this host vector',
                     'typed': event['typed'], 'selected_native_launch': launch})
    control_pass = all(not h['errors'] for h in host)
    return {'batch_size': batch, 'status': 'PASS_STOCK_OFF_COMPATIBILITY' if numerical_pass and control_pass else 'FAIL_STOCK_OFF_COMPATIBILITY',
            'numerical_comparison_pass': numerical_pass, 'host_control_and_raw_pass': control_pass,
            'model_call_comparisons': rows, 'logits_comparisons': logits,
            'first_observed_model_difference': next((x for x in rows if x['differences']), None),
            'host_comparisons': host, 'stock_pod105': decoded, 'vendor_load_evidence': vendor,
            'native_fa_launch_count': len(log['windows']),
            'probe_source_sha256': {'stock': sr['source_sha256'], 'off': fr['source_sha256']},
            'limits': ['Probe source revisions differ and are recorded; native model source hashes and numerical options must match.',
                       'Same-batch target slot 0 comparisons only; OFF full-batch input integrity does not extend stock target-only hashes.',
                       '105 public own fields decoded; nested library structures are covered by complete 1464-byte equality, not invented field names.',
                       'Stock schedule/vector values are absent; they are not filled from OFF.',
                       'Compatibility is not B/F mechanism acceptance, msprobe localization, or a historical defect reproduction.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stock-run', default='hf_stock_2048_a3')
    parser.add_argument('--off-run', default='private_off_compat_a1')
    parser.add_argument('--schema', type=Path)
    parser.add_argument('--vendor-manifest', type=Path)
    parser.add_argument('--output', default='-')
    args = parser.parse_args()
    schema_path = args.schema or args.root / 'build_off/fa_policy/fa_pod_schema.json'
    manifest_path = args.vendor_manifest or args.root / 'build_off/fa_vendor_stage_a1/vendor_manifest.json'
    evidence = {str(schema_path): sha_file(schema_path), str(manifest_path): sha_file(manifest_path)}
    report = {'schema': 'fault28.stock_off.compatibility.v1', 'status': 'FAIL_STOCK_OFF_COMPATIBILITY',
              'mechanism_accepted': False, 'cases': [], 'errors': [], 'source_evidence_sha256': evidence}
    schema = read_json(schema_path)
    for batch in (1, 8):
        try:
            report['cases'].append(compare_pair(args.root, batch, args.stock_run, args.off_run, schema, manifest_path, evidence))
        except Exception as exc:
            report['errors'].append({'batch_size': batch, 'error_type': type(exc).__name__, 'message': str(exc)})
    if not report['errors'] and all(c['status'] == 'PASS_STOCK_OFF_COMPATIBILITY' for c in report['cases']):
        report['status'] = 'PASS_STOCK_OFF_COMPATIBILITY'
    payload = json.dumps(report, ensure_ascii=True, indent=2) + '\n'
    if args.output == '-':
        sys.stdout.write(payload)
    else:
        path = Path(args.output)
        require(not path.exists(), 'Output must be fresh')
        path.write_text(payload, encoding='utf8')
    return 0 if report['status'] == 'PASS_STOCK_OFF_COMPATIBILITY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
