"""Independent stdlib-only actual native launch/POD provenance review.

Reads immutable completed call windows and source/build manifests. Never imports
the runtime/adapter, reads tensor files, starts devices, or infers actual factors
from requested profiles. Writes only the explicitly new report path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

SCHEMA_SHA = 'f5865ad7185b7fabeb71b37aababb2b1719f79861ed70f487351dbac7248638b'
VENDOR_SHA = '42173485d5d3c86b57cd8a847dbce2dfd1fc42004c1a451bb282b7aa211d9ef1'
POLICY_SHA = '15bdb7b6e63ddec53f9d3d9a4c31ef4217732f7fdcf46f724cbad52099a5a3f3'
PADDING_SHA = '7c268a7a261c4aa72ed53f60a15f021bff7aa67a4c6585b00d9ed838fc68897e'
COMMIT = 'c77c0829f1f0f0b8f882db1d9b051a758b605472'
KERNELS = {
    'attention/flash_attention_score/op_kernel/arch22/flash_attention_score_tiling.h': '6ca76bce8a92fdf63d3c5646dfe6d2af3a9b367a9e6fbf7f994f0a5cee71c8d3',
    'attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h': '18bbd703454ea8cd6fbe6936329c29513745c269e7f7ad7ea5ab05d81dcd08cb',
    'attention/flash_attention_score/op_kernel/arch22/flash_attention_score_common.h': '3e074a966515df1c9cf0cc070d2315a5978ccb44e7e75a842188de5016e4cc34',
    'attention/flash_attention_score/op_kernel/flash_attention_score.cpp': '7f4adb241e3f09ad3204159bcd1b963800e174230f80c78b8126e29b7481ba64',
    'attention/flash_attention_score/op_kernel/arch22/flash_attention_score_template_tiling_key.h': 'a8951fea89250f0f815382e9c3b2a05a4c01a12f971985bfe5d133f04062f072',
    'common/include/op_kernel/util.h': '442648c354814d149ba7f8185e4b16cc7d528d1333f8437f07c0f3c5c93187d9',
}
FORMATS = {'int64_t': 'q', 'uint64_t': 'Q', 'int32_t': 'i', 'uint32_t': 'I',
           'int16_t': 'h', 'uint16_t': 'H', 'int8_t': 'b', 'uint8_t': 'B', 'float': 'f'}
PADDING = {'inputParams.remainPH': (115, 1), 'inputParams.attenMaskS2SizePH': (118, 2),
           'inputParams.qStartIdxPH': (132, 4), 'inputParams.InputParamsPH': (154, 6),
           'coreParams.s1SparseValidSizePH': (700, 4)}
OP = r'(?P<op>aclnnFlashAttentionScore_\d+_FlashAttentionScore)'
LAUNCH = re.compile(r'\[NNOP\]\[Launch\]\[(?P<tid>\d+)\]\s+OpName:\[' + OP +
    r'\]\s+Tiling Key:\s*(?P<key>0x[0-9a-fA-F]+|\d+),\s*len:\s*(?P<size>\d+),\s*'
    r'numBlocks:\s*(?P<blocks>\d+),\s*Workspace Num:\s*(?P<workspaces>\d+)\.')
RAW = re.compile(r'\[SplitDataAndPrint\]\[(?P<tid>\d+)\]\s+OpName:\[' + OP +
    r'\]\s+PrintTilingData\[(?P<index>\d+)\]: '
    r'(?:Aligned TilingData Size: (?P<aligned>\d+), data\(uint32_t\): )?(?P<words>[0-9 ]+)$')
WS = re.compile(r'\[CommonOpExecutorRun\]\[(?P<tid>\d+)\]\s+OpName:\['
    r'(?P<op>aclnnFlashAttentionScore_\d+)\]\s+Workspace addr: \S+,\s*size:\s*(?P<size>\d+),')
PID = re.compile(r'\bOP\((\d+),[^)]*\)')
LOAD_PID = re.compile(r'\bGE\((\d+),[^)]*\)')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def decode(raw, schema):
    require(len(raw) == schema['size'] == 1464, 'POD length differs')
    out = {}
    coverage = set()
    for f in schema['fields']:
        fmt = '<' + str(f['count']) + FORMATS[f['type']]
        n = struct.calcsize(fmt)
        require(n == f['width'] * f['count'], 'POD width differs')
        value = struct.unpack_from(fmt, raw, f['offset'])
        require(struct.pack(fmt, *value) == raw[f['offset']:f['offset'] + n], 'POD roundtrip differs')
        require(f['name'] not in out, 'Duplicate POD name')
        out[f['name']] = value[0] if len(value) == 1 else list(value)
        for index in range(f['offset'], f['offset'] + n):
            require(index not in coverage, 'POD fields overlap')
            coverage.add(index)
    # The approved schema projects 205 named fields. These three SDK sections
    # have verified boundaries but their members are not expanded in that schema.
    # Keep all 192 bytes bound in raw evidence; never classify them as padding.
    require(len(out) == 205 and set(range(len(raw))) - coverage == set(range(1248, 1440)),
            'Approved POD field/section coverage changed')
    return out


def parse_window(data, call):
    markers, launches, chunks, workspaces = [], [], [], []
    for line_number, payload in enumerate(data.splitlines(keepends=True), 1):
        line = payload.decode('utf-8', 'surrogateescape').rstrip('\r\n')
        if line.startswith('{'):
            try:
                obj = json.loads(line)
                if obj.get('event') in ('native_api_enter', 'native_api_exit'):
                    markers.append((line_number, obj))
            except ValueError:
                pass
        for regex, target in ((LAUNCH, launches), (RAW, chunks), (WS, workspaces)):
            match = regex.search(line)
            if match:
                pid = PID.search(line)
                require(pid and int(pid[1]) == call['pid'], 'Native log PID differs')
                target.append({'line': line_number, 'sha256': digest(payload), **match.groupdict()})
    require(len(markers) == 2 and markers[0][1]['event'] == 'native_api_enter'
            and markers[1][1]['event'] == 'native_api_exit', 'Call window markers differ')
    for _, event in markers:
        for key in ('call_id', 'case_id', 'api', 'pid', 'thread_id', 'scope_token', 'phase', 'batch_size', 'step'):
            require(event.get(key) == call.get(key), 'Marker binding differs: ' + key)
    require(markers[-1][1]['status'] == 'returned', 'Native call did not return')
    require(len(launches) == 1, 'Expected one actual FAScore launch')
    launch = launches[0]
    require(markers[0][0] < launch['line'] < markers[1][0], 'Launch outside call')
    require(chunks and [int(x['index']) for x in chunks] == list(range(len(chunks))), 'Raw chunks missing/repeated')
    require(all(x['op'] == launch['op'] and x['tid'] == launch['tid'] for x in chunks), 'Raw/launch identity differs')
    size, aligned = int(launch['size']), int(chunks[0]['aligned'])
    require(0 < size <= aligned <= 1024 * 1024 and aligned % 4 == 0, 'Invalid raw framing')
    require(all(x['aligned'] is None for x in chunks[1:]), 'Repeated raw header')
    raw_text = ''.join(chunk['words'] for chunk in chunks)
    words = [int(word) for word in raw_text.split()]
    require(len(words) * 4 == aligned, 'Raw aligned length differs')
    raw = struct.pack('<' + str(len(words)) + 'I', *words)[:size]
    ws = [x for x in workspaces if x['op'] == launch['op'].removesuffix('_FlashAttentionScore')
          and x['tid'] == launch['tid']]
    require(len(ws) == 1, 'Enclosing API workspace ambiguous')
    return raw, {'key': str(int(launch['key'], 0) if launch['key'].startswith('0x') else int(launch['key'])),
        'block_dim': int(launch['blocks']), 'workspace_count': int(launch['workspaces']),
        'api_workspace_bytes': int(ws[0]['size']), 'native_launch_tid': int(launch['tid']),
        'raw_chunk_count': len(chunks), 'aligned_bytes': aligned, 'op_name': launch['op'],
        'raw_sha256': digest(raw), 'window_sha256': digest(data), 'native_launch_line': launch['line'],
        'native_launch_line_sha256': launch['sha256'], 'marker_count': len(markers)}


def typed_check(host, pod):
    typed = host['typed']
    scalar = {'m': 'coreParams.s1BaseSize', 's2_base': 'coreParams.s2BaseSize',
        'n_ratio': 'coreParams.nRatio', 's1_outer': 'coreParams.s1OuterSize', 's2_outer': 'coreParams.s2OuterSize',
        's1_tail': 'coreParams.s1BaseTailSize', 's2_tail': 'coreParams.s2BaseTailSize',
        'core_num': 'multiCoreParams.coreNum', 'total_query_tasks': 'multiCoreParams.totalSize',
        'split_factor': 'multiCoreParams.splitFactorSize', 'split_factor_tail': 'multiCoreParams.splitFactorTailSize',
        'sparse_start_idx': 'multiCoreParams.sparseStartIdx', 'sparse_type': 'inputParams.sparseType',
        's1_sparse_valid_size': 'coreParams.s1SparseValidSize', 's2_sparse_valid_size': 'coreParams.s2SparseValidSize',
        'b': 'inputParams.bSize', 'n2': 'inputParams.n2Size', 'g': 'inputParams.gSize',
        's1': 'inputParams.s1Size', 's2': 'inputParams.s2Size', 'd': 'inputParams.dSize',
        'keep_prob': 'inputParams.keepProb', 'impl_mode': 'inputParams.implMode'}
    for name, source in scalar.items():
        require(typed[name] == pod[source], 'Typed/public POD differs: ' + name)
    require(typed['s2_group'] == pod['coreParams.s2BaseSize'] * pod['coreParams.nRatio'], 'S2 group differs')
    require(struct.pack('<f', typed['scale']) == struct.pack('<f', pod['inputParams.scaleValue']), 'Scale differs')
    require(typed['n1'] == pod['inputParams.n2Size'] * pod['inputParams.gSize'], 'Query heads differ')
    for name in ('bmm1', 'bmm2'):
        require(len(typed[name]) == 50, 'Expected complete TCube fields')
        for key, value in typed[name].items():
            require(pod[name + 'TilingData.' + key] == value, 'TCube field differs: ' + key)
    return len(scalar) + 3 + 100


def audit_sources(root, padding_path):
    evidence = root / 'build/evidence_a1'
    schema_path = evidence / 'fa_pod_schema.json'
    require(sha(schema_path) == SCHEMA_SHA, 'Compiler POD schema changed')
    schema = read_json(schema_path)
    require(sha(padding_path) == PADDING_SHA, 'Pre-registered padding audit changed')
    padding = read_json(padding_path)
    require({x['name']: (x['offset'], x['count']) for x in padding['members']} == PADDING, 'Padding audit differs')
    manifest_path = root / 'native_vendor_stage_a1/vendor_manifest.json'
    require(sha(manifest_path) == VENDOR_SHA, 'Vendor manifest changed')
    vendor = read_json(manifest_path)
    require(vendor['source_commit'] == COMMIT and vendor['policy_source_manifest_sha256'] == POLICY_SHA, 'Vendor source binding changed')
    vroot = root / vendor['vendor_root']
    for item in vendor['artifacts']:
        p = vroot / item['path']
        require(p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], 'Vendor artifact changed: ' + item['path'])
    source = root / 'source/ops-transformer-v9.1.0'
    policy_path = evidence / 'policy_source_manifest.json'
    require(sha(policy_path) == POLICY_SHA, 'Policy manifest changed')
    for item in read_json(policy_path)['files']:
        require(sha(source / item['path']) == item['after_sha256'], 'Patched host source changed')
    for name, expected in KERNELS.items():
        require(sha(source / name) == expected, 'Matched kernel source changed: ' + name)
    return schema, {'pod_schema_sha256': SCHEMA_SHA, 'vendor_manifest_sha256': VENDOR_SHA,
        'policy_source_manifest_sha256': POLICY_SHA, 'source_commit': COMMIT,
        'vendor_artifacts_checked': len(vendor['artifacts']), 'kernel_source_sha256': KERNELS,
        'padding_source_audit_sha256': PADDING_SHA, 'complete_public_fields': 205,
        'input_semantic_fields': 34, 'execution_semantic_fields': 166, 'padding_bytes_preserved': 17}


def audit_case(path, schema):
    result = read_json(path / 'result.json')
    require(result['status'] == 'CAPTURE_COMPLETED', 'Case incomplete: ' + path.name)
    require(result['original_api_calls'] == len(result['steps']), 'Call count differs')
    log = path.with_suffix('.log')
    log_data = log.read_bytes()
    rows = []
    for step in result['steps']:
        stepdir = path / 'steps' / ('step' + str(step['step'])) / 'native_profile'
        window = (stepdir / 'native_window.log').read_bytes()
        call = step['native_call']
        raw, info = parse_window(window, call)
        require(log_data.count(window) == 1, 'Native window does not occur exactly once in source log')
        start = log_data.index(window)
        info.update(source_window_byte_range=[start, start + len(window)],
                    source_prefix_sha256=digest(log_data[:start + len(window)]),
                    source_log_sha256=digest(log_data), caller_tid=call['thread_id'], pid=call['pid'])
        pod = decode(raw, schema)
        require(pod['inputParams.bSize'] == result['batch_size'], 'Actual batch differs')
        fields = ('coreParams.s1BaseSize', 'coreParams.nRatio', 'coreParams.s2BaseSize',
                  'coreParams.s2OuterSize', 'multiCoreParams.coreNum', 'multiCoreParams.totalSize')
        info.update(case_id=result['case_id'], step=step['step'], mode=step['mode'],
                    actual_factors={'m': pod[fields[0]], 'n_ratio': pod[fields[1]]},
                    actual_profile={k: pod[k] for k in fields}, padding={k: pod[k] for k in PADDING},
                    pod_sha256=digest(json.dumps(pod, sort_keys=True, separators=(',', ':')).encode()),
                    result_sha256=sha(path / 'result.json'), host_callback_checked=False,
                    controlled_support_granted=False)
        info['unexpanded_sdk_sections'] = {item['name']: {
            'offset': item['offset'], 'size': item['size'],
            'raw_hex': raw[item['offset']:item['offset'] + item['size']].hex(),
            'sha256': digest(raw[item['offset']:item['offset'] + item['size']]),
            'semantic_members_decoded': False, 'classified_as_padding': False}
            for item in schema['sections'] if item['name'] in
            ('softmaxFlashTilingData', 'transposeTilingData', 'transposeTilingDataTailCore')}
        input_contract = {k: v for k, v in pod.items() if k.startswith('inputParams.') and k not in PADDING}
        execution_plan = {k: v for k, v in pod.items() if not k.startswith('inputParams.') and k not in PADDING}
        execution_plan.update({'@raw.' + k: v['raw_hex'] for k, v in info['unexpanded_sdk_sections'].items()})
        require(len(input_contract) == 34 and len(execution_plan) == 169, 'Independent semantic projection incomplete')
        info['input_contract_sha256'] = digest(json.dumps(input_contract, sort_keys=True, separators=(',', ':')).encode())
        info['execution_plan_sha256'] = digest(json.dumps(execution_plan, sort_keys=True, separators=(',', ':')).encode())
        info['input_contract_field_count'], info['execution_plan_item_count'] = 34, 169
        hostpath = stepdir / 'host_event.json'
        if hostpath.exists():
            host = read_json(hostpath)
            require(host['schema'] == 'nes.fa.host_observation.v1' and host['callback_status'] == 0, 'Host callback failed')
            require((host['pid'], host['tid'], host['token']) == (call['pid'], call['thread_id'], call['scope_token']), 'Host scope identity differs')
            require(host['raw_hex'] == raw.hex() and host['raw_size'] == len(raw), 'Host vs actual launch raw differs')
            require(str(host['key']) == info['key'] and host['block_dim'] == info['block_dim']
                    and host['workspace_count'] == info['workspace_count'], 'Host vs actual launch metadata differs')
            original = [json.loads(line) for line in (stepdir / 'host_events.original.jsonl').read_text().splitlines() if line.strip()]
            require([x for x in original if x.get('token') == call['scope_token']] == [host], 'Original host snapshot differs')
            plan = host['plan']
            require(plan == step['typed_plan_intent'], 'Plan snapshot differs')
            require(plan['scope']['pid'] == call['pid'] and plan['scope']['native_tid'] == call['thread_id']
                    and plan['scope']['token'] == call['scope_token'] and plan['scope']['phase'] == 'operator_execution', 'Plan scope differs')
            require(plan['certificate']['source_manifest_sha256'] == POLICY_SHA
                    and plan['certificate']['build_manifest_sha256'] == VENDOR_SHA, 'Plan build binding differs')
            require(host['eligible'] is True and host['context_guards']
                    and all(x['passed'] is True and not x['failed_guards'] for x in host['context_guards']), 'Context did not match')
            require(len(host['selector_events']) == 1, 'Ambiguous selector events')
            require(host['selected_factors'] == info['actual_factors'], 'Selected factors differ from generated actual POD')
            launcher_path = path.with_name(path.name + '.vendor_launch.json')
            launcher = read_json(launcher_path)
            require(launcher['schema'] == 'nes.vendor_launch.v1' and launcher['pid'] == call['pid']
                    and launcher['case_id'] == result['case_id']
                    and launcher['case_sha256'] == result['source_case_sha256']
                    and launcher['verified_vendor']['manifest_sha256'] == VENDOR_SHA,
                    'Independent private launcher binding differs')
            require(launcher['child_environment']['ASCEND_CUSTOM_OPP_PATH'] ==
                    '/audit/native_vendor_stage_a1/vendors/native_execution_strategy_transformer', 'Private vendor environment differs')
            loaded = []
            for lineno, line in enumerate(log_data[:start + len(window)].decode('utf-8', 'surrogateescape').splitlines(), 1):
                if 'LoadOppPackage' in line and 'load so, so_name is ' in line:
                    library = line.split('load so, so_name is ', 1)[1].strip().replace('//', '/')
                    if '/native_execution_strategy_transformer/' in library:
                        pid = LOAD_PID.search(line)
                        require(pid and int(pid[1]) == call['pid'], 'Loaded private library PID differs')
                        loaded.append({'line': lineno, 'path': library})
            for name in ('libcust_opmaster_rt2.0.so', 'libcust_opsproto_rt2.0.so'):
                require(any(x['path'].endswith('/' + name) for x in loaded), 'Private library load not observed: ' + name)
            info.update(host_callback_checked=True, host_raw_byte_exact=True, typed_fields_checked=typed_check(host, pod),
                        host_native_factors=host['native_factors'], host_selected_factors=host['selected_factors'],
                        selection_applied=host['selection_applied'], host_reason=host['reason'],
                        host_sha256=sha(hostpath), guard_reports=host['context_guards'],
                        launcher_receipt_sha256=sha(launcher_path), actual_private_library_loads=loaded)
            require(step['native_observation']['strategy']['parameters']['execution_plan'] == execution_plan,
                    'Session observation differs from independent 169-item actual plan')
        else:
            original = step['native_observation_only']
            require(original['pod'] == pod, 'Saved stock POD differs from independent decode')
            launch = original['native_launch']
            require(launch['raw_hex'] == raw.hex() and launch['window_sha256'] == digest(window)
                    and launch['source_prefix_sha256'] == info['source_prefix_sha256'], 'Saved stock launch differs')
            info['framing_limitation'] = 'Stock raw framing is independently re-parsed; public host callback unavailable for this stock call.'
        rows.append(info)
    if result['phase'] == 'controlled':
        require(len(rows) == 4 and [x['step'] for x in rows] == [0, 1, 2, 3], 'Private sequence differs')
        require(len({x['input_contract_sha256'] for x in rows}) == 1, 'Input semantics differ within private sequence')
        require(rows[0]['execution_plan_sha256'] == rows[3]['execution_plan_sha256'], 'Off did not restore complete actual execution plan')
        require(rows[1]['execution_plan_sha256'] == rows[2]['execution_plan_sha256']
                and rows[1]['execution_plan_sha256'] != rows[0]['execution_plan_sha256'], 'Repeated actual strategy switch not observed')
        require(all(x['host_native_factors'] == rows[0]['actual_factors'] for x in rows), 'Pre-selection native factors differ from observed baseline')
        for x in rows:
            x['sequence_invariants'] = {'input_contract_34_unchanged': True,
                'controlled_169_plan_repeats_exactly': True, 'off_169_plan_equals_native': True,
                'controlled_169_plan_differs_from_native': True,
                'pre_selection_native_factors_equal_actual_baseline': True}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--attempt', required=True)
    ap.add_argument('--padding-audit', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--allow-incomplete', action='store_true')
    args = ap.parse_args()
    report = {'schema': 'nes.fa.independent_native_review.v1', 'status': 'FAIL',
        'attempt': args.attempt, 'audit_source_sha256': sha(__file__), 'reads_tensor_payloads': False,
        'executes_npu_or_containers': False, 'numeric_or_model_effect_claimed': False, 'cases': []}
    try:
        schema, sources = audit_sources(args.root, args.padding_audit)
        report['source_binding'] = sources
        attempt = args.root / 'outputs' / args.attempt
        completion = read_json(attempt / 'completion.json') if (attempt / 'completion.json').exists() else None
        report['completion_status'] = None if completion is None else completion['status']
        if not args.allow_incomplete:
            require(completion and completion['status'] == 'CAPTURE_COMPLETED'
                    and completion['owned_container_removed'] is True
                    and completion['device_lock_released'] is True
                    and not completion['cleanup_errors'], 'Attempt capture/cleanup incomplete')
        sources = read_json(attempt / 'source_manifest.json')
        for relative, expected in sources.items():
            require(sha(attempt / 'source' / relative) == expected, 'Capture source snapshot changed: ' + relative)
        report['capture_sources'] = {'manifest_sha256': sha(attempt / 'source_manifest.json'),
                                    'file_count': len(sources), 'all_snapshot_hashes_verified': True}
        results = sorted(attempt.glob('*/result.json'))
        if not args.allow_incomplete:
            require({x['case_id'] for x in completion['cases']} == {p.parent.name for p in results},
                    'Completion/result case inventory differs')
        for p in results:
            if args.allow_incomplete and read_json(p).get('status') != 'CAPTURE_COMPLETED':
                continue
            original = read_json(p)
            require(original['replay_source_sha256'] == sources['replay.py']
                    and original['runtime_manifest_sha256'] == sources['runtime_manifest.json'], 'Result/capture runtime source differs')
            if not args.allow_incomplete:
                entry = next(x for x in completion['cases'] if x['case_id'] == p.parent.name)
                require(entry['result_sha256'] == sha(p), 'Completion result hash differs')
            report['cases'].extend(audit_case(p.parent, schema))
        require(report['cases'], 'No completed call to audit')
        report['actual_calls_checked'] = len(report['cases'])
        report['status'] = 'PASS_NATIVE_SOURCE_AND_POD_REVIEW' if not args.allow_incomplete else 'PARTIAL_CALL_REVIEW_NOT_ATTEMPT_ACCEPTANCE'
    except Exception as error:
        report['error'] = str(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output), 'sha256': sha(args.output),
                      'calls': len(report['cases']), 'error': report.get('error')}))
    raise SystemExit(1 if report['status'] == 'FAIL' else 0)


if __name__ == '__main__':
    main()
