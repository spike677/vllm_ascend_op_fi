"""Offline model coverage acceptance. No device operations and no certificate issuance.

CLI: --inputs suite.json --output NEW.json [--pilot-only]. Suite schema is
nes.model_coverage.acceptance_inputs.v1, cases contain case_dir,
framework_report, native_census_report, statistics_report. Paths are relative
to suite.json or explicit absolute paths. completion/log default to the case's
parent directory. Original capture JSON is never edited.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPOSITORY = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(REPOSITORY))
from model_statistics_io import load_capture, child, read, require
from audit_model_coverage import audit
from compare_model_statistics import RUN_PAIRS
from model_context import actual_native_call, validate_request_rows
from native_execution_strategy.adapters.flash_attention import FlashAttentionAdapter, EFFECTIVE_SCALE
from native_execution_strategy.certificates import load_certificate
from native_execution_strategy.adapters.fa_schema import execution_plan, input_contract

CERTIFICATE_SHA = 'be58b4e77098a8911ab97ad8711144a5a5999dc3742080726c594ad88e8fcc5b'
RUNTIME_SHA = 'ca0bf188bdc031ea3e4249c03ac13a9a8f751cd5ae8485655d9048c9f6cdba3b'
SOURCE_SHA = '15bdb7b6e63ddec53f9d3d9a4c31ef4217732f7fdcf46f724cbad52099a5a3f3'
BUILD_SHA = '42173485d5d3c86b57cd8a847dbce2dfd1fc42004c1a451bb282b7aa211d9ef1'
MODES = ['calibration', 'native', 'controlled', 'controlled', 'off']
HEX = re.compile(r'^[0-9a-f]{64}$')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def mapped_path(root, original):
    """Translate only this case's /evidence prefix, never arbitrary basenames."""
    root = Path(root).resolve(strict=True)
    original = str(original)
    prefix = '/evidence/' + root.name + '/'
    if original.startswith(prefix):
        relative = original[len(prefix):]
        require('\\' not in relative and '..' not in PurePosixPath(relative).parts,
                'Unsafe original container evidence path')
        return child(root, relative)
    path = Path(original)
    require(path.is_absolute() and path.resolve(strict=True).is_relative_to(root)
            and not path.is_symlink(), 'Evidence is not inside this case: ' + original)
    return path.resolve(strict=True)


def remap_record(root, record, call_folder):
    result = copy.deepcopy(record)
    require(result.get('evidence'), 'UNKNOWN: no original native evidence')
    for ref in result['evidence']:
        path = mapped_path(root, ref['path'])
        require(path.is_relative_to(call_folder.resolve()) and sha(path) == ref['sha256'],
                'Original native evidence missing, changed, or belongs to another call')
        # Only the in-memory evidence reference changes, never host/raw/plan JSON.
        ref['path'] = str(path)
    return result


def load_original_parser(path, expected_sha):
    """Only execute the reviewed stdlib parser, with its captured source SHA."""
    require(sha(path) == expected_sha == sha(EXPERIMENT / 'native_evidence.py'),
            'Captured native parser is not this reviewed exact-character revision')
    spec = importlib.util.spec_from_file_location('_nes_model_frozen_native_parser', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OriginalLogVerifier:
    """Single forward pass through original log, then parse each immutable window.

    The independent census marker ranges bind the window before raw decoding.
    It is not enough for two derived JSON records to agree about tiling bytes.
    """
    def __init__(self, log, parser, temporary):
        self.log, self.parser, self.temporary = Path(log), parser, Path(temporary)
        self.stream = self.log.open('rb')
        self.offset, self.line, self.checked = 0, 0, 0
        self.prefix = hashlib.sha256()

    def close(self):
        self.stream.close()

    def verify(self, call, census_call, record):
        native, host = record['native_launch'], record['host_event']
        require(call['scope_token'] == call['call_id'] and host['token'] == call['scope_token']
                and host['pid'] == call['pid'] and host['tid'] == call['thread_id'],
                'Original native host token/PID/TID does not bind this SDPA invocation')
        enter, leave = census_call['enter'], census_call['exit']
        start, end = enter['byte_range'][0], leave['byte_range'][1]
        lines = [enter['line_number'], leave['line_number']]
        require(0 <= self.offset <= start < end and native['window_byte_range'] == [start, end]
                and native['window_line_range'] == lines and native['source_prefix_bytes'] == end,
                'Derived native window is not the original census marker range')
        payloads = []
        while self.offset < end:
            payload = self.stream.readline()
            require(payload and self.offset + len(payload) <= end, 'Original log window ends inside a line or beyond EOF')
            self.line += 1
            if self.offset < start:
                require(self.offset + len(payload) <= start, 'Original window begins inside a line')
            else:
                payloads.append(payload)
            for marker in (enter, leave):
                if self.offset == marker['byte_range'][0]:
                    require([self.offset, self.offset + len(payload)] == marker['byte_range']
                            and self.line == marker['line_number']
                            and hashlib.sha256(payload).hexdigest() == marker['line_sha256'],
                            'Independent marker byte/line/SHA does not match original log')
            self.offset += len(payload)
            self.prefix.update(payload)
        original = b''.join(payloads)
        require(self.line == lines[1] and self.prefix.hexdigest() == native['source_prefix_sha256'],
                'Original log prefix SHA or line boundary differs')
        refs = [ref for ref in record['evidence'] if Path(ref['path']).name == 'native_window.log']
        require(len(refs) == 1 and Path(refs[0]['path']).read_bytes() == original
                and hashlib.sha256(original).hexdigest() == refs[0]['sha256'] == native['window_sha256'],
                'Captured native window differs from original complete log bytes')
        target = self.temporary / ('reparsed_%04d.log' % self.checked)
        parsed = self.parser.actual_launch(Path(refs[0]['path']), call, target)
        fields = ('raw_hex', 'raw_size', 'raw_sha256', 'tiling_key', 'block_dim', 'workspace_count',
                  'api_workspace_bytes', 'pid', 'tid', 'op_name', 'raw_reconstruction',
                  'raw_chunk_payload_lengths', 'raw_text_sha256', 'window_sha256')
        require(all(parsed[k] == native[k] for k in fields),
                'Reparsed ORIGINAL launch bytes/key/geometry differ from derived record')
        require(parsed['window_byte_range'] == [0, len(original)]
                and parsed['window_line_range'] == [1, lines[1] - lines[0] + 1]
                and target.read_bytes() == original, 'Reparsed marker framing differs from original window')
        self.checked += 1
        return {'raw_reparsed_from_original_log': True, 'window_sha256': parsed['window_sha256'],
                'source_prefix_sha256': self.prefix.hexdigest(), 'window_byte_range': [start, end],
                'window_line_range': lines, 'raw_sha256': parsed['raw_sha256']}


def checked_snapshot(value, shape):
    require(isinstance(value, dict) and value.get('shape') == shape
            and value.get('dtype') == 'torch.float16' and str(value.get('device')).startswith('npu:'),
            'Wrong complete tensor geometry/dtype/device')
    require(value.get('bytes') == math.prod(shape) * 2 and HEX.fullmatch(value.get('full_sha256', ''))
            and len(value.get('per_batch_sha256', [])) == shape[0]
            and all(HEX.fullmatch(x) for x in value['per_batch_sha256']), 'Missing complete/per-request logical hash')
    require(len(value.get('stride', [])) == len(shape) and type(value.get('storage_offset')) is int
            and type(value.get('object_id')) is int and type(value.get('data_ptr')) is int
            and type(value.get('actual_format')) is int and value.get('byteorder') == 'little'
            and value.get('raw_order') == 'C-contiguous logical elements', 'Tensor identity/geometry incomplete')
    return value


def input_integrity(call, batch, stage):
    q, k = (2048, 2048) if stage == 'prefill' else (1, 2049)
    shapes = {'query': [batch, 16, q, 128], 'key': [batch, 8, k, 128], 'value': [batch, 8, k, 128]}
    before, after = call.get('inputs_before', {}), call.get('inputs_after', {})
    require(set(before) == set(after) == set(shapes), 'Missing/unreviewed actual SDPA input observation')
    allowed = []
    for name, shape in shapes.items():
        a, b = (dict(checked_snapshot(x[name], shape)) for x in (before, after))
        old, new = a.pop('actual_format'), b.pop('actual_format')
        require(a == b, 'Input logical values/identity/geometry changed: ' + name)
        require(old == new or (old, new) == (0, 2), 'Unreviewed native format normalization')
        if old != new:
            allowed.append({'input': name, 'from': old, 'to': new, 'logical_bytes_unchanged': True})
        require(call['sdpa_metadata'][name] == {'shape': shape, 'dtype': 'torch.float16'}, 'Actual SDPA metadata differs')
    require(call.get('allowed_input_format_normalizations') == allowed, 'Input normalization receipt differs')
    require(call['sdpa_metadata'].get('attn_mask') is None and call['sdpa_metadata'].get('dropout_p') == 0,
            'Actual Python mask/dropout differs from fixed stock model route')
    if stage == 'prefill':
        meta = call['sdpa_metadata']
        require(meta.get('is_causal') is True and meta.get('enable_gqa') is True, 'Actual causal/GQA source precondition missing')
        import struct
        scale = 128**-0.5 if meta.get('scale') is None else meta['scale']
        require(type(scale) in (int, float) and math.isfinite(scale)
                and struct.unpack('<f', struct.pack('<f', scale))[0] == EFFECTIVE_SCALE, 'Effective Python/native scale differs')
    checked_snapshot(call['output'], shapes['query'])
    require(call.get('original_api_calls') == 1 and call.get('original_return_identity_preserved') is True,
            'Original API was omitted/repeated/replaced')


def sources_and_case(root, result):
    attempt = root.parent
    manifest, manifest_sha = read(attempt / 'source_manifest.json')
    require(isinstance(manifest, dict) and manifest, 'Missing frozen controller source manifest')
    source = attempt / 'source'
    for name, expected in manifest.items():
        require(sha(child(source, name)) == expected, 'Frozen execution source mismatch: ' + name)
    selected, _ = read(attempt / 'selected_cases.json')
    rows = [row for row in selected if row.get('case_id') == root.name]
    require(len(rows) == 1, 'Case not selected exactly once by controller')
    case_file = child(source, rows[0]['file'])
    case, case_sha = read(case_file)
    require(case == rows[0]['case'] and case_sha == result['case_sha256'] == manifest[rows[0]['file']], 'Original case binding differs')
    require(case['schema'] == 'nes.model_coverage.case.v1' and case['operator_variant'] == 'private_vendor'
            and case['runtime_manifest_sha256'] == result['runtime_manifest_sha256'] == RUNTIME_SHA,
            'Case uses a different execution/runtime scope')
    require(sha(root / 'model_config.json') == case['model_config_sha256']
            and result['loader']['manifest_sha256'] == case['approved_vendor_manifest_sha256'] == BUILD_SHA,
            'Actual model config/native vendor differs from frozen case')
    binding = result['physical_device_binding']
    require(binding['device'] == case['physical_device'] == 'Ascend910B4'
            and binding['physical_card'] == 5 and binding['logical_device'] == 0
            and binding['preflight_sha256'] == sha(attempt / 'before.json')
            and binding['container_identity_sha256'] == sha(attempt / 'container_identity.json'),
            'Physical device binding differs from original controller evidence')
    require(manifest['runtime_manifest.json'] == RUNTIME_SHA, 'Validated runtime source not bound')
    require(manifest['model_context.py'] == sha(EXPERIMENT / 'model_context.py'),
            'Analyze with the exact executed context reconstruction revision')
    package = source / 'runtime/native_execution_strategy'
    cert_path = package / 'data/fa_cann91_s1s2_v1.json'
    require(sha(cert_path) == CERTIFICATE_SHA, 'Wrong validated certificate')
    loaded = load_certificate(cert_path, evidence_root=package, allow_experimental=False)
    require(loaded.data['fingerprints'] == {'source_sha256': SOURCE_SHA, 'build_sha256': BUILD_SHA}, 'Wrong native source/build')
    for relative, expected in loaded.data['python_source_files'].items():
        require(manifest['runtime/native_execution_strategy/' + relative] == expected,
                'Core Python source is not the certified closure')
    for name, item in result.get('runtime_imports', {}).items():
        prefix = '/audit/runtime/'
        require(item['path'].startswith(prefix)
                and manifest['runtime/' + item['path'][len(prefix):]] == item['sha256'], 'Actual imported runtime differs')
    require(result.get('runtime_imports'), 'Actual runtime import evidence absent')
    for key, filename in [('checkpoint_manifest_sha256', 'model_manifest.json'),
                          ('tokenizer_manifest_sha256', 'tokenizer_manifest.json'),
                          ('source_fingerprints_sha256', 'source_fingerprints.json')]:
        require(result[key] == manifest[filename], 'Model/source fingerprint not frozen')
    installed, _ = read(source / 'source_fingerprints.json')
    expected_modules = {name: {k: item[k] for k in ('path', 'sha256')} for name, item in installed['modules'].items()}
    require(result['versions'] == installed['versions'] and result['actual_source_modules'] == expected_modules,
            'Actual installed model/backend source differs')
    inputs, input_sha = read(source / 'inputs/manifest.json')
    require(input_sha == case['input_manifest_sha256'], 'Frozen input manifest differs')
    by_id = {x['request_id']: x for x in inputs['requests']}
    requests = [by_id[x] for x in case['request_ids']]
    require(len(requests) == case['batch_size'] and len(set(case['request_ids'])) == len(requests), 'Duplicate requests')
    for row in requests:
        require(len(row['prompt_token_ids']) == 2048 and json_sha(row['prompt_token_ids']) == row['prompt_token_ids_sha256'], 'Token content SHA differs')
    require(len({r['prompt_token_ids_sha256'] for r in requests}) == len(requests), 'Peers are not distinct frozen prompts')
    from native_execution_strategy.config import from_dict
    config = from_dict(case['execution_config'])
    require(config.adapter.version == 'validated' and config.selection_target == 'joint', 'Unregistered policy target')
    return case, requests, loaded, {'source_manifest_sha256': manifest_sha, 'source_files': manifest,
        'case_sha256': case_sha, 'input_manifest_sha256': input_sha, 'certificate_sha256': CERTIFICATE_SHA}


def check_statistics(path, loaded, pilot):
    report, report_sha = read(path)
    require(report.get('analysis_source_sha256') == {p.name: sha(p) for p in
            (HERE / 'compare_model_statistics.py', HERE / 'model_statistics_io.py', HERE / 'statistics_values.py')},
            'Independent statistics analyzer revision differs')
    require(report['schema'] == 'nes.model_coverage.independent_statistics.v1'
            and report['case_identity'] == loaded['identity'] and report['evidence_sha256'] == loaded['hashes'],
            'Independent statistics report belongs to different original evidence')
    expected = 'PILOT_CAPTURE_SCHEMA_VERIFIED_NO_COMPARATIVE_RUN' if pilot else 'FORMAL_COMPARISONS_COMPLETE'
    require(report['status'] == expected and report['mechanism_accepted'] is False,
            'Missing independent original statistics inference')
    require(report['localization_inputs_only'] == ['config.json', 'original dump.json', 'original stack.json', 'original construct.json'],
            'Statistics localization used an unapproved input')
    require(report['run_comparison_count'] == (0 if pilot else 5) and report['phase_comparison_count'] == (0 if pilot else 10),
            'Incomplete independent model comparisons')
    refs = {}
    for ref in report['phase_reports']:
        original, observed_sha = read(child(Path(path).parent, ref['file']))
        require(observed_sha == ref['sha256'] and original['B_step'] == ref['B_step'] and original['F_step'] == ref['F_step'],
                'Independent phase inference report changed')
        require((ref['B_step'], ref['F_step']) not in refs, 'Duplicate independent phase comparison')
        refs[ref['B_step'], ref['F_step']] = original
    if not pilot:
        require(set(refs) == {(a * 2 + s, b * 2 + s) for a, b, _ in RUN_PAIRS for s in (0, 1)}, 'Independent pair matrix differs')
        for (a, b), original in refs.items():
            # Calibration has a deliberately different manual scope stack.
            if a >= 2:
                require(original['alignment_complete'] is True, 'Session model statistics do not align; first difference UNKNOWN')
    return {'report_sha256': report_sha, 'runs': report['runs'], 'phase_reports': report['phase_reports'],
        'collector_disclosure': report['collector_disclosure'], 'first_difference_derived_from_native_control': False,
        'boundary': 'Original framework SDPA/model API statistics; not an internal CANN fault location'}


def logits_bytes(root, phase, batch):
    snapshot = phase['logits']
    checked_snapshot(snapshot, snapshot['shape'])
    require(snapshot['shape'][:2] == [batch, 1] and len(snapshot['shape']) == 3, 'Logits request mapping differs')
    path = child(root, phase['private_logits_file'])
    payload = path.read_bytes()
    require(len(payload) == snapshot['bytes'] and hashlib.sha256(payload).hexdigest() == snapshot['full_sha256'],
            'Private logits bytes differ from original observation')
    size = len(payload) // batch
    rows = [payload[i * size:(i + 1) * size] for i in range(batch)]
    require([hashlib.sha256(x).hexdigest() for x in rows] == snapshot['per_batch_sha256'], 'Private per-request logits SHA differs')
    return rows


def compare_output_rows(calls, phases, root, batch, pilot):
    if pilot:
        return {'comparisons': [], 'numerics_observed': None, 'model_logits_changed': None}
    comparisons = []
    for a, b, name in RUN_PAIRS:
        restore = name in ('calibration_vs_native', 'fault_repeat', 'native_vs_off')
        stages = []
        for stage, offset in [('prefill', 0), ('decode', 1)]:
            changes, first = [], None
            for layer in range(28):
                left, right = calls[a * 2 + offset, layer], calls[b * 2 + offset, layer]
                changed_inputs = [key for key in ('query', 'key', 'value')
                                  if left['inputs_before'][key]['full_sha256'] != right['inputs_before'][key]['full_sha256']]
                output_changed = left['output']['full_sha256'] != right['output']['full_sha256']
                changed_slots = [i for i, (x, y) in enumerate(zip(left['output']['per_batch_sha256'], right['output']['per_batch_sha256'])) if x != y]
                require(output_changed == bool(changed_slots), 'Whole/per-request output hashes contradict')
                if restore:
                    require(not changed_inputs and not output_changed, 'Native restoration/repeat values are not exact: ' + name)
                if first is None and (changed_inputs or output_changed):
                    first = {'layer_index': layer, 'boundary': 'input' if changed_inputs else 'output',
                             'changed_inputs': changed_inputs, 'output_changed': output_changed}
                changes.append({'layer_index': layer, 'changed_inputs': changed_inputs, 'output_changed': output_changed,
                                'changed_request_slots': changed_slots})
            left_rows = logits_bytes(root, phases[a * 2 + offset], batch)
            right_rows = logits_bytes(root, phases[b * 2 + offset], batch)
            changed_logits = [i for i, (x, y) in enumerate(zip(left_rows, right_rows)) if x != y]
            if restore:
                require(not changed_logits, 'Full logits restoration/repeat bytes differ: ' + name)
            stages.append({'stage': stage, 'layers': changes, 'first_runtime_hash_difference': first,
                'logits_changed_request_slots': changed_logits, 'logits_complete_bytes_compared': True})
        comparisons.append({'comparison_id': name, 'B_run_index': a, 'F_run_index': b,
            'exact_restoration_or_repeat_required': restore, 'stages': stages})
    faults = [x for x in comparisons if x['comparison_id'].startswith('native_vs_fault')]
    return {'comparisons': comparisons,
        'numerics_observed': any(y['output_changed'] for x in faults for s in x['stages'] for y in s['layers']),
        'model_logits_changed': any(s['logits_changed_request_slots'] for x in faults for s in x['stages']),
        'private_tensor_data_in_report': False,
        'scope': 'Input/output SHA and complete per-request logits bytes; numerical difference is not an accuracy claim'}


def accept_case(spec, pilot=False):
    with ExitStack() as cleanup:
        return _accept_case(spec, pilot, cleanup)


def _accept_case(spec, pilot, cleanup):
    root = Path(spec['case_dir']).resolve(strict=True)
    completion = Path(spec.get('completion', root.parent / 'completion.json'))
    loaded = load_capture(root, completion)
    result, _ = read(root / 'result.json')
    case, requests, cert, frozen = sources_and_case(root, result)
    batch = case['batch_size']
    require(case['mode_sequence'] == (['calibration'] if pilot else MODES), 'Wrong pilot/formal execution matrix')
    count = 2 if pilot else 10
    require(len(loaded['steps']) == count, 'Missing full model phases')
    framework, framework_sha = read(spec['framework_report'])
    require(framework.get('analysis_source_sha256') == {p.name: sha(p) for p in
            (HERE / 'audit_model_coverage.py', HERE / 'model_statistics_io.py', HERE / 'statistics_values.py')},
            'Independent denominator analyzer revision differs')
    recomputed = audit(loaded, root / 'model_config.json')
    for key in ('schema', 'status', 'case_identity', 'evidence_sha256', 'calls', 'phases',
                'framework_sdpa_calls', 'fault_prefill_shape_candidates', 'fault_prefill_layer_request_pairs'):
        require(framework.get(key) == recomputed[key], 'Framework denominator report disagrees with original data: ' + key)
    census, census_sha = read(spec['native_census_report'])
    log = Path(spec.get('log', root.parent / (root.name + '.log')))
    require(census['schema'] == 'nes.model_native_launch_census.v1' and census['status'] == 'PASS_NATIVE_LAUNCH_CENSUS'
            and not census['issues'] and census['case_id'] == root.name and census['batch_size'] == batch,
            'Whole-log native execution coverage UNKNOWN')
    require(census['analysis_source_sha256'] == sha(HERE / 'native_launch_census.py')
            and census['source_log']['sha256'] == sha(log) and census['source_log']['bytes'] == log.stat().st_size,
            'Whole-log census source or original log changed')
    require(census['summary']['original_sdpa_windows'] == count * 28 and len(census['calls']) == count * 28
            and census['summary']['dispatches_outside_model_phases'] == 0
            and census['summary']['unscoped_dispatches_inside_model_phases'] == 0, 'Unaccounted original native calls')
    wrappers = result['calls']
    require(len(wrappers) == result['original_api_calls'] == count * 28, 'Wrapper count differs from independent denominator')
    require(len({x['call_id'] for x in wrappers}) == len(wrappers), 'Reused original call ID')
    actual_files = {p.relative_to(root).as_posix() for p in (root / 'native_calls').glob('call_*/call.json')}
    require(actual_files == {x['file'] for x in wrappers}, 'Extra/missing call sidecars')
    native_by_id = {x['identity']['call_id']: x for x in census['calls']}
    require(set(native_by_id) == {x['call_id'] for x in wrappers}, 'Census and wrapper call sets differ')
    parser = load_original_parser(root.parent / 'source/native_evidence.py', frozen['source_files']['native_evidence.py'])
    temporary = cleanup.enter_context(tempfile.TemporaryDirectory(prefix='.accept_native_reparse_', dir=HERE))
    log_verifier = OriginalLogVerifier(log, parser, temporary)
    cleanup.callback(log_verifier.close)
    adapter = FlashAttentionAdapter(cert.path, evidence_root=cert.path.parents[1], selection_target='joint', allow_experimental=False)
    adapter.bind_evidence_directory(root)
    by_position, proofs, observations = {}, [], {}
    for row, denominator in zip(wrappers, framework['calls']):
        path = child(root, row['file']); call, observed_sha = read(path)
        require(observed_sha == row['sha256'] and call['status'] == 'CAPTURE_COMPLETED', 'Call sidecar missing or changed')
        step, layer = denominator['step'], denominator['layer_index']
        require((step, layer) not in by_position, 'Duplicate step/layer')
        phase = loaded['steps'][step]['phase']; mapping = loaded['steps'][step]['mapping']; rows = mapping['request_rows']
        validate_request_rows(rows, prefill=phase['stage'] == 'prefill')
        require([r['request_id'] for r in rows] == case['request_ids'], 'Actual request mapping differs from frozen case')
        for original, mapped in zip(requests, rows):
            require(mapped['prompt_token_ids_sha256'] == original['prompt_token_ids_sha256'], 'Actual prompt lineage differs')
        expected_ids = [v for r in requests for v in (r['prompt_token_ids'] if phase['stage'] == 'prefill' else [r['teacher_forced_next_token']])]
        require(mapping['actual_input_ids_sha256'] == json_sha(expected_ids)
                and mapping['actual_position_ids_sha256'] == json_sha(list(range(2048)) if phase['stage'] == 'prefill' else [2048])
                and mapping['cache_source_run'] == phase['run_index'], 'Frozen token/position/cache lineage differs')
        for key in ('step', 'run_index', 'mode', 'stage', 'layer_index', 'layer_name', 'module_call_id', 'request_ids'):
            require(call.get(key) == denominator.get(key), 'Framework/call one-to-one identity differs: ' + key)
        require(row['call_id'] == call['call_id'] == phase['sdpa_call_ids'][layer]
                and call['request_mapping_file'] == phase['request_mapping_file']
                and call['request_mapping_sha256'] == phase['request_mapping_sha256'], 'Call/mapping cross binding differs')
        native = native_by_id[call['call_id']]
        require(native['metadata_source']['sha256'] == observed_sha and native['sdpa_metadata'] == call['sdpa_metadata']
                and all(call['native_call'].get(k) == v for k, v in native['identity'].items()), 'Original log-window/call binding differs')
        input_integrity(call, batch, phase['stage'])
        by_position[step, layer] = call
        if phase['stage'] == 'decode':
            require(call['native_call']['scope_token'] is None and call.get('session') is None
                    and call.get('eligibility') == 'UNSUPPORTED_SHAPE_OBSERVED' and call.get('expected_injection') is False
                    and not call.get('native_record_file') and not list(path.parent.rglob('host_events*.jsonl')),
                    'Unsupported decode unexpectedly entered NES or lacks explicit passthrough evidence')
            proofs.append({'step': step, 'layer_index': layer, 'call_id': call['call_id'], 'stage': 'decode',
                'classification': 'OBSERVED_Q1_K2049_UNSUPPORTED', 'actual_route': native['observed_route'],
                'strategy_switched': None, 'request_ids': call['request_ids']})
            continue
        require(call.get('native_record_file'), 'UNKNOWN: candidate lacks actual host/launch record')
        record_path = child(root, call['native_record_file']); original_record, record_sha = read(record_path)
        require(record_sha == call['native_record_sha256'], 'Actual native record changed')
        record = remap_record(root, original_record, path.parent)
        original_reparse = log_verifier.verify(call['native_call'], native, record)
        actual, provenance = actual_native_call(record, rows, case['physical_device'])
        require(provenance['independent_descriptor_supported'] is True, 'UNKNOWN/unsupported prefill native context')
        context = adapter.inspect(actual)
        observation = adapter.parse_native_evidence(context, record)
        require(call['actual_native_context'] == context.as_dict() and call['actual_strategy'] == observation.strategy.as_dict(),
                'Declared native context/strategy differs from original raw')
        require(call['native_descriptor_provenance'] == provenance, 'Native descriptor provenance was altered')
        own = [census['attention_dispatches'][i] for i in native['launches']]
        require(len(own) == 1 and own[0]['family'] == 'FA', 'Candidate not bound to exactly one actual FA dispatch')
        launch = record['native_launch']
        require(all(own[0][k] == launch[k] for k in ('pid', 'tid', 'op_name', 'tiling_key', 'block_dim')),
                'Actual FA dispatch and public raw profile differ')
        host = record['host_event']
        require(sha(path.parent / 'host_events.jsonl') == next(x['sha256'] for x in record['evidence']
                if Path(x['path']).name == 'host_events.original.jsonl'), 'Live host event differs from original snapshot')
        plan = execution_plan(record['pod'], raw=bytes.fromhex(launch['raw_hex']))
        require(len(plan) == 169 and len(input_contract(record['pod'])) == 34, 'Incomplete execution projection')
        observations[step, layer] = observation
        if phase['mode'] == 'controlled':
            baseline = observations[2, layer]
            switched = observation.strategy.semantic_sha256 != baseline.strategy.semantic_sha256
            require(switched and host['mode'] == 'controlled' and host['selection_applied'] is True
                    and observation.strategy.parameters['query_tile'] == 64
                    and observation.strategy.parameters['key_group_ratio'] == 4, 'Eligible call did not actually switch to certified joint plan')
            session = call['session']
            require(session['decision']['matched'] is True and session['decision']['action'] == 'request_strategy'
                    and session['decision']['requested_strategy'] == observation.strategy.as_dict()
                    and session['decision']['current_semantic_sha256'] == baseline.strategy.semantic_sha256,
                    'Session decision is not bound to that layer real baseline')
        else:
            switched = False
            require(host['mode'] == 'native' and host['selection_applied'] is False
                    and observation.strategy.parameters['query_tile'] == 128
                    and observation.strategy.parameters['key_group_ratio'] == 8, 'Native/off call changed strategy')
            if step:
                require(observation.strategy.as_dict() == observations[0, layer].strategy.as_dict(), 'Complete169 native plan did not recover')
        if phase['mode'] != 'calibration':
            assessment = call['session']['assessment']
            require(call['session']['observation_error'] is None and call['session']['original_return_identity_preserved'] is True
                    and assessment['numerics_observed'] is None and assessment['model_impact'] is None
                    and assessment['strategy_switched'] is (None if phase['mode'] == 'native' else switched),
                    'Session assessment invents or contradicts measured outcomes')
        proofs.append({'step': step, 'layer_index': layer, 'call_id': call['call_id'], 'stage': 'prefill',
            'classification': 'ACTUAL_NATIVE_CERTIFIED_CONTEXT', 'actual_raw_sha256': launch['raw_sha256'],
            'strategy_semantic_sha256': observation.strategy.semantic_sha256, 'strategy_switched': switched,
            'input_contract_fields': 34, 'execution_plan_fields': 169, 'request_ids': call['request_ids'],
            'original_record_sha256': record_sha, 'original_native_reparse': original_reparse})
    require(set(by_position) == {(s, l) for s in range(count) for l in range(28)}, 'Not all original layers/phases accounted')
    numeric = compare_output_rows(by_position, {s: x['phase'] for s, x in loaded['steps'].items()}, root, batch, pilot)
    statistics = check_statistics(Path(spec['statistics_report']), loaded, pilot)
    return {'case_id': root.name, 'batch_size': batch, 'request_ids': case['request_ids'], 'mode_sequence': case['mode_sequence'],
        'frozen_bindings': frozen, 'framework_report_sha256': framework_sha, 'native_census_sha256': census_sha,
        'completion_sha256': loaded['hashes']['completion'], 'collector_disclosure': loaded['collector'],
        'framework_calls': len(proofs), 'eligible_prefill_calls': sum(x['stage'] == 'prefill' for x in proofs),
        'unsupported_decode_calls': sum(x['stage'] == 'decode' for x in proofs),
        'actual_strategy_switches': sum(x['strategy_switched'] is True for x in proofs),
        'prefill_raw_reparsed_from_original_log': log_verifier.checked,
        'switched_layer_request_pairs': sum(len(x['request_ids']) for x in proofs if x['strategy_switched'] is True),
        'calls': proofs, 'runtime_numerics': numeric, 'independent_msprobe_inference': statistics,
        'resource_scope': 'Controller clean completion and selected-card resource records; no assertion all host devices were idle'}


def accept_suite(specs, pilot=False):
    require(len(specs) in ((1, 2) if pilot else (2,)), 'Formal acceptance requires separate B1/B8 cases')
    cases = [accept_case(x, pilot) for x in specs]
    if not pilot:
        require(sorted(x['batch_size'] for x in cases) == [1, 8], 'Formal B1/B8 matrix incomplete')
        ordered = sorted(cases, key=lambda x: x['batch_size'])
        require(ordered[0]['request_ids'] == ordered[1]['request_ids'][:1], 'Target request changes between B1/B8')
        for name in ('model_probe.py', 'model_context.py', 'model_helpers.py', 'runtime_manifest.json', 'inputs/manifest.json',
                     'model_manifest.json', 'tokenizer_manifest.json', 'source_fingerprints.json'):
            require(ordered[0]['frozen_bindings']['source_files'][name] == ordered[1]['frozen_bindings']['source_files'][name],
                    'Cases belong to different frozen execution generations: ' + name)
    totals = {key: sum(x[key] for x in cases) for key in ('framework_calls', 'eligible_prefill_calls',
        'unsupported_decode_calls', 'actual_strategy_switches', 'switched_layer_request_pairs')}
    if not pilot:
        require(totals == {'framework_calls': 560, 'eligible_prefill_calls': 280, 'unsupported_decode_calls': 280,
            'actual_strategy_switches': 112, 'switched_layer_request_pairs': 504}, 'Observed totals do not fulfill preregistered all-hit matrix')
    return {'schema': 'nes.model_coverage.execution_acceptance.v1',
        'status': 'PILOT_CALIBRATION_VERIFIED_NOT_COVERAGE_ACCEPTANCE' if pilot else 'PASS_MODEL_ALL_ELIGIBLE_NATIVE_STRATEGY_COVERAGE',
        'formal_coverage_accepted': not pilot, 'native_calls_executed_by_analyzer': 0,
        'cases': cases, 'totals': totals, 'certificate_sha256': CERTIFICATE_SHA,
        'new_certificate_issued': False, 'historical_fault_reproduced': False,
        'limits': ['Strategy coverage, observed numerical change and model logits change are independent outcomes.',
            'Every eligible prefill call is checked; decode Q1/KV2049 is explicitly outside this fixed certificate.',
            'One B8 call carries eight mapped requests, not eight native calls.',
            'Later-layer B/F inputs may differ because of propagation; they are not required to match across arms.',
            'Native first numerical difference is taken from independent original statistics reports, never control/hit metadata.',
            'Exact logits comparisons do not establish accuracy loss. Internal causal mask values are not directly captured.',
            'Decode uses original direct passthrough; this does not validate the Session unsupported branch.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot-only', action='store_true')
    args = parser.parse_args()
    require(not args.output.exists(), 'Never overwrite a previous acceptance or failure report')
    suite, suite_sha = read(args.inputs)
    require(suite['schema'] == 'nes.model_coverage.acceptance_inputs.v1' and isinstance(suite['cases'], list), 'Wrong suite input schema')
    specs = []
    for row in suite['cases']:
        require(set(row) <= {'case_dir', 'framework_report', 'native_census_report', 'statistics_report', 'completion', 'log'}
                and {'case_dir', 'framework_report', 'native_census_report', 'statistics_report'} <= set(row), 'Unknown or missing suite paths')
        specs.append({k: str((args.inputs.parent / Path(v)).resolve(strict=True)) if not Path(v).is_absolute() else v for k, v in row.items()})
    try:
        result = accept_suite(specs, args.pilot_only)
        code = 0
    except (ValueError, KeyError, TypeError, OSError, AssertionError, OverflowError) as exc:
        result = {'schema': 'nes.model_coverage.execution_acceptance.v1', 'status': 'FAIL_OR_UNKNOWN_MODEL_EXECUTION_ACCEPTANCE',
            'formal_coverage_accepted': False, 'error': type(exc).__name__ + ': ' + str(exc), 'native_calls_executed_by_analyzer': 0}
        code = 1
    result['suite_input_sha256'] = suite_sha
    result['analysis_source_sha256'] = {p.name: sha(p) for p in (Path(__file__), EXPERIMENT / 'model_context.py',
        EXPERIMENT / 'native_evidence.py',
        HERE / 'audit_model_coverage.py', HERE / 'model_statistics_io.py', HERE / 'native_launch_census.py',
        HERE / 'compare_model_statistics.py', HERE / 'statistics_values.py')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'sha256': sha(args.output), 'error': result.get('error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
