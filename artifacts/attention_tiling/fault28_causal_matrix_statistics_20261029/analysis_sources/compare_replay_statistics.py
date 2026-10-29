"""Independent native msprobe statistics comparison for a single original FA replay.

Localization reads only config/dump/stack/construct. Completion, receipt and
collector provenance establish capture integrity; profile/hit logs, replay tensor
hashes, private outputs and requested policies are never localization inputs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

STATS = ('Max', 'Min', 'Mean', 'Norm')
API = re.compile(r'^NPU\.npu_fusion_attention_v3\.\d+\.forward$')
FRAME = re.compile(r'^File (.+), line (\d+), in ([^,]+),')
ORIGINAL_SHA = 'e981aa6fe9afa1e7a1f0654db44166d7c199674b0727d8d7ffa40626b1074011'
PATCHED_SHA = '3aa7ede049bc6cb4d54a572a85f907214a98890a3c9544ed7f75e25df1306141'
MANIFEST_SHA = '743354feff5a5f91a17e68695bad167fa2584a48d257ec227d0973d071f185af'
FORBIDDEN = {'.bin', '.pt', '.pth', '.npy', '.npz', '.safetensors'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 256 * 1024**2,
            'Missing, symlinked or oversized JSON: ' + str(path))
    raw = path.read_bytes()
    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs), hashlib.sha256(raw).hexdigest()


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    return value


def canonical(name):
    if name.startswith('Module.'):
        return re.sub(r'\.(forward|backward)\.\d+$', r'.\1', name)
    return re.sub(r'\.\d+\.(forward|backward)$', r'.\1', name)


def compare_values(left, right):
    result = {'differences': [], 'metadata_differences': [], 'unobservable': [],
              'matching_finite_statistics': 0, 'tensor_slots': 0}

    def visit(a, b, path):
        ta = isinstance(a, dict) and a.get('type') == 'torch.Tensor'
        tb = isinstance(b, dict) and b.get('type') == 'torch.Tensor'
        if ta or tb:
            if not (ta and tb):
                result['metadata_differences'].append({'slot': path, 'reason': 'tensor_structure_mismatch'})
                return
            result['tensor_slots'] += 1
            for tensor in (a, b):
                require(isinstance(tensor.get('dtype'), str) and isinstance(tensor.get('shape'), list)
                        and all(type(d) is int and d >= 0 for d in tensor['shape']), 'Malformed tensor metadata')
                require(not tensor.get('data_name') and not any(k in tensor for k in ('md5', 'MD5', 'checksum')),
                        'Tensor payload/checksum is outside the statistics contract')
            ma, mb = ({k: v for k, v in x.items() if k not in STATS} for x in (a, b))
            if ma != mb:
                result['metadata_differences'].append({'slot': path, 'B': ma, 'F': mb})
                return  # Different whole-batch geometry cannot align a target request.
            for field in STATS:
                av, bv = a.get(field), b.get(field)
                item = {'slot': path + '.' + field, 'B': safe(av), 'F': safe(bv),
                        'B_present': field in a, 'F_present': field in b}
                if av is None or bv is None or type(av) not in (int, float) or type(bv) not in (int, float):
                    result['unobservable'].append(item)
                elif math.isfinite(av) and math.isfinite(bv):
                    if av != bv:
                        result['differences'].append({**item, 'kind': 'finite_statistic_difference'})
                    else:
                        result['matching_finite_statistics'] += 1
                elif safe(av) != safe(bv):
                    result['differences'].append({**item, 'kind': 'nonfinite_summary_state_difference'})
                else:
                    result['unobservable'].append(item)
            return
        if type(a) is not type(b):
            result['metadata_differences'].append({'slot': path, 'reason': 'type_mismatch'})
        elif isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    result['metadata_differences'].append({'slot': path + '.' + key, 'reason': 'missing_field'})
                else:
                    visit(a[key], b[key], path + '.' + key)
        elif isinstance(a, list):
            if len(a) != len(b):
                result['metadata_differences'].append({'slot': path, 'reason': 'sequence_length_mismatch'})
            else:
                for index, (av, bv) in enumerate(zip(a, b)):
                    visit(av, bv, path + '.' + str(index))
        elif safe(a) != safe(b):
            result['metadata_differences'].append({'slot': path, 'B': safe(a), 'F': safe(b)})

    visit(left, right, '')
    return result


def disclosure(config, provenance):
    precision = config.get('precision', 'low')
    require(precision in ('low', 'high'), 'Unsupported statistics precision')
    actual = provenance.get('actual_loaded', {})
    require(actual.get('module') == 'msprobe.core.dump.data_dump.data_processor.pytorch_processor'
            and actual.get('path') == actual.get('method_code_file'), 'Actual collector module/code path missing')
    if precision == 'high':
        require(provenance.get('variant') == 'task_local_fp32_high_v1'
                and provenance.get('precision') == 'high' and provenance.get('modified_official_collector') is True,
                'Public high mode must disclose the task-local patched collector')
        require(provenance.get('original_processor_sha256') == ORIGINAL_SHA
                and provenance.get('patched_processor_sha256') == actual.get('sha256') == PATCHED_SHA
                and provenance.get('manifest_sha256') == MANIFEST_SHA
                and provenance.get('artifact_count') == 513, 'Unreviewed high collector provenance')
        require(actual.get('path') == '/audit/collector_high_a1/packages/msprobe/core/dump/data_dump/data_processor/pytorch_processor.py',
                'Unexpected patched collector import path')
        require(isinstance(provenance.get('provenance'), str) and provenance['provenance']
                and isinstance(provenance.get('policy'), str) and provenance['policy'], 'Missing public modification disclosure')
    else:
        require(provenance.get('variant') == 'official_low' and provenance.get('precision') == 'low'
                and provenance.get('modified_official_collector') is False and actual.get('sha256') == ORIGINAL_SHA,
                'Unreviewed official-low collector')
    return {'precision': precision, 'variant': provenance['variant'], 'modified_official_collector': provenance['modified_official_collector'],
            'actual_processor_sha256': actual['sha256'], 'actual_processor_path': actual['path'],
            'stage_manifest_sha256': provenance.get('manifest_sha256'), 'declared_policy': provenance.get('policy'),
            'provenance': provenance.get('provenance'), 'stage_rehashed_by_this_tool': False}


def load(case_dir, completion_path):
    case_dir = Path(case_dir)
    hashes, documents = {}, {}
    for key, path in [('config', case_dir / 'config.json'), ('receipt', case_dir / 'receipt.json'),
                      ('collector_provenance', case_dir / 'collector_provenance.json'),
                      ('completion', Path(completion_path)), ('result_integrity_only', case_dir / 'result.json')]:
        documents[key], hashes[key] = read(path)
    config, receipt, provenance, completion, result = (documents[k] for k in
                ('config', 'receipt', 'collector_provenance', 'completion', 'result_integrity_only'))
    require(config.get('task') == 'statistics' and config.get('level') == 'mix' and config.get('step') == [0]
            and config.get('rank') == [] and config.get('async_dump') is False, 'Require synchronous single-step statistics/mix')
    settings = config.get('statistics', {})
    require(settings == {'scope': [], 'list': [], 'data_mode': ['all'], 'summary_mode': 'statistics', 'tensor_list': []},
            'Require unrestricted standard statistics configuration without tensor collection')
    collector = disclosure(config, provenance)
    require(receipt.get('schema') == 'fault28.causal.statistics_receipt.v1'
            and receipt.get('steps') == [0] and receipt.get('rank') == 0
            and receipt.get('config_sha256') == hashes['config']
            and receipt.get('collector_provenance_sha256') == hashes['collector_provenance']
            and receipt.get('statistics_collector') == provenance, 'Receipt/config/collector integrity mismatch')
    # Only these result fields are consulted. Profile, host hit, numerical fingerprints and outputs are ignored.
    require(result.get('status') == 'CAPTURE_COMPLETED' and result.get('case_id') == case_dir.name
            and result.get('receipt_sha256') == hashes['receipt'], 'Case completion/receipt binding missing')
    require(completion.get('status') == 'CAPTURE_COMPLETED' and completion.get('owned_container_removed') is True
            and completion.get('cleanup_errors') == [] and completion.get('device_lock_released') is True,
            'Attempt did not complete cleanly')
    cases = [c for c in completion.get('cases', []) if c.get('case_id') == case_dir.name]
    require(len(cases) == 1 and cases[0].get('rc') == 0
            and cases[0].get('status') == 'CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE'
            and cases[0].get('result_sha256') == hashes['result_integrity_only'], 'Attempt does not bind the completed case')
    observation = case_dir / 'observation'
    require(not any(p.is_symlink() or p.is_file() and p.suffix.lower() in FORBIDDEN for p in observation.rglob('*')),
            'Tensor payload or symlink present in statistics tree')
    expected_files = []
    for kind in ('dump', 'stack', 'construct'):
        file = observation / 'step0/rank0' / (kind + '.json')
        documents[kind], hashes[kind] = read(file)
        expected_files.append({'path': file.relative_to(case_dir).as_posix(), 'sha256': hashes[kind], 'bytes': file.stat().st_size})
    require(sorted(receipt.get('native_files', []), key=lambda r: r['path']) == sorted(expected_files, key=lambda r: r['path']),
            'Native statistics files differ from receipt')
    dump, stack, construct = (documents[k] for k in ('dump', 'stack', 'construct'))
    require(dump.get('task') == 'statistics' and dump.get('level') == 'mix' and dump.get('framework') == 'pytorch'
            and dump.get('dump_data_dir') is None and isinstance(dump.get('data'), dict) and dump['data'],
            'Wrong raw native dump contract')
    require(isinstance(stack, dict) and stack and isinstance(construct, dict) and construct, 'Missing stack/construct')
    stack_index = {}
    for row in stack.values():
        require(isinstance(row, list) and len(row) == 2 and isinstance(row[0], list) and isinstance(row[1], list),
                'Unknown native stack schema')
        for name in row[0]:
            require(name not in stack_index, 'Duplicate stack mapping')
            stack_index[name] = row[1]
    entries, issues = [], []
    for ordinal, (name, node) in enumerate(dump['data'].items()):
        require(isinstance(node, dict) and name in construct, 'Invalid/missing constructed native node')
        parent, chain, seen = construct[name], [], {name}
        while parent is not None:
            require(parent in construct and parent not in seen, 'Invalid/cyclic module ancestry')
            seen.add(parent)
            chain.append(canonical(parent))
            parent = construct[parent]
        texts, parsed = stack_index.get(name, []), []
        for text in texts:
            require(isinstance(text, str), 'Invalid stack text')
            match = FRAME.match(text)
            if not match:
                issues.append({'node': name, 'reason': 'unparsed_stack'})
            else:
                parsed.append((match[1].replace('\\', '/'), int(match[2]), match[3].strip(), text))
        if not parsed:
            issues.append({'node': name, 'reason': 'missing_stack'})
        if any(Path(p).name == 'replay.py' and fn in ('snapshot', 'raw_bytes', 'reference_queries') for p, _, fn, _ in parsed):
            issues.append({'node': name, 'reason': 'unexpected_diagnostic_inside_capture'})
        entries.append({'name': name, 'ordinal': ordinal, 'canonical': canonical(name), 'parents': chain,
                        'stack': texts, 'key': (canonical(name), tuple(chain), tuple(parsed)), 'node': node})
    native = [entry for entry in entries if API.fullmatch(entry['name'])]
    require(len(native) == 1 and receipt.get('native_api_nodes') == [native[0]['name']], 'Require one emitted original FA API call')
    require(any('ReplayModule.forward' in p for p in native[0]['parents']), 'Native API has no captured ReplayModule ancestor')
    return {'entries': entries, 'issues': issues, 'hashes': hashes, 'collector': collector,
            'config': {k: v for k, v in config.items() if k != 'dump_path'}, 'case_directory': str(case_dir)}


def public(entry):
    return {k: entry[k] for k in ('name', 'ordinal', 'canonical', 'parents', 'stack')}


def analyze(baseline, fault, baseline_completion, fault_completion):
    b, f = load(baseline, baseline_completion), load(fault, fault_completion)
    alignment = b['config'] == f['config'] and b['collector'] == f['collector'] and not b['issues'] and not f['issues']
    alignment = alignment and [e['key'] for e in b['entries']] == [e['key'] for e in f['entries']]
    rows = []
    for left, right in zip(b['entries'], f['entries']):
        if left['key'] != right['key']:
            continue
        before = compare_values({k: left['node'].get(k) for k in ('input_args', 'input_kwargs', 'parameters')},
                                {k: right['node'].get(k) for k in ('input_args', 'input_kwargs', 'parameters')})
        after = compare_values(left['node'].get('output'), right['node'].get('output'))
        row = {'B': public(left), 'F': public(right), 'input': before, 'output': after}
        if API.fullmatch(left['name']):
            ao, bo = left['node'].get('output'), right['node'].get('output')
            require(isinstance(ao, list) and isinstance(bo, list) and len(ao) == len(bo) == 6,
                    'Expected original public FA six-return statistics')
            row['return_slots'] = [{'index': i, 'name': name, **compare_values(ao[i], bo[i])} for i, name in
                enumerate(('attention_score', 'softmax_max', 'softmax_sum', 'softmax_out', 'seed', 'offset'))]
        rows.append(row)
    metadata = any(row[side]['metadata_differences'] for row in rows for side in ('input', 'output'))
    complete = alignment and not metadata
    api_rows = [row for row in rows if not row['B']['name'].startswith('Module.')]
    changed = [row for row in api_rows if row['input']['differences'] or row['output']['differences']]
    first = changed[0] if complete and changed else None
    unobservable = sum(len(row[side]['unobservable']) for row in rows for side in ('input', 'output'))
    module_changed = [row['B']['name'] for row in rows if row['B']['name'].startswith('Module.')
                      and (row['input']['differences'] or row['output']['differences'])]
    native_row = next((row for row in rows if API.fullmatch(row['B']['name'])), None)
    if not complete:
        status = 'INCONCLUSIVE_ALIGNMENT_OR_METADATA'
    elif first is not None:
        status = 'ALIGNED_OBSERVED_API_STATISTICAL_DIFFERENCE'
    elif module_changed:
        status = 'ALIGNED_MODULE_DIFFERENCE_WITHOUT_API_DIFFERENCE'
    elif unobservable:
        status = 'ALIGNED_NO_OBSERVED_API_DIFFERENCE_WITH_UNOBSERVABLE_STATISTICS'
    else:
        status = 'ALIGNED_AVAILABLE_STATISTICS_MATCH'
    return {'schema': 'fault28.causal.independent_statistics_comparison.v1', 'status': status,
        'analysis_inputs': {'localization': ['config.json', 'dump.json', 'stack.json', 'construct.json'],
            'integrity_only': ['completion.json', 'receipt.json', 'collector_provenance.json', 'result.json: status/case_id/receipt_sha256'],
            'files_not_opened': ['host/profile/hit logs', 'private tensor files'],
            'ignored_result_fields': ['profile/policy labels', 'native-call/hit annotations', 'input/output numerical fingerprints',
                                      'private output paths and numerical comparison results']},
        'evidence_sha256': {'B': b['hashes'], 'F': f['hashes']}, 'collector_disclosure': {'B': b['collector'], 'F': f['collector']},
        'alignment': {'complete': complete, 'trace_config_collector_match': alignment, 'metadata_compatible': not metadata,
            'B_nodes': len(b['entries']), 'F_nodes': len(f['entries']), 'matched_ordered_nodes': len(rows),
            'B_issues': b['issues'], 'F_issues': f['issues'],
            'method': 'Full parsed source stack including line/code text, module ancestry, API identity and original dump order; no diagnostic exclusions.'},
        'first_observed_api_difference': first,
        'first_difference_boundary': ('input' if first['input']['differences'] else 'output') if first else None,
        'attention_score_output_observed_difference': bool(native_row['return_slots'][0]['differences']) if complete and native_row else None,
        'auxiliary_output_observed_differences': [slot['name'] for slot in native_row['return_slots'][1:]
                                                if slot['differences']] if complete and native_row else None,
        'unobservable_statistic_fields': unobservable, 'module_difference_nodes': module_changed,
        'comparisons': rows, 'tensor_equality_proven': False, 'physical_fault_location_proven': False,
        'native_cann_kernel_coverage_proven': False, 'mechanism_accepted': False,
        'limits': ['First means first differing API statistics record in dump order, not a device instruction or physical fault point.',
            'Null/missing/matching nonfinite statistics are unobservable, never equal tensor values. Equal finite summaries may collide.',
            'Matching input statistics do not prove identical Q/K/V/mask values. First output difference does not by itself prove the root cause.',
            'Different batch shapes cannot align target slices using whole-tensor summaries; incompatible metadata suppresses first-difference claims.',
            'Attention-score return0 and the five auxiliary returns are reported separately. An auxiliary-only change is not a primary attention-output difference.',
            'Completion and provenance checks are integrity gates, not independent resource auditing or native profile/tiling verification.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='Completed baseline case directory')
    parser.add_argument('--fault', type=Path, required=True, help='Completed comparison case directory')
    parser.add_argument('--baseline-completion', type=Path, required=True)
    parser.add_argument('--fault-completion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh JSON report; input files remain unchanged')
    args = parser.parse_args()
    require(not args.output.exists(), 'Report must be new')
    try:
        report = analyze(args.baseline, args.fault, args.baseline_completion, args.fault_completion)
        rc = 0 if report['alignment']['complete'] else 2
    except (ValueError, KeyError, TypeError, OSError) as error:
        report = {'schema': 'fault28.causal.independent_statistics_comparison.v1', 'status': 'INVALID_CAPTURE_EVIDENCE',
                  'error': str(error), 'mechanism_accepted': False, 'physical_fault_location_proven': False}
        rc = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output), 'mechanism_accepted': False}))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
