"""Independent model-capture integrity and raw msprobe trace reader; no host logs."""
import hashlib
import json
from pathlib import Path
import re

from statistics_values import FRAME, FORBIDDEN, canonical, disclosure, pairs, read, require

SDPA = re.compile(r'^Functional\.scaled_dot_product_attention\.\d+\.forward$')
LAYER = re.compile(r'^Module\.model\.layers\.(\d+)\.self_attn\.Qwen3Attention\.forward\.\d+$')
MODES = ('calibration', 'native', 'controlled', 'controlled', 'off')
DIAGNOSTIC = ('/audit/model_helpers.py', 'tensor_snapshot')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute() and '\\' not in name
            and '..' not in Path(name).parts, 'Unsafe relative evidence path')
    path = root / name
    require(path.resolve(strict=True).is_relative_to(root.resolve(strict=True)) and not path.is_symlink(), 'Evidence path escapes case')
    return path


def indexed_stack(stack):
    index = {}
    for row in stack.values():
        require(isinstance(row, list) and len(row) == 2 and isinstance(row[0], list) and isinstance(row[1], list), 'Unknown raw stack schema')
        for name in row[0]:
            require(name not in index, 'Duplicate stack membership')
            index[name] = row[1]
    return index


def parse_entries(dump, stack, construct):
    require(dump.get('task') == 'statistics' and dump.get('level') == 'mix'
            and dump.get('framework') == 'pytorch' and dump.get('dump_data_dir') is None
            and isinstance(dump.get('data'), dict) and dump['data'], 'Wrong original statistics format')
    index = indexed_stack(stack)
    entries, issues = [], []
    for ordinal, (name, node) in enumerate(dump['data'].items()):
        require(isinstance(node, dict) and name in construct, 'Node missing from original construct')
        parent, parents, seen = construct[name], [], {name}
        while parent is not None:
            require(parent in construct and parent not in seen, 'Invalid/cyclic original ancestry')
            seen.add(parent); parents.append(parent); parent = construct[parent]
        texts = index.get(name, [])
        parsed = []
        for frame in texts:
            require(isinstance(frame, str), 'Stack frame is not text')
            match = FRAME.match(frame)
            if match:
                parsed.append((match[1].replace('\\', '/'), int(match[2]), match[3].strip(), frame))
            else:
                issues.append({'node': name, 'reason': 'unparsed_stack_frame'})
        if not parsed:
            issues.append({'node': name, 'reason': 'missing_stack'})
        diagnostic = any((p, fn) == DIAGNOSTIC for p, _, fn, _ in parsed)
        require(not (SDPA.fullmatch(name) and diagnostic), 'Real SDPA incorrectly appears inside tensor_snapshot diagnostic')
        layer = next((int(m[1]) for p in parents if (m := LAYER.fullmatch(p))), None)
        if LAYER.fullmatch(name):
            layer = int(LAYER.fullmatch(name)[1])
        entries.append({'name': name, 'ordinal': ordinal, 'parents': parents, 'stack': texts, 'node': node,
            'layer_index': layer, 'diagnostic': diagnostic,
            'key': (canonical(name), tuple(canonical(p) for p in parents), tuple(parsed))})
    return entries, issues


def load_capture(directory, completion_file):
    root = Path(directory).resolve(strict=True)
    completion_file = Path(completion_file).resolve(strict=True)
    documents, hashes, files = {}, {}, {}
    names = {'result': 'result.json', 'config': 'config.json', 'receipt': 'receipt.json',
             'collector': 'collector_provenance.json', 'phases': 'phase_manifest.json',
             'inventory': 'module_inventory.json'}
    for key, relative in names.items():
        path = child(root, relative)
        documents[key], hashes[key] = read(path); files[key] = str(path)
    completion, hashes['completion'] = read(completion_file); files['completion'] = str(completion_file)
    result, config, receipt, provenance, phases, inventory = (documents[k] for k in names)
    require(result.get('schema') == 'nes.model_coverage.capture.v1' and result.get('status') == 'CAPTURE_COMPLETED'
            and result.get('case_id') == root.name and result.get('receipt_sha256') == hashes['receipt']
            and result.get('phase_manifest_file') == 'phase_manifest.json' and result.get('phase_manifest_sha256') == hashes['phases'],
            'Case result/phase/receipt binding missing')
    require(result.get('module_inventory_sha256') == hashes['inventory'], 'Module inventory is not sealed by result')
    require(all(re.fullmatch('[0-9a-f]{64}', result.get(key, '')) for key in ('case_sha256', 'runtime_manifest_sha256',
                'checkpoint_manifest_sha256', 'tokenizer_manifest_sha256', 'source_fingerprints_sha256')),
            'Missing frozen model/runtime/input source identity')
    require(completion.get('status') == 'CAPTURE_COMPLETED' and completion.get('owned_container_removed') is True
            and completion.get('cleanup_errors') == [] and completion.get('device_lock_released') is True, 'Attempt is not cleanly complete')
    rows = [row for row in completion.get('cases', []) if row.get('case_id') == root.name]
    require(len(rows) == 1 and rows[0].get('rc') == 0 and rows[0].get('result_sha256') == hashes['result']
            and rows[0].get('status') == 'CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE', 'Controller does not bind this complete case')
    require(config.get('task') == 'statistics' and config.get('level') == 'mix' and config.get('precision') == 'high'
            and config.get('async_dump') is False and config.get('rank') == []
            and config.get('step') in ([0, 1], list(range(10))), 'Require registered pilot/formal high statistics mix steps')
    require(config.get('statistics') == {'scope': [], 'list': [], 'data_mode': ['all'], 'summary_mode': 'statistics', 'tensor_list': []},
            'Nonstandard statistics selection or tensor dumping')
    collector = disclosure(config, provenance)
    require(receipt.get('schema') == 'nes.model_coverage.statistics_receipt.v1'
            and receipt.get('config_sha256') == hashes['config']
            and receipt.get('collector_provenance_sha256') == hashes['collector']
            and receipt.get('statistics_collector') == provenance, 'Receipt/collector binding differs')
    require(receipt.get('diagnostic_functions') == ['tensor_snapshot'], 'Unreviewed diagnostic exclusion requested')
    require(phases.get('schema') == 'nes.model_phases.v1' and isinstance(phases.get('phases'), list), 'Missing phase manifest')
    phase_rows = phases['phases']
    require([row.get('step') for row in phase_rows] == config['step'], 'Phase manifest missing or reorders steps')
    inventory_rows = inventory.get('layers', [])
    require(len(inventory_rows) == 28 and [r.get('layer_index') for r in inventory_rows] == list(range(28))
            and all(r.get('layer_name') == f'model.layers.{i}.self_attn' and r.get('class') == 'Qwen3Attention'
                    for i, r in enumerate(inventory_rows)), 'Model attention inventory differs from 28-layer architecture')
    require(isinstance(inventory.get('config_sha256'), str) and re.fullmatch('[0-9a-f]{64}', inventory['config_sha256']), 'Missing actual model config fingerprint')
    observations = root / 'observation'
    require(not any(p.is_symlink() or p.is_file() and p.suffix.lower() in FORBIDDEN for p in observations.rglob('*')), 'Tensor payload/symlink in statistics')
    expected_files, captures, mapping_rows = [], {}, {}
    for phase in phase_rows:
        step = phase['step']; run_index, phase_index = divmod(step, 2)
        stage = ('prefill', 'decode')[phase_index]
        require(phase.get('run_index') == run_index and phase.get('mode') == MODES[run_index]
                and phase.get('stage') == stage and phase.get('layer_count') == 28, 'Incorrect model phase declaration')
        mapping_path = child(root, phase['request_mapping_file'])
        mapping, mapping_sha = read(mapping_path)
        require(mapping_sha == phase['request_mapping_sha256'] and all(mapping.get(k) == phase[k] for k in ('step', 'run_index', 'mode', 'stage')), 'Request mapping not bound to phase')
        files[f'step{step}_request_mapping'] = str(mapping_path); hashes[f'step{step}_request_mapping'] = mapping_sha
        mapping_rows[step] = mapping
        payloads = {}
        for kind in ('dump', 'stack', 'construct'):
            path = child(root, f'observation/step{step}/rank0/{kind}.json')
            payloads[kind], fingerprint = read(path)
            hashes[f'step{step}_{kind}'] = fingerprint; files[f'step{step}_{kind}'] = str(path)
            expected_files.append({'path': path.relative_to(root).as_posix(), 'sha256': fingerprint, 'bytes': path.stat().st_size})
        entries, issues = parse_entries(**payloads)
        apis = [entry for entry in entries if SDPA.fullmatch(entry['name'])]
        require(len(apis) == 28 and [row['layer_index'] for row in apis] == list(range(28))
                and all(row['parents'] and LAYER.fullmatch(row['parents'][0]) for row in apis),
                'Full-model statistics lack exactly one real SDPA per attention layer')
        declared = [row for row in receipt.get('steps', []) if row.get('step') == step]
        require(len(declared) == 1 and declared[0].get('functional_sdpa_nodes') == [x['name'] for x in apis]
                and declared[0].get('parents') == [payloads['construct'][x['name']] for x in apis]
                and declared[0].get('dump_nodes') == len(entries), 'Raw SDPA nodes disagree with receipt')
        captures[step] = {'entries': entries, 'issues': issues, 'sdpa': apis, 'phase': phase, 'mapping': mapping}
    require(sorted(receipt.get('native_files', []), key=lambda x: x['path']) == sorted(expected_files, key=lambda x: x['path']), 'Raw native files do not match receipt')
    require(len(receipt.get('steps', [])) == len(captures), 'Extra receipt step')
    event_path = child(root, 'module_events.jsonl')
    require(event_path.stat().st_size <= 64 * 1024**2, 'Oversized module event ledger')
    events = [json.loads(line, object_pairs_hook=pairs) for line in event_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    files['module_events'] = str(event_path); hashes['module_events'] = sha(event_path)
    require(result.get('module_events_sha256') == hashes['module_events'], 'Module events are not sealed by result')
    # Only integrity/frozen-source identifiers from result are retained, not native call or output values.
    identity = {k: result.get(k) for k in ('case_id', 'case_sha256', 'runtime_manifest_sha256',
        'checkpoint_manifest_sha256', 'tokenizer_manifest_sha256', 'source_fingerprints_sha256', 'versions')}
    return {'root': root, 'steps': captures, 'events': events, 'inventory': inventory, 'hashes': hashes,
            'files': files, 'collector': collector, 'identity': identity,
            'config': {k: v for k, v in config.items() if k not in ('dump_path', 'step')}}
