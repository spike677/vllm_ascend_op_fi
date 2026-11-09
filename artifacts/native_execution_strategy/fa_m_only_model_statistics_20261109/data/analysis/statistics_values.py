"""Shared statistics value/disclosure checks copied from reviewed c9865474; no device access."""
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
    if precision == 'high' and provenance.get('variant') == 'task_local_cpu_fp64_high_v1':
        require(provenance.get('precision') == 'high' and provenance.get('modified_official_collector') is True
                and provenance.get('original_processor_sha256') == ORIGINAL_SHA
                and provenance.get('patched_processor_sha256') == actual.get('sha256') == '6688aa3e90849027b12458add4c9da6ac4b334b57e835b742df8403c2fc474b2'
                and provenance.get('manifest_sha256') == '06ce36bae3463f44a143047d184f53915dbea4dd04b39d79960518e6878d8010'
                and provenance.get('artifact_count') == 512, 'Unknown CPU FP64 collector')
        require(actual.get('path') == '/audit/collector_fp64_a1/packages/msprobe/core/dump/data_dump/data_processor/pytorch_processor.py',
                'Unexpected CPU FP64 collector import path')
        require(isinstance(provenance.get('policy'), dict) and provenance['policy'].get('operator_data_changed') is False,
                'Missing CPU FP64 uniform observer disclosure')
    elif precision == 'high':
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

