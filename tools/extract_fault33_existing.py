"""Read-only N2 extraction. JSON pointers select evidence, never supply missing values."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from tools.review_launch_capture_v2 import HEADER, review

YELLOW_SOURCE = '1fecdc87bb4ea3f923d25d9d57e4bad83da41060d7c986bdc1c2635b0b24a4dc'
FIELDS = ('rank', 'prompt_length', 'api', 'module', 'phase', 'q_shape', 'k_shape',
          'v_shape', 'sparse_mode', 'mask', 'actual_seq_lengths', 'kv_length',
          'block_table_valid_range', 'launch_relation')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pointer(document, path):
    if not isinstance(path, str) or not path.startswith('/'):
        raise ValueError('JSON_POINTER_REQUIRED')
    value = document
    for token in path[1:].split('/'):
        token = token.replace('~1', '/').replace('~0', '~')
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def extract(root, api_path, plan):
    root = Path(root).resolve()
    api_raw = Path(api_path).read_bytes()
    api = json.loads(api_raw)
    if plan.get('source_sha256') != YELLOW_SOURCE:
        raise ValueError('UNREVIEWED_LEGACY_COLLECTOR')
    if len(plan.get('calls', [])) != 6:
        raise ValueError('THREE_SHAPES_TWO_RANKS_REQUIRED')
    rows, captures, identities = [], {}, set()
    for call in plan['calls']:
        issues, values = [], {}
        for field in FIELDS:
            try:
                values[field] = pointer(api, call.get('pointers', {}).get(field))
            except (KeyError, IndexError, TypeError, ValueError):
                values[field] = None
            if values[field] is None:
                issues.append('MISSING:' + field)
        file = (root / call['capture']).resolve()
        if not file.is_relative_to(root):
            raise ValueError('CAPTURE_OUTSIDE_ROOT')
        raw = file.read_bytes()
        if len(raw) < HEADER.size:
            raise ValueError('TRUNCATED_CAPTURE')
        magic, pid, serial, key, blocks, size, count, width, rc, ph = HEADER.unpack_from(raw)
        if (magic != 0x444341544c415531 or not 0 < size <= 65536 or count > 128
                or width != 8 or ph != count * width or len(raw) != 80 + size + ph
                or blocks == 0 or rc != 0 or file.name != f'launch-{pid}-{serial}.bin'):
            raise ValueError('INVALID_CAPTURE')
        capture_id = (str(file.parent), pid)
        if capture_id not in captures:
            captures[capture_id] = review(file.parent, pid)
        complete = captures[capture_id]['complete']
        if not complete:
            issues.append('CAPTURE_INCOMPLETE')
        real_key = key if 0 < key < 1 << 63 else None
        if real_key is None:
            issues.append('REAL_ENTRY_UNPROVEN_NAME_MARKER_OR_ZERO')
        # A pointer must reference an actual recorded correlation, not an inferred neighbour.
        relation = values['launch_relation']
        if (not isinstance(relation, dict) or relation.get('pid') != pid
                or relation.get('serial') != serial or relation.get('unique') is not True
                or relation.get('method') not in ('recorded_correlation', 'nested_trace')):
            issues.append('UNIQUE_API_LAUNCH_RELATION_UNPROVEN')
        identity = (values['prompt_length'], values['rank'])
        if identity in identities:
            issues.append('DUPLICATE_SHAPE_RANK')
        identities.add(identity)
        q = values['q_shape']
        if values['phase'] != 'decode' or not isinstance(q, list) or not q or q[0] != 1:
            issues.append('NOT_DECODE_Q1')
        if not isinstance(values['api'], str) or 'fused_infer_attention_score' not in values['api'].lower():
            issues.append('NOT_FIA_API')
        tiling = dict(bytes=None, sha256=None, host_offset=None, evidence=None)
        spec = call.get('tiling')
        if spec is None:
            issues.append('TILING_BOUNDARY_UNPROVEN')
        else:
            index, length = spec.get('holder_index'), spec.get('bytes')
            if (type(index) is not int or not 0 <= index < count or type(length) is not int
                    or length <= 0 or not spec.get('boundary_evidence')):
                raise ValueError('TILING_BOUNDARY_INVALID')
            ptr, start = struct.unpack_from('<II', raw, 80 + size + index * width)
            if (ptr + 8 > size or start + length > size
                    or max(ptr, start) < min(ptr + 8, start + length)):
                raise ValueError('TILING_OUTSIDE_HOST_OR_OVERLAPS_POINTER')
            tiling = dict(bytes=length, sha256=sha(raw[80+start:80+start+length]),
                          host_offset=start, evidence=spec['boundary_evidence'])
        rows.append(dict(capture=call['capture'], capture_sha256=sha(raw), pid=pid,
                         serial=serial, captured_key=key, launch_key=real_key, block_dim=blocks,
                         key_evidence='YELLOW_DIFF_TEXT_REVIEW_NOT_FULL_SOURCE_AUDIT',
                         host_bytes=size, host_sha256=sha(raw[80:80+size]), tiling=tiling,
                         api=values, pointers=call.get('pointers', {}), issues=issues))
    expected = {(n, rank) for n in (128, 512, 2304) for rank in (0, 1)}
    issues = [] if identities == expected else ['THREE_SHAPES_TWO_RANKS_NOT_COVERED']
    return dict(schema='dcat.fault33.n2.existing.v1', api_sha256=sha(api_raw),
                plan_sha256=sha(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()),
                source_sha256=YELLOW_SOURCE, rows=rows, issues=issues,
                field_extraction_complete=not issues and all(not r['issues'] for r in rows),
                captures=[dict(directory=k[0], pid=k[1], complete=v['complete'],
                               issues=v['issues'], manifest_sha256=v.get('manifest_sha256'))
                          for k, v in captures.items()],
                serializer_validated=False, qualification_granted=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-root', required=True, type=Path)
    p.add_argument('--api', required=True, type=Path)
    p.add_argument('--plan', required=True, type=Path)
    a = p.parse_args()
    result = extract(a.capture_root, a.api, json.loads(a.plan.read_bytes()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['field_extraction_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
