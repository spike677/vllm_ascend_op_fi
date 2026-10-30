"""Pinned public CANN POD, never inferred private offsets or raw-field mutation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

from ..models import json_value, require

POD_SCHEMA_SHA256 = 'f5865ad7185b7fabeb71b37aababb2b1719f79861ed70f487351dbac7248638b'
SCHEMA_FILE = Path(__file__).with_name('fa_cann91_schema.json')
_raw = SCHEMA_FILE.read_bytes()
require(hashlib.sha256(_raw).hexdigest() == POD_SCHEMA_SHA256, 'Public FA POD schema changed')
POD_SCHEMA = json.loads(_raw)
FORMATS = {'int64_t': 'q', 'uint64_t': 'Q', 'int32_t': 'i', 'uint32_t': 'I',
           'int16_t': 'h', 'uint16_t': 'H', 'int8_t': 'b', 'uint8_t': 'B', 'float': 'f'}
FIELDS = {field['name']: field for field in POD_SCHEMA['fields']}
PADDING_FIELDS = {'inputParams.remainPH': (115, 1), 'inputParams.attenMaskS2SizePH': (118, 2),
                  'inputParams.qStartIdxPH': (132, 4), 'inputParams.InputParamsPH': (154, 6),
                  'coreParams.s1SparseValidSizePH': (700, 4)}
# Exact source-audited fields only; never a PH-suffix heuristic. All205 fields
# and complete raw bytes remain in original observations and launch bindings.
for _name, (_offset, _count) in PADDING_FIELDS.items():
    require(FIELDS[_name]['offset'] == _offset and FIELDS[_name]['width'] * FIELDS[_name]['count'] == _count,
            'Source-audited alignment layout changed')
PROJECTION_FIELDS = tuple(name for name in FIELDS if not name.startswith('inputParams.') and name not in PADDING_FIELDS)
INPUT_FIELDS = tuple(name for name in FIELDS if name.startswith('inputParams.') and name not in PADDING_FIELDS)
OPAQUE_SECTIONS = {
    '@raw.softmaxFlashTilingData': (1248, 64),
    '@raw.transposeTilingData': (1312, 64),
    '@raw.transposeTilingDataTailCore': (1376, 64),
}
require(len(FIELDS) == 205 and len(PROJECTION_FIELDS) == 166 and len(INPUT_FIELDS) == 34 and POD_SCHEMA['size'] == 1464,
        'Unexpected public schema coverage')
for _name, (_offset, _size) in OPAQUE_SECTIONS.items():
    require({'name': _name.removeprefix('@raw.'), 'offset': _offset, 'size': _size} in POD_SCHEMA['sections'],
            'Public opaque section layout changed')
_covered = set()
for _field in FIELDS.values():
    _bytes = set(range(_field['offset'], _field['offset'] + _field['width'] * _field['count']))
    require(not (_covered & _bytes), 'Overlapping public POD fields')
    _covered.update(_bytes)
for _offset, _size in OPAQUE_SECTIONS.values():
    _bytes = set(range(_offset, _offset + _size))
    require(not (_covered & _bytes), 'Overlapping public opaque section')
    _covered.update(_bytes)
require(_covered == set(range(1464)), 'Unbound bytes in public POD')


def decode(raw: bytes) -> dict:
    require(type(raw) is bytes and len(raw) == 1464, 'Expected complete1464-byte public POD')
    result = {}
    for name, field in FIELDS.items():
        fmt = '<' + str(field['count']) + FORMATS[field['type']]
        require(struct.calcsize(fmt) == field['width'] * field['count'], 'Public POD width mismatch')
        value = struct.unpack_from(fmt, raw, field['offset'])
        result[name] = value[0] if len(value) == 1 else list(value)
    return json_value(result)


def execution_plan(pod: dict, *, raw: bytes) -> dict:
    require(set(pod) == set(FIELDS), 'Incomplete decoded public POD')
    require(decode(raw) == pod, 'Execution plan raw bytes differ from decoded public POD')
    result = {name: pod[name] for name in PROJECTION_FIELDS}
    # These public SDK sections have known bounds but unexpanded members.
    # Preserve every byte, without guessing internal fields or ignoring padding.
    result.update({name: raw[offset:offset + size].hex() for name, (offset, size) in OPAQUE_SECTIONS.items()})
    validate_execution_plan(result)
    return result


def validate_execution_plan(plan: dict) -> None:
    require(type(plan) is dict and set(plan) == set(PROJECTION_FIELDS) | set(OPAQUE_SECTIONS), 'Incomplete execution-plan projection')
    _validate_fields({name: plan[name] for name in PROJECTION_FIELDS})
    for name, (_, size) in OPAQUE_SECTIONS.items():
        value = plan[name]
        require(type(value) is str and len(value) == size * 2
                and all(c in '0123456789abcdef' for c in value), 'Invalid public opaque section bytes: ' + name)


def _validate_fields(plan: dict) -> None:
    json_value(plan)
    for name, value in plan.items():
        field = FIELDS[name]
        values = value if field['count'] != 1 else [value]
        require(type(values) is list and len(values) == field['count'], 'Wrong public field count: ' + name)
        require(all(type(x) in ((int, float) if field['type'] == 'float' else (int,)) for x in values),
                'Wrong public field type: ' + name)
        try:
            struct.pack('<' + str(field['count']) + FORMATS[field['type']], *values)
        except (ValueError, OverflowError, struct.error) as exc:
            raise ValueError('Public field value out of range: ' + name) from exc


def input_contract(pod: dict) -> dict:
    require(set(pod) == set(FIELDS), 'Incomplete decoded public POD')
    return {name: pod[name] for name in INPUT_FIELDS}


def validate_input_contract(contract: dict) -> None:
    require(type(contract) is dict and set(contract) == set(INPUT_FIELDS), 'Incomplete native input contract')
    _validate_fields(contract)
