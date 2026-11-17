"""Authoritative registry, with JSON-compatible YAML 1.2 and no device imports."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def definitions():
    document = json.loads((ROOT / 'registry.yaml').read_text(encoding='utf-8'))
    if document['schema'] != 'demoncat.fault_registry.v1':
        raise ValueError('REGISTRY_VERSION_UNSUPPORTED')
    rows = document['mechanisms']
    aliases = [value for row in rows for value in (row['legacy_id'], row['canonical_id'])]
    if len(aliases) != len(set(aliases)):
        raise ValueError('DUPLICATE_MECHANISM_ID')
    return deepcopy(rows)


def resolve(mechanism):
    for row in definitions():
        if mechanism in (row['legacy_id'], row['canonical_id']):
            return row
    raise ValueError('UNKNOWN_MECHANISM:' + str(mechanism))
