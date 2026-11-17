"""Validate reviewed, bounded NPU jobs without granting runtime qualification."""
from pathlib import PurePosixPath
import re


ROOT = PurePosixPath('/home/17-fault-c2c3-20261117')
LABEL = '17-fault-c2c3-20261117'
LOCK = '/var/tmp/fault22_npu_2_5.lock'
REQUIRED = ('experiment_id', 'provider', 'model', 'cann', 'soc',
            'physical_cards', 'ranks', 'worker', 'build_fingerprint',
            'expected_hits', 'resource_budget', 'recovery_plan')


def validate_cases(cases, cards):
    if (not isinstance(cases, list) or not 1 <= len(cases) <= 8 or
            not cards or len(set(cards)) != len(cards) or set(cards) - {2, 5}):
        raise ValueError('BOUNDED_CASES_AND_DISTINCT_CARDS_REQUIRED')
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError('CASE_OBJECT_REQUIRED')
        name = case.get('id', '')
        if not re.fullmatch(r'[a-z0-9_-]{1,48}', name) or name in seen:
            raise ValueError('UNIQUE_SAFE_CASE_ID_REQUIRED')
        seen.add(name)
        metadata = case.get('review', {})
        if any(key not in metadata or metadata[key] is None for key in REQUIRED):
            raise ValueError('REVIEW_MANIFEST_INCOMPLETE:' + name)
        if metadata['experiment_id'] != name or metadata['physical_cards'] != cards:
            raise ValueError('CASE_RESOURCE_REVIEW_MISMATCH:' + name)
        budget = metadata['resource_budget']
        if (not isinstance(budget, dict) or not 0 < budget.get('timeout_seconds', 0) <= 1800
                or not 0 < budget.get('memory_gib', 0) <= 80
                or not 0 < budget.get('cpus', 0) <= 32):
            raise ValueError('RESOURCE_BUDGET_INVALID:' + name)
        if not isinstance(case.get('command'), str) or not case['command'].strip():
            raise ValueError('COMMAND_REQUIRED:' + name)
        if not isinstance(case.get('image'), str) or not case['image']:
            raise ValueError('IMAGE_REQUIRED:' + name)
        destinations = {'/evidence', '/msprobe_packages', '/etc/ld.so.preload',
                        '/usr/local/Ascend/driver', '/usr/local/dcmi'}
        for mount in case.get('mounts', []):
            if not isinstance(mount, list) or len(mount) != 3:
                raise ValueError('MOUNT_INVALID:' + name)
            source, destination, mode = mount
            if mode != 'ro':
                raise ValueError('ONLY_OWN_EVIDENCE_MAY_BE_WRITABLE:' + name)
            if (not isinstance(source, str) or not source.startswith('/') or
                    not isinstance(destination, str) or not destination.startswith('/') or
                    '..' in PurePosixPath(destination).parts or destination == '/' or
                    destination in destinations or ',' in source + destination):
                raise ValueError('MOUNT_PATH_INVALID:' + name)
            destinations.add(destination)
        for key, value in case.get('env', {}).items():
            if not re.fullmatch(r'[A-Z_][A-Z0-9_]*', key) or not isinstance(value, str):
                raise ValueError('ENV_INVALID:' + name)
    return cases
