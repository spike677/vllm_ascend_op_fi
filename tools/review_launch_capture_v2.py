"""Validate capture completeness; does not decode tiling or grant 33 admission."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

HEADER = struct.Struct('<10Q')
ERRORS = ('unknown', 'oversize', 'capacity_drop', 'open_error', 'short_write',
          'close_error', 'map_drop', 'copy_error', 'launch_error')
COUNTS = ('seen', 'unarmed', 'filtered', 'selected', 'written', 'reserved_bytes') + ERRORS


def review(directory, pid):
    root = Path(directory)
    manifest = root / f'capture-{pid}.json'
    issues = []
    try:
        raw = manifest.read_bytes()
        report = json.loads(raw)
        if (report.get('schema') != 'dcat.launch.capture.v2' or report.get('pid') != pid
                or report.get('config_valid') is not True
                or report.get('finalized') is not True or report.get('pending_writes') != 0
                or any(type(report.get(k)) is not int or report[k] < 0 for k in COUNTS)
                or report['selected'] > 1000000):
            raise ValueError('invalid manifest fields')
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return dict(schema='dcat.launch.capture.review.v2', complete=False,
                    issues=['FINAL_MANIFEST_MISSING_OR_INVALID:' + str(error)], qualification_granted=False)
    if report['seen'] != report['unknown'] + report['filtered'] + report['selected']:
        issues.append('CENSUS_ACCOUNTING_MISMATCH')
    dropped = sum(report[k] for k in ('oversize', 'capacity_drop', 'open_error',
                                     'short_write', 'close_error', 'copy_error'))
    if report['selected'] != report['written'] + dropped:
        issues.append('WRITE_ACCOUNTING_MISMATCH')
    issues += [k.upper() for k in ERRORS if report[k]]
    if report['selected'] == 0:
        issues.append('NO_SELECTED_LAUNCHES')
    records = []
    serials = set()
    for path in sorted(root.glob(f'launch-{pid}-*.bin')):
        data = path.read_bytes()
        try:
            magic, actual_pid, serial, key, blocks, size, count, holder_size, rc, ph_size = HEADER.unpack_from(data)
            if (magic != 0x444341544c415531 or actual_pid != pid or blocks == 0
                    or not 0 < size <= 65536 or count > 128 or holder_size == 0
                    or ph_size != count * holder_size or rc != 0
                    or len(data) != HEADER.size + size + ph_size
                    or path.name != f'launch-{pid}-{serial}.bin' or serial in serials):
                raise ValueError('invalid header, length, serial, or native return')
            serials.add(serial)
            record = dict(file=path.name, serial=serial, key=key, blocks=blocks,
                          bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            if report.get('binding_metadata') is True:
                binding_raw = Path(str(path) + '.binding.json').read_bytes()
                binding = json.loads(binding_raw)
                if (binding.get('schema') != 'dcat.launch.binding.v1'
                        or binding.get('pid') != pid or binding.get('serial') != serial
                        or binding.get('key_source') not in ('entry', 'name_only')
                        or not isinstance(binding.get('function_name'), str)):
                    raise ValueError('BINDING_IDENTITY_INVALID')
                entry = binding['key_source'] == 'entry'
                if ((entry and (type(binding.get('entry')) is not int or binding['entry'] != key))
                        or (not entry and (binding.get('entry') is not None or key != 0
                                          or not binding['function_name']))):
                    raise ValueError('BINDING_KEY_INVALID')
                record.update(key=key if entry else None, captured_key=key,
                              key_source=binding['key_source'], function_name=binding['function_name'],
                              binding_sha256=hashlib.sha256(binding_raw).hexdigest())
            else:
                record['key_source'] = 'legacy_unverified'
            records.append(record)
        except (OSError, TypeError, AttributeError, struct.error, ValueError) as error:
            issues.append('INVALID_RECORD:' + path.name + ':' + str(error))
    if len(records) != report['written']:
        issues.append('FILE_COUNT_MISMATCH')
    if serials != set(range(report['selected'])):
        issues.append('SELECTED_SERIALS_NOT_COMPLETE')
    return dict(schema='dcat.launch.capture.review.v2', complete=not issues, issues=issues,
                manifest_sha256=hashlib.sha256(raw).hexdigest(), census=report, records=records,
                name_only_records=sum(r.get('key_source') == 'name_only' for r in records),
                qualification_granted=False, serializer_validated=False,
                scope='ARMED_WINDOW_SELECTED_KEYS_ONLY')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--pid', required=True, type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.directory, args.pid)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('complete', 'issues', 'qualification_granted')}))
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
