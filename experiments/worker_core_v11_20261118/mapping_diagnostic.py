"""Device-free, read-only mapped-file diagnosis; never grants qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu
    from fault_injection.environment import inspect_environment, _mapped_images
    assert not torch_npu.npu.is_initialized(), 'DEVICE_INITIALIZATION_FORBIDDEN'
    before = Path('/proc/self/maps').read_text()
    inventory = _mapped_images(before, ())
    rows = []
    def stat_record(info):
        return dict(device=info.st_dev, major=os.major(info.st_dev), minor=os.minor(info.st_dev),
                    inode=info.st_ino, size=info.st_size, mtime_ns=info.st_mtime_ns)
    for (path, device, inode, deleted), mappings in sorted(inventory.items()):
        row = dict(path=path, mapped_device=device, mapped_inode=inode, deleted=deleted,
                   executable_mappings=mappings)
        try:
            row.update(realpath=os.path.realpath(path), path_stat=stat_record(os.stat(path)))
        except OSError as error:
            row['path_error'] = repr(error)
        row['map_files'] = []
        for address, offset in mappings:
            link = '/proc/self/map_files/' + address
            item = dict(address=address, offset=offset)
            try:
                item['link'] = os.readlink(link)
                item['stat'] = stat_record(os.stat(link))
                with open(link, 'rb') as stream:
                    item['opened_stat'] = stat_record(os.fstat(stream.fileno()))
                    item['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
            except OSError as error:
                item['error'] = repr(error)
            row['map_files'].append(item)
        rows.append(row)
    result = dict(schema='demoncat.mapping_diagnostic.v1', qualification_granted=False,
                  actual_npu_calls=0, device_initialized=torch_npu.npu.is_initialized(),
                  torch=str(torch.__version__), rows=rows, environment=inspect_environment(),
                  maps_before=before, maps_after=Path('/proc/self/maps').read_text(),
                  mountinfo=Path('/proc/self/mountinfo').read_text())
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(path=str(args.out), mappings=len(rows), actual_npu_calls=0)))


if __name__ == '__main__':
    main()
