"""Read-only inventory of actual links in one freshly built vendor."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import stat


def main():
    p=argparse.ArgumentParser();p.add_argument('--vendor',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.vendor.resolve(strict=True);rows=[]
    for path in sorted(root.rglob('*')):
        if not path.is_symlink():continue
        resolved=path.resolve(strict=True)
        rows.append(dict(path=path.relative_to(root).as_posix(),target=os.readlink(path),
            resolved_path=str(resolved),resolved_relative=resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else None,
            lstat_mode=stat.filemode(path.lstat().st_mode),is_file=resolved.is_file(),is_dir=resolved.is_dir(),
            bytes=resolved.stat().st_size if resolved.is_file() else None,
            sha256=hashlib.sha256(resolved.read_bytes()).hexdigest() if resolved.is_file() else None))
    with a.out.open('x') as stream:json.dump(dict(vendor=str(root),links=rows,actual_npu_calls=0),stream,indent=2)
    print(json.dumps(dict(links=len(rows),output=str(a.out))))


if __name__=='__main__':main()
