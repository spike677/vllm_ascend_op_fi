"""Offline saved-report consistency audit; it grants no runtime permission."""
import argparse
import hashlib
import json
from pathlib import Path

from fault_injection.coverage_audit import audit_saved_coverage, index_artifacts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--pointer',default='',help='JSON pointer to coverage report in original receipt')
    parser.add_argument('--c2-report',type=Path)
    parser.add_argument('--artifact',type=Path,action='append',default=[])
    parser.add_argument('--artifact-directory',type=Path,action='append',default=[])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    raw=args.report.read_bytes();report=json.loads(raw)
    if args.pointer:
        if not args.pointer.startswith('/'):raise ValueError('JSON_POINTER_REQUIRED')
        for part in args.pointer.split('/')[1:]:
            key=part.replace('~1','/').replace('~0','~')
            report=report[int(key)] if isinstance(report,list) else report[key]
    paths=list(args.artifact)
    for directory in args.artifact_directory:
        paths.extend(sorted(directory.glob('*.json')))
    result=audit_saved_coverage(report,artifact_index=index_artifacts(paths),
        c2_report=json.loads(args.c2_report.read_bytes()) if args.c2_report else None)
    result.update(original_receipt_sha256=hashlib.sha256(raw).hexdigest(),json_pointer=args.pointer)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
