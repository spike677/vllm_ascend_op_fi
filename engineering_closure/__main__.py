"""CLI for offline engineering audits and reviewed NPU case preparation."""
import argparse
import json
from pathlib import Path
import sys


def emit(value, output=None):
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if output:
        with Path(output).open('x', encoding='utf-8') as stream:
            stream.write(raw)
    else:
        print(raw, end='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit', help='Audit the 17-mechanism checkpoint and source hashes')
    audit.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    audit.add_argument('--out', type=Path)
    evidence = commands.add_parser('evidence', help='Join original artifacts in one explicit run scope')
    evidence.add_argument('bundle', type=Path)
    evidence.add_argument('--artifact-root', type=Path, required=True)
    evidence.add_argument('--out', type=Path)
    queue = commands.add_parser('queue', help='Show missing prerequisites for all five batches')
    queue.add_argument('--directory', type=Path, default=Path(__file__).with_name('npu_execution'))
    queue.add_argument('--receipts', type=Path)
    queue.add_argument('--evidence-root', type=Path)
    queue.add_argument('--out', type=Path)
    prepare = commands.add_parser('prepare', help='Generate cases for the existing host controller')
    prepare.add_argument('kind', choices=('worker18-statistics', '39.output_proxy', '41.tail_column'))
    prepare.add_argument('--options', type=Path, required=True)
    prepare.add_argument('--out', type=Path, required=True)
    graph = commands.add_parser('path', help='Normalize a saved actual NaturalTensorObserver census')
    graph.add_argument('census', type=Path)
    graph.add_argument('--out', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'audit':
            from .evidence import audit_checkpoint
            result = audit_checkpoint(args.repo)
        elif args.command == 'evidence':
            from .evidence import audit_bundle
            result = audit_bundle(json.loads(args.bundle.read_bytes()), artifact_root=args.artifact_root)
        elif args.command == 'queue':
            from .queue import plan_queue
            receipts = json.loads(args.receipts.read_bytes()) if args.receipts else []
            result = plan_queue(args.directory, receipts=receipts, evidence_root=args.evidence_root)
        elif args.command == 'prepare':
            from .queue import prepare_cases
            result = prepare_cases(args.kind, json.loads(args.options.read_bytes()))
        else:
            from .path_discovery import normalize_execution_path
            result = normalize_execution_path(json.loads(args.census.read_bytes()))
        emit(result, args.out)
        return 2 if isinstance(result, dict) and result.get('errors') else 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
