"""Safe configuration and discovery CLI. Execution uses a locally bound model."""
import argparse
import json
from pathlib import Path

from . import ConfigError, example_config, registry, validate_config


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('catalog')
    p=sub.add_parser('example');p.add_argument('mechanism')
    p=sub.add_parser('validate');p.add_argument('config',type=Path)
    p=sub.add_parser('inspect');p.add_argument('mechanism')
    p=sub.add_parser('generate');p.add_argument('--check',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='catalog':data=registry.definitions()
        elif args.command=='example':data=example_config(args.mechanism)
        elif args.command=='validate':data=validate_config(json.loads(args.config.read_text(encoding='utf-8')))
        elif args.command=='inspect':data=registry.resolve(args.mechanism).inspect()
        else:
            from fault_registry.generate import generate
            data=generate(check=args.check)
        print(json.dumps(dict(ok=True,data=data,qualification_granted=False),ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    except (ConfigError,ValueError,OSError) as error:
        detail=error.to_dict() if isinstance(error,ConfigError) else dict(code='INVALID_INPUT',message=str(error))
        print(json.dumps(dict(ok=False,error=detail),ensure_ascii=False,indent=2));return 2


if __name__=='__main__':raise SystemExit(main())
