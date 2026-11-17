"""CLI: python -m fault_injection.native_build inspect|build|verify|loaded."""
import argparse
import json
from pathlib import Path

from .manager import BuildError, NativeBuildManager
from .recipes import recipes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--cache", type=Path)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("recipes")
    for name in ("inspect", "build"):
        sub = commands.add_parser(name)
        sub.add_argument("--mechanism", required=True)
        sub.add_argument("--operator")
        if name == "build":
            sub.add_argument("--timeout", type=int, default=1800)
    for name in ("verify", "loaded"):
        sub = commands.add_parser(name)
        sub.add_argument("--artifact", required=True, type=Path)
        if name == "loaded":
            sub.add_argument("--pid", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        manager = NativeBuildManager(repository=args.repository, cache=args.cache)
        if args.action == "recipes":
            result = [r.as_dict() for r in recipes()]
        elif args.action == "inspect":
            result = manager.inspect(args.mechanism, args.operator)
        elif args.action == "build":
            result = manager.build(args.mechanism, args.operator, timeout=args.timeout)
        elif args.action == "verify":
            result = manager.verify(args.artifact)
        else:
            result = manager.verify_loaded(args.artifact, pid=args.pid)
    except (BuildError, ValueError, OSError) as error:
        print(json.dumps(dict(status="REFUSED", reason=str(error), actual_npu_calls=0), ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
