"""Read-only catalog/contract export; no service, model loading or injection."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='catalog', choices=('catalog', 'host-manifest'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'catalog':
        from .capabilities import list_mechanisms
        data = {'schema': 'demoncat.remaining_mechanisms.catalog.v1',
                'mechanisms': list_mechanisms(), 'fault39': 'registered in common catalog; separate demoncat_fault39 execution package',
                'registry': 'fault_registry/registry.yaml', 'mechanism_count': 17}
    else:
        from .integration import integration_manifest
        data = integration_manifest()
    content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        # A catalog snapshot must not silently overwrite an earlier receipt.
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(content)
    else:
        print(content, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
