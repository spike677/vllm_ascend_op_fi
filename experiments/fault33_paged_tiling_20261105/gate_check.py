"""Read-only selected-card gate; writes an unmodified resource snapshot."""
import argparse
import json
from pathlib import Path
import controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    controller.ALLOW_IDLE_IDS.update(controller.REVIEWED_IDLE_CONTAINERS)
    report = controller.resource_snapshot(5, require_idle=True)
    with Path(args.output).open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'eligible': report['eligible'],
          'unapproved_containers': [{'id': x['id'], 'name': x['metadata']['name'],
              'reasons': x['selected_card_capability_reasons']}
              for x in report['unapproved_device_capable_containers']],
          'foreign_card5_processes': report['selected_card_foreign_npu_processes']}))


if __name__ == '__main__':
    main()
