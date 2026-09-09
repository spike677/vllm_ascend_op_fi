"""Developer CLI: list -> describe -> observe -> candidates -> export -> validate.

No model requests or mutation by default. Observe opens a bounded window and returns.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ascend_fi.inspect.client import InspectClient
from ascend_fi.inspect.plans import validate_plan
from ascend_fi.inspect.operators import OperatorEvidenceAdapter
from ascend_fi.discovery.adapters import TraceContext
from ascend_fi.discovery.resolver import load_capabilities


def read_json(path):
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('INPUT_FILE_BUDGET_EXCEEDED')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['status', 'list', 'describe', 'observe', 'poll', 'stop',
                                       'candidates', 'export', 'validate', 'recorded'])
    p.add_argument('--base-url')
    p.add_argument('--args', type=Path, help='JSON kwargs, not executable Python')
    p.add_argument('--output', type=Path)
    p.add_argument('--adapter', help='explicit trusted LOCAL lifecycle factory module:function; only validate')
    p.add_argument('--confirm-lossless', action='store_true')
    a = p.parse_args()
    args = read_json(a.args) if a.args else {}
    if a.command == 'recorded':
        adapter = OperatorEvidenceAdapter(load_capabilities(Path(__file__).resolve().parents[1] / 'profiles/capabilities'))
        data = read_json(Path(args['input']))
        if args['kind'] == 'candidate':
            result = adapter.recorded_candidate(data, source_ref=args['input'])
        elif args['kind'] == 'typed_qmm':
            result = adapter.typed_probe(data, source_ref=args['input'], runtime=args.get('runtime'))
        elif args['kind'] == 'qmm_run':
            result = adapter.native_run(data, context=TraceContext(rank=args.get('rank'),
                runtime=args.get('runtime', {}), evidence_ref=args['input']))
        elif args['kind'] == 'native_parameter':
            result = adapter.native_parameter(data, source_ref=args['input'], runtime=args.get('runtime'))
        elif args['kind'] == 'native_parameter':
            result = adapter.native_parameter(data, source_ref=args['input'], runtime=args.get('runtime'))
        else:
            raise ValueError('UNKNOWN_RECORDED_SCHEMA')
    elif a.command == 'validate':
        if not a.confirm_lossless or not a.adapter:
            p.error('validate requires --confirm-lossless and an existing leased lifecycle factory')
        module, sep, factory = a.adapter.partition(':')
        if not sep:
            p.error('adapter must be module:function')
        executor = getattr(importlib.import_module(module), factory)(args)
        result = validate_plan(args['plan'], epoch=args['service_epoch'],
                               fingerprint=args['runtime_fingerprint'], executor=executor).to_dict()
    else:
        if not a.base_url:
            p.error('--base-url required (private developer endpoint only)')
        action = {'list': 'list_modules', 'describe': 'describe_target', 'observe': 'observe_start',
                  'poll': 'observe_poll', 'stop': 'observe_stop', 'candidates': 'list_candidates',
                  'export': 'export_plan'}.get(a.command, a.command)
        result = InspectClient(a.base_url).call(action, **args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if a.output:
        with a.output.open('x', encoding='utf-8') as handle:
            handle.write(rendered + '\n')
    else:
        print(rendered)


if __name__ == '__main__':
    main()
