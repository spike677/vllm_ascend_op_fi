"""Root-only, device-free managed DSO load/query; never installs a group.

The root runner supplies the reviewed immutable SDK image and read-only source
and bundle mounts. This program does not start containers or import torch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback


def require_zero_state(state):
    """A loaded ABI is not an installed group or a Native invocation."""
    import ctypes as C
    from fault_injection.quant_group_native import METADATA_ABI_TYPES, METADATA_CAPACITY, METADATA_DIMENSIONS
    metadata = dict(version=1, size=C.sizeof(METADATA_ABI_TYPES['state']),
        snapshot_size=C.sizeof(METADATA_ABI_TYPES['snapshot']), record_size=C.sizeof(METADATA_ABI_TYPES['record']),
        dimensions=METADATA_DIMENSIONS, capacity=METADATA_CAPACITY, total=0, retained=0, overflow=0,
        storage_bytes=METADATA_CAPACITY*C.sizeof(METADATA_ABI_TYPES['record']))
    if state.get('metadata_diagnostics') != metadata or any(type(value) is not int for value in state['metadata_diagnostics'].values()):
        raise RuntimeError('LOAD_QUERY_HAS_ACTIVITY_OR_RESOURCES:metadata_diagnostics')
    def check(value, path=()):
        if path == ('metadata_diagnostics',):
            return  # Exact fixed log ABI/storage and zero activity checked above.
        if isinstance(value, dict):
            for name, item in value.items():
                check(item, path + (name,))
        elif path in (('version',), ('size',), ('census', 'capacity')):
            if type(value) is not int or value <= 0:
                raise RuntimeError('LOADED_ABI_IDENTITY_MISSING:' + '.'.join(path))
        elif value not in (0, [], None):
            raise RuntimeError('LOAD_QUERY_HAS_ACTIVITY_OR_RESOURCES:' + '.'.join(path))
    check(state)
    if state.get('version') != 1 or state.get('size') != 416 or state.get('census', {}).get('capacity') != 32768:
        raise RuntimeError('REVIEWED_A3_ABI_AND_CENSUS_CAPACITY_REQUIRED')


def probe(bundle, approved_build_sha256):
    devices = sorted(str(path) for pattern in ('davinci*', 'devmm_svm', 'hisi_hdc') for path in Path('/dev').glob(pattern))
    if devices:
        raise RuntimeError('DEVICE_FREE_PROBE_REQUIRED:' + ','.join(devices))
    from fault_injection.quant_group_native import QuantGroupNative
    runtime = QuantGroupNative.load(bundle, approved_build_sha256=approved_build_sha256)
    state = runtime.query()
    require_zero_state(state)
    return dict(status='MANAGED_DSO_LOAD_AND_ZERO_STATE_PASS_NOT_NATIVE_ACCEPTANCE', identity=runtime.identity,
                state=state, device_nodes=devices, actual_npu_calls=0, native_qualified_targets=0, c2=False, c3=False,
                installed=False, model_invocations=0, native_helper_invocations=0,
                limitations=['Read-only ABI/state queries; no install, public operation, numerical result or healthy OFF.',
                             'Immutable container image identity and mount isolation are attested by the root runner.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--approved-build-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise RuntimeError('NEW_PROBE_DIRECTORY_REQUIRED')
    out.mkdir(parents=True)
    receipt = dict(schema='demoncat.quant.group_load_probe.v1', status='FAILED', bundle=str(args.bundle),
                   approved_build_sha256=args.approved_build_sha256, native_qualified_targets=0, c2=False, c3=False,
                   probe_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        receipt.update(probe(args.bundle, args.approved_build_sha256))
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        (out / 'result.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if receipt['status'] == 'FAILED':
        raise RuntimeError('MANAGED_LOAD_PROBE_FAILED_RECEIPT_PRESERVED')
    print(json.dumps(dict(status=receipt['status'], result=str(out / 'result.json'))))


if __name__ == '__main__':
    main()
