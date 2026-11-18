"""Device-free dispatcher registration contract; executes zero operators."""
import argparse
import json
import os


def inspect_override(expected_torch='2.9.0+cpu'):
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    import torch
    if str(torch.__version__) != expected_torch:
        raise RuntimeError('EXACT_TORCH_BUILD_REQUIRED:' + str(torch.__version__))
    namespace = 'dcat_output39_registration_contract'
    name = namespace + '::identity'
    original = torch.library.Library(namespace, 'DEF')
    replacement = None
    snapshots = {}
    def record(label):
        dump = torch._C._dispatch_dump(name)
        private = [line for line in dump.splitlines() if line.startswith('PrivateUse1')]
        snapshots[label] = dict(dispatch_dump=dump, privateuse1_lines=private)
        return private
    try:
        original.define('identity(Tensor x) -> Tensor')
        original.impl('identity', lambda x: x, 'PrivateUse1')
        first = record('original')
        if len(first) != 1 or not first[0].startswith('PrivateUse1: registered at '):
            raise RuntimeError('ORIGINAL_REGISTRATION_NOT_UNIQUE')
        replacement = torch.library.Library(namespace, 'IMPL')
        replacement.impl('identity', lambda x: x, 'PrivateUse1', allow_override=True)
        overwritten = record('new_library_override')
        if (len(overwritten) != 2 or overwritten[0] == first[0]
                or overwritten[1] != first[0].replace('PrivateUse1:', 'PrivateUse1 (inactive):', 1)):
            raise RuntimeError('NEW_LIBRARY_SHADOW_NOT_VISIBLE')
        replacement._destroy()
        replacement = None
        if record('original_restored') != first:
            raise RuntimeError('OVERRIDE_REMOVAL_DID_NOT_RESTORE_ORIGINAL')
        original.impl('identity', lambda x: x, 'PrivateUse1', allow_override=True)
        same = record('same_library_override')
        if (len(same) != 2 or same[0] != first[0]
                or same[1] != first[0].replace('PrivateUse1:', 'PrivateUse1 (inactive):', 1)):
            raise RuntimeError('SAME_LIBRARY_SHADOW_NOT_VISIBLE')
    finally:
        if replacement is not None:
            replacement._destroy()
        original._destroy()
    return dict(schema='demoncat.output39.dispatch_registration_cpu.v1',
        status='PASS_DISPATCH_METADATA_ONLY_NOT_NPU_ACCEPTANCE', torch_version=str(torch.__version__),
        operator_executions=0, tensor_allocations=0, device_calls=0,
        snapshots=snapshots, owned_libraries_destroyed=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-torch', default='2.9.0+cpu')
    args = parser.parse_args(argv)
    print(json.dumps(inspect_override(args.expected_torch), indent=2))


if __name__ == '__main__':
    main()
