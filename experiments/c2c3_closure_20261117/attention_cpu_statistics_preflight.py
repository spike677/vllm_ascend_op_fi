"""Exercise the actual reviewed collector and writer using CPU tensors only."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'experiments/engineering_20261117'))
from attention_cpu_statistics import CpuFp64Statistics, require, sha


def tensors(value):
    if isinstance(value, dict):
        if value.get('type') == 'torch.Tensor':
            yield value
        else:
            for child in value.values():
                yield from tensors(child)
    elif isinstance(value, list):
        for child in value:
            yield from tensors(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'FRESH_CPU_PREFLIGHT_DIRECTORY_REQUIRED')
    require(not list(Path('/dev').glob('davinci*')), 'CPU_PREFLIGHT_MUST_HAVE_NO_NPU_DEVICES')
    args.out.mkdir(parents=True)
    result = dict(schema='demoncat.attention.cpu_statistics_preflight.v1', status='RUNNING',
                  actual_npu_tensor_operations=0, native_qualification=False, errors=[])
    observer = CpuFp64Statistics()
    try:
        import torch
        from fresh_fia_qualification import collector
        from model_capture import statistics_model_call
        torch.set_num_threads(1)

        class CpuProbe(torch.nn.Module):
            def forward(self, value):
                return torch.add(value, 0.25)

        model = CpuProbe().eval()
        value = torch.tensor([1., 2., 3., 4.], dtype=torch.float16)
        observer.install()
        debugger = collector(args.out, observer.identity())
        with statistics_model_call(debugger, model, lambda: None), torch.inference_mode():
            output = model(value)
        debugger.step()
        observer.close()
        require(output.device.type == 'cpu' and torch.equal(output, value + 0.25),
                'CPU_PREFLIGHT_MODEL_OUTPUT_CHANGED')
        raw = args.out/'observation/step0/rank0'
        dump = json.loads((raw/'dump.json').read_bytes())
        require(dump['task'] == 'statistics' and dump['level'] == 'mix'
                and dump.get('dump_data_dir') is None, 'CPU_PREFLIGHT_RAW_CONFIGURATION_CHANGED')
        rows = list(tensors(dump['data']))
        require(rows and all(row.get('Mean') is not None and row.get('Norm') is not None for row in rows),
                'CPU_PREFLIGHT_FINITE_STATISTICS_MISSING')
        require(any(row['Mean'] == 2.75 and math.isclose(row['Norm'], math.sqrt(35.25), rel_tol=1e-14)
                    for row in rows), 'CPU_PREFLIGHT_SERIALIZED_FP64_VALUES_DIFFER')
        result.update(status='CPU_COLLECTOR_AND_WRITER_VERIFIED', raw_tensor_rows=len(rows),
                      files={name:sha(raw/name) for name in ('dump.json','stack.json','construct.json')},
                      config_sha256=sha(args.out/'config.json'),
                      collector_identity_sha256=sha(args.out/'collector_identity.json'))
    except BaseException as error:
        result.update(status='CPU_PREFLIGHT_FAILED')
        result['errors'].append(dict(error=repr(error), traceback=traceback.format_exc()))
    finally:
        if observer.installed:
            try:
                observer.close()
            except BaseException as error:
                result['errors'].append(dict(stage='cleanup', error=repr(error)))
                result['status'] = 'CPU_PREFLIGHT_FAILED'
        result['backend'] = observer.report()
        if result['backend']['status'] != 'NEW_COLLECTION_COMPLETE':
            result['status'] = 'CPU_PREFLIGHT_FAILED'
        with (args.out/'result.json').open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({key:result[key] for key in ('status','actual_npu_tensor_operations','errors')}))
    return 0 if result['status'] == 'CPU_COLLECTOR_AND_WRITER_VERIFIED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
