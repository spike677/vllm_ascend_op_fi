"""Bounded real CPU torch checks. This test never creates an NPU tensor."""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    from collector_stage import select_collector, verify_loaded
    manifest_path = args.stage / 'stage_manifest.json'
    selection = select_collector({'statistics_variant': 'task_local_fp32_high_v1', 'capture_msprobe': True,
        'approved_collector_manifest_file': str(manifest_path),
        'approved_collector_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}, stage=args.stage)
    import torch
    module = importlib.import_module('msprobe.core.dump.data_dump.data_processor.pytorch_processor')
    actual = verify_loaded(selection, module)
    processor = module.PytorchDataProcessor.__new__(module.PytorchDataProcessor)
    processor.tensor_handler = module.TensorHandler()
    processor.config = SimpleNamespace(summary_mode='statistics', async_dump=False, precision='high')
    from msprobe.core.dump.data_dump.json_writer import DataWriter
    writer = DataWriter()
    writer.config = SimpleNamespace(summary_mode='statistics')
    processor.data_writer = writer
    original_tree = ast.parse((args.stage / 'original_pytorch_processor.py').read_text(encoding='utf-8'))
    original_class = next(n for n in original_tree.body if isinstance(n, ast.ClassDef) and n.name == 'PytorchDataProcessor')
    original_method = next(n for n in original_class.body if isinstance(n, ast.FunctionDef) and n.name == 'get_stat_info')
    namespace = dict(module.__dict__)
    exec(compile(ast.Module(body=[original_method], type_ignores=[]), 'original_reviewed_get_stat_info', 'exec'), namespace)
    original_get = namespace['get_stat_info']
    rows = []

    def identity(x):
        return (id(x), x.data_ptr(), x.dtype, tuple(x.shape), tuple(x.stride()), str(x.device), x.storage_offset(), x._version,
                x.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())

    def check(name, x):
        before = identity(x)
        hi = processor.get_stat_info(x, precision='high')
        lo = processor.get_stat_info(x, precision='low')
        original_low = original_get(processor, x, precision='low')
        for key in ('max', 'min', 'mean', 'norm', 'check_sum'):
            expected, actual = getattr(original_low, key), getattr(lo, key)
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            else:
                assert actual == expected or actual is None and expected is None, (name, key)
        metadata = processor._analyze_tensor(x, 'output')
        assert metadata['dtype'] == str(x.dtype) and tuple(metadata['shape']) == tuple(x.shape)
        values = writer.flush_stat_stack()
        transformed = writer._replace_stat_placeholders(metadata, values)
        metadata = metadata if transformed is None else transformed
        serialized = json.loads(json.dumps(metadata))
        assert serialized['dtype'] == str(x.dtype) and tuple(serialized['shape']) == tuple(x.shape)
        if hi.mean is not None and isinstance(hi.mean, torch.Tensor) and torch.isfinite(hi.mean):
            assert serialized['Mean'] == hi.mean.item(), name + ': writer rounded or changed the statistics'
        assert identity(x) == before, name + ': original tensor changed'
        fp64_errors = None
        expected_fp32 = x.dtype in (torch.float16, torch.bfloat16, torch.float32) and x.numel() > 0
        if expected_fp32:
            assert hi.mean is not None and hi.norm is not None
            assert hi.mean.dtype == hi.norm.dtype == torch.float32
            reference = x.detach().float()
            if x.shape:
                torch.testing.assert_close(hi.mean, reference.mean(), rtol=0, atol=0, equal_nan=True)
                torch.testing.assert_close(hi.norm, reference.norm(), rtol=0, atol=0, equal_nan=True)
            else:
                torch.testing.assert_close(hi.mean, reference, rtol=0, atol=0, equal_nan=True)
                torch.testing.assert_close(hi.norm, reference, rtol=0, atol=0, equal_nan=True)
            reference64 = x.detach().double()
            if bool(torch.isfinite(reference64).all()):
                ref_mean = reference64.mean() if x.shape else reference64
                ref_norm = reference64.norm() if x.shape else reference64
                torch.testing.assert_close(hi.mean.double(), ref_mean, rtol=2e-6, atol=2e-7)
                torch.testing.assert_close(hi.norm.double(), ref_norm, rtol=2e-6, atol=2e-7)
                fp64_errors = {'mean_abs_error': abs(hi.mean.item() - ref_mean.item()),
                               'norm_abs_error': abs(hi.norm.item() - ref_norm.item()),
                               'reference_mean': ref_mean.item(), 'reference_norm': ref_norm.item(),
                               'rtol': 2e-6, 'atol': 2e-7}
        elif x.dtype == torch.float64 and x.shape:
            assert hi.mean is None and hi.norm is None and lo.mean is None and lo.norm is None
        elif x.dtype in (torch.bool, torch.int64) and x.shape:
            assert hi.mean is None and hi.norm is None and lo.mean is None and lo.norm is None
        rows.append({'name': name, 'dtype': str(x.dtype), 'shape': list(x.shape),
                     'high_mean_dtype': str(hi.mean.dtype) if hasattr(hi.mean, 'dtype') else None,
                     'original_identity_metadata_bytes_unchanged': True,
                     'analyze_tensor_preserves_original_dtype_shape': True, 'low_matches_original_function': True,
                     'official_writer_json_preserves_statistics': True,
                     'represented_values_fp64_reference': fp64_errors})
        return hi, lo

    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        check(str(dtype) + '_vector', torch.tensor([0.3, -0.7, 0.2, 0.9], dtype=dtype))
        check(str(dtype) + '_noncontiguous', torch.arange(12, dtype=dtype).reshape(3, 4).T)
        check(str(dtype) + '_empty', torch.tensor([], dtype=dtype))
        check(str(dtype) + '_scalar', torch.tensor(-0.125, dtype=dtype))
    check('boolean', torch.tensor([True, False]))
    check('integer', torch.tensor([2, -1]))
    check('complex', torch.tensor([1 + 2j, -2 - 3j]))
    check('nan', torch.tensor([1.0, float('nan')], dtype=torch.float16))
    check('inf', torch.tensor([1.0, float('inf')], dtype=torch.bfloat16))
    # A small stored-value change hidden by FP16 scalar rounding is visible in FP32 aggregation.
    a = torch.ones(4096, dtype=torch.float16)
    b = a.clone()
    b[0] = 1.0009765625
    ah, al = check('collision_baseline', a)
    bh, bl = check('collision_changed', b)
    assert al.mean.item() == bl.mean.item() and al.norm.item() == bl.norm.item()
    assert ah.mean.item() != bh.mean.item() and ah.norm.item() != bh.norm.item()
    result = {'schema': 'fault28.msprobe.collector_cpu_check.v1', 'status': 'CPU_CHECKS_PASSED',
        'torch_version': torch.__version__, 'actual_loaded': actual, 'collector': selection, 'checks': rows,
        'npu_tensors_created': False, 'uniform_policy': 'No arm/layer/target inputs to patched collector method.',
        'collision': {'low_mean': al.mean.item(), 'high_baseline_mean': ah.mean.item(),
                      'high_changed_mean': bh.mean.item(), 'low_norm': al.norm.item(),
                      'high_baseline_norm': ah.norm.item(), 'high_changed_norm': bh.norm.item()},
        'limits': 'CPU measurement and import checks only; no NPU capture or fault-localization acceptance.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checks': len(rows), 'loaded': actual, 'collision': result['collision']}))


if __name__ == '__main__':
    main()
