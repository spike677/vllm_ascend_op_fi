"""CPU fixtures for independent single-call replay statistics; no framework/device imports."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import compare_replay_statistics as comparator

HERE = Path(__file__).resolve().parent
API = 'NPU.npu_fusion_attention_v3.0.forward'
MODULE = 'Module.ReplayModule.forward.0'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor(shape, mean=0.25, unknown=False):
    return {'type': 'torch.Tensor', 'dtype': 'torch.float16', 'shape': shape, 'requires_grad': False,
            **{key: None if unknown else value for key, value in
               [('Max', 1.0), ('Min', -1.0), ('Mean', mean), ('Norm', 32.0)]}}


def payload(batch=1, unknown=False):
    args = [tensor([batch, heads, 2048, 128]) for heads in (16, 8, 8)]
    outputs = [tensor([batch, 16, 2048, 128])]
    outputs += [tensor([batch, 16, 2048, 8]) for _ in range(2)]
    outputs += [tensor([1], unknown=unknown) for _ in range(3)]
    node = {'input_args': args, 'input_kwargs': {'head_num': {'type': 'int', 'value': 16}}, 'output': outputs}
    module = {'input_args': copy.deepcopy(args), 'input_kwargs': {}, 'output': copy.deepcopy(outputs)}
    stack = {'0': [[API], ['File /audit/replay.py, line 292, in forward, \n return torch_npu.npu_fusion_attention_v3(query, key, value, **kwargs)']],
             '1': [[MODULE], ['File /audit/replay.py, line 330, in main, \n native = module(query, key, value)']]}
    return {'task': 'statistics', 'level': 'mix', 'framework': 'pytorch', 'dump_data_dir': None,
            'data': {API: node, MODULE: module}}, stack, {API: MODULE, MODULE: None}


def provenance(high=True):
    path = '/audit/collector_high_a1/packages/msprobe/core/dump/data_dump/data_processor/pytorch_processor.py'
    if not high:
        return {'variant': 'official_low', 'precision': 'low', 'modified_official_collector': False,
                'actual_loaded': {'module': 'msprobe.core.dump.data_dump.data_processor.pytorch_processor',
                    'path': '/msprobe_packages/processor.py', 'method_code_file': '/msprobe_packages/processor.py',
                    'sha256': comparator.ORIGINAL_SHA}}
    return {'variant': 'task_local_fp32_high_v1', 'precision': 'high', 'modified_official_collector': True,
        'manifest_sha256': comparator.MANIFEST_SHA, 'original_processor_sha256': comparator.ORIGINAL_SHA,
        'patched_processor_sha256': comparator.PATCHED_SHA, 'artifact_count': 513,
        'provenance': 'Fixture explicitly discloses task-local modified collector.', 'policy': 'Uniform FP32 statistics temporary.',
        'actual_loaded': {'module': 'msprobe.core.dump.data_dump.data_processor.pytorch_processor', 'path': path,
                         'method_code_file': path, 'sha256': comparator.PATCHED_SHA}}


def capture(root, name, documents=None, high=True):
    directory = root / name
    dump, stack, construct = copy.deepcopy(documents or payload())
    cfg = {'task': 'statistics', 'level': 'mix', 'rank': [], 'step': [0], 'async_dump': False,
        'dump_path': '/evidence/' + name + '/observation', 'statistics': {'scope': [], 'list': [],
        'data_mode': ['all'], 'summary_mode': 'statistics', 'tensor_list': []}}
    if high:
        cfg['precision'] = 'high'
    cfg_sha = write(directory / 'config.json', cfg)
    prov = provenance(high)
    prov_sha = write(directory / 'collector_provenance.json', prov)
    native_files = []
    for kind, value in zip(('dump', 'stack', 'construct'), (dump, stack, construct)):
        file = directory / 'observation/step0/rank0' / (kind + '.json')
        fingerprint = write(file, value)
        native_files.append({'path': file.relative_to(directory).as_posix(), 'sha256': fingerprint, 'bytes': file.stat().st_size})
    receipt_sha = write(directory / 'receipt.json', {'schema': 'fault28.causal.statistics_receipt.v1', 'rank': 0,
        'steps': [0], 'config_sha256': cfg_sha, 'collector_provenance_sha256': prov_sha,
        'statistics_collector': prov, 'native_files': native_files, 'native_api_nodes': [API]})
    result_sha = write(directory / 'result.json', {'status': 'CAPTURE_COMPLETED', 'case_id': name,
        'receipt_sha256': receipt_sha, 'profile': 'POISON_NOT_LOCALIZATION_INPUT',
        'injected_location': 'Torch.wrong.999.forward'})
    completion = directory / 'completion.json'
    write(completion, {'status': 'CAPTURE_COMPLETED', 'owned_container_removed': True, 'cleanup_errors': [],
        'device_lock_released': True, 'cases': [{'case_id': name, 'rc': 0,
        'status': 'CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE', 'result_sha256': result_sha}]})
    # These intentionally invalid files must never be read or interpreted by the comparator.
    (directory / 'host_events.jsonl').write_bytes(b'\xffHOST_PROFILE_SHOULD_NOT_BE_READ')
    (directory / 'private_output').mkdir()
    (directory / 'private_output/target_fp16.bin').write_bytes(b'PRIVATE_TENSOR_SHOULD_NOT_BE_READ')
    return directory, completion


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='statistics_fixture_', dir=HERE) as temp:
        root = Path(temp).resolve()
        assert root.parent == HERE and root.is_relative_to(HERE)
        def run(name, b_data=None, f_data=None, high_b=True, high_f=True):
            bd, bc = capture(root / name, 'B', b_data, high_b)
            fd, fc = capture(root / name, 'F', f_data, high_f)
            return comparator.analyze(bd, fd, bc, fc), (bd, bc, fd, fc)
        def passed(name):
            checks.append({'name': name, 'passed': True})

        report, _ = run('identical')
        assert report['status'] == 'ALIGNED_AVAILABLE_STATISTICS_MATCH' and report['first_observed_api_difference'] is None
        passed('finite_equal_is_valid_no_difference_not_tensor_equality')
        f = payload(); f[0]['data'][API]['output'][0]['Mean'] += 0.00000001
        f[0]['data'][MODULE]['output'][0]['Mean'] += 0.00000001
        report, _ = run('primary_output', f_data=f)
        assert report['first_difference_boundary'] == 'output' and report['attention_score_output_observed_difference'] is True
        assert report['auxiliary_output_observed_differences'] == [] and not report['mechanism_accepted']
        passed('original_api_output_difference_independent_of_poison_profile_and_private_bytes')
        f = payload(); f[0]['data'][API]['output'][1]['Norm'] += 1
        report, _ = run('auxiliary_only', f_data=f)
        assert report['attention_score_output_observed_difference'] is False and report['auxiliary_output_observed_differences'] == ['softmax_max']
        passed('auxiliary_difference_is_not_attention_score_difference')
        f = payload(); f[0]['data'][API]['input_args'][0]['Mean'] += 0.01
        report, _ = run('input_difference', f_data=f)
        assert report['first_difference_boundary'] == 'input'
        passed('input_difference_reported_before_output')
        report, _ = run('null_equal', payload(unknown=True), payload(unknown=True))
        assert report['status'] == 'ALIGNED_NO_OBSERVED_API_DIFFERENCE_WITH_UNOBSERVABLE_STATISTICS'
        assert report['unobservable_statistic_fields'] > 0 and report['first_observed_api_difference'] is None
        passed('null_null_remains_unobservable')
        b = payload(); b[0]['data'][API]['output'][0]['Mean'] = None
        report, _ = run('null_finite', b_data=b)
        assert report['first_observed_api_difference'] is None and report['unobservable_statistic_fields'] > 0
        passed('null_finite_is_not_observed_numeric_difference')
        report, _ = run('batch_mismatch', payload(batch=1), payload(batch=8))
        assert report['status'] == 'INCONCLUSIVE_ALIGNMENT_OR_METADATA' and report['first_observed_api_difference'] is None
        passed('B1_B8_whole_statistics_cannot_localize_target')
        report, _ = run('precision_mismatch', high_f=False)
        assert report['status'] == 'INCONCLUSIVE_ALIGNMENT_OR_METADATA'
        passed('different_collectors_are_not_aligned')
        report, _ = run('both_low', high_b=False, high_f=False)
        assert report['status'] == 'ALIGNED_AVAILABLE_STATISTICS_MATCH'
        passed('explicit_official_low_pair_supported')
        f = payload(); f[0]['data'] = {MODULE: f[0]['data'][MODULE], API: f[0]['data'][API]}
        report, _ = run('order_mismatch', f_data=f)
        assert not report['alignment']['complete'] and report['first_observed_api_difference'] is None
        passed('reordered_trace_suppresses_global_first')
        f = payload(); f[1]['0'][1][0] = 'File /audit/replay.py, line 100, in snapshot, \n value.cpu()'
        report, _ = run('diagnostic_contamination', f_data=f)
        assert not report['alignment']['complete']
        passed('diagnostic_stack_inside_capture_is_not_silently_excluded')
        b, f = payload(), payload()
        for docs, mean in ((b, .25), (f, .5)):
            extra = {'input_args': [], 'input_kwargs': {}, 'output': [tensor([1], mean=mean)]}
            docs[0]['data'] = {'Torch.add.0.forward': extra, **docs[0]['data']}
            docs[1]['2'] = [['Torch.add.0.forward'], ['File /audit/replay.py, line 290, in forward, \n x = x + y']]
            docs[2]['Torch.add.0.forward'] = MODULE
        report, _ = run('earlier_api', b, f)
        assert report['first_observed_api_difference']['B']['name'] == 'Torch.add.0.forward'
        passed('first_difference_not_forced_to_FA_API')

        for name, mutation in [
            ('failed_completion', lambda d, c: write(c, {'status': 'FAIL'})),
            ('tampered_dump', lambda d, c: (d / 'observation/step0/rank0/dump.json').write_text('{}')),
            ('tensor_payload', lambda d, c: (d / 'observation/step0/rank0/activation.bin').write_bytes(b'x')),
            ('undisclosed_high', lambda d, c: write(d / 'collector_provenance.json', provenance(False))),
        ]:
            _, paths = run(name)
            bd, bc, fd, fc = paths
            mutation(fd, fc)
            try:
                comparator.analyze(bd, fd, bc, fc)
                raise AssertionError(name + ' accepted')
            except ValueError:
                passed(name + '_rejected')
    result = {'schema': 'fault28.causal.statistics_comparator_cpu_tests.v1', 'status': 'CPU_FIXTURES_PASSED',
        'tests': checks, 'count': len(checks), 'npu_calls': 0, 'containers_started': 0, 'real_fault_evidence': False,
        'tool_sha256': hashlib.sha256((HERE / 'compare_replay_statistics.py').read_bytes()).hexdigest(),
        'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output = HERE / 'replay_statistics_cpu_checks.json'
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'count': len(checks), 'tool_sha256': result['tool_sha256']}))


if __name__ == '__main__':
    main()
