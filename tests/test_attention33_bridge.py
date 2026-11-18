"""CPU refusal contracts for the natural FD stock/private bridge."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

FOLDER = Path(__file__).parents[1]/'experiments/c2c3_closure_20261117'
with patch.object(sys, 'path', [str(FOLDER), *sys.path]):
    import attention33_contract as contract
    import attention33_bridge as bridge
    import prepare_attention33_cases as preparer


class BridgeContracts(unittest.TestCase):
    def test_execution_contract_binds_complete_model_package_and_driver_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);model = root/'model';installed = root/'installed'
            model.mkdir();driver = root/'driver.py';driver.write_text('entry = 1\n')
            config = dict(model_type='qwen3', hidden_size=1024, num_hidden_layers=28,
                num_attention_heads=16, num_key_value_heads=8, head_dim=128, vocab_size=151936)
            (model/'config.json').write_text(json.dumps(config));weights = model/'model.safetensors';weights.write_bytes(b'fixture')
            for name in ('demoncat', 'fault_injection'):
                folder = installed/name;folder.mkdir(parents=True);(folder/'__init__.py').write_text('# fixture\n')
            a = contract.execution_contract(model, installed, driver, .35, {'DCAT33_REVIEWED_PHYSICAL_CARD': '2'})
            weights.write_bytes(b'changed')
            b = contract.execution_contract(model, installed, driver, .35, {'DCAT33_REVIEWED_PHYSICAL_CARD': '2'})
            self.assertNotEqual(a, b)
            self.assertEqual(a['runner']['num_gpu_blocks_override'], 128)
            self.assertEqual(a['sampling']['max_tokens'], 2)
            with self.assertRaisesRegex(RuntimeError, 'MEMORY_REQUIRED'):
                contract.execution_contract(model, installed, driver, float('nan'), {})

    def test_bridge_header_refuses_old_key_unknown_platform_or_copied_input_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);path = root/'bridge.json';reviewer = root/'review.py';reviewer.write_text('# reviewed\n')
            base = dict(schema='demoncat.paged33.stock_private_bridge.v1', status='PASS_EXACT_COMPLETE_STOCK_PRIVATE_BRIDGE',
                execution_contract={'real': 'input'}, completion_sha256='new-build',
                reviewer_source_sha256=contract.sha(reviewer), full_payload_file_pairs=435,
                native_baseline_launches=112, native_keys=[5100000000010200103], actual_aic=[24],
                raw_payloads_finite=True, runtime_launches_reverified=True, c2=False, c3=False)
            def check(data):
                path.write_text(json.dumps(data))
                return contract.verify_bridge_header(path, contract.sha(path), {'real': 'input'}, 'new-build', reviewer)
            self.assertEqual(check(base), base)
            for key, value in (('actual_aic', []), ('actual_aic', [16]),
                ('native_keys', [5000000000010200103]), ('completion_sha256', 'old-six-key'),
                ('execution_contract', {'different': 'input'}), ('full_payload_file_pairs', 434),
                ('runtime_launches_reverified', False), ('c2', True)):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(RuntimeError, 'PLATFORM_DIFFERS'):
                    changed = dict(base);changed[key] = value;check(changed)

    def test_raw_nonfinite_mutation_size_and_escape_are_independently_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary);folder = run/'measured/private';folder.mkdir(parents=True)
            path = folder/'tensor.bin';raw = np.array([1, 2], dtype='<f2').tobytes();path.write_bytes(raw)
            row = dict(path='/evidence/run/measured/private/tensor.bin', shape=[2], dtype='torch.float16',
                bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), finite=True)
            self.assertEqual(bridge.raw_tensor(run, row, [2])['bytes'], 4)
            path.write_bytes(np.array([1, float('nan')], dtype='<f2').tobytes())
            with self.assertRaisesRegex(RuntimeError, 'RAW_PAYLOAD_NONFINITE'):bridge.raw_tensor(run, row, [2])
            path.write_bytes(np.array([1, 3], dtype='<f2').tobytes())
            with self.assertRaisesRegex(RuntimeError, 'RAW_PAYLOAD_HASH_DIFFERS'):bridge.raw_tensor(run, row, [2])
            path.write_bytes(raw+b'xx')
            with self.assertRaisesRegex(RuntimeError, 'FILE_SIZE_DIFFERS'):bridge.raw_tensor(run, row, [2])
            with self.assertRaisesRegex(RuntimeError, 'PATH_INVALID'):
                bridge.evidence_path(run, '/evidence/run/../outside.bin')

    def test_comparison_reads_every_byte_even_when_metadata_hashes_are_copied(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);a = root/'a.bin';b = root/'b.bin';a.write_bytes(b'ab');b.write_bytes(b'ab')
            left = {str(i): dict(path=str(a), shape=[1], dtype='torch.float16', bytes=2, sha256='same') for i in range(87)}
            right = {k: dict(v, path=str(b)) for k, v in left.items()}
            self.assertEqual(bridge.compare_payload(left, right), 174)
            b.write_bytes(b'ac')
            with self.assertRaisesRegex(RuntimeError, 'RAW_BYTES_DIFFER'):bridge.compare_payload(left, right)
            right.pop('86')
            with self.assertRaisesRegex(RuntimeError, 'CENSUS_DIFFERS'):bridge.compare_payload(left, right)

    def test_json_duplicate_fields_and_nonfinite_are_not_silently_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'bad.json'
            for data in ('{"a":1,"a":2}', '{"a":NaN}'):
                path.write_text(data)
                with self.assertRaises(RuntimeError):contract.read_json(path)

    def test_queue_has_only_stock_and_baseline_until_actual_raw_bridge_exists(self):
        from types import SimpleNamespace
        from demoncat.runtime.vllm_ascend.worker_extension import paged33_tnd
        from fault_injection.native_adapters.attention28 import launch
        from fault_injection.native_adapters.attention28.fia import binary_registration
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);source = root/'source';folder = source/'experiments/c2c3_closure_20261117'
            folder.mkdir(parents=True);files = []
            for name in ('attention33_worker_probe.py', 'attention33_contract.py', 'attention33_bridge.py', 'prepare_attention33_cases.py'):
                p = folder/name;p.write_text('# source fixture\n');files.append(dict(path=p.relative_to(source).as_posix(), sha256=contract.sha(p)))
            (source/'source_manifest.json').write_text(json.dumps(dict(schema='demoncat.c2c3.source.v1', files=files, base_commit='frozen')))
            observer_source = source/'fault_injection/native_adapters/attention28/launch_observer.cpp'
            observer_source.parent.mkdir(parents=True);observer_source.write_text('// observer fixture\n')
            build = root/'fresh/build/run';(build/'evidence').mkdir(parents=True)
            (build/'evidence/completion.json').write_text('{}')
            (root/'fresh/controller.json').write_text(json.dumps(dict(status='BUILD_COMMAND_COMPLETED', actual_npu_calls=0,
                image_id='sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0')))
            observer = root/'observer/attention-preflight';observer.mkdir(parents=True)
            library = observer/'observer.so';library.write_bytes(b'CPU fixture')
            (observer/'result.json').write_text(json.dumps(dict(observer_compiled=True, actual_npu_calls=0,
                observer_memory_selftest=dict(exit_code=0), observer_source=dict(sha256=contract.sha(observer_source)),
                observer_library=dict(path='/build/attention-preflight/observer.so', sha256=contract.sha(library)))))
            bundle = dict(artifacts={'new.o': {}}, completion=dict(kernel_binary_sha256='new',
                variant='tnd_page16_fai_fd_s2_256', fai_public_schema_sha256='schema'))
            args = SimpleNamespace(source=source, installed=root/'installed', model=root/'model',
                build_root=build, observer=observer.parent, stage='baseline', card=2, timeout=1800,
                bridge=None, stock_run=None, baseline_run=None)
            with patch.object(paged33_tnd, 'verify_bundle', return_value=bundle), \
                patch.object(launch, 'bundle_kernel_abi', return_value=({}, {})), \
                patch.object(binary_registration, 'validate_metadata'), \
                patch.object(preparer, 'execution_contract', return_value={'bound': 'actual inputs'}), \
                patch.object(sys, 'path', list(sys.path)):
                cases = preparer.prepare(args)
                self.assertEqual(len(cases), 2)
                self.assertTrue(cases[0]['id'].endswith('stock'))
                self.assertTrue(cases[1]['id'].endswith('baseline'))
                self.assertNotIn('--statistics', cases[0]['command'])
                self.assertIn('--statistics', cases[1]['command'])
                self.assertIn('dcat_f33_fai_fd_transformer', cases[1]['command'])
                self.assertEqual(cases[1]['review']['physical_cards'], [2])
                args.stage = 'fault'
                with self.assertRaisesRegex(RuntimeError, 'PRIOR_STOCK_AND_BASELINE_REQUIRED'):preparer.prepare(args)


if __name__ == '__main__':unittest.main()
