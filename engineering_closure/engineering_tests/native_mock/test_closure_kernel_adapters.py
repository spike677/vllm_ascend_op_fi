import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock

from engineering_closure.kernel_adapters import verify_fa_launch, verify_fia_launch, verify_native_build
from experiments.engineering_20261117 import fresh_fia_qualification as fresh
from fault_injection.core import UnsupportedCall
from tests.test_attention_launch_observer import fixture as launch_fixture
from tests.test_fresh_fia_qualification import fixture as build_fixture, raw_fixture
from tests.test_strategy_bridge import fixture as fa_fixture, ROOT
from fault_injection.strategy import attention_context
from native_execution_strategy.adapters.flash_attention import FlashAttentionAdapter


class KernelRouteTests(unittest.TestCase):
    def test_native_build_reuses_manager_and_rejects_other_api_receipt(self):
        manager = Mock()
        manager.verify.return_value = dict(mechanism='20.buffer_substitute', operator='aclnnMatmulWeightNz',
            recipe='fault20.weight_nz.abi0.v1', build_key='a' * 64)
        manager.verify_loaded.return_value = dict(build_key='a' * 64, pid=9)
        with self.assertRaisesRegex(UnsupportedCall, 'DOES_NOT_MATCH'):
            verify_native_build('20.buffer_substitute', 'aclnnQuantMatmulWeightNz', 'build', pid=9, manager=manager)
        manager.verify_loaded.assert_not_called()
        result = verify_native_build('20.buffer_substitute', 'aclnnMatmulWeightNz', 'build', pid=9, manager=manager)
        self.assertEqual(result['loaded']['pid'], 9)
        self.assertFalse(result['execution_verified'])

    def test_fa_refuses_fia_context_and_host_only_evidence(self):
        adapter = FlashAttentionAdapter(ROOT / 'data/fa_cann91_s1s2_v1.json', evidence_root=ROOT)
        call, native = fa_fixture()
        context = attention_context(call, adapter, native)
        with self.assertRaisesRegex(UnsupportedCall, 'S1S2_ADAPTER_AND_CONTEXT'):
            from dataclasses import replace
            verify_fa_launch(adapter, replace(context, operator='FusedInferAttentionScore'), {})
        with self.assertRaisesRegex(UnsupportedCall, 'HOST_AND_NATIVE_RECORD'):
            verify_fa_launch(adapter, context, {})

    def test_fia_verified_final_bytes_are_route_and_build_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor, _, completion, receipt = build_fixture(root)
            _, binary_sha, _, host, records = launch_fixture(root)
            binary = vendor / 'vendor/kernel/ascend910b/kernel.o'
            binary.parent.mkdir(parents=True)
            binary.write_bytes((root / 'registered-41-1.bin').read_bytes())
            metadata = binary.with_suffix('.json')
            metadata.write_text(json.dumps(dict(sha256=binary_sha, parameters=[None])))
            source = vendor / 'vendor/fused_infer_attention_score/op_kernel/fused_infer_attention_score.cpp'
            source.parent.mkdir(parents=True)
            source.write_text('void fused_infer_attention_score(__gm__ uint8_t* workspace, __gm__ uint8_t* tiling) {}')
            receipt['kernel_binary_sha256'] = binary_sha
            for path in (binary, metadata, source):
                receipt['artifacts'].append(dict(path=path.relative_to(vendor).as_posix(),
                    bytes=path.stat().st_size, sha256=fresh.sha(path)))
            completion.write_text(json.dumps(receipt))
            _, schema = raw_fixture()
            schema['fields'][0]['name'] = 'innerSplitParams.mBaseSize'
            schema['fields'][1]['name'] = 'innerSplitParams.s2BaseSize'
            schema_path = root / 'schema.json'
            schema_path.write_text(json.dumps(schema))
            mode, query_tile, key_tile, key = fresh.plan('legal_tiling', 'fault')
            values = [query_tile, key_tile, *range(2, 11)]
            raw = struct.pack('<11I', *values)
            host.update(mode=mode, eligible=True, selected_m=query_tile, tiling_key=str(key),
                serializer_roundtrip_matches=True, raw_unchanged_by_observer=True, kernel_argument_raw_observed=False,
                raw_hex=raw.hex(), raw_size=len(raw), typed={field['name']: value for field, value in zip(schema['fields'], values)})
            records[2].update(tiling_key=str(key), args_hex=(bytes(16) + raw).hex(), args_size=16 + len(raw))
            runtime = root / 'runtime.jsonl'
            runtime.write_text(''.join(json.dumps(row) + '\n' for row in records))
            inputs = dict(completion=completion, completion_sha256=fresh.sha(completion), vendor=vendor,
                public_schema=schema_path, schema_sha256=fresh.sha(schema_path), runtime_directory=root, host=host)
            verified = verify_fia_launch('28.fia.legal_tiling', phase='fault', **inputs)
            self.assertTrue(verified['launch']['kernel_argument_raw_observed'])
            self.assertIsNone(verified['numerical_effect'])
            self.assertEqual(verified['new_npu_calls'], 0)
            with self.assertRaisesRegex(UnsupportedCall, 'VARIANT_DOES_NOT_MATCH'):
                verify_fia_launch('28.fia.denominator_replay', phase='fault', **inputs)
            with self.assertRaisesRegex(UnsupportedCall, 'HOST_ROUTE_DOES_NOT_MATCH'):
                verify_fia_launch('28.fia.legal_tiling', phase='baseline', **inputs)
            records[2]['args_hex'] = (bytes(16) + struct.pack('<11I', 999, *values[1:])).hex()
            runtime.write_text(''.join(json.dumps(row) + '\n' for row in records))
            with self.assertRaisesRegex(RuntimeError, 'HOST_FINAL_ARGUMENT_BYTES_DIFFER'):
                verify_fia_launch('28.fia.legal_tiling', phase='fault', **inputs)
