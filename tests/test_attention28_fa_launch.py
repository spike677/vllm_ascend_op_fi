"""FA CPU decoding/refusal contracts; synthetic traces are not NPU evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import uuid

from fault_injection.native_adapters.attention28.fa import launch as fa
from tests.test_attention_launch_observer import fixture as runtime_fixture

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / 'native_execution_strategy/adapters/fa_cann91_schema.json').read_bytes())
spec = importlib.util.spec_from_file_location('new_fa_probe_tests', ROOT / 'experiments/c2c3_closure_20261117/attention28_fa_launch_probe.py')
probe = importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def pod_bytes():
    raw = bytearray(1464)
    values = {'coreParams.s1BaseSize': 128, 'coreParams.s2BaseSize': 128, 'coreParams.nRatio': 8,
              'inputParams.bSize': 1, 'inputParams.n2Size': 8, 'inputParams.gSize': 2,
              'inputParams.s1Size': 2048, 'inputParams.s2Size': 2048, 'inputParams.dSize': 128,
              'coreParams.d2Size': 128, 'inputParams.keepProb': 1, 'inputParams.scaleValue': probe.EFFECTIVE_SCALE}
    for field in SCHEMA['fields']:
        if field['name'] in values:struct.pack_into('<' + fa.FORMATS[field['type']], raw, field['offset'], values[field['name']])
    return bytes(raw)


def metadata(digest):
    return dict(sha256=digest, parameters=[None] * 24, coreType='MIX', intercoreSync=1,
                taskRation='tilingKey', opParaSize=1472, kernelList=[dict(tilingKey=fa.KEY,
                kernelType='MIX_AIC', crossCoreSync=1, taskRation='1:2')])


def host_projection(pod):
    mapping = {'m': 'coreParams.s1BaseSize', 's2_base': 'coreParams.s2BaseSize', 'n_ratio': 'coreParams.nRatio',
        's1_outer': 'coreParams.s1OuterSize', 's2_outer': 'coreParams.s2OuterSize',
        's1_tail': 'coreParams.s1BaseTailSize', 's2_tail': 'coreParams.s2BaseTailSize',
        'core_num': 'multiCoreParams.coreNum', 'total_query_tasks': 'multiCoreParams.totalSize',
        'split_factor': 'multiCoreParams.splitFactorSize', 'split_factor_tail': 'multiCoreParams.splitFactorTailSize',
        'sparse_start_idx': 'multiCoreParams.sparseStartIdx', 'sparse_type': 'inputParams.sparseType',
        's1_sparse_valid_size': 'coreParams.s1SparseValidSize', 's2_sparse_valid_size': 'coreParams.s2SparseValidSize',
        'b': 'inputParams.bSize', 'n2': 'inputParams.n2Size', 'g': 'inputParams.gSize',
        's1': 'inputParams.s1Size', 's2': 'inputParams.s2Size', 'd': 'inputParams.dSize', 'dv': 'coreParams.d2Size',
        'impl_mode': 'inputParams.implMode', 'keep_prob': 'inputParams.keepProb'}
    result = {name: pod[field] for name, field in mapping.items()}
    result.update(scale=pod['inputParams.scaleValue'], n1=16, s2_group=1024)
    for key, prefix in (('bmm1', 'bmm1TilingData.'), ('bmm2', 'bmm2TilingData.')):
        result[key] = {name[len(prefix):]: value for name, value in pod.items() if name.startswith(prefix)}
    return result


def fa_fixture(root):
    call, digest, _, host, records = runtime_fixture(root)
    raw = pod_bytes();pod = fa.decode_public_pod(raw, SCHEMA)
    host.update(schema='nes.fa.host_observation.v1', callback_status=0, eligible=True,
        certificate_verified_by_host=False, new_build_runtime_validation_inherited=False,
        raw_hex=raw.hex(), raw_size=len(raw), tiling_key=str(fa.KEY), typed=host_projection(pod),
        guard_reports=[dict(passed=True, failed_guards=[])], workspace_sizes=[16384],
        selected_factors=dict(m=128, n_ratio=8))
    observed = records[2]
    observed.update(tiling_key=str(fa.KEY), argument_layout='aclrt_host_placeholders', function_handle=99,
        placeholders_hex=struct.pack('<II', 200, 216).hex(), args_size=216 + 1464,
        args_hex=(bytes(216) + raw).hex(), has_tiling=0, tiling_addr_offset=0, tiling_data_offset=0)
    records.insert(1, dict(records[0], event='function_bind', monotonic_ns=2, function_handle=99, tiling_key=str(fa.KEY)))
    bundle = dict(completion=dict(kernel_binary_sha256=digest, public_pod_decoder_sha256='d' * 64),
        schema=SCHEMA, metadata=metadata(digest), decoder='CPU fixture decoder',
        abi=dict(operator='FlashAttentionScore', pointer_bytes=8, names=list(fa.NAMES), tiling_pointer_offset=200))
    return call, host, records, bundle, pod


class FaLaunchTests(unittest.TestCase):
    def verify(self, root, fixture, *, off=False, decoder=None):
        call, host, records, bundle, pod = fixture
        (root / 'runtime.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
        decoded = pod if decoder is None else decoder
        document = dict(schema='nes.fa.public_pod_values.v1', raw_size=1464, fields=decoded)
        with patch.object(fa.subprocess, 'run', return_value=NS(stdout=json.dumps(document))):
            return fa.verify_fa_launch(root, call, bundle, None if off else host,
                                      root / (uuid.uuid4().hex + '.bin'), unscoped_off=off)

    def test_full_public_schema_retains_opaque_and_padding_bytes(self):
        raw = bytearray(pod_bytes());raw[1248] = 0x7a;raw[115] = 0x55
        pod = fa.decode_public_pod(bytes(raw), SCHEMA)
        self.assertEqual(len(pod), 205);self.assertEqual(pod['inputParams.remainPH'], 0x55)
        projection = fa.semantic_pod(bytes(raw), SCHEMA)
        self.assertEqual(projection['opaque']['softmaxFlashTilingData'][:2], '7a')
        self.assertNotIn('inputParams.remainPH', projection['fields'])
        old = copy.deepcopy(SCHEMA);old['size'] = 1576
        with self.assertRaisesRegex(RuntimeError, 'FA_FRESH_PUBLIC_POD_SCHEMA_REQUIRED'):fa.decode_public_pod(bytes(1576), old)
        bad = copy.deepcopy(SCHEMA);bad['fields'][0]['offset'] += 1
        with self.assertRaises(RuntimeError):fa.validate_schema(bad)

    def test_independent_fa_slot_and_new_decoder_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);f = fa_fixture(root)
            result = self.verify(root, f)
            self.assertEqual(result['tiling_pointer_offset'], 200)
            self.assertEqual(result['raw_size'], 1464);self.assertEqual(result['decoded_field_count'], 205)
            self.assertEqual(result['opaque_bytes_preserved'], 192)
            self.assertFalse(result['historical_qualification_inherited']);self.assertFalse(result['c2'])
            wrong = dict(f[-1]);wrong['coreParams.s1BaseSize'] = 64
            with self.assertRaisesRegex(RuntimeError, 'NEW_COMPILED_DECODER_DISAGREES'):self.verify(root, f, decoder=wrong)

    def test_real_decoder_envelope_requires_exact_schema_size_and_all_public_fields(self):
        pod = fa.decode_public_pod(pod_bytes(), SCHEMA)
        document = dict(schema='nes.fa.public_pod_values.v1', raw_size=1464, fields=pod)
        fa.verify_compiled_decoder_document(document, pod)
        invalid = [dict(pod), {**document, 'unknown': 1}, {**document, 'schema': 'nes.fia.values.v1'},
                   {**document, 'raw_size': 1576}, {**document, 'raw_size': 1464.0}]
        missing = copy.deepcopy(document);del missing['fields']['inputParams.scaleValue'];invalid.append(missing)
        extra = copy.deepcopy(document);extra['fields']['unknown'] = 0;invalid.append(extra)
        swapped = copy.deepcopy(document);del swapped['fields']['inputParams.scaleValue']
        swapped['fields']['unknown'] = 0;invalid.append(swapped)
        for candidate in invalid:
            with self.subTest(candidate_keys=list(candidate)):
                with self.assertRaisesRegex(RuntimeError, 'DECODER_ENVELOPE_DIFFERS'):
                    fa.verify_compiled_decoder_document(candidate, pod)
        # Neither semantic values nor source padding can be dropped or tolerated.
        for name, value in (('coreParams.s1BaseSize', 64), ('inputParams.scaleValue', 0.0883883535861969),
                            ('inputParams.remainPH', 1)):
            changed = copy.deepcopy(document);changed['fields'][name] = value
            with self.subTest(field=name):
                with self.assertRaisesRegex(RuntimeError, 'DECODER_DISAGREES'):
                    fa.verify_compiled_decoder_document(changed, pod)

    def test_fia_slot_or_matching_host_bytes_elsewhere_cannot_select_fa_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);f = fa_fixture(root);event = f[2][3]
            event['placeholders_hex'] = struct.pack('<II', 280, 300).hex()
            with self.assertRaisesRegex(RuntimeError, 'TILING_PLACEHOLDER_MISSING'):self.verify(root, f)
            event['placeholders_hex'] = struct.pack('<II', 200, 216).hex()
            wrong = bytearray(pod_bytes());struct.pack_into('<i', wrong, 576, 64)
            event['args_hex'] = (bytes(216) + bytes(wrong) + pod_bytes()).hex()
            event['args_size'] = 216 + 2 * 1464
            with self.assertRaisesRegex(RuntimeError, 'HOST_FINAL_RAW_BYTES_DIFFER'):self.verify(root, f)

    def test_unscoped_off_uses_actual_launch_with_no_host_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);f = fa_fixture(root)
            result = self.verify(root, f, off=True)
            self.assertTrue(result['native_launch_verified']);self.assertTrue(result['unscoped_off'])
            self.assertFalse(result['host_observation_present'])
            f[2][3]['tiling_key'] = '109000000000400000'
            with self.assertRaises(RuntimeError):self.verify(root, f, off=True)

    def test_host_projection_mismatch_cannot_be_replaced_by_selected_factor_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);f = fa_fixture(root)
            f[1]['typed']['bmm1']['baseN'] = 999
            with self.assertRaisesRegex(RuntimeError, 'MATMUL_PUBLIC_FIELDS_DIFFER'):self.verify(root, f)

    def test_exact_fa_entry_and_runtime_source_hashes_derive_200_byte_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);entry = root / 'entry.cpp'
            declaration = 'void flash_attention_score(' + ','.join('__gm__ uint8_t* ' + name for name in fa.NAMES) + '){}'
            entry.write_text(declaration)
            with self.assertRaisesRegex(RuntimeError, 'PINNED_ENTRY_SOURCE_DIFFERS'):
                fa.derive_kernel_abi(entry, metadata('a' * 64), SCHEMA, root)
            def digest(path):
                if Path(path) == entry:return fa.SOURCE_SHA
                return fa.OPBASE[Path(path).relative_to(root).as_posix()]
            with patch.object(fa, 'sha', side_effect=digest):
                abi = fa.derive_kernel_abi(entry, metadata('a' * 64), SCHEMA, root)
                self.assertEqual(abi['tiling_pointer_offset'], 200)
                self.assertEqual(abi['declared_tiling_pointer_offset'], 192)
                wrong = metadata('a' * 64);wrong['parameters'] = [None] * 34
                with self.assertRaisesRegex(RuntimeError, 'PARAMETER_ABI_DIFFERS'):fa.derive_kernel_abi(entry, wrong, SCHEMA, root)
                entry.write_text(declaration.replace('uint8_t* query,', 'uint8_t* unreviewed,'))
                with self.assertRaisesRegex(RuntimeError, 'PARAMETER_ABI_DIFFERS'):fa.derive_kernel_abi(entry, metadata('a' * 64), SCHEMA, root)

    def test_selection_uses_this_runs_native_pair_and_plan_is_experimental(self):
        self.assertEqual(probe.choose_factors(dict(m=128, n_ratio=8), 'query_tile'), dict(m=64, n_ratio=8))
        self.assertEqual(probe.choose_factors(dict(m=64, n_ratio=4), 'joint'), dict(m=128, n_ratio=8))
        with self.assertRaisesRegex(RuntimeError, 'OUTSIDE_SOURCE_CANDIDATES'):
            probe.choose_factors(dict(m=128, n_ratio=16), 'query_tile')
        bundle = dict(completion=dict(source_commit='c77c0829f1f0f0b8f882db1d9b051a758b605472', patch_receipt_sha256='a' * 64),
                      completion_sha256='b' * 64)
        plan = probe.typed_plan(bundle, batch=8, call_id='c' * 32, native=None, selected=None, experiment_sha='d' * 64)
        self.assertEqual(plan['mode'], 'native');self.assertIsNone(plan['requested']['native'])
        self.assertEqual(plan['certificate']['validation_state'], 'experimental')
        self.assertEqual(plan['certificate']['build_manifest_sha256'], 'b' * 64)


if __name__ == '__main__':unittest.main()
