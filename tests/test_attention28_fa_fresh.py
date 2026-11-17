"""FA rebuild contracts; these are CPU checks, never runtime admission."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fa_build_tests', ROOT / 'fault_injection/native_adapters/attention28/fa/build_fresh.py')
build = importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
reg = build.registration


class FaBuildTests(unittest.TestCase):
    def receipt(self):
        return dict(schema='nes.fa.policy_source.v1', source_commit=build.COMMIT,
            status='PATCHED_NOT_BUILT_NOT_NPU_VALIDATED', kernel_source_modified=False,
            files=[dict(path=p, after_sha256=s) for p, s in build.PATCHED.items()])

    def test_only_frozen_four_source_patch_accepted(self):
        original = self.receipt();build.validate_receipt(original)
        for change in ('extra', 'changed', 'duplicate', 'kernel'):
            data = copy.deepcopy(original)
            if change == 'extra':data['files'].append(dict(path='attention/other.cpp', after_sha256='a' * 64))
            if change == 'changed':data['files'][0]['after_sha256'] = 'a' * 64
            if change == 'duplicate':data['files'].append(data['files'][0])
            if change == 'kernel':data['kernel_source_modified'] = True
            with self.assertRaises(RuntimeError):build.validate_receipt(data)
        self.assertEqual(len(build.verify_imported()), 7)

    def test_fa_never_uses_fia_sdk_probe_or_variant(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(work=Path(folder) / 'new', sdk_interface_manifest=Path('fia-probe.json'))
            obj = build.Builder(args);obj.receipt = self.receipt()
            with self.assertRaisesRegex(RuntimeError, 'FA_EXACT_PINNED_CANN91_REQUIRED'):obj.configure_policy(None)
            args.sdk_interface_manifest = None;obj.configure_policy(None)
            self.assertEqual(obj.operator, 'flash_attention_score')
            self.assertEqual(obj.vendor_name, 'dcat_f28_fa')
            self.assertFalse(obj.report['historical_certificate_inherited'])

    def metadata(self, binary):
        return dict(binFileName=reg.BINARY, binFileSuffix='.o', sha256=reg.sha(binary),
            coreType='MIX', taskRation='tilingKey', kernelList=[dict(tilingKey=reg.KEY)],
            supportInfo=dict(simplifiedKeyMode=0, simplifiedKey=['FA/actual'], optionalInputMode='gen_placeholder',
                             optionalOutputMode='gen_placeholder', tilingKey=[reg.KEY]))

    def test_duplicate_or_other_kernel_key_and_old_binary_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / 'fresh.o';binary.write_bytes(b'new compiled bytes')
            good = self.metadata(binary);reg.validate_metadata(good, binary)
            for name in ('duplicate', 'fia', 'extra_support', 'wrong_binary'):
                bad = copy.deepcopy(good)
                if name == 'duplicate':bad['kernelList'] *= 2
                if name == 'fia':bad['kernelList'][0]['tilingKey'] = 109000000000000000
                if name == 'extra_support':bad['supportInfo']['tilingKey'].append(reg.KEY)
                if name == 'wrong_binary':bad['sha256'] = '0' * 64
                with self.assertRaises(RuntimeError):reg.validate_metadata(bad, binary)

    def test_summary_preserves_compiler_representation_without_fia_assumption(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / 'fresh.o';binary.write_bytes(b'new')
            meta = self.metadata(binary)
            path = 'ascend910b/flash_attention_score/' + reg.BINARY
            row = dict(coreType=0, simplifiedKey=['FA/actual'], multiKernelType=1, binPath=path + '.o', jsonPath=path + '.json')
            op = dict(binaryList=[row], simplifiedKeyMode=0, optionalInputMode='gen_placeholder', optionalOutputMode='gen_placeholder')
            generated = {'binary_info_config.json': {'FlashAttentionScore': op}, 'flash_attention_score.json': {'binList': ['fresh']}}
            existing = copy.deepcopy(generated['flash_attention_score.json'])
            reg.validate_generated(generated, existing, meta)
            meta['taskRation'] = 1
            with self.assertRaisesRegex(RuntimeError, 'MULTIKERNEL_DISAGREES'):reg.validate_generated(generated, existing, meta)
            row.pop('multiKernelType');reg.validate_generated(generated, existing, meta)
            row['binPath'] = 'ascend910b/fused_infer_attention_score/old.o'
            with self.assertRaisesRegex(RuntimeError, 'SUMMARY_BINARY_DIFFERS'):reg.validate_generated(generated, existing, meta)


if __name__ == '__main__':unittest.main()
