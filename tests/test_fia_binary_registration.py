"""Registration repair contracts: no NPU and no old vendor fixture is consumed."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

FILE=Path(__file__).parents[1]/'fault_injection/native_adapters/attention28/fia/binary_registration.py'
spec=importlib.util.spec_from_file_location('fia_binary_registration',FILE)
registration=importlib.util.module_from_spec(spec);spec.loader.exec_module(registration)

class FiaBinaryRegistrationTests(unittest.TestCase):
    def metadata(self,binary):
        return dict(binFileName=registration.BINARY,binFileSuffix='.o',sha256=registration.sha(binary),
            coreType='MIX',taskRation='tilingKey',
            kernelList=[dict(tilingKey=k) for k in sorted(registration.KEYS['legal_tiling'])],
            supportInfo=dict(simplifiedKeyMode=0,simplifiedKey=['exact-new-metadata-key'],
                optionalInputMode='gen_placeholder',optionalOutputMode='gen_placeholder',
                tilingKey=[str(k) for k in sorted(registration.KEYS['legal_tiling'])]))

    def test_metadata_requires_exact_variant_keys_and_actual_binary_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary=Path(tmp)/'new.o';binary.write_bytes(b'new-test-binary')
            base=self.metadata(binary);registration.validate_metadata(base,binary,'legal_tiling')
            for mutation in ('missing','wrong_variant','hash','support'):
                with self.subTest(mutation=mutation):
                    m=copy.deepcopy(base);variant='legal_tiling'
                    if mutation=='missing':m['kernelList'].pop()
                    if mutation=='wrong_variant':variant='denominator_replay'
                    if mutation=='hash':m['sha256']='1'*64
                    if mutation=='support':m['supportInfo']['tilingKey']=['103000000000000000']
                    with self.assertRaises(RuntimeError):registration.validate_metadata(m,binary,variant)

    def generated(self):
        base='ascend910b/fused_infer_attention_score/'+registration.BINARY
        metadata=dict(supportInfo=dict(simplifiedKey=['matching']))
        existing={'binList':[{'binInfo':{'jsonFilePath':base+'.json'},'inputs':['same-new-signature']}]}
        op=dict(simplifiedKeyMode=0,optionalInputMode='gen_placeholder',optionalOutputMode='gen_placeholder',
            binaryList=[dict(binPath=base+'.o',jsonPath=base+'.json',coreType=0,multiKernelType=1,simplifiedKey=['matching'])])
        generated={'binary_info_config.json':{'FusedInferAttentionScore':op},
                   'fused_infer_attention_score.json':copy.deepcopy(existing)}
        return generated,existing,metadata

    def test_generated_index_must_bind_existing_registry_and_new_metadata(self):
        generated,existing,metadata=self.generated()
        self.assertEqual(registration.validate_generated(generated,existing,metadata),generated['binary_info_config.json'])
        for mutation in ('signature','binary','key','extra'):
            with self.subTest(mutation=mutation):
                generated,existing,metadata=self.generated()
                if mutation=='signature':existing['binList'][0]['inputs']=['different']
                if mutation=='binary':generated['binary_info_config.json']['FusedInferAttentionScore']['binaryList'][0]['binPath']='old/vendor.o'
                if mutation=='key':metadata['supportInfo']['simplifiedKey']=['other']
                if mutation=='extra':generated['binary_info_config.json']['UnexpectedOperator']={}
                with self.assertRaises(RuntimeError):registration.validate_generated(generated,existing,metadata)

    def test_fixed_official_source_fingerprint_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'cmake/scripts/util';folder.mkdir(parents=True)
            (folder/'ascendc_ops_config.py').write_text('raise Exception("do not execute")')
            (folder/'const_var.py').write_text('unused')
            with self.assertRaisesRegex(RuntimeError,'OFFICIAL_GENERATOR_SOURCE_DIFFERS'):
                registration.load_official(Path(tmp))

    def test_native_fd_package_requires_eight_unique_matching_entries(self):
        from fault_injection.native_adapters.attention28.fia.fai_fd_geometry import KEYS, VARIANT
        self.assertEqual(registration.KEYS[VARIANT],set(KEYS))
        with tempfile.TemporaryDirectory() as temporary:
            binary=Path(temporary)/'new.o';binary.write_bytes(b'new-fd-test-binary')
            metadata=self.metadata(binary)
            metadata['kernelList']=[dict(tilingKey=key) for key in KEYS]
            metadata['supportInfo']['tilingKey']=[str(key) for key in KEYS]
            registration.validate_metadata(metadata,binary,VARIANT)
            for part in ('kernelList','supportInfo'):
                changed=copy.deepcopy(metadata)
                entries=changed[part] if part=='kernelList' else changed[part]['tilingKey']
                entries.append(entries[0])
                with self.assertRaises(RuntimeError):registration.validate_metadata(changed,binary,VARIANT)
            changed=copy.deepcopy(metadata)
            changed['kernelList'][6]['tilingKey']=5900000000010200100
            with self.assertRaisesRegex(RuntimeError,'KEY_SET_DIFFERS'):
                registration.validate_metadata(changed,binary,VARIANT)

if __name__=='__main__':unittest.main()

