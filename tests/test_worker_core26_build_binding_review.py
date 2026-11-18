"""CPU byte fixtures for build-to-held-address joins; no native binaries loaded."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.worker_core_v11_20261118 import review26_environment_originals as e
from experiments.worker_core_v11_20261118 import review26_group_originals as r


class Native26BuildBindingReviewTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.files={}
        names=e.PARENT_FILES|{'libdcat26_group_v2.so','metadata-array-group-cpu.json'}
        for name in names:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(('explicit CPU non-library fixture:'+name).encode())
            self.files[name]=r.C.file_sha(path)
        self.parent=dict(schema='demoncat.writer26.metadata_ownership.v1',
            images={name:dict(sha256=self.files[name],symbol=symbol) for name,symbol in e.METADATA_SYMBOLS.items()},
            cpu_receipt_sha256=self.files['metadata-array-cpu.json'])
        self.files['metadata-array-ownership.json']=self.save('metadata-array-ownership.json',self.parent)
        source={'fixture/source.cpp':'1'*64}
        self.receipt=dict(schema='demoncat.writer26.group_build.v2',sources=source,actual_npu_calls=0,device_mounts=[],
            parent_manifest_sha256='2'*64,parent_files={name:self.files[name] for name in e.PARENT_FILES},
            rebuilt_dependencies=['libcust_opapi.so'],
            custom_bound_addresses_contract=dict(status='PASS_CPU_EXISTING_CUSTOM_CACHE_ONLY',actual_npu_calls=0))
        # Rebuilt custom differs from its parent; the final manifest is the
        # expected loaded getter image, not this old parent receipt value.
        self.receipt['parent_files']['libcust_opapi.so']='3'*64
        self.build=dict(schema='demoncat.writer26.group_deployment.v2',actual_npu_calls=0,legacy_writers_absent=True,
            sources=source,files=self.files,bound_symbol_profile='demoncat.writer26.cached_addresses.v1',
            metadata_array_ownership=dict(parent_receipt_sha256=self.files['metadata-array-ownership.json']))
        self.freeze()

    def save(self,name,value):
        path=self.root/name;path.write_text(json.dumps(value,sort_keys=True),encoding='utf-8')
        return r.C.file_sha(path)

    def freeze(self):
        self.build['build_receipt_sha256']=self.save('build-receipt.json',self.receipt)
        self.sha=self.save('build.json',self.build)

    def inspect(self):
        return e.load_build_binding(r.C,self.root,self.sha)

    def test_final_custom_image_and_parent_domain_are_separate(self):
        value=self.inspect()
        self.assertEqual(value['files']['libcust_opapi.so'],r.C.file_sha(self.root/'libcust_opapi.so'))
        self.assertNotEqual(value['files']['libcust_opapi.so'],self.receipt['parent_files']['libcust_opapi.so'])
        self.assertEqual(value['files_rehashed'],12)
        self.assertFalse(value['complete_parent_bundle_rehashed'])

    def test_no_build_pin_or_modified_custom_file_refused(self):
        with self.assertRaisesRegex(ValueError,'NATIVE_BUILD_ORIGINAL_BYTES'):
            e.load_build_binding(r.C,self.root,'a'*64)
        (self.root/'libcust_opapi.so').write_bytes(b'changed after manifest')
        with self.assertRaisesRegex(ValueError,'NATIVE_BUILD_FILE_BYTES:libcust_opapi.so'):self.inspect()

    def test_altered_build_receipt_refused(self):
        self.receipt['rebuilt_dependencies']=[];self.save('build-receipt.json',self.receipt)
        with self.assertRaisesRegex(ValueError,'NATIVE_BUILD_ORIGINAL_BYTES:build-receipt.json'):self.inspect()

    def test_saved_false_is_not_an_integer_zero_compute_count(self):
        self.receipt['custom_bound_addresses_contract']['actual_npu_calls']=False;self.freeze()
        with self.assertRaisesRegex(ValueError,'FRESH_CUSTOM_GETTER_CPU_BUILD_CONTRACT'):self.inspect()

    def test_parent_dependency_cannot_be_silently_rebuilt_or_removed(self):
        for change in (lambda v:v['parent_files'].update({'libopapi.so':'a'*64}),
                lambda v:v['parent_files'].pop('libopapi.so'),
                lambda v:v.update(rebuilt_dependencies=['libcust_opapi.so','libopapi.so'])):
            with self.subTest(change=change):
                saved=deepcopy(self.receipt);change(self.receipt);self.freeze()
                with self.assertRaisesRegex(ValueError,'EXACT_PARENT_COPY_AND_REBUILT_CUSTOM_IMAGE'):self.inspect()
                self.receipt=saved

    def test_parent_metadata_cannot_be_replaced_with_custom_image(self):
        self.parent['images']['libcust_opapi.so']=dict(sha256=self.files['libcust_opapi.so'],symbol='invented')
        new_sha=self.save('metadata-array-ownership.json',self.parent)
        self.files['metadata-array-ownership.json']=new_sha
        self.receipt['parent_files']['metadata-array-ownership.json']=new_sha
        self.build['metadata_array_ownership']['parent_receipt_sha256']=new_sha;self.freeze()
        with self.assertRaisesRegex(ValueError,'PARENT_METADATA_IMAGE_DOMAIN_PRESERVED'):self.inspect()

    def test_unsafe_path_and_missing_file_refused(self):
        self.build['files']={**self.files,'../outside':'a'*64};self.freeze()
        with self.assertRaisesRegex(ValueError,'UNSAFE_ORIGINAL_PATH'):self.inspect()
        self.build['files']=self.files;self.freeze();(self.root/'libcust_opapi.so').unlink()
        with self.assertRaises(OSError):self.inspect()


if __name__=='__main__':unittest.main()
