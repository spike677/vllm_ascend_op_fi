"""Device-free provenance contracts; the byte fixture is not a Native DSO."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.worker_core_v11_20261118 import review39_group_originals as r


class GroupNativeProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.source = self.folder/'source'; self.source.mkdir()
        root = Path(__file__).parents[1]
        _, projection = r.OLD.helpers(r.OLD.CUSTOM_PROFILE)
        modules = r.OLD.MODULES+r.EXTRA_MODULES+projection.custom.EXTRA_MODULES
        sources = r.OLD.SOURCES+r.EXTRA_MODULES+projection.custom.EXTRA_MODULES
        files = []; hashes = {}
        for name in sources:
            raw = (root/name).read_bytes(); path = self.source/name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
            hashes[name] = hashlib.sha256(raw).hexdigest()
            files.append(dict(path=name, sha256=hashes[name], bytes=len(raw)))
        self.commit = '1'*40
        manifest = self.source/'source_manifest.json'
        manifest.write_text(json.dumps(dict(base_commit=self.commit,files=files)),encoding='utf-8')
        self.manifest_sha = r.C.file_sha(manifest)
        self.library = self.folder/'explicit-cpu-byte-fixture.bin'
        self.library.write_bytes(b'CPU provenance bytes only, not ELF or device code')
        self.pin = r.C.file_sha(self.library)
        worker = dict(pid=123,worker_start_identity='CPU_SCHEMA_ONLY')
        native = dict(abi=1,sha256=self.pin,loaded_mapping_verified=True,read_span_available=True,
            simulation=False,qualification_granted=False)
        model = dict(files={},classes=[])
        weights = dict(format='safetensors',files={'CPU-METADATA-ONLY':dict(bytes=4,sha256='a'*64)})
        self.host = dict(worker=worker)
        self.document = dict(schema='demoncat.output39.worker_core_probe.v1',mechanism='39.output_proxy',
            stage='c2_c3',status='REFUSED_OR_FAILED',error='CPU_SCHEMA_RETAINED_FAILURE',standard_worker=True,
            native_sha256=self.pin,build_fingerprint='b'*64,sources=hashes,
            binding=dict(worker=worker,binding=dict(status='SERVING_DEPLOYMENT_BOUND_NOT_QUALIFIED',
                mechanism='39.output_proxy',workers_created=0,hooks_installed=0,native_leases_opened=0,
                native=native,qualification_granted=False,c2=False,c3=False),
                loaded_sources={name:dict(path='/installed/'+(name[4:] if name.startswith('src/') else name),
                    sha256=hashes[name]) for name in modules},
                model_source=dict(manifest=model,source_sha256=r.C.digest(model))),
            weights=dict(manifest=weights,weights_sha256=r.C.digest(weights)))

    def review(self, document=None, *, explicit=True):
        return r.provenance(self.document if document is None else document,self.host,dict(finished_groups={}),
            self.source,self.manifest_sha,self.commit,self.library,profile=r.OLD.CUSTOM_PROFILE,
            native_sha256=self.pin if explicit else None)

    def codes(self, result):
        return [x['code'] for name in ('loaded_source_review','native_identity_review') for x in result[name]['issues']]

    def test_explicit_byte_pin_reuses_all_original_source_and_inert_checks(self):
        result = self.review()
        self.assertTrue(result['consistent'],result)
        self.assertEqual(result['expected_native_sha256'],self.pin)
        self.assertEqual(result['native_pin_basis'],'EXPLICIT_REVIEW_ARGUMENT')
        self.assertEqual(result['loaded_source_review']['source_files_rehashed'],len(self.document['sources']))
        self.assertFalse(result['native_identity_review']['public_qualification_granted'])

    def test_omitted_pin_keeps_fixed_legacy_refusal_and_does_not_leak_previous_pin(self):
        old = r.OLD.NATIVE_SHA
        self.assertTrue(self.review()['consistent'])
        self.assertEqual(r.OLD.NATIVE_SHA,old)
        result = self.review(explicit=False)
        self.assertFalse(result['consistent'])
        self.assertIn('NEW_WORKER_ENTRY_AND_EXACT_NATIVE_DSO',self.codes(result))
        self.assertEqual(result['expected_native_sha256'],old)

    def test_probe_hash_is_not_the_external_review_pin(self):
        document=deepcopy(self.document);document['native_sha256']='c'*64
        self.assertFalse(self.review(document)['consistent'])

    def test_library_byte_change_is_not_accepted_by_matching_saved_strings(self):
        self.library.write_bytes(b'changed byte fixture')
        result=self.review()
        self.assertFalse(result['consistent'])
        self.assertIn('NEW_WORKER_ENTRY_AND_EXACT_NATIVE_DSO',self.codes(result))

    def test_initial_activity_or_boolean_zero_is_not_inert(self):
        for field in ('workers_created','hooks_installed','native_leases_opened'):
            for value in (1,False):
                document=deepcopy(self.document);document['binding']['binding'][field]=value
                result=self.review(document)
                self.assertFalse(result['consistent'],(field,value))
                self.assertIn('TYPED_INERT_GROUP_DEPLOYMENT_NOT_ACCEPTANCE',self.codes(result))

    def test_native_abi_simulation_mapping_and_read_span_must_be_actual_types(self):
        for field,value in [('abi',True),('abi',2),('simulation',True),('simulation','false'),
                            ('loaded_mapping_verified',False),('read_span_available',False),('sha256','d'*64)]:
            document=deepcopy(self.document);document['binding']['binding']['native'][field]=value
            result=self.review(document)
            self.assertFalse(result['consistent'],field)
            self.assertIn('EXACT_PINNED_NONSIMULATED_NATIVE_ABI_AND_MAPPING',self.codes(result))

    def test_c1_stage_or_unreviewed_status_cannot_be_silently_grouped(self):
        for field,value in [('stage','c1'),('stage','c2_isolated_once'),('standard_worker',1),('status','SUCCESS')]:
            document=deepcopy(self.document);document[field]=value
            self.assertFalse(self.review(document)['consistent'],field)

    def test_loaded_source_and_worker_identity_still_checked(self):
        document=deepcopy(self.document)
        name=next(iter(document['binding']['loaded_sources']))
        document['binding']['loaded_sources'][name]['sha256']='e'*64
        self.assertIn('ACTUAL_FROZEN_LOADED_MODULE_HASH',self.codes(self.review(document)))
        document=deepcopy(self.document);document['binding']['worker']['pid']+=1
        self.assertFalse(self.review(document)['consistent'])

    def test_source_file_bytes_are_rehashed(self):
        path=self.source/(r.OLD.CORE+'serving39_output_discovery.py')
        path.write_bytes(path.read_bytes()+b'\n# changed CPU source fixture\n')
        result=self.review()
        self.assertFalse(result['consistent'])
        self.assertIn('SOURCE_FILE_BYTES',self.codes(result))

    def test_missing_discovery_source_cannot_be_hidden_in_an_old_expected_set(self):
        document=deepcopy(self.document)
        document['sources'].pop(r.OLD.CORE+'serving39_output_discovery.py')
        with self.assertRaises(KeyError):self.review(document)


if __name__=='__main__':unittest.main()
