"""CPU-only contracts for the experimental fresh FIA launcher."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

FILE=Path(__file__).parents[1]/'experiments/engineering_20261117/fresh_fia_qualification.py'
spec=importlib.util.spec_from_file_location('fresh_fia_qualification',FILE)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

def fixture(tmp):
    root=tmp/'vendor';file=root/'op_api/lib/libcust_opapi.so'
    file.parent.mkdir(parents=True);file.write_bytes(b'new-test-bytes-not-ELF')
    receipt=dict(schema='demoncat.fia.fresh_build.v1',status='CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION',
        loading_allowed=False,source_commit=probe.COMMIT,npu_executed=False,
        previous_experiment_objects_used=False,variant='legal_tiling',
        compiled_keys=sorted(probe.KEYS['legal_tiling']),kernel_binary_sha256='1'*64,
        artifacts=[dict(path='op_api/lib/libcust_opapi.so',bytes=file.stat().st_size,sha256=probe.sha(file))])
    completion=tmp/'completion.json';completion.write_text(json.dumps(receipt))
    return root,file,completion,receipt

def raw_fixture():
    fields=[dict(name='field'+str(i),offset=i*4,width=4,count=1,dtype='uint32_t') for i in range(11)]
    raw=struct.pack('<11I',*range(11))
    row=dict(raw_hex=raw.hex(),raw_size=len(raw),typed={f['name']:i for i,f in enumerate(fields)})
    schema=dict(schema='fault28.fia_graph.public_serializer_layout.v1',
        kernel_argument_raw_observed=False,size=len(raw),fields=fields)
    return row,schema

class FreshFiaQualificationTests(unittest.TestCase):
    def test_only_audited_internal_tiling_link_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            try:(Path(tmp)/'probe').symlink_to('not-required-to-exist')
            except OSError:self.skipTest('Real symlink creation unavailable; execute this test on Linux')
        for mutation in ('valid','outside','changed_target','manifest_hash','chain','directory'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root,_,completion,receipt=fixture(Path(tmp))
                link=root/probe.TILING_LINK;target=link.parent/probe.TILING_TARGET
                target.parent.mkdir(parents=True);target.write_bytes(b'new-host-test')
                link.symlink_to(probe.TILING_TARGET)
                for item in (link,target):
                    receipt['artifacts'].append(dict(path=item.relative_to(root).as_posix(),
                        bytes=item.stat().st_size,sha256=probe.sha(item)))
                if mutation=='outside':
                    link.unlink();link.symlink_to(Path(tmp)/'completion.json')
                if mutation=='changed_target':
                    other=link.parent/'other.so';other.write_bytes(target.read_bytes())
                    link.unlink();link.symlink_to('other.so')
                if mutation=='manifest_hash':receipt['artifacts'][-1]['sha256']='1'*64
                if mutation=='chain':
                    other=target.parent/'other.so';other.write_bytes(target.read_bytes())
                    target.unlink();target.symlink_to('other.so')
                if mutation=='directory':
                    (root/'external').symlink_to(target.parent,target_is_directory=True)
                completion.write_text(json.dumps(receipt))
                if mutation=='valid':probe.verify_build(completion,probe.sha(completion),root)
                else:
                    with self.assertRaises(RuntimeError):probe.verify_build(completion,probe.sha(completion),root)

    def test_tree_verification_does_not_promote_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,_,completion,_=fixture(Path(tmp))
            result=probe.verify_build(completion,probe.sha(completion),root)
            self.assertIs(result['production_loading_allowed'],False)
            self.assertNotIn('native_launch_verified',result)

    def test_wrong_or_extra_artifacts_and_old_objects_refused(self):
        for mutation in ('bytes','extra','path','keys','old_object','promoted'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root,file,completion,receipt=fixture(Path(tmp))
                if mutation=='bytes':file.write_bytes(b'wrong')
                if mutation=='extra':(root/'old.so').write_bytes(b'old')
                if mutation=='path':receipt['artifacts'][0]['path']='../outside'
                if mutation=='keys':receipt['compiled_keys'].append(119000000000400000)
                if mutation=='old_object':receipt['previous_experiment_objects_used']=True
                if mutation=='promoted':receipt['loading_allowed']=True
                completion.write_text(json.dumps(receipt))
                with self.assertRaises(RuntimeError):probe.verify_build(completion,probe.sha(completion),root)

    def test_two_mechanisms_keep_distinct_plans(self):
        self.assertEqual(probe.plan('legal_tiling','baseline'),('coupled_base512',512,512,109000000000000000))
        self.assertEqual(probe.plan('legal_tiling','fault'),('coupled_m128',128,128,109000000000400000))
        self.assertEqual(probe.plan('denominator_replay','baseline'),('coupled_m128',128,128,109000000000400000))
        self.assertEqual(probe.plan('denominator_replay','fault'),('coupled_replay_m128',128,128,119000000000400000))
        self.assertEqual(probe.plan('legal_tiling','restore'),probe.plan('denominator_replay','off'))

    def test_public_byte_decode_must_agree_with_host_typed(self):
        row,schema=raw_fixture()
        self.assertEqual(probe.decode_host(row,schema),row['typed'])
        row['typed']['field2']=400
        with self.assertRaisesRegex(RuntimeError,'HOST_TYPED_RAW_DIFFERS'):probe.decode_host(row,schema)

    def test_public_host_schema_cannot_claim_kernel_argument_proof(self):
        row,schema=raw_fixture();schema['kernel_argument_raw_observed']=True
        with self.assertRaisesRegex(RuntimeError,'PUBLIC_HOST_SCHEMA_REQUIRED'):probe.decode_host(row,schema)

    def test_byte_length_and_field_bounds_checked(self):
        row,schema=raw_fixture();row['raw_size']-=1
        with self.assertRaisesRegex(RuntimeError,'RAW_SERIALIZER_SIZE_DIFFERS'):probe.decode_host(row,schema)
        row,schema=raw_fixture();schema['fields'][3]['offset']=1000
        with self.assertRaisesRegex(RuntimeError,'PUBLIC_FIELD_BOUNDS_INVALID'):probe.decode_host(row,schema)

if __name__=='__main__':unittest.main()

