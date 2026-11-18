"""Controller contracts with explicit synthetic source/compiler fixtures; no SDK/NPU run."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fault_injection.native_adapters.attention28.fia import build_fresh as build
from fault_injection.native_adapters.attention28.fia import binary_registration as registration
from fault_injection.native_adapters.attention28.fia import apply_tnd28_prefill_normal as recipe
from test_tnd28_prefill_normal import source_fixture


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def json_file(path, value):
    write(path, (json.dumps(value, sort_keys=True)+'\n').encode())


SERIALIZER = '''const uint32_t FIA_MAX_AIC_CORE_NUM = 26;
BEGIN_TILING_DATA_DEF(Base)
TILING_DATA_FIELD_DEF(uint32_t, a);
END_TILING_DATA_DEF;
BEGIN_TILING_DATA_DEF(Inner)
TILING_DATA_FIELD_DEF(uint64_t, b);
END_TILING_DATA_DEF;
BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreTilingData)
TILING_DATA_FIELD_DEF_STRUCT(Base, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(Inner, innerSplitParams);
END_TILING_DATA_DEF;
'''


class Tnd28BuildContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root/'official'
        self.sdk = self.root/'cann-8.5.1'
        self.sdk.mkdir()
        self.interfaces = self.root/'interfaces.json'
        self.interfaces.write_bytes(b'EXPLICIT_CPU_INTERFACE_FIXTURE_NOT_ACTUAL_SDK')
        self.raw = {p: ('UNCHANGED_CPU_SOURCE:'+p).encode() for p in (*recipe.SOURCE_SHA,*recipe.EXTRA_UNCHANGED)}
        self.raw.update({p: s.encode() for p,s in source_fixture().items()})
        self.raw[recipe.EXTRA_UNCHANGED[0]] = SERIALIZER.encode()
        pins = {p: digest(self.raw[p]) for p in recipe.SOURCE_SHA}
        for owner,name,value in ((recipe,'SOURCE_SHA',pins),(build,'tnd28_recipe',lambda:recipe),
                (build,'git_blob',lambda source,name:self.raw[name]),
                (build,'TND28_SDK_INTERFACE_SHA256',build.sha(self.interfaces))):
            patcher = patch.object(owner,name,value)
            patcher.start(); self.addCleanup(patcher.stop)
        updated = recipe.patched({p: self.raw[p].decode() for p in recipe.PATHS})
        typed,self.projection,self.fields = recipe.base.typed_materials(SERIALIZER)
        typed = typed.replace('fault28_fia','tnd28_prefill_normal').replace(
            'DEMONCAT_FAULT28_FIA_TYPED_H','DEMONCAT_TND28_PREFILL_TYPED_H')
        self.expected = dict(self.raw)
        self.expected.update({p:s.encode() for p,s in updated.items()})
        self.expected.update({recipe.base.HOST+'tnd28_prefill_normal.h':(build.HERE/'tnd28_prefill_normal.h').read_bytes(),
            recipe.base.HOST+'tnd28_prefill_contract.h':(build.HERE/'tnd28_prefill_contract.h').read_bytes(),
            recipe.base.HOST+'tnd28_prefill_typed.h':typed.encode()})
        for p,v in self.expected.items():
            write(self.source/p,v)
        self.receipt = dict(schema='demoncat.attention28.tnd_prefill_normal_source.v1',
            mechanism='28.fia.normal_candidate', variant=recipe.VARIANT, vendor=recipe.VENDOR,
            source_commit=build.COMMIT,status='PATCHED_NOT_BUILT_NOT_NPU_VALIDATED',
            normal_only=True,normal_route_preference_changed=True,
            scope=dict(layout='TND',batch=1,s1=2048,s2=2048,heads=16,kv_heads=8,dim=128,
                       page_attention=False,api_block_size=128,m=512,s2_base=512),
            recipe_sha256=build.sha(build.HERE/'apply_tnd28_prefill_normal.py'),
            base_recipe_sha256=build.sha(build.HERE/'base_policy.py'),normal_key=str(recipe.NORMAL_KEY),
            compile_keys=[str(k) for k in recipe.KEYS],projection_sha256=digest(self.projection.encode()),
            typed_field_count=len(self.fields),
            files=[dict(path=p,before_sha256=digest(self.raw[p]) if p in self.raw else None,
                        after_sha256=digest(v),bytes=len(v)) for p,v in self.expected.items()])
        self.receipt.update({n:False for n in (*build.TND28_DENIED,'tensor_payload_mutated',
                                             'kernel_source_modified','tiling_arithmetic_modified')})
        args = SimpleNamespace(work=self.root/'build',python=Path(sys.executable),
            sdk_interface_manifest=self.interfaces,sdk_interface_sha256=build.sha(self.interfaces),
            opdef_compat='ascend910b-cann851')
        self.b = build.Builder(args)
        self.b.sdk,self.b.source,self.b.receipt,self.b.env = self.sdk,self.source,self.receipt,{}
        self.b.excluded_test_fixtures=[]
        self.b.frozen={str(self.source/p):digest(v) for p,v in self.expected.items()}
        self.b.vendor=self.root/'vendor'
        write(self.b.vendor/'op_api/lib/libcust_opapi.so',b'CPU_TEST_ARTIFACT_NOT_A_LIBRARY')
        self.calls=[]

    def configure(self):
        self.b.configure_policy(self.source)

    def fake_cpu_run(self,label,argv,**kwargs):
        self.calls.append((label,[str(a) for a in argv]))
        write(self.b.ev/(label+'.log'),b'CPU_TEST_COMMAND_FIXTURE\n')
        if label=='tnd28_normal_contract_compile':
            write(self.b.ev/'tnd28_normal_contract',b'CPU_TEST_EXECUTABLE_FIXTURE')
        if label=='tnd28_normal_contract':
            write(self.b.ev/(label+'.log'),b'PASS_CPU_TND28_NORMAL_GEOMETRY_COORDINATES_NOT_SDK_OR_NPU\n')

    def schema_fixture(self):
        folder=self.b.ev/'tnd28_schema'
        public=self.source/recipe.EXTRA_UNCHANGED[0]
        write(folder/'fia_schema_projection.h',self.projection.encode())
        write(folder/'fia_abi_decoder',b'CPU_TEST_SCHEMA_EXECUTABLE_FIXTURE')
        entries=[dict(file='/fresh/attention/common/op_host/arch22/fia_tiling_nonquant.cpp',
                      directory='/fresh/build',arguments=['g++','-c','fresh-source.cpp'])]
        json_file(self.source/'build/compile_commands.json',entries)
        fields=[dict(name='baseParams.a',dtype='uint32_t',count=1,width=4,offset=0),
                dict(name='innerSplitParams.b',dtype='uint64_t',count=1,width=8,offset=8)]
        schema=dict(schema='fault28.fia_graph.public_serializer_layout.v1',
            type='FusedInferAttentionScoreTilingData',size=1576,fields=fields,
            projection_sha256=build.sha(folder/'fia_schema_projection.h'),public_source_sha256=build.sha(public),
            layout_kind='public_host_serializer_projection',kernel_argument_raw_observed=False)
        selftest=dict(status='PASS_CPU_DERIVED_LAYOUT_SENTINEL',serialized_bytes=1576,
            fields={f['name']:{k:v for k,v in f.items() if k!='name'} for f in fields})
        json_file(folder/'fia_abi_schema.json',schema)
        json_file(folder/'fia_abi_selftest.json',selftest)
        compiled=dict(schema='demoncat.fia.fresh_abi_compile.v1',npu_executed=False,loading_allowed=False,
            schema_sha256=build.sha(folder/'fia_abi_schema.json'),projection_sha256=schema['projection_sha256'],
            public_source_sha256=build.sha(public),actual_new_fia_compile_entry=entries[0],
            compiler_source_sha256=build.sha(build.HERE/'fia_abi_decoder.cpp'),
            decoder_sha256=build.sha(folder/'fia_abi_decoder'))
        json_file(folder/'compile_receipt.json',compiled)
        return folder,schema,compiled,selftest

    def ready_to_finish(self):
        self.configure()
        with patch.object(self.b,'run',self.fake_cpu_run):
            self.b.cpu_checks()
        self.schema_fixture()
        with patch.object(self.b,'run'):
            self.b.tnd28_public_schema()
        self.b.report['compiled_keys']=list(build.TND28_NORMAL_KEYS)

    def test_exact_reconstructed_normal_source_cannot_grant_loading(self):
        self.configure()
        self.assertEqual(self.b.variant,recipe.VARIANT)
        self.assertEqual(self.b.vendor_name,recipe.VENDOR)
        self.assertEqual(len(self.b.report['tnd28_source_closure']),17)
        self.assertEqual(self.b.report['tnd28_source_closure'],{p:digest(v) for p,v in self.expected.items()})
        self.assertTrue(self.b.report['tnd28_source_closure_reconstructed'])
        for field in build.TND28_DENIED:
            self.assertIs(self.b.report[field],False)

    def test_each_identity_or_qualification_flag_is_required(self):
        cases=[('schema','other'),('mechanism','33.paged_decode'),('vendor','dcat_f33_arch22_normal'),
               ('source_commit','a'*40),('recipe_sha256','0'*64),('base_recipe_sha256','0'*64),
               ('normal_key','103000000010200003'),('status','BUILT'),('normal_only',False)]
        cases += [(name,True) for name in (*build.TND28_DENIED,'tensor_payload_mutated',
                                          'kernel_source_modified','tiling_arithmetic_modified')]
        for name,value in cases:
            with self.subTest(name=name):
                self.b.receipt=copy.deepcopy(self.receipt);self.b.receipt[name]=value
                with self.assertRaisesRegex(RuntimeError,'EXACT_SOURCE_RECEIPT'):
                    self.configure()
                self.assertFalse(self.b.report['loading_allowed'])

    def test_exact_sdk_profile_required_before_any_build_command(self):
        for field,value in (('sdk_interface_sha256','0'*64),('opdef_compat',None),('sdk_interface_manifest',None)):
            old=getattr(self.b.args,field);setattr(self.b.args,field,value)
            with self.subTest(field=field),patch.object(self.b,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'EXACT_REVIEWED_CANN851'):
                    self.configure()
                run.assert_not_called()
            setattr(self.b.args,field,old)
        self.b.sdk=self.root/'cann-9.1.0'
        with self.assertRaisesRegex(RuntimeError,'EXACT_REVIEWED_CANN851'):
            self.configure()

    def test_scope_and_key_set_cannot_borrow_decode_or_fault_entries(self):
        for name,value in (('m',128),('s2_base',256),('page_attention',True),('s1',1),('api_block_size',16)):
            self.b.receipt=copy.deepcopy(self.receipt);self.b.receipt['scope'][name]=value
            with self.subTest(name=name),self.assertRaisesRegex(RuntimeError,'EXACT_SOURCE_RECEIPT'):
                self.configure()
        self.b.receipt=copy.deepcopy(self.receipt)
        self.b.receipt['compile_keys'].append('119000000000400000')
        with self.assertRaisesRegex(RuntimeError,'EXACT_SOURCE_RECEIPT'):
            self.configure()

    def test_complete_unique_source_set_required(self):
        for mutation in ('missing','duplicate','extra'):
            self.b.receipt=copy.deepcopy(self.receipt);rows=self.b.receipt['files']
            if mutation=='missing':rows.pop()
            elif mutation=='duplicate':rows[-1]=dict(rows[0])
            else:rows.append(dict(rows[0],path='another.cpp'))
            with self.subTest(mutation=mutation),self.assertRaisesRegex(RuntimeError,'COMPLETE_UNIQUE_SOURCE'):
                self.configure()

    def test_receipt_cannot_authorize_altered_host_kernel_or_generated_header(self):
        for name in (recipe.PATHS[1],recipe.EXTRA_UNCHANGED[1],recipe.base.HOST+'tnd28_prefill_typed.h'):
            changed=self.expected[name]+b' altered'
            write(self.source/name,changed)
            self.b.receipt=copy.deepcopy(self.receipt)
            row=next(r for r in self.b.receipt['files'] if r['path']==name)
            row.update(after_sha256=digest(changed),bytes=len(changed))
            with self.subTest(name=name),self.assertRaisesRegex(RuntimeError,'RECONSTRUCTED_SOURCE_BYTES'):
                self.configure()
            write(self.source/name,self.expected[name])

    def test_unmodified_pristine_pin_and_before_hash_are_independent(self):
        name=recipe.PATHS[0];old=self.raw[name];self.raw[name]+=b' changed'
        with self.assertRaisesRegex(ValueError,'PRISTINE_SOURCE_DIFFERS'):
            self.configure()
        self.raw[name]=old
        self.b.receipt=copy.deepcopy(self.receipt);self.b.receipt['files'][0]['before_sha256']='0'*64
        with self.assertRaisesRegex(RuntimeError,'RECONSTRUCTED_SOURCE_BYTES'):
            self.configure()

    def test_projection_or_field_count_cannot_relabel_old_schema(self):
        for name,value in (('projection_sha256','0'*64),('typed_field_count',1)):
            self.b.receipt=copy.deepcopy(self.receipt);self.b.receipt[name]=value
            with self.subTest(name=name),self.assertRaisesRegex(RuntimeError,'GENERATED_TYPED_SCHEMA'):
                self.configure()

    def test_cpu_compile_is_strict_and_binds_executable_and_logs(self):
        self.configure()
        with patch.object(self.b,'run',self.fake_cpu_run):
            self.b.cpu_checks()
        labels=dict(self.calls);argv=labels['tnd28_normal_contract_compile']
        for flag in ('-std=c++17','-Wall','-Wextra','-Werror','-pedantic','-pthread'):
            self.assertIn(flag,argv)
        receipt=json.loads((self.b.ev/'tnd28_normal_cpu_contract.json').read_bytes())
        self.assertEqual(receipt['source_sha256'],build.sha(build.HERE/'test_tnd28_prefill_contract.cpp'))
        self.assertEqual(receipt['executable_sha256'],build.sha(self.b.ev/'tnd28_normal_contract'))
        self.assertEqual(receipt['run_log_sha256'],build.sha(self.b.ev/'tnd28_normal_contract.log'))
        self.assertFalse(receipt['npu_executed']);self.assertFalse(receipt['loading_allowed'])

    def test_wrong_cpu_result_and_compile_failure_cannot_mint_receipt(self):
        self.configure()
        def wrong(label,argv,**kwargs):
            self.fake_cpu_run(label,argv,**kwargs)
            if label=='tnd28_normal_contract':write(self.b.ev/(label+'.log'),b'PASS_OTHER_TEST')
        with patch.object(self.b,'run',wrong),self.assertRaisesRegex(RuntimeError,'CONTRACT_OUTPUT_DIFFERS'):
            self.b.cpu_checks()
        self.assertNotIn('tnd28_cpu_contract_sha256',self.b.report)
        with patch.object(self.b,'run',side_effect=RuntimeError('compiler failure')):
            with self.assertRaisesRegex(RuntimeError,'compiler failure'):self.b.cpu_checks()
        self.assertFalse((self.b.ev/'tnd28_normal_cpu_contract.json').exists())

    def test_schema_is_actual_compile_entry_and_selftest_bound(self):
        self.configure();self.schema_fixture()
        with patch.object(self.b,'run') as run:self.b.tnd28_public_schema()
        self.assertIn('--source',run.call_args.args[1])
        self.assertTrue(self.b.report['tnd28_host_header_compiled_with_actual_sdk'])
        self.assertFalse(self.b.report['loading_allowed'])

    def test_schema_wrong_field_type_overlap_or_substitution_refused(self):
        self.configure()
        for changed in ('dtype','offset','duplicate','count','out_of_bounds'):
            folder,schema,compiled,_=self.schema_fixture()
            if changed=='dtype':schema['fields'][0]['dtype']='float'
            elif changed=='offset':schema['fields'][1]['offset']=0
            elif changed=='duplicate':schema['fields'][1]=dict(schema['fields'][0])
            elif changed=='count':schema['fields'][0]['count']=True
            else:schema['fields'][1]['offset']=1575
            json_file(folder/'fia_abi_schema.json',schema)
            compiled['schema_sha256']=build.sha(folder/'fia_abi_schema.json')
            json_file(folder/'compile_receipt.json',compiled)
            with self.subTest(changed=changed),patch.object(self.b,'run'),self.assertRaisesRegex(RuntimeError,'TYPED_FIELDS'):
                self.b.tnd28_public_schema()

    def test_schema_wrong_compiler_decoder_selftest_or_source_refused(self):
        self.configure()
        for changed in ('entry','decoder','source','projection','layout','selftest'):
            folder,schema,compiled,selftest=self.schema_fixture()
            if changed=='entry':compiled['actual_new_fia_compile_entry']['directory']='/old-build'
            elif changed=='decoder':write(folder/'fia_abi_decoder',b'OLD_BINARY')
            elif changed=='source':compiled['public_source_sha256']='0'*64
            elif changed=='projection':schema['projection_sha256']='0'*64
            elif changed=='layout':schema['layout_kind']='kernel_native_pointer'
            else:selftest['fields']['baseParams.a']['offset']=64
            json_file(folder/'fia_abi_schema.json',schema);json_file(folder/'fia_abi_selftest.json',selftest)
            compiled['schema_sha256']=build.sha(folder/'fia_abi_schema.json')
            json_file(folder/'compile_receipt.json',compiled)
            with self.subTest(changed=changed),patch.object(self.b,'run'),self.assertRaisesRegex(RuntimeError,'ACTUAL_SDK'):
                self.b.tnd28_public_schema()
            self.assertNotIn('tnd28_host_header_compiled_with_actual_sdk',self.b.report)

    def test_completion_remains_unqualified_and_binds_full_artifact_bytes(self):
        self.ready_to_finish();self.b.finish()
        report=json.loads((self.b.ev/'completion.json').read_bytes())
        self.assertEqual(report['status'],'CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION')
        self.assertEqual(report['artifacts'][0]['sha256'],build.sha(self.b.vendor/'op_api/lib/libcust_opapi.so'))
        self.assertIn('actual public V view and prepared pointer/copy/stride lineage',report['qualification_missing'])
        for name in build.TND28_DENIED:self.assertIs(report[name],False)

    def test_completion_cannot_silently_reset_a_prior_loading_grant(self):
        self.ready_to_finish()
        for name in build.TND28_DENIED:
            self.b.report[name]=True
            with self.subTest(name=name),self.assertRaisesRegex(RuntimeError,'AUTHORIZATION_MUST_REMAIN_FALSE'):
                self.b.finish()
            self.b.report[name]=False
        self.assertFalse((self.b.ev/'completion.json').exists())

    def test_completion_rehashes_cpu_schema_and_decoder_not_just_pass_flags(self):
        self.ready_to_finish()
        for relative in ('tnd28_normal_cpu_contract.json','tnd28_schema/fia_abi_schema.json',
                         'tnd28_schema/compile_receipt.json','tnd28_schema/fia_abi_selftest.json',
                         'tnd28_schema/fia_abi_decoder'):
            path=self.b.ev/relative;before=path.read_bytes();path.write_bytes(before+b'changed')
            with self.subTest(path=relative),self.assertRaisesRegex(RuntimeError,'COMPLETION_EVIDENCE'):
                self.b.finish()
            path.write_bytes(before)
        self.assertFalse((self.b.ev/'completion.json').exists())

    def test_exact_normal_binary_registration_cannot_borrow_33_or_fault_key(self):
        binary=self.root/'test.o';binary.write_bytes(b'CPU_BINARY_FIXTURE')
        metadata=dict(binFileName=registration.BINARY,binFileSuffix='.o',sha256=registration.sha(binary),
            coreType='MIX',taskRation='tilingKey',kernelList=[dict(tilingKey=k) for k in recipe.KEYS],
            supportInfo=dict(tilingKey=[str(k) for k in recipe.KEYS],simplifiedKeyMode=0,simplifiedKey=['CPU'],
                             optionalInputMode='gen_placeholder',optionalOutputMode='gen_placeholder'))
        registration.validate_metadata(metadata,binary,recipe.VARIANT)
        self.assertEqual(registration.KEYS[recipe.VARIANT],set(recipe.KEYS))
        for changed in ('missing','duplicate','extra','wrong33','hash'):
            bad=copy.deepcopy(metadata)
            if changed=='missing':bad['kernelList'].pop()
            elif changed=='duplicate':bad['supportInfo']['tilingKey'].append(str(recipe.NORMAL_KEY))
            elif changed=='extra':bad['kernelList'].append(dict(tilingKey=119000000000400000))
            elif changed=='wrong33':bad['kernelList'][0]['tilingKey']=103000000010200003
            else:bad['sha256']='0'*64
            with self.subTest(changed=changed),self.assertRaises(RuntimeError):
                registration.validate_metadata(bad,binary,recipe.VARIANT)


if __name__=='__main__':
    unittest.main()
