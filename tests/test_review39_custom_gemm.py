"""Stdlib-only receipt doubles; no simulated data grants device qualification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.c2c3_postwrite_20261117 import review39_custom_gemm as custom
from experiments.c2c3_postwrite_20261117 import review39_worker_core_result as review


def geometry(shape, identity, dtype='torch.bfloat16'):
    size = 4 if dtype == 'torch.float32' else 2
    stride = []; step = 1
    for n in reversed(shape): stride.insert(0,step); step *= n
    return dict(shape=shape, stride=stride, storage_offset=0, dtype=dtype, device='npu:0',
        data_ptr=identity*4096, storage_identity=identity, storage_bytes=step*size, logical_bytes=step*size)


def producer(output=None):
    output = output or geometry([2,8],30)
    path = '/vllm-workspace/vllm/vllm/utils/torch_utils.py'
    active = 'PrivateUse1: registered at '+path+':789 :: (none) [ boxed ]'
    definition = 'debug: registered at '+path+':789'
    current = dict(guard_profile='TORCH_2_9_SINGLE_PRIVATEUSE1_NO_SHADOW_SOURCE_ORIGIN_V1', torch_version=custom.TORCH_VERSION,
        current_privateuse1_registration=active, definition_registration=definition,
        source_module='vllm.utils.torch_utils',source_path=path,source_sha256=custom.SOURCES['vllm.utils.torch_utils'],
        construction_lines=[789,789],actual_registration_line=789,shadowed_privateuse1_registrations=0)
    implementation = dict(schema=custom.SCHEMA,sources=dict(custom.SOURCES),public_operator=custom.OPERATOR,
        lowering='torch.nn.functional.linear(x, weight, None)',module_weight_object_identity=99,
        registered_library_object_identity=100,registered_dispatch_key='PrivateUse1',current_registration=current,
        actual_dispatch_dump='\n'.join(['name: vllm::unquantized_gemm','schema: '+custom.SCHEMA,definition,
            'alias analysis kind: FROM_SCHEMA',active]),source_and_loaded_code_verified=True,owned_weight_verified=True)
    return dict(operator=custom.OPERATOR,boundary='reviewed_vllm_custom_gemm_return',schema=custom.SCHEMA,
        schema_sha256=hashlib.sha256(custom.SCHEMA.encode()).hexdigest(),expected_output_shape=output['shape'],
        operands={'x':geometry([2,4],10),'weight':geometry([8,4],20)},returned_tensor=output,
        original_return_storage=True,registered_implementation=implementation,native_matmul_claim=False)


def consumer(value, operator=custom.RMS):
    contract = custom.CONSUMERS[operator]; rms = operator == custom.RMS
    outputs = [geometry(value['shape'] if rms else [value['shape'][0],value['shape'][1]//2],40)]
    if rms: outputs.append(geometry(value['shape'][:-1]+[1],41,'torch.float32'))
    active = ('PrivateUse1: registered at build/CMakeFiles/torch_npu.dir/compiler_depend.ts:8413 :: '
        +contract['signature']+' [ boxed unboxed ]')
    proof = dict(operator=operator,schema=contract['schema'],schema_sha256=hashlib.sha256(contract['schema'].encode()).hexdigest(),
        torch_version=custom.TORCH_VERSION,torch_npu_version=custom.TORCH_NPU_VERSION,
        meta_shape_source=dict(module=custom.META_MODULE,sha256=custom.META_SHA,function=contract['function']),
        actual_dispatch_dump='\n'.join(['schema: '+contract['schema'],'alias analysis kind: FROM_SCHEMA',active]),
        schema_has_no_alias_or_write=True,original_returned_once=True,input=value,outputs=outputs,
        formats=[2]*(4 if rms else 2),scalars={'epsilon':1e-6} if rms else {'dim':-1})
    if rms: proof['gamma'] = geometry([value['shape'][-1]],42)
    return dict(operator=operator,module_path='projection',original_returned=True,selected_elements_read=[0],
        operand_views=[value],reviewed_consumer=proof)


def inventory(output=None, operator=custom.RMS):
    proof = producer(output); value = proof['returned_tensor']; read = consumer(value,operator)
    refused = dict(key=[77,custom.OPERATOR,0],module_path='projection',module_object_identity=77,operator=custom.OPERATOR,
        schema=custom.SCHEMA,output_geometry=geometry([2,8],31),original_calls=1,rejection='CPU_METADATA_UNOWNED_WEIGHT',
        eligible=False,selected_indices=[],consumers=[],producer_proof=None,consumer_observations=[])
    selected = dict(key=[77,custom.OPERATOR,1],module_path='projection',module_object_identity=77,operator=custom.OPERATOR,
        schema=custom.SCHEMA,output_geometry=value,original_calls=1,rejection=None,eligible=True,selected_indices=[0],
        consumers=[operator],producer_proof=proof,consumer_observations=[read])
    return dict(admission_independent=True,dropped_calls=0,calls=[refused,selected],
        modules=[dict(path='projection',object_identity=77,calls=1)],operator_counts={custom.OPERATOR:2,operator:1})


def native_receipt(output, owner, generation, phase):
    events = []
    for sequence,operation in enumerate(('open','apply','restore','release','live'),1):
        event = dict(sequence=sequence,operation=operation,handle=500+generation,owner=owner,generation=generation,
            returned=True,return_code=0,error=None)
        if operation == 'open': event.update(data_ptr=output['data_ptr'],bytes=output['logical_bytes'],device=0)
        if operation == 'apply': event.update(mode=int(phase=='fault'),indices=[0],changed_elements=int(phase=='fault'))
        if operation == 'live': event['native_live'] = False
        events.append(event)
    return dict(schema='demoncat.postwrite.lease_receipt.v1',owner=owner,generation=generation,handle=500+generation,thread=333,
        geometry=output,element_offset=0,elements=output['logical_bytes']//2,open=events[0],applications=[events[1]],
        restore=events[2],release=events[3],live_readbacks=[events[4]],events=events,first_failure=None,closed=True)


def c1_details():
    rows = []; inventories = []; owner = 71
    for index,phase in enumerate(review.PHASES):
        generation = (1,1,2,3,4,4)[index]
        item = inventory(geometry([2,8],30+index)); selected = item['calls'][1]
        report = dict(actual_target=dict(producer_proof=selected['producer_proof'],consumer_observations=selected['consumer_observations']),
            native_lease_receipt=None,intervention=None)
        if phase in ('baseline','fault'):
            report['native_lease_receipt'] = native_receipt(selected['output_geometry'],owner,generation,phase)
            report['intervention'] = dict(selected_elements=1,readback_exact=True,changed_elements=int(phase=='fault'),new_fault_hits=int(phase=='fault'))
        if phase != 'health': inventories.append(item)
        rows.append(dict(phase=phase,report=report if phase!='health' else None,
            complete_inventory=item if phase!='health' else None,native_generation=generation))
    selected = inventories[0]['calls'][1]
    return dict(rows=rows,inventories=inventories,native_owner=owner,
        selection=dict(producer_proof=selected['producer_proof'],consumer_observations=selected['consumer_observations']))


class CustomReceiptTests(unittest.TestCase):
    def setUp(self):
        self.core,self.legacy = review.helpers()
        self.profile = custom.ProjectionProfile(self.legacy)

    def examine(self, item):
        audit = self.core.Audit()
        audit.section('custom_inventory',self.profile.inventory,audit,item,'metadata-double','npu')
        return audit.report()

    def refuses(self,change,code,operator=custom.RMS):
        item = inventory(operator=operator); change(item)
        report = self.examine(item)
        self.assertFalse(report['consistent'],report)
        self.assertIn(code,[row['code'] for row in report['issues']],report)

    def test_actual_receipt_geometry_for_rms_and_swiglu(self):
        for op in custom.CONSUMERS:
            with self.subTest(op=op):
                result = self.examine(inventory(operator=op))
                self.assertTrue(result['consistent'],result)

    def test_old_profile_keeps_custom_unqualified_and_source51_constants(self):
        item = inventory(); audit = self.core.Audit()
        self.legacy.inventory(audit,item,'legacy','npu')
        self.assertFalse(audit.report()['consistent'])
        self.assertNotIn(custom.OPERATOR,self.legacy.PRODUCERS)
        self.assertNotIn(custom.RMS,self.legacy.CONSUMERS)

    def test_three_exact_sources_and_schema_required(self):
        self.refuses(lambda item:item['calls'][1]['producer_proof']['registered_implementation']['sources'].update(
            {'vllm.utils.torch_utils':'0'*64}),'CUSTOM_THREE_SOURCES_LOADED_CODE_AND_WEIGHT_OBJECT')
        self.refuses(lambda item:item['calls'][1].update(schema=custom.SCHEMA+' '),'EVERY_CUSTOM_CALL_EXACT_SCHEMA')

    def test_bias_tensor_or_cloned_original_return_cannot_be_accepted(self):
        self.refuses(lambda item:item['calls'][1]['producer_proof']['operands'].update(bias=geometry([8],50)),
            'CUSTOM_ORIGINAL_GEOMETRY_AND_NO_BIAS')
        self.refuses(lambda item:item['calls'][1]['producer_proof'].update(returned_tensor=geometry([2,8],51)),
            'CUSTOM_ORIGINAL_GEOMETRY_AND_NO_BIAS')

    def test_foreign_or_same_library_override_leaves_inactive_original(self):
        def change(item):
            proof = item['calls'][1]['producer_proof']['registered_implementation']
            proof['actual_dispatch_dump'] += '\nPrivateUse1 (inactive): registered at /foreign:1 :: (none) [ boxed ]'
        self.refuses(change,'CURRENT_UNSHADOWED_PRIVATEUSE1_DEFINITION_AND_SOURCE_OWNER')

    def test_active_origin_and_definition_source_line_are_recomputed(self):
        def change(item):
            proof = item['calls'][1]['producer_proof']['registered_implementation']
            proof['actual_dispatch_dump'] = proof['actual_dispatch_dump'].replace(':789',':790')
            current = proof['current_registration']
            current['actual_registration_line'] = 790
            current['construction_lines'] = [790,790]
            for key in ('current_privateuse1_registration','definition_registration'):
                current[key] = current[key].replace(':789',':790')
        self.refuses(change,'CURRENT_UNSHADOWED_PRIVATEUSE1_DEFINITION_AND_SOURCE_OWNER')

    def test_bool_guard_without_original_current_dispatch_receipt_fails(self):
        self.refuses(lambda item:item['calls'][1]['producer_proof']['registered_implementation'].pop('current_registration'),
            'MISSING_OR_INVALID_SCHEMA:KeyError')

    def test_all_actual_ordinals_include_earlier_refused_call(self):
        self.refuses(lambda item:item['calls'][1]['key'].__setitem__(2,0),'DUPLICATE_ACTUAL_CALL_OR_MODULE')
        self.refuses(lambda item:item['calls'].pop(0),'PROJECTION_OPERATOR_DENOMINATOR')

    def test_eligible_custom_requires_full_original_producer_and_consumer(self):
        self.refuses(lambda item:item['calls'][1].update(producer_proof=None),'ELIGIBLE_CUSTOM_HAS_ORIGINAL_PRODUCER_PROOF')
        self.refuses(lambda item:item['calls'][1].update(consumer_observations=[]),'CONSUMER_CENSUS')

    def test_npu_schema_version_alias_and_inactive_registration_are_rejected(self):
        for key,value in (('torch_version','2.10.0+cpu'),('torch_npu_version','foreign'),('schema',custom.SCHEMA),
                          ('schema_has_no_alias_or_write',False)):
            with self.subTest(key=key):
                self.refuses(lambda item:item['calls'][1]['consumer_observations'][0]['reviewed_consumer'].update({key:value}),
                    'CURRENT_READ_ONLY_NPU_CONSUMER_REGISTRATION')
        self.refuses(lambda item:item['calls'][1]['consumer_observations'][0]['reviewed_consumer'].update(
            actual_dispatch_dump='PrivateUse1 (inactive): foreign'),'CURRENT_READ_ONLY_NPU_CONSUMER_REGISTRATION')

    def test_schema_meta_example_does_not_establish_actual_rstd_shape(self):
        self.refuses(lambda item:item['calls'][1]['consumer_observations'][0]['reviewed_consumer']['outputs'].__setitem__(
            1,geometry([2,8],41,'torch.float32')),'ACTUAL_RMS_GAMMA_AND_TWO_RETURN_GEOMETRIES')

    def test_swiglu_scalar_and_full_actual_output_shape(self):
        self.refuses(lambda item:item['calls'][1]['consumer_observations'][0]['reviewed_consumer'].update(scalars={'dim':0}),
            'ACTUAL_SWIGLU_LAST_EVEN_DIM_RETURN_GEOMETRY',custom.SWI)
        self.refuses(lambda item:item['calls'][1]['consumer_observations'][0]['reviewed_consumer']['outputs'].__setitem__(
            0,geometry([2,8],40)),'ACTUAL_SWIGLU_LAST_EVEN_DIM_RETURN_GEOMETRY',custom.SWI)

    def test_consumer_gamma_alias_or_non_nd_format_is_not_a_value_read(self):
        self.refuses(lambda item:item['calls'][1]['consumer_observations'][0]['reviewed_consumer']['gamma'].update(storage_identity=30),
            'ACTUAL_ND_READ_ONLY_NONALIASED_CONSUMER_STORAGE')
        self.refuses(lambda item:item['calls'][1]['consumer_observations'][0]['reviewed_consumer']['formats'].__setitem__(0,47),
            'ACTUAL_ND_READ_ONLY_NONALIASED_CONSUMER_STORAGE')
        self.refuses(lambda item:item['calls'][1]['consumer_observations'][0]['reviewed_consumer']['outputs'][0].update(data_ptr=30*4096),
            'ACTUAL_ND_READ_ONLY_NONALIASED_CONSUMER_STORAGE')

    def test_disjoint_consumer_view_cannot_claim_selected_index_zero(self):
        def change(item):
            row = item['calls'][1]['consumer_observations'][0]
            other = deepcopy(row['reviewed_consumer']['input'])
            other.update(shape=[1,8],storage_offset=8,data_ptr=other['data_ptr']+16,logical_bytes=16)
            row['reviewed_consumer']['input'] = other; row['operand_views'] = [other]
        self.refuses(change,'ACTUAL_NPU_INPUT_READS_SELECTED_STORAGE')


class CustomC1JoinTests(unittest.TestCase):
    def setUp(self):
        self.core,legacy = review.helpers(); self.profile = custom.ProjectionProfile(legacy)

    def examine(self,details):
        audit = self.core.Audit()
        for index,item in enumerate(details['inventories']):
            audit.section('inventory',self.profile.inventory,audit,item,str(index),'npu')
        audit.section('c1',custom.completed_c1,audit,self.profile,details)
        return audit.report()

    def refuses(self,change,code):
        details = c1_details(); change(details); report = self.examine(details)
        self.assertFalse(report['consistent'],report)
        self.assertIn(code,[row['code'] for row in report['issues']],report)

    def test_six_request_metadata_joins_original_spans_two_generations_and_off(self):
        report = self.examine(c1_details())
        self.assertTrue(report['consistent'],report)

    def test_same_value_weight_clone_and_foreign_library_object_rejected(self):
        for key in ('module_weight_object_identity','registered_library_object_identity'):
            with self.subTest(key=key):
                self.refuses(lambda d:d['inventories'][3]['calls'][1]['producer_proof']['registered_implementation'].update({key:999}),
                    'CUSTOM_SAME_CURRENT_LIBRARY_AND_OWNED_WEIGHT_OBJECT')

    def test_native_open_changed_pointer_bytes_device_or_original_geometry(self):
        for key,value in (('data_ptr',99999),('bytes',2),('device',1)):
            with self.subTest(key=key):
                self.refuses(lambda d:d['rows'][3]['report']['native_lease_receipt']['open'].update({key:value}),
                    'ACTUAL_NATIVE_OPEN_USES_EXACT_ORIGINAL_DEVICE_SPAN')
        self.refuses(lambda d:d['rows'][3]['report']['native_lease_receipt'].update(geometry=geometry([2,8],99)),
            'ACTUAL_NATIVE_ORIGINAL_SPAN_OWNER_GENERATION_AND_CLOSE')

    def test_native_owner_generation_and_returned_handle_each_join_original(self):
        for key,value in (('owner',999),('generation',999),('handle',999)):
            with self.subTest(key=key):
                self.refuses(lambda d:d['rows'][3]['report']['native_lease_receipt']['release'].update({key:value}),
                    'ACTUAL_NATIVE_SUCCESSFUL_EXACT_TICKET_RETURNS')

    def test_native_apply_selected_index_mode_and_changed_count(self):
        for key,value in (('indices',[1]),('mode',4),('changed_elements',0)):
            with self.subTest(key=key):
                self.refuses(lambda d:d['rows'][3]['report']['native_lease_receipt']['applications'][0].update({key:value}),
                    'ACTUAL_NATIVE_APPLY_MODE_SELECTED_INDEX_AND_EFFECT')

    def test_native_release_failure_or_live_owner_cannot_be_zero_by_claim(self):
        self.refuses(lambda d:d['rows'][3]['report']['native_lease_receipt']['release'].update(return_code=3),
            'ACTUAL_NATIVE_SUCCESSFUL_EXACT_TICKET_RETURNS')
        self.refuses(lambda d:d['rows'][3]['report']['native_lease_receipt']['live_readbacks'][0].update(native_live=True),
            'ACTUAL_NATIVE_RELEASE_THEN_EXACT_TICKET_ABSENT_READBACK')
        self.refuses(lambda d:d['rows'][3]['report']['native_lease_receipt'].update(first_failure={'error':'original failure'}),
            'ACTUAL_NATIVE_ORIGINAL_SPAN_OWNER_GENERATION_AND_CLOSE')

    def test_native_close_is_not_inferred_from_missing_or_reordered_events(self):
        self.refuses(lambda d:d['rows'][3]['report']['native_lease_receipt']['events'].reverse(),
            'ACTUAL_NATIVE_CALL_ORDER_AND_UNMODIFIED_EVENT_PROJECTIONS')
        self.refuses(lambda d:d['rows'][3]['report'].pop('native_lease_receipt'),'MISSING_OR_INVALID_SCHEMA:KeyError')
        self.refuses(lambda d:d['rows'][3]['report']['native_lease_receipt'].update(closed=False),
            'ACTUAL_NATIVE_ORIGINAL_SPAN_OWNER_GENERATION_AND_CLOSE')

    def test_off_cannot_reuse_previous_native_receipt(self):
        self.refuses(lambda d:d['rows'][4]['report'].update(native_lease_receipt=d['rows'][3]['report']['native_lease_receipt']),
            'CUSTOM_OFF_OPENS_NO_NATIVE_LEASE')

    def test_fault_phase_cannot_drop_an_earlier_refused_ordinal(self):
        self.refuses(lambda d:d['inventories'][3]['calls'].pop(0),'CUSTOM_EVERY_PHASE_FULL_ACTUAL_ORDINAL_DENOMINATOR')

    def test_actual_target_lineage_cannot_be_substituted_after_inventory(self):
        self.refuses(lambda d:d['rows'][3]['report']['actual_target'].update(producer_proof=producer(geometry([2,8],99))),
            'CUSTOM_SELECTED_RAW_ORDINAL_LINEAGE_EQUALS_ACTUAL_TARGET')

    def test_original_setup_and_worker_identity_cannot_be_replaced_by_status(self):
        worker = dict(pid=1,model_instance_id='77')
        binding = dict(worker=worker,loaded_sources={name:{'sha256':'1'*64} for name in custom.EXTRA_MODULES})
        records = [dict(method='actual.setup',responses=[binding]),dict(method='demoncat_worker_identity',responses=[dict(worker=worker)])]
        def examine(raw):
            audit = self.core.Audit()
            audit.section('setup',custom.deployment,audit,dict(binding=binding),dict(worker=worker),raw)
            return audit.report()
        self.assertTrue(examine(records)['consistent'])
        self.assertFalse(examine(records[1:])['consistent'])
        self.assertFalse(examine(records[:1])['consistent'])


class CustomOfflineTests(unittest.TestCase):
    def test_custom_helper_import_and_checks_never_load_torch_or_dso(self):
        code = ('import sys;from experiments.c2c3_postwrite_20261117 import review39_worker_core_result as r;'
            'sys.addaudithook(r.deny_runtime);r.helpers(r.CUSTOM_PROFILE);'
            'from test_review39_custom_gemm import CustomReceiptTests;'
            't=CustomReceiptTests();t.setUp();t.test_actual_receipt_geometry_for_rms_and_swiglu();'
            'assert not any(x.split(".")[0] in ("torch","torch_npu","vllm","ctypes") for x in sys.modules)')
        run = subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)

    def test_custom_source_profile_rehashes_loaded_dependencies_and_manifest(self):
        core,_ = review.helpers()
        with tempfile.TemporaryDirectory(prefix='.review39-',dir=Path.cwd()) as folder:
            root = Path(folder).resolve()
            self.assertTrue(root.is_relative_to(Path.cwd().resolve()))
            files = []; hashes = {}; loaded = {}
            for name in review.SOURCES+custom.EXTRA_MODULES:
                path = root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(name.encode())
                digest = hashlib.sha256(path.read_bytes()).hexdigest(); hashes[name] = digest
                files.append(dict(path=name,sha256=digest,bytes=path.stat().st_size))
                if name in review.MODULES+custom.EXTRA_MODULES:
                    loaded[name] = dict(path='/source/'+name,sha256=digest)
            manifest = root/'source_manifest.json'; manifest.write_text(json.dumps(dict(base_commit='a'*40,files=files)))
            sha = core.file_sha(manifest); native = root/'native.so'; native.write_bytes(b'CPU receipt hash double; never loaded')
            weights = dict(format='safetensors',files={'actual':dict(sha256='1'*64,bytes=1)})
            doc = dict(schema='demoncat.output39.worker_core_probe.v1',mechanism='39.output_proxy',standard_worker=True,
                native_sha256=core.file_sha(native),build_fingerprint='2'*64,sources=hashes,
                weights=dict(manifest=weights,weights_sha256=core.digest(weights)),
                binding=dict(worker={},loaded_sources=loaded,model_source=dict(manifest={},source_sha256=core.digest({})),
                    binding=dict(workers_created=0,hooks_installed=0,native_leases_opened=0,
                        native=dict(sha256=core.file_sha(native),loaded_mapping_verified=True))))
            def check(value=doc):
                with patch.object(review,'NATIVE_SHA',core.file_sha(native)):
                    return review.provenance(core,value,dict(worker={}),root,sha,'a'*40,native,profile=review.CUSTOM_PROFILE)
            self.assertTrue(check()['consistent'])
            missing = deepcopy(doc); missing['binding']['loaded_sources'].pop(custom.EXTRA_MODULES[0])
            self.assertIn('LOADED_SOURCE_AND_WORKER_SET',[row['code'] for row in check(missing)['issues']])
            foreign = deepcopy(doc); foreign['binding']['loaded_sources'][custom.EXTRA_MODULES[1]]['sha256'] = '0'*64
            self.assertIn('ACTUAL_FROZEN_LOADED_MODULE_HASH',[row['code'] for row in check(foreign)['issues']])
            absent = deepcopy(doc); absent.update(binding=None,requests=[{'actual_request':'metadata double'}])
            self.assertIn('CUSTOM_REQUESTS_REQUIRE_ORIGINAL_DEPLOYMENT',[row['code'] for row in check(absent)['issues']])
            (root/custom.EXTRA_MODULES[0]).write_bytes(b'changed after frozen manifest')
            self.assertIn('SOURCE_FILE_BYTES',[row['code'] for row in check()['issues']])

    def test_profile_is_explicit_and_unknown_profile_never_falls_back(self):
        with self.assertRaisesRegex(ValueError,'UNKNOWN_PROJECTION_PROFILE'):
            review.helpers('anything-successful')


if __name__ == '__main__': unittest.main()
