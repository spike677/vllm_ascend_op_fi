"""Current public CPU provider originals; never NPU/standard-Worker evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import shutil
import subprocess
import sys

from experiments.worker_core_v11_20261118 import review17_worker_originals as r


class LLMEngine:
    def has_unfinished_requests(self):return False
    def reset_encoder_cache(self):return None


class LLM:
    def __init__(self):self.llm_engine=LLMEngine()
    def reset_mm_cache(self):return None


class Worker17OriginalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_core_vision17_group import Vision17GroupTests,REQUESTS
        from demoncat.runtime.vllm_ascend.worker_extension.providers.vision17_group import (
            install_vision17_group_binding,remove_vision17_group_binding)
        from experiments.worker_core_v11_20261118.vision17_cache_control import reset_idle_mm_cache
        environment=dict(cpu_test_fixture=True)
        environment_sha=hashlib.sha256(json.dumps(environment,sort_keys=True).encode()).hexdigest()
        setup={}
        def bind(case,identity,config):
            value=install_vision17_group_binding(case.worker,prompt_token_ids=[1,2,3],image_sha256='a'*64,
                weights_sha256='b'*64,environment_sha256=environment_sha,first_live_validation=True,_simulation=True,
                _processor=case.model.processor,_synchronize=lambda:None)
            setup.update(value,environment=environment,environment_sha256=environment_sha)
            return {}
        case=Vision17GroupTests()
        with patch.object(Vision17GroupTests,'bind',bind):case.setUp()
        cls.addClassCleanup(case.tearDown)
        document=dict(status='CPU_TEST_STARTED',requests=[],mm_cache_resets=[])
        llm=LLM();case.start()
        for index,(phase,_) in enumerate(REQUESTS):
            document['mm_cache_resets'].append(reset_idle_mm_cache(llm,phase,reset_encoder_cache=True))
            output=case.request();row=case.provider.engine.rows[-1]
            document['requests'].append(dict(phase=phase,prompt_token_ids=row['request_binding']['prompt_token_ids'],
                output_token_ids=[int(output.argmax())],text='CPU fixture',mapping=dict(internal_request_id=row['request_id'],
                    external_request_id=str(index),request_state_object_id=id(case.runner.requests[row['request_id']]),
                    source='EXPLICIT_CPU_RUNNER_MAPPING')))
        final=case.ok(case.client.finish());case.ok(case.client.unregister());host=case.client.detach()
        document.update(phase_rows=final['phase_rows'],coverage=final['reports'],rpc=host['history'],
            last_scope={k:v for k,v in host.items() if k not in ('history','capability_enumeration')},
            setup=setup,unbind=remove_vision17_group_binding(case.worker),binding_still_owned=False,
            input=dict(image_sha256='a'*64,weights_sha256='b'*64,config_sha256='c'*64,processor_overrides={},raw_PIL_image=True),
            probe_sha256=r.C.file_sha(Path(r.__file__).parents[2]/r.PROBE))
        cls.document=json.loads(json.dumps(document,allow_nan=False))

    def inspect(self,change=None,natural=False):
        d=deepcopy(self.document)
        if change:change(d)
        return r.review(d,natural=natural)

    def test_actual_current_cpu_public_ten_phase_originals(self):
        result=self.inspect();self.assertTrue(result['consistent'],result['issues'])
        self.assertTrue(result['originals_reconciled'])
        self.assertEqual(result['details']['public_protocol']['actions'],28)
        self.assertEqual(sum(x['logits_changed'] for x in result['details']['phases']),3)
        self.assertFalse(result['c2'] or result['c3'] or result['e5'] or result['native_qualification'] or result['multi_target_verified'])

    def test_saved_success_or_summary_hit_counts_are_not_the_oracle(self):
        result=self.inspect(lambda d:d.update(status='SUCCESS',c2=True,c3=True,E5=True,fault_injections=0,
                                             logits_changed={'all':False}))
        self.assertTrue(result['consistent'],result['issues']);self.assertFalse(result['c2'] or result['e5'])

    def test_cpu_cannot_qualify_natural_npu_scope(self):
        self.assertFalse(self.inspect(natural=True)['consistent'])

    def test_final_padding_cannot_be_replaced_by_saved_applied_bool(self):
        self.assertFalse(self.inspect(lambda d:d['phase_rows'][4]['conv_observation']['observed_calls'][0]['actual_geometry'].update(
            padding=[0,0,0]))['consistent'])

    def test_upper_original_parameters_and_non_padding_arguments_are_fixed(self):
        for change in (lambda d:d['phase_rows'][4]['initial_observation']['observed_calls'][0]['actual_geometry'].update(padding=[0,7,7]),
                       lambda d:d['phase_rows'][4]['conv_observation']['observed_calls'][0]['actual_geometry'].update(groups=2)):
            with self.subTest(change=change):self.assertFalse(self.inspect(change)['consistent'])

    def test_weight_bias_or_input_change_refused(self):
        for index in range(3):
            def change(d):
                value=d['phase_rows'][4]['conv_observation']['observed_calls'][0]['inputs_after'][index]
                if value is None:d['phase_rows'][4]['conv_observation']['observed_calls'][0]['inputs_after'][index]={'sha256':'0'*64,'shape':[1],'dtype':'torch.float32'}
                else:value['sha256']='0'*64
            with self.subTest(index=index):self.assertFalse(self.inspect(change)['consistent'])

    def test_consumer_equal_values_without_exact_object_is_refused(self):
        self.assertFalse(self.inspect(lambda d:d['phase_rows'][4]['conv_observation']['observed_calls'][0]['consumer']
            ['actual_input_metadata'][0].update(object_id=999))['consistent'])

    def test_alias_storage_or_unknown_first_consumer_is_refused(self):
        for change in (lambda d:d['phase_rows'][4]['conv_observation']['observed_calls'][0]['lineage_edges'][0]['output_metadata'][0].update(storage_id=999),
                       lambda d:d['phase_rows'][4]['conv_observation']['observed_calls'][0]['consumer'].update(operator='aten.unknown.default')):
            with self.subTest(change=change):self.assertFalse(self.inspect(change)['consistent'])

    def test_fault_logits_off_or_repeat_mismatch_is_refused(self):
        for index,source in ((4,0),(5,4),(7,0),(9,4)):
            self.assertFalse(self.inspect(lambda d:d['phase_rows'][index].update(logits=d['phase_rows'][source]['logits']))['consistent'])

    def test_cache_reset_success_cannot_replace_encoder_execution(self):
        self.assertFalse(self.inspect(lambda d:d['phase_rows'][6].update(encoder_observed=False))['consistent'])

    def test_request_mapping_scope_or_proof_object_drift_refused(self):
        for change in (lambda d:d['requests'][5]['mapping'].update(internal_request_id='foreign'),
                       lambda d:d['coverage']['C3']['scope'].update(model_weights_sha256='0'*64),
                       lambda d:d['coverage']['C3']['live_c2_proof']['consumed']['object_binding'].update(python_object_id=1)):
            with self.subTest(change=change):self.assertFalse(self.inspect(change)['consistent'])

    def test_missing_ledger_call_lease_leak_or_detach_is_refused(self):
        for change in (lambda d:d['coverage']['C3']['calls'].pop(),
                       lambda d:d['coverage']['C3']['recovery']['workers'][0].update(host_bytes=16),
                       lambda d:d['rpc'].pop(),lambda d:d.update(binding_still_owned=True)):
            with self.subTest(change=change):self.assertFalse(self.inspect(change)['consistent'])

    def test_extract_only_and_failed_attempt_never_gain_original_authority(self):
        result=self.inspect(lambda d:d.update(source_original=dict(extraction_only=True)))
        self.assertEqual(result['classification'],'EXTRACT_REJECTED_NOT_ORIGINAL');self.assertFalse(result['consistent'])
        result=self.inspect(lambda d:d.update(error='actual encoder failure',status='SUCCESS'))
        self.assertEqual(result['classification'],'FAILED_OR_PARTIAL_NOT_QUALIFIED');self.assertFalse(result['originals_reconciled'])

    def test_summary_does_not_embed_rpc_or_full_original_objects(self):
        output=json.dumps(self.inspect())
        self.assertLess(len(output),18000)
        self.assertNotIn('actual_input_metadata',output);self.assertNotIn('prompt_token_ids',output)

    def test_source_provenance_rehashes_current_producer_and_rejects_changed_file_or_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=Path(r.__file__).parents[2]
            paths=['fault_injection/'+name for name in r.SOURCE_NAMES]
            paths.extend(('src/' if name.startswith('demoncat.') else '')+name.replace('.','/')+'.py' for name in r.SOURCE_MODULES)
            paths.extend((r.PROBE,'experiments/worker_core_v11_20261118/vision17_cache_control.py',
                'experiments/c2c3_closure_20261117/vision17_worker_census_probe.py',
                'experiments/c2c3_closure_20261117/natural_tensor_worker_core_probe.py',
                'experiments/c2c3_closure_20261117/natural_vision_input.py'))
            paths.extend('src/demoncat/runtime/vllm_ascend/worker_extension/'+name+'.py' for name in
                         ('host','runtime','identity','config','registry','serving_context','providers/_scope'))
            for name in paths:
                dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,dest)
            commit='a'*40;manifest=root/'source_manifest.json'
            manifest.write_text(json.dumps(dict(base_commit=commit,files=[dict(path=name,sha256=r.C.file_sha(root/name),
                bytes=(root/name).stat().st_size) for name in paths])),encoding='utf-8')
            sha=r.C.file_sha(manifest);a=r.C.Audit()
            proof=r.source_review(a,self.document,root,sha,commit)
            self.assertTrue(a.report()['consistent'],a.issues);self.assertEqual(proof['files_rehashed'],len(paths))
            a=r.C.Audit();r.source_review(a,self.document,root,'0'*64,commit)
            self.assertFalse(a.report()['consistent'])
            (root/r.PROBE).write_text('# changed after run\n',encoding='utf-8')
            a=r.C.Audit();r.source_review(a,self.document,root,sha,commit)
            self.assertFalse(a.report()['consistent'])

    def test_cli_rejects_extract_and_wrong_original_sha_without_runtime_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'extract.json';path.write_text(json.dumps(dict(source_original=dict(extraction_only=True))))
            args=[sys.executable,r.__file__,'--result',str(path),'--source-root',directory,'--source-commit','a'*40,
                  '--source-manifest-sha256','b'*64,'--result-sha256']
            for sha in (r.C.file_sha(path),'0'*64):
                run=subprocess.run(args+[sha],capture_output=True,text=True,check=False)
                self.assertEqual(run.returncode,2,run.stderr)
                result=json.loads(run.stdout);self.assertFalse(result['consistent'] or result['originals_reconciled'])
                self.assertFalse(result['runtime_executed'] or result['c2'] or result['c3'] or result['e5'])


if __name__=='__main__':unittest.main()
