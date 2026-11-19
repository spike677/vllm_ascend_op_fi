"""Local harness contracts; real CPU Host/Runtime, explicit non-CANN Native double."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from experiments.worker_core_v11_20261118 import probe_quant_ordinary as probe
from experiments.worker_core_v11_20261118 import prepare_quant_ordinary_case as prepare


def model_files(folder):
    description = {'model_quant_type': 'W8A8_DYNAMIC', 'model.embed_tokens.weight': 'FLOAT',
        'model.layers.0.self_attn.q_proj.weight': 'W8A8_DYNAMIC',
        'model.layers.0.self_attn.q_proj.weight_scale': 'FLOAT',
        'test_nested_metadata': {'retained': [1, {'same': True}]}}
    config = dict(model_type='qwen3', quantize='w8a8_dynamic', torch_dtype='bfloat16',
                  quantization_config=description)
    values = {'config.json': json.dumps(config).encode(),
        'quant_model_weight_w8a8_dynamic.safetensors': b'EXPLICIT_CPU_IDENTITY_FIXTURE_NOT_REAL_WEIGHTS',
        'quant_model_description_w8a8_dynamic.json': json.dumps(description).encode(),
        'tokenizer.json': b'{}', 'tokenizer_config.json': b'{}', 'generation_config.json': b'{}'}
    for name, raw in values.items():
        (folder / name).write_bytes(raw)
    return values


class IdentityAndCaseContracts(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.values = model_files(self.folder)

    def test_exact_nonstandard_single_file_identity_uses_real_names_and_all_bytes(self):
        result = probe.model_identity(self.folder)
        self.assertEqual(set(result['manifest']['files']), set(self.values))
        self.assertEqual(result['manifest']['selected_safetensors'], ['quant_model_weight_w8a8_dynamic.safetensors'])
        self.assertFalse(result['manifest']['qualification_granted'])
        for name, raw in self.values.items():
            self.assertEqual(result['manifest']['files'][name], dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        before = result['sha256']
        (self.folder / 'tokenizer.json').write_bytes(b'{"changed":true}')
        self.assertNotEqual(probe.model_identity(self.folder)['sha256'], before)

    def test_extra_weight_or_index_or_directory_is_not_guessed(self):
        for name in ('model.safetensors', 'model.safetensors.index.json', 'unknown_directory'):
            with self.subTest(name=name):
                path = self.folder / name
                path.mkdir() if name.endswith('directory') else path.write_bytes(b'{}')
                with self.assertRaisesRegex(ValueError, 'EXACT_SINGLE_FILE'):
                    probe.model_identity(self.folder)
                path.rmdir() if path.is_dir() else path.unlink()

    def test_config_moe_static_wrong_dtype_or_missing_description_refused(self):
        path = self.folder / 'config.json'
        for change in ({'model_type': 'qwen3_moe'}, {'torch_dtype': 'float16'}):
            value = json.loads(self.values['config.json'])
            value.update(change)
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'ORDINARY_QWEN_DYNAMIC'):
                probe.model_identity(self.folder)
        path.write_bytes(self.values['config.json'])
        (self.folder / 'quant_model_description_w8a8_dynamic.json').unlink()
        with self.assertRaisesRegex(ValueError, 'EXACT_SINGLE_FILE'):
            probe.model_identity(self.folder)

    def test_actual_generation_options_are_not_a_per_layer_loader_description(self):
        path = self.folder / 'config.json'
        config = json.loads(self.values['config.json'])
        config['quantization_config'] = dict(w_bit=8, a_bit=8, is_dynamic=True, dev_type='cpu',
            disable_level='L0', do_smooth=False, open_outlier=True, anti_method='')
        path.write_text(json.dumps(config))
        before = {item.name: item.read_bytes() for item in self.folder.iterdir()}
        with self.assertRaisesRegex(ValueError, 'INLINE_MODELSLIM_DESCRIPTION'):
            probe.model_identity(self.folder)
        self.assertEqual(before, {item.name: item.read_bytes() for item in self.folder.iterdir()})

    def test_inline_description_full_dictionary_missing_extra_and_nested_changes_refused(self):
        for mutation in ('missing_layer', 'extra_option', 'nested_value', 'nested_type', 'type_changed'):
            config = json.loads(self.values['config.json'])
            inline = config['quantization_config']
            if mutation == 'missing_layer':
                del inline['model.layers.0.self_attn.q_proj.weight']
            elif mutation == 'extra_option':
                inline['w_bit'] = 8
            elif mutation == 'nested_value':
                inline['test_nested_metadata']['retained'][1]['same'] = False
            elif mutation == 'nested_type':
                inline['test_nested_metadata']['retained'][1]['same'] = 1  # bool/int equality must not mask this change.
            else:
                config['quantization_config'] = list(inline)
            (self.folder / 'config.json').write_text(json.dumps(config))
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'INLINE_MODELSLIM_DESCRIPTION'):
                probe.model_identity(self.folder)

    def test_equal_but_missing_embedding_or_wrong_quant_type_description_refused(self):
        for description in ({'model_quant_type': 'W8A8_DYNAMIC'},
                            {'model_quant_type': 'W8A8', 'model.embed_tokens.weight': 'FLOAT'}):
            config = json.loads(self.values['config.json'])
            config['quantization_config'] = description
            (self.folder / 'config.json').write_text(json.dumps(config))
            (self.folder / 'quant_model_description_w8a8_dynamic.json').write_text(json.dumps(description))
            with self.assertRaisesRegex(ValueError, 'ACTUAL_ORDINARY_QWEN_DYNAMIC'):
                probe.model_identity(self.folder)

    def test_installed_source_must_equal_frozen_source(self):
        source, installed = self.folder / 'source', self.folder / 'installed'
        name = 'demoncat.runtime.example'
        original = probe.source_path(source, name)
        target = installed / 'demoncat/runtime/example.py'
        original.parent.mkdir(parents=True)
        target.parent.mkdir(parents=True)
        original.write_bytes(b'x=1\n')
        target.write_bytes(b'x=1\n')
        with patch.object(probe, 'MODULES', (name,)):
            self.assertEqual(probe.installed_source_pins(source, installed)[0]['sha256'], probe.sha_file(original))
            target.write_bytes(b'x=2\n')
            with self.assertRaisesRegex(ValueError, 'FRESH_INSTALLED_SOURCE'):
                probe.installed_source_pins(source, installed)

    def test_phase_original_exclusive_and_bounded(self):
        path = self.folder / 'original.json'
        result = probe.write_original(path, {'complete': False, 'reason': 'PRIMARY'})
        self.assertEqual(result['sha256'], probe.sha_file(path))
        with self.assertRaises(FileExistsError):
            probe.write_original(path, {'complete': True})
        with self.assertRaisesRegex(RuntimeError, 'BOUNDED_ACCEPTANCE_ORIGINAL'):
            probe.write_original(self.folder / 'too-large.json', 'x' * (8 * 1024 * 1024))
        self.assertFalse((self.folder / 'too-large.json').exists())

    @staticmethod
    def base():
        return dict(id='old', image=prepare.IMAGE, timeout=1800,
            mounts=[['/installed', '/installed', 'ro'], [prepare.MODEL, '/model', 'ro'],
                    ['/source/probe.py', '/probe.py', 'ro'], ['/bundle', '/quant', 'ro']],
            env={'TASK_QUEUE_ENABLE': '0', 'DCAT_DQ_GROUP_V1_ALLOWED': '1'},
            command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
                'export ASCEND_CUSTOM_OPP_PATH=/quant/runtime/vendor:/original/vendor\n'
                'export LD_PRELOAD=/original/libcust_opapi.so\n'
                'export PYTHONPATH=/installed:/msprobe_packages:$PYTHONPATH\n'
                'exec python3 /probe.py --build-sha256 ' + 'b'*64 + ' --group-abi',
            review=dict(experiment_id='old', provider='old', model=prepare.MODEL, cann='8.5.1', soc='910B4',
                physical_cards=[5], ranks=[0], worker='old', build_fingerprint='b'*64, expected_hits=0,
                resource_budget=dict(timeout_seconds=1800, memory_gib=80, cpus=16,
                    native_metadata_records=8192, native_metadata_static_bytes=777), recovery_plan='old'))

    def adapted(self, base=None, **updates):
        args = dict(source_root='/source-new', source_commit='c'*40, source_sha='d'*64,
                    model_sha='e'*64, build_sha='b'*64)
        args.update(updates)
        return prepare.adapt(self.base() if base is None else base, **args)

    def test_case_keeps_loading_order_no_new_build_and_exact_queue_bounds(self):
        base = self.base()
        saved = copy.deepcopy(base)
        case = self.adapted(base)
        self.assertEqual(base, saved)
        self.assertEqual(case['timeout'], 900)
        self.assertEqual(case['image'], prepare.IMAGE)
        budget = case['review']['resource_budget']
        self.assertEqual((budget['protocol_batches'], budget['health_batches']), (6, 2))
        self.assertEqual((budget['memory_gib'], budget['cpus']), (32, 8))
        self.assertEqual(budget['native_metadata_static_bytes'], 777)
        self.assertEqual(case['review']['expected_hits'], 1)
        self.assertFalse(case['review']['main_project_integration_verified'])
        self.assertIn('LD_PRELOAD=/original/libcust_opapi.so', case['command'])
        self.assertIn('PYTHONPATH=/installed:/source:$PYTHONPATH', case['command'])
        self.assertNotIn('--group-abi', case['command'])
        self.assertIn(['/source-new', '/source', 'ro'], case['mounts'])

    def test_case_rejects_mutable_image_invalid_sha_missing_startup_and_writable_mount(self):
        for mutation in ('image', 'command', 'mount'):
            base = self.base()
            if mutation == 'image':
                base['image'] = 'quay.io/ascend/vllm-ascend:latest'
            elif mutation == 'command':
                base['command'] = 'exec python3 /probe.py'
            else:
                base['mounts'][0][2] = 'rw'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.adapted(base)
        with self.assertRaisesRegex(ValueError, 'EXACT_REVIEWED'):
            self.adapted(build_sha='bad')


class BatchSchedulingContracts(unittest.TestCase):
    def setUp(self):
        self.events, self.saves, self.record = [], [], {}
        self.ids = ['actual-%d' % index for index in range(8)]
        self.mapping = [dict(internal_request_id=value, external_request_id='external-'+value) for value in self.ids]
        self.engine = NS(has_unfinished_requests=lambda: bool(self.events), output_processor=object(),
                         step=lambda: self.events.append('actual_prefill') or [])
        self.llm = NS(llm_engine=self.engine,
            reset_prefix_cache=lambda **kwargs: self.events.append('reset') or True,
            enqueue=lambda *args, **kwargs: self.events.append('enqueue') or self.ids,
            wait_for_completion=lambda **kwargs: self.events.append('actual_decode') or
                [NS(request_id=row['external_request_id'], outputs=[NS(token_ids=[1, 2])]) for row in self.mapping])
        def action(name, *args):
            self.events.append(name)
            return dict(status='QUERY_OK' if name == 'query' else 'ACTION_COMPLETED',
                response=dict(recovery_required=False, provider_state=dict(serving={'next_phase': 'normal'}, native={})))
        self.client = NS(advance_batch=lambda ids: action('advance', ids), begin_decode_capture=lambda: action('capture'),
                         collect_replay=lambda: action('collect'), query=lambda: action('query'))

    def run_batch(self):
        return probe.one_batch(self.llm, self.client, 'normal', [], object(), seen=set(), record=self.record,
            save=lambda: self.saves.append(copy.deepcopy(self.record)), map_requests=lambda _: self.mapping)

    def test_real_enqueue_prefill_capture_wait_collect_query_order(self):
        self.run_batch()
        self.assertEqual(self.events, ['reset', 'enqueue', 'advance', 'actual_prefill', 'capture', 'actual_decode', 'collect', 'query'])
        self.assertFalse(self.saves[0]['completed'])
        self.assertIsNone(self.saves[0]['mapping'])
        self.assertEqual(self.saves[1]['mapping'], self.mapping)
        self.assertTrue(self.record['completed'])

    def test_invalid_actual_mapping_is_saved_before_any_advance(self):
        self.mapping[0]['internal_request_id'] = 'foreign'
        with self.assertRaisesRegex(RuntimeError, 'ENQUEUED_B8_MAPPING_CHANGED'):
            self.run_batch()
        self.assertEqual(self.events, ['reset', 'enqueue'])
        self.assertEqual(self.saves[-1]['mapping'][0]['internal_request_id'], 'foreign')
        self.assertFalse(self.record['completed'])

    def test_duplicate_external_ids_and_extra_mapping_refused(self):
        self.mapping[0]['external_request_id'] = self.mapping[1]['external_request_id']
        with self.assertRaisesRegex(RuntimeError, 'ENQUEUED_B8_MAPPING_CHANGED'):
            self.run_batch()
        self.assertNotIn('advance', self.events)

    def test_failure_after_decode_preserves_attempt_without_collect_or_repeat(self):
        def failed_wait(**kwargs):
            self.events.append('actual_decode')
            raise RuntimeError('ACTUAL_WAIT_FAILED')
        self.llm.wait_for_completion = failed_wait
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_WAIT_FAILED'):
            self.run_batch()
        self.assertEqual(self.events.count('enqueue'), 1)
        self.assertEqual(self.events.count('actual_decode'), 1)
        self.assertNotIn('collect', self.events)
        self.assertFalse(self.record['completed'])


class ActualPublicRuntimeHarness(unittest.TestCase):
    """Run the complete driver over real CPU hooks/Host/Runtime, not real vLLM/CANN."""
    def setUp(self):
        import torch
        from test_quant_projection_runtime import OrdinaryQuantTests
        from demoncat.runtime.vllm_ascend.worker_extension import serving_context
        fixture = OrdinaryQuantTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        old = fixture.host_client()  # Prepare actual Core/factory, then release its unrelated fixture owner.
        for action in ('recover', 'unregister', 'detach'):
            reply = getattr(old, action)()
            self.assertEqual(reply['status'], 'DETACHED' if action == 'detach' else 'ACTION_COMPLETED', reply)
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.source, self.evidence = self.folder / 'source', self.folder / 'evidence'
        self.source.mkdir()
        self.evidence.mkdir()
        (self.source / 'source_manifest.json').write_text(json.dumps({'base_commit': 'a'*40}))
        self.model_identity = dict(manifest={'explicit_cpu_fixture': True}, sha256='m'*64)
        self.enterContext(patch.object(probe, 'SOURCE', self.source))
        self.enterContext(patch.object(probe, 'EVIDENCE', self.evidence))
        self.enterContext(patch.object(probe, 'installed_source_pins', return_value=[]))
        self.enterContext(patch.object(probe, 'model_identity', return_value=self.model_identity))
        self.enterContext(patch.object(serving_context, 'require_host_driven_core', return_value=True))
        self.removals, self.generations, self.scheduled = [], [], []
        self.fail_after_fault = self.fail_drain = False
        owner = self

        class CpuEngine:
            def __init__(self):
                self.pending = False
                self.output_processor = NS(request_states={})
            def has_unfinished_requests(self):
                return self.pending
            def step(self):
                owner.scheduled.append('prefill')
                fixture.runner.with_prefill = True
                with torch.inference_mode():
                    hidden = fixture.model(input_ids=torch.tensor([1, 2]*8), positions=torch.arange(16))
                    fixture.model.processor(fixture.model.head, hidden[-8:])
                for value in fixture.runner.requests.values():
                    value.output_token_ids = [3]
                return []

        class CpuLLM:
            def __init__(self):
                self.llm_engine = CpuEngine()
                self.count = 0
            def collective_rpc(self, method, *, timeout, args=()):
                if callable(method):
                    return [method(fixture.worker, *args)]
                return fixture.executor.collective_rpc(method, timeout=timeout, args=args)
            def get_tokenizer(self):
                return NS(encode=lambda text, **kwargs: [1, 2])
            def generate(self, prompts, params, **kwargs):
                owner.generations.append(len(prompts))
                return [NS(outputs=[NS(token_ids=[3, 4])]) for _ in prompts]
            def reset_prefix_cache(self, **kwargs):
                return not self.llm_engine.pending
            def enqueue(self, prompts, params, **kwargs):
                self.count += 1
                ids = ['harness-%d-%d' % (self.count, index) for index in range(8)]
                owner.scheduled.append(('enqueue', fixture.session.phase_name()))
                self.llm_engine.pending = True
                self.llm_engine.output_processor.request_states = {value: NS(request_id=value, external_req_id='ext-'+value) for value in ids}
                fixture.runner.requests = {value: NS(prompt_token_ids=[1, 2], output_token_ids=[]) for value in ids}
                fixture.runner.input_batch = NS(req_ids=ids)
                return ids
            def wait_for_completion(self, **kwargs):
                owner.scheduled.append('decode')
                fixture.runner.with_prefill = False
                with torch.inference_mode():
                    hidden = fixture.model(input_ids=torch.tensor([3]*8), positions=torch.tensor([2]*8))
                    fixture.model.processor(fixture.model.head, hidden)
                self.llm_engine.pending = False
                outputs = [NS(request_id=value.external_req_id, outputs=[NS(token_ids=[3, 4])])
                           for value in self.llm_engine.output_processor.request_states.values()]
                self.llm_engine.output_processor.request_states.clear()
                if owner.fail_after_fault and fixture.session.phase_name() == 'fault':
                    if owner.fail_drain:
                        fixture.binding.drain_result = False
                    raise RuntimeError('PRIMARY_AFTER_ACTUAL_FAULT')
                return outputs

        self.llm = CpuLLM()
        self.enterContext(patch.object(sys.modules['vllm'], 'LLM', create=True, return_value=self.llm))
        self.enterContext(patch.object(sys.modules['vllm'], 'SamplingParams', create=True, side_effect=lambda **kwargs: NS(**kwargs)))
        def setup(worker, identity, build, pins):
            self.assertIs(worker, fixture.worker)
            return {'worker': worker.demoncat_worker_identity(), 'binding': {'status': 'EXPLICIT_CPU_DEPLOYMENT_DOUBLE'}}
        def remove(worker):
            self.assertIs(worker, fixture.worker)
            self.assertFalse(fixture.session.installed)
            self.removals.append(True)
            return {'status': 'REMOVED'}
        self.enterContext(patch.object(probe, 'setup', setup))
        self.enterContext(patch.object(probe, 'remove', remove))

    def run_probe(self):
        code = probe.main(['--build-sha256', 'b'*64, '--source-commit', 'a'*40,
            '--source-manifest-sha256', probe.sha_file(self.source / 'source_manifest.json'), '--model-sha256', 'm'*64])
        return code, json.loads((self.evidence / 'result.json').read_bytes()), json.loads((self.evidence / 'host-core-receipts.json').read_bytes())

    def test_six_batches_public_host_runtime_and_two_health_calls_cleanly_release(self):
        code, result, host = self.run_probe()
        self.assertEqual(code, 0, result)
        self.assertEqual([row['phase'] for row in result['phases']], list(probe.PHASES))
        self.assertEqual(self.generations, [8, 8])
        self.assertEqual(self.scheduled.count('prefill'), 6)
        self.assertEqual(self.scheduled.count('decode'), 6)
        self.assertEqual(self.fixture.binding.selected_modes, [1, 2, 4, 3])
        self.assertFalse(host['ownership_retained'])
        self.assertEqual(self.removals, [True])
        self.assertFalse(result['qualification_granted'] or result['c1'] or result['c2'] or result['c3'])
        self.assertFalse(result['main_project_integration_verified'])
        for row in result['phases']:
            self.assertEqual(probe.sha_file(row['original']['path']), row['original']['sha256'])

    def test_after_fault_error_retained_then_actual_off_and_owner_release(self):
        self.fail_after_fault = True
        code, result, host = self.run_probe()
        self.assertEqual(code, 2)
        self.assertIn('PRIMARY_AFTER_ACTUAL_FAULT', result['error'])
        self.assertTrue(result['cleanup_off']['completed'])
        self.assertFalse(host['ownership_retained'])
        self.assertEqual(self.fixture.binding.selected_modes, [1, 2, 4, 3])
        self.assertEqual(self.scheduled.count('decode'), 6)
        self.assertEqual(self.removals, [True])
        self.assertEqual(probe.sha_file(result['failure_readback']['path']), result['failure_readback']['sha256'])
        self.assertFalse(result['phases'][-1]['completed'])

    def test_failed_native_drain_retains_owner_and_never_removes_or_repeats_fault(self):
        self.fail_after_fault = self.fail_drain = True
        code, result, host = self.run_probe()
        self.assertEqual(code, 2)
        self.assertIn('PRIMARY_AFTER_ACTUAL_FAULT', result['error'])
        self.assertTrue(host['ownership_retained'])
        self.assertNotIn('cleanup_off', result)
        self.assertEqual(self.removals, [])
        self.assertEqual(self.fixture.binding.selected_modes, [1, 2, 4, 3])
        self.assertTrue(self.fixture.session.installed)
        self.assertIn('OWNERSHIP_RETAINED_OR_UNVERIFIED', [row.get('status') for row in result['cleanup']])


if __name__ == '__main__':
    unittest.main()
