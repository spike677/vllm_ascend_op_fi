"""CPU metadata/ABI doubles; no actual CANN, NPU, model or row qualification."""
import copy
import ctypes as C
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fault_injection import quant_gmm2_native as native
from fault_injection import quant_container_native as base


IDENTITY = dict(pid=9, thread=11, process_start_identity='proc:cpu-fixture:9:123')


def tensor(descriptor=500):
    row = base.record(base.Tensor())
    row.update(descriptor=descriptor, address=descriptor+1000, counts=[1, 1, 1], called_mask=127,
               shape=[8]+[0]*7, strides=[1]+[0]*7, storage=[8]+[0]*7, complete=1)
    return row


def fixture(count=1):
    """Fresh registrations reuse the same handle/executor addresses each scope."""
    state = base.record(base.instance(base.State))
    state.update(installed=1, stopped=1, owner_hi=7, owner_lo=8, generation=1, owner_thread=11)
    state['arrays'].update(acquired_arrays=9, released_arrays=9)
    raw, scopes, events = [], [], []
    def scope(token, stage):
        row = dict(native_scope=dict(token=token, forward=1, operator_sequence=token*10,
            boundary_sequence=token*9, stage=stage, thread=11), status='RETURNED', original_calls=1, original_error=None)
        scopes.append(row); return row['native_scope']
    def original(operation, sc, handle, registration=0, members=1):
        call = len(raw)//2 + 1
        row = base.record(base.instance(base.Event))
        row.update(operation=operation, call=call, pid=9, thread=11, caller_kind=2 if sc['stage']==1 else 1,
            function=1000+operation, caller=2000+sc['stage'], scope_token=sc['token'],
            forward=sc['forward'], operator_sequence=sc['operator_sequence'], boundary_sequence=sc['boundary_sequence'],
            stage=sc['stage'], count=members if operation in (1, 3) else 0)
        if operation in (1, 3):
            registration = call
            row.update(payload_offset=len(raw)*500, payload_bytes=members*(C.sizeof(base.Tensor) if operation==1 else 8))
            if operation == 1:
                row['tensor_descriptors'] = [tensor(handle+10+i) for i in range(members)]
            else:
                row['integer_values'] = [1]*members
        for phase in (1, 2):
            event = copy.deepcopy(row)
            event.update(phase=phase, sequence=len(raw)+1, original_calls=phase-1,
                handle=handle if phase==2 or operation in (2, 4) else 0,
                registration=registration if phase==2 or operation in (2, 4) else 0)
            raw.append(event)
        state['entries'][operation-1] += 1; state['returns'][operation-1] += 1; state['original_calls'][operation-1] += 1
        return raw[-2], raw[-1]
    sc = scope(1, 1)
    _, old = original(3, sc, 91); original(4, sc, 91, old['registration'])
    scope(2, 2)
    owners = {key: dict(address=100000000+offset, base=100000000, offset=offset, symbol=symbol)
              for key, (_, symbol, offset) in native.FUNCTIONS.items()}
    for number in range(count):
        sc = scope(3+number, 3)
        slots = []
        for index in native.CONTAINERS:
            kind = 3 if index == 16 else 1; handle = 100+index
            first, last = original(kind, sc, handle, members=0 if index in (2, 16) else 1)
            slot = base.record(native.Slot())
            slot.update(formal_index=index, kind=kind, handle=handle, registration=last['registration'],
                creator_entry=first['sequence'], creator_return=last['sequence'], count=first['count'],
                payload_offset=first['payload_offset'], payload_bytes=first['payload_bytes'],
                actual_count=first['count'], joined=1)
            slots.append(slot)
        prepare = base.record(base.instance(native.Event))
        call = len(events)//2 + 1
        prepare.update(operation=1, call=call, prepare_registration=call, owner_hi=7, owner_lo=8, generation=1,
            scope_token=sc['token'], forward=1, operator_sequence=sc['operator_sequence'], boundary_sequence=sc['boundary_sequence'],
            stage=3, pid=9, thread=11, device=0, function=owners['prepare_function']['address'], function_base=100000000,
            function_offset=owners['prepare_function']['offset'], caller_base=200000000,
            caller=200000000+native.CALLER_OFFSETS[1], caller_offset=native.CALLER_OFFSETS[1],
            container_events=len(raw), containers=slots, group_list=tensor(800), scalars=[-1, 0, 1, 0, -11])
        for slot in slots:
            prepare['formals'][slot['formal_index']] = slot['handle']
        prepare['formals'][8] = 800; prepare['formals'][21:] = [3001, 3002]
        for n, index in enumerate((12, 13, 14, 15, 17)):
            prepare['formals'][index] = prepare['scalars'][n] & ((1 << 64)-1)
        for phase in (1, 2):
            event = copy.deepcopy(prepare)
            event.update(sequence=len(events)+1, phase=phase, monotonic_ns=len(events)+1, original_calls=phase-1)
            if phase == 2:
                event.update(workspace_bytes=16, executor=900)
                for slot in event['containers']:
                    slot['live_at_return'] = 1
            events.append(event)
        execute = base.record(base.instance(native.Event))
        execute.update({key: prepare[key] for key in ('owner_hi', 'owner_lo', 'generation', 'scope_token',
            'forward', 'operator_sequence', 'boundary_sequence', 'stage', 'pid', 'thread', 'device', 'container_events')})
        execute.update(operation=2, call=call+1, prepare_registration=call,
            function=owners['execute_function']['address'], function_base=100000000,
            function_offset=owners['execute_function']['offset'], caller_base=200000000,
            caller=200000000+native.CALLER_OFFSETS[2], caller_offset=native.CALLER_OFFSETS[2],
            workspace=4000, workspace_bytes=32, executor=900, stream=7000)
        for phase in (1, 2):
            event = copy.deepcopy(execute)
            event.update(sequence=len(events)+1, phase=phase, monotonic_ns=len(events)+1, original_calls=phase-1)
            events.append(event)
        for slot in slots:
            original(slot['kind']+1, sc, slot['handle'], slot['registration'])
    scope(count+3, 4)
    state.update(total_calls=len(raw)//2, total_events=len(raw), retained_events=len(raw))
    gstate = base.record(base.instance(native.State))
    gstate.update(installed=1, stopped=1, resolver_ready=1, owner_hi=7, owner_lo=8, generation=1, thread=11,
        total_calls=len(events)//2, total_events=len(events), retained_events=len(events))
    for key in ('prepares', 'prepare_returns', 'executes', 'execute_returns', 'original_prepares',
                'original_executes', 'paired_execute_returns'):
        gstate[key] = count
    gstate.update({key: row['address'] for key, row in owners.items()})
    worker = dict(IDENTITY, rank=0, logical_device=0, configured_logical_device=0, observed_logical_device=0)
    return dict(schema='demoncat.quant.container_trace.v1', state=state, events=raw, public_scopes=scopes,
        gmm2=dict(schema='demoncat.quant.gmm2_trace.v1', state=gstate, events=events, failures=[], formal_names=list(native.FORMALS),
            startup=dict(status='RESOLVED_BEFORE_LLM', identity=copy.deepcopy(IDENTITY), original_functions=owners),
            worker_identity=dict(status='WORKER_IDENTIFIED', worker=worker)))


class RawAssessmentTests(unittest.TestCase):
    def test_real_shape_cpu_fixture_rejoins_every_actual_slot_without_row_grant(self):
        result = native.assess_observation(fixture(), expected_pairs=1)
        self.assertEqual(result['reasons'], [])
        self.assertEqual(result['joined_nonnull_container_slots'], 15)
        for name in ('qualified', 'request_rows_proven', 'c1', 'c2', 'c3', 'device_drain_proven'):
            self.assertFalse(result[name])

    def test_full_twenty_six_and_reused_addresses_require_fresh_registration(self):
        value = fixture(26)
        self.assertEqual(native.assess_observation(value)['reasons'], [])
        value['gmm2']['events'][6]['prepare_registration'] = 1
        self.assertIn('TYPED_RETURN_IDENTITY_CHANGED', native.assess_observation(value)['reasons'])

    def test_missing_extra_or_duplicate_native_calls_cannot_hide(self):
        for mutation in ('missing', 'duplicate', 'invented_count'):
            value = fixture()
            if mutation == 'missing': value['gmm2']['events'].pop()
            if mutation == 'duplicate': value['gmm2']['events'].append(copy.deepcopy(value['gmm2']['events'][-1]))
            if mutation == 'invented_count': value['gmm2']['state']['original_executes'] = 0
            self.assertIn('TYPED_INDEPENDENT_DENOMINATOR', native.assess_observation(value, expected_pairs=1)['reasons'])

    def test_old_registration_same_handle_does_not_prove_current_gws(self):
        value = fixture(2)
        for event in value['gmm2']['events'][4:6]:
            event['containers'][0] = copy.deepcopy(value['gmm2']['events'][event['phase']-1]['containers'][0])
        reasons = native.assess_observation(value, expected_pairs=2)['reasons']
        self.assertIn('CONTAINER_LIVE_INTERVAL_NOT_CURRENT_GWS', reasons)
        self.assertIn('CONTAINER_SOURCE_OR_PAYLOAD_CHANGED', reasons)

    def test_actual_parameter_order_cannot_be_reconstructed_from_creator_order(self):
        value = fixture()
        for event in value['gmm2']['events'][:2]:
            event['formals'][0], event['formals'][1] = event['formals'][1], event['formals'][0]
        self.assertIn('CURRENT_CONTAINER_SLOT_JOIN_FAILED', native.assess_observation(value, expected_pairs=1)['reasons'])

    def test_missing_optional_is_null_not_an_invented_empty_container(self):
        value = fixture()
        for event in value['gmm2']['events'][:2]:
            event['formals'][2] = 0; event['containers'][2]['handle'] = 0
        self.assertIn('NULL_CONTAINER_INVENTED_REGISTRATION', native.assess_observation(value, expected_pairs=1)['reasons'])

    def test_signed_quant_group_size_is_observed_not_filled_from_public_defaults(self):
        value = fixture()
        for event in value['gmm2']['events'][:2]: event['scalars'][4] = 0
        self.assertIn('ACTUAL_23_FORMALS_REQUIRED', native.assess_observation(value, expected_pairs=1)['reasons'])

    def test_gws_current_typed_group_list_cannot_be_public_backfilled(self):
        for key, replacement in (('descriptor', 12345), ('complete', 0), ('called_mask', 31)):
            value = fixture()
            for event in value['gmm2']['events'][:2]: event['group_list'][key] = replacement
            self.assertTrue(native.assess_observation(value, expected_pairs=1)['reasons'])

    def test_wrong_scope_owner_device_and_workspace_refuse(self):
        for key, changed in (('scope_token', 1), ('owner_hi', 9), ('device', 5), ('executor', 901), ('workspace_bytes', 0)):
            value = fixture()
            for event in value['gmm2']['events'][2:]: event[key] = changed
            self.assertTrue(native.assess_observation(value, expected_pairs=1)['reasons'], key)

    def test_pending_failed_or_overflowed_raw_observation_never_completes(self):
        for key in ('pending_executors', 'untracked_executors', 'overflow', 'first_error', 'abi_exceptions',
                    'outside_scope_calls', 'foreign_thread_calls', 'getter_failures', 'failed_with_executor'):
            value = fixture(); value['gmm2']['state'][key] = 1
            self.assertIn('TYPED_OBSERVATION_NOT_COMPLETE_AND_IDLE', native.assess_observation(value, expected_pairs=1)['reasons'])

    def test_actual_caller_and_original_pointer_are_both_bound(self):
        for key in ('caller', 'caller_offset', 'function', 'function_offset'):
            value = fixture(); value['gmm2']['events'][0][key] += 1
            self.assertIn('ACTUAL_TYPED_ORIGINAL_OR_REVIEWED_CALLER_CHANGED', native.assess_observation(value, expected_pairs=1)['reasons'])

    def test_originals_and_first_error_are_not_repaired_by_cleanup(self):
        value = fixture(); value['gmm2']['failures'] = [dict(action='first', error='pending')]
        self.assertIn('TYPED_OBSERVATION_NOT_COMPLETE_AND_IDLE', native.assess_observation(value, expected_pairs=1)['reasons'])
        value['gmm2']['failures'] = []; value['gmm2']['events'][-1]['original_calls'] = 2
        self.assertIn('EXACTLY_ONE_TYPED_ORIGINAL_REQUIRED', native.assess_observation(value, expected_pairs=1)['reasons'])


class Object:
    pass


class ApiDouble:
    def __init__(self):
        self.c = base.instance(base.State); self.g = base.instance(native.State)
        self.calls = []; self.fail = {}
        for key, (_, _, offset) in native.FUNCTIONS.items(): setattr(self.g, key, 100000000+offset)
    def method(self, journal, name):
        def call(*args):
            self.calls.append((journal, name))
            if (journal, name) in self.fail: return self.fail[journal, name]
            state = self.c if journal == 'containers' else self.g
            if name == 'state': C.memmove(args[0], C.byref(state), C.sizeof(state))
            if name == 'resolve': self.g.resolver_ready = 1
            if name == 'begin': state.installed = 1
            if name == 'stop': state.stopped = 1
            if name == 'release': state.installed = 0; state.retired = 1
            return 0
        return call
    def binding(self):
        result = native.NativeGmm2.__new__(native.NativeGmm2)
        result.functions = {name: self.method('containers', name) for name in ('state', 'begin', 'stop', 'release')}
        result.gfunctions = {name: self.method('gmm2', name) for name in ('state', 'resolve', 'begin', 'stop', 'release')}
        result.current = Mock(); result.config = base.instance(base.Config)
        result.config.owner_hi, result.config.owner_lo, result.config.generation, result.config.thread = 7, 8, 1, 11
        result.limits = base.instance(native.Limits); result.started = False; result.startup = None
        result.worker_identity = result.worker_refs = None; result.failures = []
        result.pid, result.thread = 9, 11; result.scopes = []; result.active_scope = None
        result.profile = dict(images={name: {} for name in ('opapi', 'nn', 'device_runtime')})
        result.manifest_sha256 = 'a'*64; result.build = dict(profile_sha256='b'*64, source_commit='c'*40)
        result.mapping = dict(path='CPU_FUNCTION_DOUBLE')
        return result


class BindingContracts(unittest.TestCase):
    def setUp(self):
        self.api = ApiDouble(); self.binding = self.api.binding()
        self.addCleanup(patch.stopall)
        patch.object(native, 'process_identity', return_value=copy.deepcopy(IDENTITY)).start()
        patch.dict(native.os.environ, VLLM_ENABLE_V1_MULTIPROCESSING='0', TASK_QUEUE_ENABLE='0').start()
        patch.object(native, 'function_owner', side_effect=lambda address, expected, **kwargs: dict(address=address, **kwargs)).start()
    def worker(self):
        worker = Object(); worker.model_runner = Object(); worker.model_runner.model = Object()
        identity = dict(status='WORKER_IDENTIFIED', quarantined=False, worker=dict(IDENTITY,
            rank=0, logical_device=0, configured_logical_device=0, observed_logical_device=0,
            worker_start_identity='current-worker', model_instance_id='current-model'))
        worker.demoncat_worker_identity = Mock(return_value=identity)
        return worker
    def bind(self):
        self.binding.preflight(); worker = self.worker(); self.binding.bind_worker(worker); return worker

    def test_preflight_resolves_only_then_requires_same_actual_worker(self):
        startup = self.binding.preflight()
        self.assertEqual(startup['status'], 'RESOLVED_BEFORE_LLM')
        self.assertEqual([x for x in self.api.calls if x[1] not in ('state',)], [('gmm2', 'resolve')])
        self.assertEqual(startup['explicit_device_query_calls'], 0)
        worker = self.worker(); self.binding.bind_worker(worker); self.binding.verify_worker(worker)
        startup['identity']['pid'] = 77
        self.assertEqual(self.binding.startup['identity']['pid'], 9)

    def test_unresolved_missing_startup_or_multiprocess_scope_refuses_before_install(self):
        with patch.dict(native.os.environ, VLLM_ENABLE_V1_MULTIPROCESSING='1'):
            with self.assertRaisesRegex(RuntimeError, 'SAME_PROCESS_SYNCHRONOUS'): self.binding.preflight()
        self.api.fail['gmm2', 'resolve'] = 1
        with self.assertRaisesRegex(RuntimeError, 'RESOLVE_FAILED'): self.binding.preflight()
        self.assertNotIn(('gmm2', 'begin'), self.api.calls)
        self.assertIsNone(self.binding.startup)

    def test_every_required_original_getter_and_device_pointer_must_be_resolved(self):
        def owner(address, expected, **kwargs):
            native.require(address > 0, 'RESOLVED_POINTER_MISSING:' + kwargs['symbol'])
            return dict(address=address, **kwargs)
        for key in native.FUNCTIONS:
            with self.subTest(key=key):
                api = ApiDouble(); binding = api.binding(); setattr(api.g, key, 0)
                with patch.object(native, 'function_owner', side_effect=owner):
                    with self.assertRaisesRegex(RuntimeError, 'RESOLVED_POINTER_MISSING'): binding.preflight()
                self.assertIsNone(binding.startup)
                self.assertFalse(any(name == 'begin' for _, name in api.calls))

    def test_foreign_resolved_owner_never_leaves_successful_preflight(self):
        with patch.object(native, 'function_owner', side_effect=RuntimeError('FOREIGN_REAL_SYMBOL_OWNER')):
            with self.assertRaisesRegex(RuntimeError, 'FOREIGN_REAL_SYMBOL_OWNER'): self.binding.preflight()
        self.assertIsNone(self.binding.startup)
        self.assertFalse(any(name == 'begin' for _, name in self.api.calls))

    def test_worker_process_start_rank_device_or_readiness_mismatch_refuses(self):
        self.binding.preflight()
        for key, value in (('pid', 10), ('process_start_identity', 'proc:other:9:123'), ('rank', 1),
                           ('observed_logical_device', 5), ('model_instance_id', None)):
            worker = self.worker(); worker.demoncat_worker_identity.return_value['worker'][key] = value
            with self.assertRaisesRegex(RuntimeError, 'ACTUAL_SAME_PROCESS_WORKER_REQUIRED'):
                self.binding.bind_worker(worker)
        self.assertFalse(any(name == 'begin' for _, name in self.api.calls))

    def test_thread_change_or_new_model_object_does_not_reuse_preflight(self):
        worker = self.bind()
        with patch.object(native, 'process_identity', return_value=dict(IDENTITY, thread=12)):
            with self.assertRaisesRegex(RuntimeError, 'PREFLIGHT_PROCESS_OR_THREAD_CHANGED'): self.binding.verify_worker(worker)
        worker.model_runner.model = Object()
        with self.assertRaisesRegex(RuntimeError, 'HELD_WORKER_RUNNER_OR_MODEL_CHANGED'): self.binding.verify_worker(worker)

    def test_pure_query_does_not_collect_resolve_or_install(self):
        before = self.binding.query_all(); calls = len(self.api.calls)
        self.assertEqual(self.binding.query_all(), before)
        self.assertEqual(self.api.calls[calls:], [('containers', 'state'), ('gmm2', 'state')])

    def test_partial_second_begin_failure_keeps_first_journal_for_cleanup(self):
        self.bind(); self.api.fail['gmm2', 'begin'] = 3
        with self.assertRaisesRegex(RuntimeError, 'BEGIN_FAILED:3'): self.binding.begin()
        self.assertTrue(self.binding.started); self.assertEqual(self.api.c.installed, 1)
        self.assertEqual(self.binding.failures[0]['action'], 'begin_gmm2')
        self.binding.stop(); self.binding.release()
        self.assertEqual(self.api.c.installed, 0)
        self.assertEqual(len(self.binding.failures), 1)

    def test_failed_gmm2_release_retains_creator_journal_and_retry_keeps_order(self):
        self.bind(); self.binding.begin(); self.binding.stop()
        self.api.fail['gmm2', 'release'] = 1
        with self.assertRaisesRegex(RuntimeError, 'RELEASE_FAILED:1'): self.binding.release()
        self.assertNotIn(('containers', 'release'), self.api.calls)
        self.assertEqual(self.api.c.installed, 1)
        del self.api.fail['gmm2', 'release']; self.binding.release()
        releases = [item for item in self.api.calls if item[1] == 'release']
        self.assertEqual(releases, [('gmm2', 'release'), ('gmm2', 'release'), ('containers', 'release')])
        self.assertIsNone(self.binding.worker_refs)

    def test_partial_release_retry_does_not_release_gmm2_twice(self):
        self.bind(); self.binding.begin(); self.binding.stop(); self.api.fail['containers', 'release'] = 1
        with self.assertRaisesRegex(RuntimeError, 'RELEASE_FAILED'): self.binding.release()
        self.assertEqual(self.api.g.installed, 0)
        del self.api.fail['containers', 'release']; self.binding.release()
        self.assertEqual(self.api.calls.count(('gmm2', 'release')), 1)

    def test_preflight_output_and_raw_nested_arrays_cannot_mutate_binding(self):
        startup = self.binding.preflight(); startup['state_after']['gmm2']['total_calls'] = 99
        self.assertEqual(self.binding.startup['state_after']['gmm2']['total_calls'], 0)
        event = base.instance(native.Event); event.containers[0].registration = 123
        snapshot = base.record(event); snapshot['containers'][0]['registration'] = 0
        self.assertEqual(event.containers[0].registration, 123)

    def test_returned_snapshot_cannot_relabel_held_mapped_image(self):
        value = self.binding.snapshot()
        value['mapping']['path'] = 'foreign'
        self.assertEqual(self.binding.mapping['path'], 'CPU_FUNCTION_DOUBLE')

    def test_held_preflight_cannot_be_claimed_in_another_module_process(self):
        with patch.object(native, '_startup_trace', None):
            with self.assertRaisesRegex(RuntimeError, 'SAME_PROCESS_PRE_LLM_HANDLE_MISSING'): native.startup_trace()
        with patch.object(native, '_startup_trace', self.binding):
            self.api.g.pending_executors = 1
            with self.assertRaisesRegex(RuntimeError, 'RETAINED_DIAGNOSTIC_OWNER'): native.discard_startup(self.binding)
            self.assertIs(native.startup_trace(), self.binding)
            self.api.g.pending_executors = 0; native.discard_startup(self.binding)
            with self.assertRaisesRegex(RuntimeError, 'SAME_PROCESS_PRE_LLM_HANDLE_MISSING'): native.startup_trace()

    def test_idle_preflight_still_retains_any_live_sdk_getter_array(self):
        with patch.object(native, '_startup_trace', self.binding):
            self.api.c.arrays.live_arrays = 1
            with self.assertRaisesRegex(RuntimeError, 'RETAINED_DIAGNOSTIC_OWNER'): native.discard_startup(self.binding)
            self.assertIs(native.startup_trace(), self.binding)


class OwnerAndLayoutTests(unittest.TestCase):
    def test_typed_layouts_have_all_fields_and_stay_inside_byte_budget(self):
        self.assertEqual(C.sizeof(base.Tensor), 304)
        self.assertEqual(C.sizeof(native.Slot), 96)
        self.assertEqual(C.sizeof(native.Event), 2216)
        self.assertEqual(C.sizeof(native.State), 288)
        self.assertLess(C.sizeof(native.Event)*256, native.EXTENSION['max_bytes'])
        for cls in (native.Limits, native.Event, native.State, native.Sizes):
            self.assertEqual(base.instance(cls).size, C.sizeof(cls))

    def test_wrong_new_build_refuses_without_loading_any_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'build.json').write_text('{}')
            with patch.object(native.C, 'CDLL') as load:
                with self.assertRaisesRegex(RuntimeError, 'BUILD_SHA_CHANGED'): native.NativeGmm2(root, '0'*64)
                load.assert_not_called()

    def test_actual_pointer_owner_needs_full_file_and_exact_symbol_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); library = root/'original.so'; library.write_bytes(b'CPU ELF owner bytes')
            expected = dict(path=str(library), file_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                file_bytes=library.stat().st_size, segments=[dict(vaddr=200, size=200)])
            address, start, name = 1300, 1000, b'actual_symbol'
            def locate(pointer, target):
                info = C.cast(target, C.POINTER(base._DlInfo)).contents
                info.dli_fname = str(library).encode(); info.dli_fbase = start
                info.dli_sname = name; info.dli_saddr = address
                return 1
            lookup = Mock(side_effect=locate)
            with patch.object(native.C, 'CDLL', return_value=Mock(dladdr=lookup)):
                result = native.function_owner(address, expected, symbol='actual_symbol', offset=300)
                self.assertEqual(result['offset'], 300)
                for change in ('offset', 'symbol', 'file'):
                    if change == 'file': library.write_bytes(b'replaced bytes')
                    with self.assertRaises(RuntimeError):
                        native.function_owner(address, expected, symbol='wrong' if change=='symbol' else 'actual_symbol',
                                              offset=301 if change=='offset' else 300)


if __name__ == '__main__':
    unittest.main()
