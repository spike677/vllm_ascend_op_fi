from types import SimpleNamespace
import unittest
import json
from pathlib import Path
import tempfile
from vllm_validate import _gate_lossless
from vllm_validate import WorkerRpc, RpcContractError, observe_request_ids, request_evidence, finish_group_capture


class FakeProcessor:
    @staticmethod
    def assign_request_id(request):
        request.external_req_id = request.request_id
        request.request_id = 'opaque-internal-id'


class RpcTests(unittest.TestCase):
    def test_lossless_receipt_is_bound_to_exact_native_build(self):
        report=dict(model='/model',tp=1,ranks=[0],environment={},targets={'0':'actual-signature'},native_build_profiles={'0':'a'*64})
        receipt=dict(report,status='LOSSLESS_VALIDATED',stage='lossless')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'lossless.json';path.write_text(json.dumps(receipt))
            _gate_lossless(path,dict(report))
            for profiles in ({'0':'b'*64},None):
                invalid=dict(receipt,native_build_profiles=profiles);path.write_text(json.dumps(invalid))
                with self.assertRaisesRegex(RuntimeError,'LOSSLESS_GATE_ENVIRONMENT_MISMATCH: native_build_profiles'):
                    _gate_lossless(path,dict(report))

    def test_unidentified_build_cannot_use_legacy_lossless_receipt(self):
        with self.assertRaisesRegex(RuntimeError,'NATIVE_BUILD_PROFILE_REQUIRED'):
            _gate_lossless('/not-read',dict(ranks=[0]))

    def test_observes_actual_mapping_without_suffix_assumption(self):
        processor = FakeProcessor()
        request = SimpleNamespace(request_id='client-17')
        with observe_request_ids(processor) as mappings:
            processor.assign_request_id(request)
        self.assertEqual(mappings, [{'external_req_id': 'client-17', 'request_id': 'opaque-internal-id'}])
        self.assertNotIn('assign_request_id', vars(processor))
    def test_mapping_wrapper_restores_on_failure(self):
        processor = FakeProcessor()
        with self.assertRaises(ValueError):
            with observe_request_ids(processor):
                raise ValueError('inference failure')
        self.assertNotIn('assign_request_id', vars(processor))
    def test_scheduler_mapping_requires_exact_recorded_id(self):
        after = {'contexts': [dict(index=0, completed=True, instrumentation_complete=True, instrumentation_errors=[], request_ids=['opaque'])]}
        self.assertEqual(request_evidence({}, after, 'client')['request_mapping'], 'UNPROVEN_REQUEST_MAPPING')
        self.assertEqual(request_evidence({}, after, 'opaque')['request_mapping'], 'SINGLE_REQUEST_SCHEDULER_MAPPING_VERIFIED')
    def test_tp2_dispatch_checks_unselected_worker(self):
        class Executor:
            def collective_rpc(self, method, *, timeout, args):
                self.last = (method, args)
                return [{'rank': 0, 'result': 'selected'}, {'rank': 1, 'status': 'RPC_NOT_TARGETED'}]
        executor = Executor()
        self.assertEqual(WorkerRpc(executor, 2).call('snapshot', rank=0)['rank'], 0)
        self.assertEqual(executor.last[1][1]['dispatch_rank'], 0)
    def test_two_selected_replies_are_rejected(self):
        executor = SimpleNamespace(collective_rpc=lambda *a, **k: [{'rank': 0}, {'rank': 1}])
        with self.assertRaises(RpcContractError):
            WorkerRpc(executor, 2).call('snapshot', rank=0)
    def test_finish_capture_attempts_every_rank_after_one_rpc_failure(self):
        calls = []
        class Rpc:
            def call(self, action, payload, rank):
                calls.append((action, payload, rank))
                if rank == 0:
                    raise RuntimeError('transport failed')
                return dict(status='CAPTURE_FINISHED', rank=rank, owner=payload['owner'], capture_active=False)
        providers = {rank: SimpleNamespace(rpc=Rpc()) for rank in (0, 1)}
        result = finish_group_capture(providers, 'actual-owner')
        self.assertEqual(result['status'], 'CAPTURE_FINISH_FAILED')
        self.assertEqual([item[2] for item in calls], [0, 1])
        self.assertEqual(result['workers'][1]['status'], 'CAPTURE_FINISHED')
    def test_finish_capture_requires_owned_inactive_receipt(self):
        provider = SimpleNamespace(rpc=SimpleNamespace(call=lambda *a, **k:
            dict(status='CAPTURE_FINISHED', rank=0, owner='different-owner', capture_active=False)))
        self.assertEqual(finish_group_capture({0: provider}, 'actual-owner')['status'], 'CAPTURE_FINISH_FAILED')


if __name__ == '__main__':
    unittest.main()
