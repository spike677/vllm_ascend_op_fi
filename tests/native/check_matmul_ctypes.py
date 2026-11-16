"""Run with the local host-memory fixture DSO already preloaded."""
import ctypes
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fault_injection.deployments.matmul import CtypesMatmulRuntime
from fault_injection.native import _STACK, _read_proofs
from fault_injection.qualification import _validate_receipt
from fault_injection.runtime_discovery import native_facts


os.environ['TASK_QUEUE_ENABLE'] = '0'
os.environ['VLLM_SERVER_DEV_MODE'] = '1'
mechanism = '20.buffer_substitute'


def verify(operator, symbol, formal):
    run = getattr(ctypes.CDLL(None), symbol)
    run.argtypes, run.restype = [], ctypes.c_int
    baseline = run()
    with tempfile.TemporaryDirectory() as directory:
        worker = dict(pid=os.getpid(), rank=0, logical_device=0, tp=1,
                      execution_mode='eager', soc='host_memory_fixture')
        runtime = CtypesMatmulRuntime(mechanism, worker, directory, operator=operator)
        assert all(not row['native_registered'] and row['runtime_matched']
                   for row in runtime.snapshot().values())
        module = object()
        receipts = []
        arms = iter([(1, 'reference'), (2, 'lossless'), (1, 'reference'), (3, 'fault')])

        def callback(call, original):
            assert call.owner is module
            active = runtime.snapshot()[operator]
            assert active['native_owned'] and active['inflight'] == 1 and active['leases'] == 1
            _, kinds = _read_proofs(call, mechanism)
            facts = native_facts(kinds['runtime_call'], kinds.keys(), mechanism)
            assert facts['build_profile_sha256'] == runtime.build_profile
            assert facts['cache_lifecycle'] == '1'
            arm, stage = next(arms)
            _, receipt = runtime.execute_scoped(call, original, arm)
            receipts.append(_validate_receipt(receipt, stage, operator, 2))

        remove = runtime.install_observer(callback, [('fixture', module, [])], mechanism)
        active = runtime.snapshot()[operator]
        assert active['native_owned'] and active['native_registered'] and active['inflight'] == 0
        assert run() == baseline  # outside actual scope
        token = _STACK.set(((0, module, 0),))
        try:
            assert run() == baseline
            assert run() == baseline
            assert run() == baseline
            assert run() != baseline
        finally:
            _STACK.reset(token)
            remove()
        assert all(not row['native_registered'] and row['leases'] == 0
                   for row in runtime.snapshot().values())
        assert run() == baseline and runtime.last_error is None
        assert receipts[0]['output_sha256'] == receipts[1]['output_sha256']
        assert receipts[2]['output_sha256'] == receipts[0]['output_sha256']
        assert receipts[3]['output_sha256'] != receipts[0]['output_sha256']
        assert receipts[1]['replaced_formals'] == [formal]
    print('PASS scoped host callback, D2D fixture and OFF for', operator)


verify('aclnnQuantMatmulWeightNz', 'fixture_qmm_run', 'x1')
verify('aclnnMatmulWeightNz', 'fixture_weight_nz_run', 'self')
