"""Request lifecycle over the owned host-memory ABI fixture; no NPU/model."""
import ctypes
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fault_injection.core import UnsupportedCall
from fault_injection.deployments.matmul import CtypesMatmulRuntime, CtypesMatmulGroupRuntime
from fault_injection.native import _STACK, _read_proofs

os.environ.update(TASK_QUEUE_ENABLE='0', VLLM_SERVER_DEV_MODE='1', DEMONCAT_F20_REQUIRE_TYPED_GWS='0')
operator = 'aclnnQuantMatmulWeightNz'
mechanism = '20.buffer_substitute'
run = ctypes.CDLL(None).fixture_qmm_run_current
run.argtypes, run.restype = [ctypes.c_uint64], ctypes.c_int
baseline = run(1)
with tempfile.TemporaryDirectory() as directory:
    runtime = CtypesMatmulRuntime(mechanism, dict(pid=os.getpid(), rank=0, logical_device=0, tp=1,
            execution_mode='eager', soc='host_memory_fixture'), directory, operator=operator)
    runtime.configure_request_lifecycle(owner='CPU-fixture-lease', generation=19)
    module = object()
    modes = iter([1, 2, 1, 3, 1, 3])
    receipts, tokens = [], []

    def callback(call, original):
        state = runtime.snapshot()[operator]
        assert state['enabled'] and not state['paused'] and state['request_controlled']
        assert state['native_owned'] and state['inflight'] == state['leases'] == 1
        try:
            runtime.pause()
        except UnsupportedCall:
            pass
        else:
            raise AssertionError('pause accepted inside active continuation')
        _, proof = _read_proofs(call, mechanism)
        tokens.append(proof['runtime_call']['current_typed_call']['gws_token'])
        _, receipt = runtime.execute_scoped(call, original, next(modes))
        receipts.append(receipt)

    remove = runtime.install_observer(callback, [('CPU.fixture', module, [])], mechanism)
    state = runtime.snapshot()[operator]
    assert state['native_registered'] and state['native_owned'] and state['paused']
    assert not state['enabled'] and state['inflight'] == state['leases'] == 0
    context = _STACK.set(((0, module, 0),))
    try:
        assert run(99) == baseline and not tokens  # even matching scope stays inactive while paused
        outputs = []
        for token in range(101, 107):
            runtime.resume()
            try:
                try:
                    remove()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('active owner was removed')
                outputs.append(run(token))
            finally:
                runtime.pause()
            state = runtime.snapshot()[operator]
            assert state['native_registered'] and state['paused'] and not state['enabled']
            assert state['inflight'] == state['leases'] == 0
            assert run(token + 1000) == baseline
        assert outputs[:3] == [baseline] * 3
        assert outputs[3] != baseline and outputs[4] == baseline and outputs[5] == 9999
        assert tokens == list(range(101, 107))
        assert len(receipts) == 5 and runtime.last_error is not None
    finally:
        _STACK.reset(context)
        remove()
    assert not runtime.snapshot()[operator]['native_registered']
    assert run(9999) == baseline
print('PASS CPU ctypes request ABI, atomic paused install, current tokens, qualification, one-fault, OFF and clean')

with tempfile.TemporaryDirectory() as directory:
    runtime = CtypesMatmulGroupRuntime(mechanism, dict(pid=os.getpid(), rank=0, logical_device=0, tp=1,
        execution_mode='eager', soc='host_memory_fixture'), directory, operator=operator)
    runtime.configure_group_lifecycle(owner='CPU-group-lease', generation=27,
                                      max_targets=2, max_faults=1, max_calls=100)
    modules = [object(), object()]
    keys = {id(module): character * 64 for module, character in zip(modules, 'ab')}
    actual_owner, current_mode = None, 0
    owners, receipts = [], []

    def group_callback(call, original, packet):
        owners.append(call.owner)
        assert runtime.independent_inventory()['complete']
        if call.owner is None:
            assert original() == 0
        else:
            _, receipt = runtime.execute_group(call, original, current_mode, keys[id(call.owner)])
            receipts.append(receipt)

    remove = runtime.install_group_observer(group_callback, lambda: actual_owner)
    before = runtime.independent_inventory()['eligible_run_calls']
    assert run(2000) == baseline and not owners
    assert runtime.independent_inventory()['eligible_run_calls'] == before + 1
    for current_mode in (1, 2, 1):
        runtime.resume()
        for actual_owner in modules:
            assert run(2000 + current_mode) == baseline
        actual_owner = None
        assert run(2100 + current_mode) == baseline
        runtime.pause()
    assert owners.count(None) == 3  # not dropped because no module was selected
    assert runtime.seal_group(keys.values())['targets'] == 2
    current_mode = 3
    runtime.resume()
    for actual_owner in modules:
        assert run(2300) != baseline
    runtime.pause()
    current_mode = 1
    runtime.resume()
    for actual_owner in modules:
        assert run(2400) == baseline
    runtime.pause()
    assert all(row['group_v3']['target_key'] in keys.values() for row in receipts)
    assert runtime.group_state()['sealed'] and not runtime.group_state()['failed']
    assert runtime.group_state()['calls'] == 13
    remove()
    assert runtime.group_state()['targets'] == 0
    assert not runtime.snapshot()[operator]['native_registered']
    assert run(2500) == baseline
print('PASS CPU ctypes v3 target groups, independent paused/unowned calls, original-once continuations and complete OFF')
