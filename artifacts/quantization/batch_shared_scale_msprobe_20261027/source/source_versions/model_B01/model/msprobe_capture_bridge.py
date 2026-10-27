"""Real msprobe 26.1 debug tensor save, bounded to one native call/session.

No package imports occur until the capture-enabled worker begins a session.
The legacy capture implementation and package source are SHA-pinned. Its real
PrecisionDebugger.save API writes debug.json and .debug.pt, followed by exact
dtype/shape/bytes readback. We never manufacture those standard output files.
"""
import hashlib
import inspect
import os
from pathlib import Path

CAPTURE_SHA='084ff6740391bab653411ff185fcf46b221f5c4217b67f77f4cdd2334562ab0d'
_factory=None


def factory():
    global _factory
    if _factory is None:
        root=Path(os.environ['BATCH_OUTLIER_LEGACY_ROOT']).resolve()
        expected=root/'src/ascend_fi/quantization/msprobe_capture.py'
        if hashlib.sha256(expected.read_bytes()).hexdigest()!=CAPTURE_SHA:
            raise RuntimeError('Unreviewed legacy msprobe capture source')
        from ascend_fi.quantization import msprobe_capture
        if Path(inspect.getsourcefile(msprobe_capture)).resolve()!=expected:
            raise RuntimeError('Wrong cached msprobe capture module origin')
        _factory=msprobe_capture.create_capture
    return _factory


def begin(runner,layer,module_name,generation,rank,mode,rows,contract,call_id):
    if os.environ.get('BATCH_MSPROBE_CAPTURE')!='1':
        raise RuntimeError('msprobe bridge called while capture is disabled')
    root=Path(os.environ['BATCH_MSPROBE_DUMP_ROOT'])/('rank'+str(rank))
    capture=factory()(model=runner.model,layer=layer,module_name=module_name,
        output_dir=root/'sessions'/generation,
        run_metadata={'generation_id':generation,'rank':rank,'pid':os.getpid(),'tp_size':2,
            'mode':mode,'experiment_mode':mode,'call_id':call_id,'phase':'decode',
            'actual_contract':contract,'scheduler':rows,
            'logical_mapping':dict(zip(rows['request_ids'],rows['logical_ids'])),
            'capture_interface':'PrecisionDebugger.save(save_backward=False), debug level, tensor task',
            'format':'native msprobe debug.json and .debug.pt; not automatic API dump',
            'scope':'selected down_proj first actual all-request pure-decode call; local TP feature shard'},
        config={'dump_root':str(root/'native_msprobe'),'max_records':6,
                'max_bytes':134217728,'max_disk_bytes':268435456})
    capture.__enter__()
    return capture


def finish(capture,error):
    capture.__exit__(type(error) if error is not None else None,error,None)
