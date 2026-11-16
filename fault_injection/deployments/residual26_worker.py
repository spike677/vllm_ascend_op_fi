"""Dedicated eager vLLM worker deployment, automatic structural ownership."""
import json,os
from pathlib import Path
from .. import attach
from .residual26 import Residual26Runtime

class Residual26WorkerExtension:
    def dcat26_install(self):
        import torch
        from torch import distributed as dist
        from capture import capture
        rank=dist.get_rank();runner=self.model_runner;model=runner.model
        if not runner.model_config.enforce_eager:raise RuntimeError('Eager runtime required')
        kind=os.environ['DCAT26_KIND'];arm=os.environ['DCAT26_ARM']
        mechanism='26.residual_stomp' if kind=='qmm' else '26.fia_destination'
        filename='libfault26_qmm_output_underflow_abi0.so' if kind=='qmm' else 'libfault26_fia_writer_proxy_abi0.so'
        root=Path(f'/evidence/rank{rank}');root.mkdir()
        # Each worker owns its log and producer/consumer state.
        os.environ['FAULT26_QMM_UNDERFLOW_LOG_PATH']=str(root/'writer.jsonl')
        os.environ['FAULT26_FIA_WRITER_LOG_PATH']=str(root/'writer.jsonl')
        build=json.loads(Path('/native/build.json').read_text())
        runtime=Residual26Runtime(model,mechanism,library='/native/'+filename,sha256=build['files'][filename],
            evidence_dir=root/'native',log_path=root/'writer.jsonl')
        session=attach(model,mechanism,arm=arm,runtime=runtime,unsupported='passthrough')
        debugger=capture('/evidence/'+arm+f'/rank{rank}')
        session.__enter__()
        original=runner.execute_model
        def execute(*args,**kwargs):
            debugger.start(model=model,rank_id=rank)
            try:return original(*args,**kwargs)
            finally:debugger.stop();debugger.step()
        runner.execute_model=execute
        self._dcat26=(runtime,session,runner,original,execute,rank)
        return dict(rank=rank,candidates=len(session.report()['candidates']),status='INSTALLED')

    def dcat26_remove(self):
        runtime,session,runner,original,execute,rank=self._dcat26
        if runner.execute_model is not execute:raise RuntimeError('Runner ownership changed')
        runner.execute_model=original;session.close()
        report=dict(rank=rank,coverage=session.report(),supported=runtime.supported,claims=runtime.claims,
                    completed=runtime.completed,error=runtime.error,pending=runtime.pending())
        Path(f'/evidence/rank{rank}/provider_result.json').write_text(json.dumps(report,indent=2))
        if runtime.supported==0 or runtime.pending() or runtime.error:raise RuntimeError('No verified supported writer, or recovery failed')
        return dict(rank=rank,supported=runtime.supported,claims=runtime.claims,
                    completed=len(runtime.completed),pending=runtime.pending())
