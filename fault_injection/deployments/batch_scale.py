"""Pinned vLLM runner integration; first observed eligible quantized module.

Uses real scheduler metadata and a separate healthy forward. Later candidates
pass through explicitly; this provider does not claim all-layer fault coverage.
"""
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from ..core import UnsupportedCall
from ..quantization import PinnedQuantRuntime,QuantizationSession,_discover

HOOK_SHA='33b650ef368de1f5e8beb771e64aa6e06f1c64f99746dcc07a3b151253b8cdd0'
BUILD_SHA='7629180c5756cc107631be32b48e0434265bf10e7605cc6bbfc43df79d05ad9b'

class QuantWorkerExtension:
    def dcat_quant_install(self):
        import torch
        from torch import distributed as dist
        root=Path('/quant');path=Path('/quant_original/model_hook.py')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=HOOK_SHA:raise UnsupportedCall('Scheduler adapter changed')
        # Only its read-only row validation is reused; install()/install_model()
        # (the historical hard-coded layer selector) are never called.
        sys.path.insert(0,str(root))
        spec=importlib.util.spec_from_file_location('_dcat_quant_rows',path)
        rows_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(rows_module)
        runner=self.model_runner
        loaded=sys.modules[rows_module.RUNNER]
        if hashlib.sha256(Path(loaded.__file__).read_bytes()).hexdigest()!=rows_module.RUNNER_SHA:
            raise UnsupportedCall('Actual runner ABI differs')
        rows_module.validate_runtime(runner)
        state=dict(active=None,selected=None,healthy=None,events=[],generations=set(),error=None)
        self._dcat_quant_state=state
        model=runner.model;rank=dist.get_rank()
        candidates,rejected=_discover(model)
        if not candidates:raise UnsupportedCall('No actual native dynamic W8A8 candidates')
        byid={id(m):(p,m,method) for p,m,method,_ in candidates}
        originals=[]
        for method in {id(method):method for _,_,method,_ in candidates}.values():
            original=method.apply
            @functools.wraps(original)
            def observe(module,*args,_original=original,**kwargs):
                active=state['active']
                if active and active.get('selected') and active['control']['mode']=='reference' and state['healthy'] is None and id(module) in byid:
                    x=args[0] if args else kwargs['x']
                    if x.ndim==2 and x.shape[0]==8 and x.is_contiguous() and x.storage_offset()==0:
                        rows_module.BackendControl.validate_input(x)
                        state['selected']=module
                        state['healthy']=dict(path=byid[id(module)][0],input_sha256=rows_module.digest(x),
                            positions=active['rows']['row_token_positions'],logical_ids=active['rows']['logical_ids'],
                            prefixes={v['logical_id']:v['output_token_ids'] for v in active['rows']['prefixes'].values()})
                return _original(module,*args,**kwargs)
            method.apply=observe;originals.append((method,original,observe))
        def context(call):
            active=state['active'];healthy=state['healthy']
            if not active or not active.get('selected') or call.module is not state['selected']:
                raise UnsupportedCall('Outside automatically selected healthy decode call')
            rows=active['rows']
            if (rows['row_token_positions']!=healthy['positions'] or rows['logical_ids']!=healthy['logical_ids']
                or {v['logical_id']:v['output_token_ids'] for v in rows['prefixes'].values()}!=healthy['prefixes']):
                raise UnsupportedCall('Decode prefixes changed from independent healthy forward')
            return dict(rows,model=model,rank=rank,world_size=dist.get_world_size(),
                baseline_input_sha256=healthy['input_sha256'],healthy_run_id='independent_reference_forward')
        runtime=PinnedQuantRuntime.from_scoped_backend(root,context_provider=context,
            approved_build_sha256=os.environ['DCAT_QUANT_BUILD_SHA256'])
        self._dcat_quant_runtime=runtime
        from capture import capture
        debugger=capture('/evidence/'+os.environ['DCAT_QUANT_ARM']+f'/rank{rank}')
        old_execute,old_prepare=runner.execute_model,runner._prepare_inputs
        def execute(*args,**kwargs):
            control=json.loads(Path('/evidence/control.json').read_text())
            if not control.get('active'):return old_execute(*args,**kwargs)
            if state['active'] is not None:raise UnsupportedCall('Concurrent runner execution')
            active=dict(control=control,rows=None,selected=False,session=None,observer=False)
            state['active']=active
            try:return old_execute(*args,**kwargs)
            finally:
                try:
                    if active['observer']:debugger.stop();debugger.step()
                    if active['session']:
                        active['session'].close();report=active['session'].report()
                        Path(f'/evidence/rank{rank}-{control["generation_id"]}-session.json').write_text(json.dumps(report,indent=2))
                        routed=[e for e in report['events'] if e['status']=='routed']
                        if sys.exc_info()[1] is None and (len(routed)!=1 or not routed[0]['native_execution_proven']):
                            raise RuntimeError('Expected one actual selected DynamicQuantV2 execution')
                        state['events'].append(dict(generation=control['generation_id'],report=report))
                finally:state['active']=None
        def prepare(*args,**kwargs):
            result=old_prepare(*args,**kwargs)
            active=state['active']
            if active:
                scheduler=args[0] if args else kwargs['scheduler_output']
                counts=args[1] if len(args)>1 else kwargs['num_scheduled_tokens']
                rows=rows_module.capture_rows(runner,scheduler,counts,result,active['control']);active['rows']=rows
                gid=active['control']['generation_id']
                if rows['pure_decode'] and rows['all_expected_present'] and gid not in state['generations']:
                    state['generations'].add(gid);active['selected']=True
                    if active['control']['mode'] in ('baseline','fault'):
                        active['session']=QuantizationSession(model,arm=active['control']['mode'],unsupported='passthrough',runtime=runtime)
                        active['session'].__enter__();debugger.start(model=model,rank_id=rank);active['observer']=True
            return result
        runner.execute_model,runner._prepare_inputs=execute,prepare
        self._dcat_quant_remove=(runner,old_execute,old_prepare,execute,prepare,originals)
        return dict(rank=rank,status='INSTALLED',candidates=len(candidates),unsupported=len(rejected))

    def dcat_quant_report(self):
        from torch import distributed as dist
        s=self._dcat_quant_state
        report=dict(rank=dist.get_rank(),healthy=s['healthy'],events=s['events'],
            native_poisoned=self._dcat_quant_runtime.poisoned,
            coverage_scope='first observed eligible module; other modules explicitly pass through')
        Path(f'/evidence/rank{dist.get_rank()}-runtime.json').write_text(json.dumps(report,indent=2))
        return dict(rank=dist.get_rank(),healthy=s['healthy'],events=len(s['events']),native_poisoned=report['native_poisoned'])

    def dcat_quant_remove(self):
        runner,old_execute,old_prepare,execute,prepare,originals=self._dcat_quant_remove
        if self._dcat_quant_state['active'] is not None:raise RuntimeError('Runner still active')
        if runner.execute_model is not execute or runner._prepare_inputs is not prepare:raise RuntimeError('Runner ownership changed')
        runner.execute_model,runner._prepare_inputs=old_execute,old_prepare
        for method,original,wrapper in originals:
            if method.apply is not wrapper:raise RuntimeError('Quant method ownership changed')
            method.apply=original
        return {'status':'REMOVED'}
