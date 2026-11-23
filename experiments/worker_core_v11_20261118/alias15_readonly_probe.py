"""Opt-in trusted read-only Worker extension and isolated A2 HTTP census.

No ProductClient, permits, fault registration, or numeric substitution.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import urllib.request


class AliasCensusExtension:
    def alias_census(self, action):
        import torch_npu
        from vllm.distributed import get_tensor_model_parallel_rank
        from tools.alias_chain_census import AliasChainCensus
        rank = get_tensor_model_parallel_rank()
        if action == 'start':
            assert not hasattr(self, '_alias_census_original')
            runner = self.model_runner
            assert 'execute_model' not in vars(runner)
            self._alias_census_original = runner.execute_model
            self._alias_census_reports = []
            self._alias_module_stack = []
            self._alias_handles = []
            model = runner.model
            def pre(name):
                return lambda *args: self._alias_module_stack.append(name)
            def post(*args):
                self._alias_module_stack.pop()
            for name, module in model.named_modules():
                self._alias_handles.append(module.register_forward_pre_hook(pre(name)))
                self._alias_handles.append(module.register_forward_hook(post, always_call=True))
            def execute(scheduler, *args, **kwargs):
                count = len(self._alias_census_reports)
                if count >= 16:
                    raise RuntimeError('ALIAS_CENSUS_FORWARD_BUDGET_EXCEEDED')
                census = AliasChainCensus(torch_npu.get_npu_format, max_samples=96,
                                          module=lambda: self._alias_module_stack[-1] if self._alias_module_stack else '')
                scheduled = dict(scheduler.num_scheduled_tokens)
                new_ids = {r.req_id for r in scheduler.scheduled_new_reqs}
                try:
                    with census:
                        return self._alias_census_original(scheduler, *args, **kwargs)
                finally:
                    self._alias_census_reports.append(dict(forward=count,rank=rank,
                        scheduled_token_counts=list(scheduled.values()),new_requests=len(new_ids),
                        phase='prefill' if new_ids else 'decode_or_cached', report=census.report()))
            self._alias_census_wrapper = execute
            runner.execute_model = execute
            return dict(status='READ_ONLY_CENSUS_ARMED',rank=rank)
        if action == 'stop':
            assert self.model_runner.execute_model is self._alias_census_wrapper
            del self.model_runner.execute_model
            for h in self._alias_handles:
                h.remove()
            self._alias_handles.clear()
            self._alias_module_stack.clear()
            del self._alias_census_original
            return dict(rank=rank,reports=self._alias_census_reports,hooks_remaining=0,
                        execute_wrapper_removed='execute_model' not in vars(self.model_runner),
                        injection_enabled=False)
        raise ValueError('READONLY_START_OR_STOP_ONLY')


def main():
    from product_http_probe import HTTPExecutor
    root=Path('/evidence');server=None
    report=dict(schema='dcat.alias.http.census.v1',status='STARTED',injection_enabled=False)
    try:
        cmd=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model','/model',
            '--served-model-name','acceptance','--host','127.0.0.1','--port','18937',
            '--dtype','bfloat16','--quantization','ascend','--tensor-parallel-size','2',
            '--distributed-executor-backend','mp','--enforce-eager','--no-enable-prefix-caching',
            '--no-async-scheduling','--enable-chunked-prefill','--max-num-batched-tokens','1024',
            '--max-model-len','1024','--max-num-seqs','1','--gpu-memory-utilization','0.8',
            '--seed','42','--trust-remote-code','--worker-extension-cls',os.environ.get('ALIAS15_EXTENSION','alias15_readonly_probe.AliasCensusExtension'),
            '--limit-mm-per-prompt','{"image":0,"video":0}']
        server=subprocess.Popen(cmd,env=dict(os.environ,VLLM_WORKER_MULTIPROC_METHOD='spawn'),
                                stdout=(root/'http-server.log').open('w'),stderr=subprocess.STDOUT)
        ex=HTTPExecutor('http://127.0.0.1:18937')
        for _ in range(360):
            if server.poll() is not None:raise RuntimeError('HTTP_SERVER_START_FAILED')
            try:
                with urllib.request.urlopen(ex.base+'/health',timeout=2):break
            except Exception:time.sleep(1)
        else:raise RuntimeError('HTTP_SERVER_READY_TIMEOUT')
        def infer():
            result=ex.post('/v1/completions',dict(model='acceptance',prompt='Explain briefly why rivers flow towards the ocean.',
                                                temperature=0,seed=42,max_tokens=4,min_tokens=4,ignore_eos=True))
            return result['choices']
        before=[infer(),infer()]
        report['before_choices']=before
        report['arm']=ex.collective_rpc('alias_census',args=('start',))
        try: report['observed_choices']=[infer(),infer()]
        finally: report['ranks']=ex.collective_rpc('alias_census',args=('stop',))
        report['after_choices']=infer()
        assert all(x==before[0] for x in before+report['observed_choices']+[report['after_choices']])
        assert len(report['ranks'])==2 and all(r['hooks_remaining']==0 and r['execute_wrapper_removed'] for r in report['ranks'])
        report['source_manifest_sha256']=hashlib.sha256(Path('/source/source_manifest.json').read_bytes()).hexdigest()
        report.update(status='READONLY_CENSUS_COMPLETED',token_text_equal=True,
                      full_hidden_logits_equality_proven=False,qualification_granted=False)
    except BaseException as error:
        report.update(status='FAILED',error=repr(error),traceback=traceback.format_exc())
    finally:
        (root/'alias_census.json').write_text(json.dumps(report,indent=2)+'\n')
        if server is not None:
            server.terminate()
            try:server.wait(timeout=60)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=30)
    return 0 if report['status']=='READONLY_CENSUS_COMPLETED' else 2


if __name__=='__main__':raise SystemExit(main())
