"""Read-only natural model writer/consumer chain confirmation, no fault enabled."""
import os
from alias15_readonly_probe import main

class AliasOutputExtension:
    def alias_census(self,action):
        import torch,torch_npu
        from vllm.distributed import get_tensor_model_parallel_rank
        from tools.alias_output_census import AliasOutputCensus
        rank=get_tensor_model_parallel_rank()
        if action=='start':
            assert not hasattr(self,'_alias_output')
            def complete():
                event=torch.npu.Event();event.record(torch.npu.current_stream());event.synchronize()
            self._alias_output=AliasOutputCensus(self.model_runner.model,format_reader=torch_npu.get_npu_format,
                complete=complete,source_sha='4ba903ee8afb23012e78870034b1280039ccf81edb41b8e16f00e6e5bbc4c076')
            return dict(status='READ_ONLY_OUTPUT_CHAIN_ARMED',rank=rank,sites=self._alias_output.install())
        if action=='stop':
            census=self._alias_output;census.close();del self._alias_output
            return dict(rank=rank,events=census.events,sites=census.sites,hooks_remaining=len(census.hooks),
                        methods_remaining=len(census.methods),execute_wrapper_removed=True,injection_enabled=False)
        raise ValueError('READONLY_START_OR_STOP_ONLY')

if __name__=='__main__':
    os.environ['ALIAS15_EXTENSION']='alias15_output_probe.AliasOutputExtension'
    raise SystemExit(main())
