"""Real Host/Runtime/Attention provider states; only Native/scheduler are CPU seams."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import test_core_attention_serving as core
from experiments.c2c3_closure_20261117 import review_attention28_stock_census as review


class OriginalRuntimeProtocolTests(unittest.TestCase):
    def setUp(self):
        self.core=core.CoreAttentionTests()
        self.core.setUp()
        self.addCleanup(self.core.doCleanups)
        self.root=Path(self.enterContext(tempfile.TemporaryDirectory()))

    def actual_protocol(self):
        fixture=self.core
        client=fixture.client(mechanism='28.fia.denominator_replay',native=True)
        native_state=fixture.driver.state
        # This is explicitly simulation=True. The independent structural
        # decoder must understand it; the full reviewer must refuse it as NPU
        # evidence. No fake state changes an actual Runtime owner or phase.
        self.enterContext(patch.object(fixture.driver,'state',side_effect=lambda:dict(native_state(),
            observer_only=True,pending_drains=0,pending_references=0,first_error=None)))
        for action in ('prepare','register','check','enable'):
            fixture.ok(getattr(client,action)())
        requests=[]
        for ordinal,phase in enumerate(('observe','observe_repeat')):
            request='actual-cpu-'+str(ordinal)
            fixture.ok(client.advance(request))
            with torch.inference_mode():
                hidden=fixture.model(torch.tensor([1,2,3]),torch.tensor([0,1,2]))
                fixture.model.logits(hidden)
            self.assertEqual(client.query()['status'],'QUERY_OK')
            state=fixture.ok(client.collect_replay())
            requests.append(dict(phase=phase,mapping=dict(internal_request_id=request),
                                 worker=state['evidence']['requests'][-1]))
        finish=fixture.ok(client.finish())
        fixture.ok(client.disable());fixture.ok(client.unregister())
        host=client.detach();self.assertEqual(host['status'],'DETACHED')
        self.assertFalse(host['ownership_retained'])
        self.assertEqual(len(host['history']),15)
        self.assertEqual((fixture.model.calls,len(fixture.api_calls)),(2,4))
        self.assertEqual(fixture.binding.leases,set())
        core.remove_attention_serving_binding(fixture.worker,client.config.mechanism)
        return host,requests,finish

    def decode(self,host,requests):
        return review.protocol_states(host,requests,host['worker'],'28.fia.denominator_replay')

    def test_actual_runtime_private_owner_and_each_provider_phase_are_read_without_aliasing(self):
        host,requests,finish=self.actual_protocol()
        binding=self.decode(host,requests)
        self.assertNotEqual(binding['host_owner'],binding['provider_owner'])
        self.assertEqual(binding['provider_owner'],self.core.providers[0]._ticket[0])
        self.assertTrue(binding['simulation'])
        self.assertEqual(binding['runtime_phases'][1:-1],
            ['PREPARED','REGISTERED','CHECKED']+['ATTEMPTED']*7+['FINISHED','STOPPED','UNREGISTERED'])
        self.assertFalse(host['history'][1]['response']['provider_state']['registered'])
        self.assertTrue(finish['registered'] and finish['binding_reserved'])
        self.assertEqual(finish['owned_hooks'],0)
        self.assertFalse(host['history'][-2]['response']['provider_state']['binding_reserved'])
        settings=[args[1]['settings'] for method,args in self.core.executor.calls
                  if method=='demoncat_fault_rpc' and args[0]=='advance']
        self.assertEqual(settings,[dict(request_id='actual-cpu-0'),dict(collect=True),
                                   dict(request_id='actual-cpu-1'),dict(collect=True)])
        # Exercise the actual file reader: a structurally correct CPU run may
        # not acquire the new offline actual-stock observation conclusion.
        def save(name,value):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value),encoding='utf-8')
            return dict(path=name,sha256=review.E.sha(path))
        refs=[dict(save('rpc/'+str(i)+'.json',row),action=row['action'])
              for i,row in enumerate(host['history'])]
        data=dict(host_receipt=save('host_receipt.json',host),worker_core=dict(receipts=refs),
            setup=dict(worker=host['worker']),requests=requests,mechanism='28.fia.denominator_replay')
        with self.assertRaisesRegex(ValueError,'CPU_SIMULATION_IS_NOT_ACTUAL_STOCK_OBSERVATION'):
            review.protocol(review.Review(self.root),data)

    def test_owner_generation_query_and_partial_cleanup_cannot_be_hidden_in_actual_runtime_history(self):
        original,requests,_=self.actual_protocol()
        def state_at(host,index,field,value):
            response=host['history'][index]['response']
            for name in ('provider_state','native','provider'):
                if name in response:response[name][field]=value
        mutations=(lambda h:state_at(h,7,'owner','f'*32),
            lambda h:state_at(h,7,'generation',99),
            lambda h:state_at(h,11,'binding_reserved',False),
            lambda h:state_at(h,1,'registered',True),
            lambda h:state_at(h,12,'owned_hooks',1),
            lambda h:state_at(h,7,'replay_pending',False),
            lambda h:h['history'][6]['response']['provider_state']['evidence'].__setitem__('requests',[requests[0]['worker']]),
            lambda h:h['history'][10]['response'].__setitem__('phase','FINISHED'),
            lambda h:h['history'][2]['response']['ticket'].__setitem__('owner','foreign-public-owner'),
            lambda h:h['history'][7]['response']['provider_state']['native_state'].__setitem__('pending_drains',1))
        for mutation in mutations:
            host=deepcopy(original);mutation(host)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.decode(host,requests)


if __name__ == '__main__':unittest.main()
