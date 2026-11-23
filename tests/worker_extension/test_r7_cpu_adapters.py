"""CPU transport/scheduler tests, not A3 Native or live model qualification."""
import hashlib
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import torch
from test_native_regions import NativeRegionsTests
from test_product_batch import Runner
from fault_injection.sites.mla28 import fingerprint
from fault_injection.sites.dsv4_request_rows import combine_row_owners, within_request_shift
from fault_injection.sites.dsv4_observer import reason
from demoncat.runtime.vllm_ascend.worker_extension.product_observer import AutomaticRequests
from demoncat.runtime.vllm_ascend.worker_extension.product_request import ProductRequest
from demoncat.runtime.vllm_ascend.worker_extension.product_native import MECHANISMS, request_evidence
from tools.review_paged33_scope import review


class Projection(torch.nn.Module):
    def forward(self, x):
        return x @ torch.eye(4, dtype=x.dtype)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.projection = Projection()
    def forward(self, input_ids, inputs_embeds, positions, intermediate_tensors):
        x = (input_ids[:, None] + torch.arange(4)).bfloat16()
        return self.projection(x)


class CPUSite:
    """Explicit scheduler/lease test double; production site is never replaced."""
    def __init__(self, b):
        self.binding = b; self.native = NativeRegionsTests().manager()
        self.events = []; self.sites = ['projection']; self.uncovered = []; self.hooks = []
        self.current = None; self.consumers = []
    @property
    def pending(self): return self.native.count()
    def install(self):
        def enter(module, args):
            x = args[0]; target = x[-1]; source = target.clone() + 10
            e = dict(site='projection', request_ids=list(self.binding.worker.model_runner.input_batch.req_ids),
                     eligible=not (self.binding.mechanism == '41.tail_column' and len(x) > 1),
                     applied=False, changed=0, consumed=False, restored=False, failed=False)
            self.events.append(e)
            if not e['eligible']:
                e['reason'] = 'PREFILL_DIRECT_KV_NOT_PAGED_DECODE'; return
            e.update(before_sha256=fingerprint(target), source_sha256=fingerprint(source), region_elements=4,
                     writer='CPU_NATIVE_TRANSPORT_FIXTURE', consumer_operator='aten::mm')
            dst, src = self.native.acquire(target), self.native.acquire(source)
            e.update(changed=self.native.copy(dst, src), applied=True, after_sha256=fingerprint(target))
            self.current = (dst, src, e, target, source)
        def leave(module, args, output):
            if self.current is None: return
            dst, src, e, target, source = self.current
            self.consumers.append(torch.equal(output[-1], source))
            assert self.consumers[-1], 'CPU value consumer did not read the overwritten region'
            self.native.consumed(dst); e['consumed'] = True
            self.native.restore(dst); self.native.restore(src)
            e.update(restored=True, restored_sha256=fingerprint(target)); self.current = None
        self.hooks = [self.binding.model.projection.register_forward_pre_hook(enter),
                      self.binding.model.projection.register_forward_hook(leave)]
    def close(self):
        assert not self.pending
        for h in self.hooks: h.remove()
        self.hooks.clear()


class CPURequest(ProductRequest):
    def install_site(self):
        self.site = CPUSite(self.binding); self.site.install()


class NativeProductTests(unittest.TestCase):
    def run_request(self, runner, rid):
        output = []
        for step in range(4):
            new = [NS(req_id=rid, prompt_token_ids=[1, 2], max_tokens=4)] if step == 0 else []
            count = 2 if new else 1
            scheduler = NS(num_scheduled_tokens={rid: count}, total_num_scheduled_tokens=count,
                scheduled_new_reqs=new, finished_req_ids=set(), scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={})
            runner.execute_model(scheduler)
            output.append(runner.execute_model_state.logits.clone()); runner.sample_tokens()
        return output

    def test_two_scopes_same_workers_all_ranks_real_cpu_consumer_restore(self):
        class Event:
            def record(self, stream): pass
            def synchronize(self): pass
        with patch.object(torch, 'npu', NS(Event=Event), create=True):
            for mechanism in MECHANISMS:
                for rank in range(2):
                    runner = Runner(Model())
                    b = NS(worker=NS(model_runner=runner), model=runner.model, mechanism=mechanism,
                        check=Mock(), prompts=((1, 2),), identity={'rank': rank},
                        native_library=('CPU_TEST_ONLY', 'a'*64), warmups=[])
                    normal = self.run_request(runner, 'normal')
                    for scope in range(2):
                        with self.subTest(mechanism=mechanism, rank=rank, scope=scope):
                            b.group = {'owner': f'owner-{rank}-{scope}', 'group_generation': scope+1}
                            observer = AutomaticRequests(b, factory=CPURequest); observer.install()
                            fault = self.run_request(runner, f'http-{scope}')
                            record = observer.records[0]
                            expected = 3 if mechanism == '41.tail_column' else 4
                            for key in ('eligible','attempted','applied','consumed','restored'):
                                self.assertEqual(record['coverage'][key], expected)
                            self.assertEqual(record['coverage']['failed'], 0)
                            self.assertEqual(record['native_memory_effect']['leases'], 0)
                            self.assertTrue(record['bytes_restored'])
                            self.assertTrue(any(not torch.equal(a, z) for a,z in zip(normal,fault)))
                            observer.close()
                            self.assertEqual(observer.snapshot()['owned_hooks'], 0)
                            for phase in ('off','health'):
                                values = self.run_request(runner, f'{phase}-{scope}')
                                self.assertTrue(all(torch.equal(a,z) for a,z in zip(normal,values)))
    def test_request_join_pending_lease_and_missing_source_refuse(self):
        b = NS(mechanism='20.buffer_substitute',native_library=('library','a'*64))
        site = NS(native=NS(count=lambda: 0))
        event = dict(request_ids=['r'],eligible=True,applied=True,consumed=True,restored=True,
            region_elements=4,consumer_operator='actual',before_sha256='b'*64,restored_sha256='b'*64,
            source_sha256='c'*64,after_sha256='c'*64)
        self.assertFalse(request_evidence(b,site,[event],['r'])['qualification_granted'])
        for field,value in [('request_ids',['other']),('consumed',False),('restored_sha256','d'*64),('source_sha256',None)]:
            with self.subTest(field=field),self.assertRaises(RuntimeError):
                request_evidence(b,site,[dict(event,**{field:value})],['r'])
        site.native.count=lambda: 1
        with self.assertRaisesRegex(RuntimeError,'LEASES_REMAIN'):request_evidence(b,site,[event],['r'])
    def test_public_native_product_candidates_remain_closed(self):
        from fault_injection.product_contract import PRODUCT_CANDIDATES
        self.assertFalse(set(MECHANISMS) & set(PRODUCT_CANDIDATES))


class Dsv4RowsTests(unittest.TestCase):
    def test_dynamic_request_order_padding_and_no_cross_request_shift(self):
        owners=combine_row_owners(request_order=['b','a'],scheduled_counts={'a':1,'b':2},
            computed_tokens={'a':12,'b':4},output_token_rows=[2,0,None,1],active_mask=[True,True,False,True])
        self.assertEqual(owners,[('a',12),('b',4),None,('b',5)])
        self.assertEqual(within_request_shift(owners),[0,3,2,1])
        empty=combine_row_owners(request_order=[],scheduled_counts={},computed_tokens={},output_token_rows=[],active_mask=[])
        self.assertEqual(empty,[])
    def test_duplicates_padding_and_baseline_mapping_are_rejected(self):
        base=dict(request_order=['r'],scheduled_counts={'r':2},computed_tokens={'r':5},output_token_rows=[0,1],active_mask=[True,True])
        for changes in (dict(output_token_rows=[0,0]),dict(output_token_rows=[0,2]),
                        dict(active_mask=[True,False]),dict(scheduled_counts={'old':2})):
            with self.subTest(changes=changes),self.assertRaises(ValueError):combine_row_owners(**dict(base,**changes))
    def test_combine_shape_and_request_list_do_not_prove_row_ownership(self):
        x=torch.ones(2,4)
        self.assertEqual(reason('8',(x,),x,['r']),'COMBINE_OUTPUT_ROW_JOIN_UNPROVEN')


class PagedScopeTests(unittest.TestCase):
    def fixture(self, tp=1):
        scope=dict(hardware_name='CPU_A3_SCOPE_FIXTURE',hardware_family='A3',tp=tp,rank=0,
            q_heads=24//tp,kv_heads=4//tp,head_dim=256,kv_length=2305,block_size=128,runtime_source_sha256='d'*64)
        launch=dict(schema='dcat.paged33.observed_spans.v1',operator='FIA_paged_decode',
            arithmetic='BF16_MM_FP32_OUT_PBF16_LSE_FP32',spans=[[0,1024],[1024,2305]],
            qkv_sha256='a'*64,launch_sha256='b'*64,serializer_sha256='c'*64)
        raw=json.dumps(launch).encode();n=scope['q_heads']*scope['head_dim']
        check=dict(schema='dcat.paged33.checkpoint.v1',passed=True,device=scope['hardware_name'],
            qkv_sha256=launch['qkv_sha256'],launch_review_sha256=hashlib.sha256(raw).hexdigest(),spans=launch['spans'],
            native_checks=[dict(elements=n,bitwise_equal=n,checkpoint_99pct_passed=True)]*2,
            repeat=dict(elements=n,bitwise_equal=n))
        return check,scope,raw
    def test_tp1_and_tp2_each_have_independent_geometry_and_no_grant(self):
        for tp in (1,2):
            c,s,raw=self.fixture(tp);result=review(c,s,s,raw)
            self.assertTrue(result['checkpoint_scope_consistent']);self.assertFalse(result['qualification_granted'])
        c,s,raw=self.fixture(1);expected=dict(s,tp=2)
        with self.assertRaisesRegex(ValueError,'IDENTITY'):review(c,s,expected,raw)
    def test_forged_pass_and_cross_launch_cannot_replace_counts(self):
        c,s,raw=self.fixture()
        for field,value in [('launch_review_sha256','f'*64),('device','CPU_A2_OTHER')]:
            with self.assertRaises(ValueError):review(dict(c,**{field:value}),s,s,raw)
        bad=deepcopy(c);bad['native_checks'][0]['bitwise_equal']=0
        with self.assertRaisesRegex(ValueError,'THRESHOLD'):review(bad,s,s,raw)
        bad=deepcopy(c);bad['repeat']['bitwise_equal']-=1
        with self.assertRaisesRegex(ValueError,'REPEAT'):review(bad,s,s,raw)


if __name__=='__main__': unittest.main()
