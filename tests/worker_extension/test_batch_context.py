from types import SimpleNamespace as NS
import unittest

from worker_extension.test_serving_context import fixture,MetadataOnlyTensor
from demoncat.runtime.vllm_ascend.worker_extension.batch_context import snapshot_first_decode
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import ContextRefused,require_serial_eager


def batch_fixture():
    runner,model,actual,ctx,_,_=fixture()
    runner.vllm_config.scheduler_config.max_num_seqs=8
    ids=['CPU_ONLY_'+str(i) for i in range(8)];expected={};requests={}
    for i,key in enumerate(ids):
        prompt=[5,6,i+8];expected[key]=dict(logical_id=i,prompt=prompt)
        requests[key]=NS(req_id=key,num_computed_tokens=3,prompt_token_ids=prompt,
            output_token_ids=[i+32],num_tokens=4,mm_features=[],prompt_embeds=None,lora_request=None,sampling_params=NS(n=1))
    runner.requests=requests;runner.with_prefill=False
    runner.input_batch=NS(req_ids=ids,num_reqs=8,req_id_to_index={key:i for i,key in enumerate(ids)},
        num_computed_tokens_cpu=[3]*8,token_ids_cpu=[requests[k].prompt_token_ids+requests[k].output_token_ids for k in ids])
    runner.query_lens=[1]*8;runner.query_start_loc=NS(cpu=list(range(9)));runner.seq_lens=NS(cpu=[4]*8)
    runner.input_ids=NS(cpu=list(range(32,40)),gpu=MetadataOnlyTensor(1,[8]))
    runner.positions=NS(cpu=[3]*8,gpu=MetadataOnlyTensor(2,[8],'torch.int64'))
    actual.update(input_ids=runner.input_ids.gpu,positions=runner.positions.gpu);ctx.num_tokens=8
    return runner,model,actual,ctx,expected


class BatchContextTests(unittest.TestCase):
    def test_actual_order_and_prefix_are_read_not_invented(self):
        args=batch_fixture();r=snapshot_first_decode(*args)
        self.assertEqual(r['row_request_indices'],list(range(8)));self.assertEqual(r['input_ids'],list(range(32,40)))

    def test_default_guard_still_refuses_b8(self):
        with self.assertRaisesRegex(ContextRefused,'MUST_BE_ONE'):require_serial_eager(batch_fixture()[0])

    def test_duplicate_or_foreign_request_refused(self):
        for change in ('duplicate','foreign','indices'):
            args=batch_fixture();r=args[0]
            if change=='duplicate':r.input_batch.req_ids[1]=r.input_batch.req_ids[0]
            elif change=='foreign':args[-1].pop(r.input_batch.req_ids[0])
            else:r.input_batch.req_id_to_index[r.input_batch.req_ids[0]]=7
            with self.subTest(change=change),self.assertRaises(ContextRefused):snapshot_first_decode(*args)

    def test_prefill_and_multiple_token_rows_cannot_qualify(self):
        for change in ('prefill','tokens','padding'):
            args=batch_fixture()
            if change=='prefill':args[0].with_prefill=True
            elif change=='tokens':args[0].query_lens=[2]*8
            else:args[3].pad_size=8
            with self.subTest(change=change),self.assertRaises(ContextRefused):snapshot_first_decode(*args)

    def test_changed_prefix_or_device_lineage_refused(self):
        for change in ('prefix','position','pointer','cpu_token','logical'):
            args=batch_fixture();r=args[0];key=r.input_batch.req_ids[0]
            if change=='prefix':r.requests[key].output_token_ids.append(7)
            elif change=='position':r.positions.cpu[0]=4
            elif change=='pointer':args[2]['input_ids']=MetadataOnlyTensor(99,[8])
            elif change=='cpu_token':r.input_ids.cpu[0]=123
            else:args[-1][key]['logical_id']=1
            with self.subTest(change=change),self.assertRaises(ContextRefused):snapshot_first_decode(*args)


if __name__=='__main__':unittest.main()
