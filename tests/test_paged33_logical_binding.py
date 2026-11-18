"""Live CPU object/storage fixtures. No source-only flag authorizes Native F."""
import copy
import threading
from types import SimpleNamespace as NS
import unittest

import torch

from demoncat.runtime.vllm_ascend.worker_extension.paged33_audit import _meta
from demoncat.runtime.vllm_ascend.worker_extension.paged33_logical_binding import LogicalPagedInput,compare_logical_inputs

PATH='model.layers.0.self_attn.attn'


class Fixture:
    def __init__(self):
        self.owner=object();self.generation=7
        packed=torch.zeros((2,80,16,8,128),dtype=torch.float16)
        self.module=torch.nn.Module();self.module.layer_name='actual.layer.0'
        self.module.impl=NS(key_cache=packed[0],value_cache=packed[1],num_kv_heads=8,num_heads=16,
                            head_size=128,sliding_window=None,sinks=None)
        self.module.kv_cache=[packed]
        model=torch.nn.Module();model.model=torch.nn.Module();model.model.layers=torch.nn.ModuleList([torch.nn.Module()])
        model.model.layers[0].self_attn=torch.nn.Module();model.model.layers[0].self_attn.attn=self.module
        self.worker=NS(model_runner=NS(model=model))
        self.mask=torch.ones(2048,2048,dtype=torch.int8).triu(1)
        self.logical=(torch.arange(1025,dtype=torch.float32)%128).to(torch.float16)

    def arguments(self,request='A',pages=None):
        pages=list(range(65)) if pages is None else list(pages)
        for kind,cache in enumerate((self.module.impl.key_cache,self.module.impl.value_cache)):
            for index,page in enumerate(pages):
                count=min(16,1025-16*index)
                cache[page,:count].copy_((self.logical[16*index:16*index+count]+kind).reshape(count,1,1).expand(count,8,128))
        table=torch.tensor([pages+[0]*15],dtype=torch.int32)
        slot=pages[-1]*16
        metadata=NS(block_tables=table,slot_mapping=torch.tensor([slot],dtype=torch.int32),
            attn_state=NS(name='DecodeOnly'),num_actual_tokens=1,seq_lens_list=[1025],actual_seq_lengths_q=[1])
        context=NS(attn_metadata={self.module.layer_name:metadata},
            no_compile_layers={self.module.layer_name:self.module},virtual_engine=0)
        actual=dict(query=torch.ones(1,16,128,dtype=torch.float16),
            key=self.module.impl.key_cache.view(80,16,1024),value=self.module.impl.value_cache.view(80,16,1024),
            atten_mask=self.mask,block_table=table,input_layout='TND',block_size=16,sparse_mode=3,
            actual_seq_lengths=[1],actual_seq_lengths_kv=[1025],num_heads=16,num_key_value_heads=8,scale=128**-.5)
        writer=dict(returned=True,module_path=PATH,key_cache=_meta(self.module.impl.key_cache),
            value_cache=_meta(self.module.impl.value_cache),slots=[slot])
        scheduler=dict(request_count=1,counts=[1],token_row_count=1,row_token_positions=[1024],request_ids=[request])
        return dict(worker=self.worker,module=self.module,module_path=PATH,forward_context=context,
            metadata=metadata,actual=actual,scheduler=scheduler,writer=writer,owner=self.owner,
            generation=self.generation,allow_cpu_test=True)

    def run(self,arguments):
        binding=LogicalPagedInput(**arguments)
        output=(arguments['actual']['query'].clone(),torch.empty(0,dtype=torch.float32))
        calls=[]
        def original():calls.append(1);return output
        returned=binding.execute(original,owner=self.owner,generation=self.generation)
        assert returned is output and calls==[1]
        return binding


class LogicalPagedContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.old_threads)

    def test_actual_relocated_pages_keep_raw_difference_and_logical_equivalence(self):
        fixture=Fixture();first=fixture.run(fixture.arguments())
        second=fixture.run(fixture.arguments('B',reversed(range(65))))
        checked=compare_logical_inputs(first,second,owner=fixture.owner,generation=7)
        self.assertFalse(checked['raw_input_exact'])
        self.assertTrue(checked['logical_input_equivalent'])
        self.assertNotEqual(*checked['raw_table_sha256'])
        self.assertEqual(checked['writer_consumer_slots'],[[1024],[0]])
        self.assertFalse(checked['qualification_granted'])
        self.assertFalse(checked['full_input_arrays_preserved'])
        # The report is a copy; changing it cannot mutate the live binding.
        report=first.report();report['evidence']['before']['signature'].clear()
        self.assertTrue(first.report()['evidence']['before']['signature'])

    def test_raw_equal_repeat_is_explicit_and_all_arguments_return_once(self):
        fixture=Fixture();args=fixture.arguments();actual=args['actual']
        before={k:id(v) for k,v in actual.items()}
        first=fixture.run(args);second=fixture.run(fixture.arguments('B'))
        result=compare_logical_inputs(first,second,owner=fixture.owner,generation=7)
        self.assertTrue(result['raw_input_exact'])
        self.assertEqual(before,{k:id(v) for k,v in actual.items()})
        with self.assertRaisesRegex(RuntimeError,'ALREADY_ATTEMPTED'):
            first.execute(lambda:None,owner=fixture.owner,generation=7)
        self.assertEqual(first.report()['actual_original_calls'],1)

    def test_equal_tensor_clones_never_establish_cache_or_metadata_identity(self):
        for name in ('key','value','block_table'):
            fixture=Fixture();args=fixture.arguments();args['actual'][name]=args['actual'][name].clone()
            with self.subTest(name=name),self.assertRaises(RuntimeError):LogicalPagedInput(**args)

    def test_duplicate_unused_outside_pages_and_wrong_writer_slot_refuse(self):
        for kind in ('duplicate','out_of_bounds','unused_out_of_bounds','slot','writer','context'):
            fixture=Fixture();args=fixture.arguments()
            if kind=='duplicate':args['actual']['block_table'][0,1]=0
            if kind=='out_of_bounds':args['actual']['block_table'][0,0]=80
            if kind=='unused_out_of_bounds':args['actual']['block_table'][0,79]=80
            if kind=='slot':args['metadata'].slot_mapping[0]=1
            if kind=='writer':args['writer']['module_path']='model.layers.1.self_attn.attn'
            if kind=='context':args['forward_context'].no_compile_layers[fixture.module.layer_name]=object()
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):LogicalPagedInput(**args)

    def test_kv_mask_query_unused_table_and_supplied_default_changes_are_not_normalized(self):
        for kind in ('logical_key','logical_value','query','mask','unused_table','supplied_default'):
            fixture=Fixture();first=fixture.run(fixture.arguments());args=fixture.arguments('B',reversed(range(65)))
            if kind=='logical_key':fixture.module.impl.key_cache[64,0,0,0]+=1
            if kind=='logical_value':fixture.module.impl.value_cache[64,0,0,0]+=1
            if kind=='query':args['actual']['query'][0,0,0]+=1
            if kind=='mask':args['actual']['atten_mask']=args['actual']['atten_mask'].clone();args['actual']['atten_mask'][0,0]=1
            if kind=='unused_table':args['actual']['block_table'][0,79]=1
            if kind=='supplied_default':args['actual']['pre_tokens']=2147483647
            second=fixture.run(args)
            with self.subTest(kind=kind),self.assertRaisesRegex(RuntimeError,'BEYOND_PAGE_RELOCATION'):
                compare_logical_inputs(first,second,owner=fixture.owner,generation=7)

    def test_original_input_mutation_or_failed_return_keeps_first_failure_and_references(self):
        for kind in ('mutate','raise','bad_output'):
            fixture=Fixture();args=fixture.arguments();binding=LogicalPagedInput(**args);calls=[]
            def original():
                calls.append(1)
                if kind=='raise':raise RuntimeError('original failed')
                if kind=='mutate':args['actual']['query'][0,0,0]+=1
                if kind=='bad_output':return object()
                return (args['actual']['query'].clone(),torch.empty(0,dtype=torch.float32))
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):
                binding.execute(original,owner=fixture.owner,generation=7)
            first=binding.first_error
            self.assertIs(binding.worker,fixture.worker)
            self.assertFalse(binding.complete)
            self.assertEqual(calls,[1])
            with self.assertRaisesRegex(RuntimeError,'ALREADY_ATTEMPTED'):
                binding.execute(original,owner=fixture.owner,generation=7)
            self.assertEqual(binding.first_error,first)
            released=binding.release(owner=fixture.owner,generation=7)
            self.assertEqual(released['first_error'],first)
            self.assertIsNone(binding.worker)

    def test_owner_generation_thread_and_inflight_release_refuse(self):
        fixture=Fixture();binding=LogicalPagedInput(**fixture.arguments())
        for owner,generation in ((object(),7),(fixture.owner,8),(fixture.owner,True)):
            with self.subTest(generation=generation),self.assertRaises(RuntimeError):
                binding.execute(lambda:None,owner=owner,generation=generation)
        errors=[]
        def foreign():
            try:binding.release(owner=fixture.owner,generation=7)
            except RuntimeError as error:errors.append(str(error))
        thread=threading.Thread(target=foreign);thread.start();thread.join()
        self.assertEqual(len(errors),1)
        def original():
            with self.assertRaisesRegex(RuntimeError,'INFLIGHT'):binding.release(owner=fixture.owner,generation=7)
            return (binding.actual['query'].clone(),torch.empty(0,dtype=torch.float32))
        binding.execute(original,owner=fixture.owner,generation=7)
        self.assertTrue(binding.release(owner=fixture.owner,generation=7)['released'])

    def test_changed_worker_or_argument_before_call_prevents_invocation(self):
        for kind in ('model','cache','query','request'):
            fixture=Fixture();args=fixture.arguments();binding=LogicalPagedInput(**args);calls=[]
            if kind=='model':fixture.worker.model_runner.model=torch.nn.Module()
            if kind=='cache':fixture.module.impl.key_cache=fixture.module.impl.key_cache.clone()
            if kind=='query':args['actual']['query']=args['actual']['query'].clone()
            if kind=='request':args['scheduler']['request_ids']=['foreign']
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):
                binding.execute(lambda:calls.append(1),owner=fixture.owner,generation=7)
            self.assertEqual(calls,[])
            self.assertIsNotNone(binding.worker)

    def test_cpu_or_other_layer_cannot_be_confused_with_actual_native_scope(self):
        fixture=Fixture();args=fixture.arguments();args['allow_cpu_test']=False
        with self.assertRaisesRegex(RuntimeError,'NPU_QUERY_REQUIRED'):LogicalPagedInput(**args)
        args['allow_cpu_test']=True;args['module_path']='model.layers.1.self_attn.attn'
        with self.assertRaisesRegex(RuntimeError,'FIRST_LAYER'):LogicalPagedInput(**args)

    def test_released_serialized_same_request_or_foreign_worker_bindings_cannot_join(self):
        fixture=Fixture();first=fixture.run(fixture.arguments());second=fixture.run(fixture.arguments())
        with self.assertRaisesRegex(RuntimeError,'TWO_REQUESTS'):
            compare_logical_inputs(first,second,owner=fixture.owner,generation=7)
        with self.assertRaisesRegex(RuntimeError,'ACTUAL_BINDING_OBJECTS'):
            compare_logical_inputs(first.report(),second.report(),owner=fixture.owner,generation=7)
        second=fixture.run(fixture.arguments('B'));first.release(owner=fixture.owner,generation=7)
        with self.assertRaisesRegex(RuntimeError,'INCOMPLETE_OR_FOREIGN'):
            compare_logical_inputs(first,second,owner=fixture.owner,generation=7)


if __name__=='__main__':unittest.main()
