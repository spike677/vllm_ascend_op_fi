"""CPU fixtures execute authentic-shaped nested calls, never actual NPU claims."""
from types import SimpleNamespace as NS
import unittest
import torch

from fault_injection.vision17_context import Vision17RequestContext
from test_vision17_trace import Encoder


class Visual(Encoder):
    spatial_merge_size=2
    def forward(self,pixel_values,grid_thw):
        return super().forward(pixel_values.view(-1,3,2,16,16))


class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.visual=Visual();self.foreign_pixels=False
    def embed_multimodal(self,**kwargs):
        image_input=dict(type='pixel_values',**kwargs)
        return self._process_image_input(image_input)
    def _process_image_input(self,image_input):
        grid_thw=image_input['image_grid_thw'];pixel_values=image_input['pixel_values']
        if self.foreign_pixels:pixel_values=pixel_values+1
        return self.visual(pixel_values,grid_thw=grid_thw)
    def forward(self,*args):raise AssertionError('Language forward is not encoder scope')


class Runner:
    def __init__(self,model):
        self.model=model;self.request_id='actual-internal';self.foreign_refs=False
        self.pixels=torch.arange(4*3*2*16*16.).reshape(4,-1)/1000
        feature=NS(modality='image',identifier='actual-processor-hash',mm_position=NS(offset=1,length=1,is_embed=None),
                   data={'pixel_values':NS(data=self.pixels),'image_grid_thw':NS(data=torch.tensor([1,2,2]))})
        request=NS(req_id=self.request_id,num_computed_tokens=0,output_token_ids=[],prompt_embeds=None,lora_request=None,
                   prompt_token_ids=[1,2,3],mm_features=[feature])
        self.requests={self.request_id:request};self.input_batch=NS(req_ids=[self.request_id],req_id_to_index={self.request_id:0})
    def _batch_mm_inputs_from_scheduler(self,scheduler_output):
        feature=self.requests[self.request_id].mm_features[0]
        return [feature.identifier],[('image',feature.data)],[(self.request_id,feature.mm_position)]
    def _execute_mm_encoder(self,scheduler_output):
        mm_hashes,mm_kwargs,mm_lora_refs=self._batch_mm_inputs_from_scheduler(scheduler_output)
        model=self.model;modality='image';num_items=1;current_item_idx=0
        if self.foreign_refs:mm_lora_refs=[('other-request',mm_lora_refs[0][1])]
        mm_kwargs_batch=dict(pixel_values=self.pixels.clone(),image_grid_thw=torch.tensor([[1,2,2]]))
        return model.embed_multimodal(**mm_kwargs_batch)


class VisionContextTests(unittest.TestCase):
    def setUp(self):
        self.model=Model().eval();self.runner=Runner(self.model);self.worker=NS(model_runner=self.runner)
        self.context=Vision17RequestContext(self.worker,simulation=True)
        self.scheduler=NS(scheduled_encoder_inputs={'actual-internal':[0]},num_scheduled_tokens={'actual-internal':3},
                          total_num_scheduled_tokens=3,scheduled_spec_decode_tokens={})
        self.expected='actual-internal';self.rows=[]
        self.handle=self.model.visual.register_forward_pre_hook(self.observe,with_kwargs=True)
    def observe(self,module,args,kwargs):
        self.rows.append(self.context.snapshot(module,args,kwargs,expected_request_id=self.expected,expected_prompt=[1,2,3]))
    def tearDown(self):self.handle.remove();self.context.close()
    def run_request(self):return self.runner._execute_mm_encoder(self.scheduler)
    def test_actual_scheduler_feature_batch_parse_and_visual_objects_join(self):
        self.run_request();row,=self.rows
        self.assertTrue(row['single_image_request_join_proven']);self.assertTrue(row['simulation'])
        self.assertFalse(row['standard_worker_qualified'])
        self.assertEqual(row['request_id'],'actual-internal');self.assertEqual(row['pixel_values']['shape'],[4,1536])
    def test_direct_visual_call_cannot_borrow_saved_request(self):
        with self.assertRaisesRegex(RuntimeError,'ACTUAL_EXECUTING_FRAME_REQUIRED'):
            self.model.visual(self.runner.pixels,grid_thw=torch.tensor([[1,2,2]]))
    def test_request_must_be_real_scheduler_request(self):
        self.expected='foreign'
        with self.assertRaisesRegex(RuntimeError,'ACTUAL_REQUEST_SET_MISMATCH'):self.run_request()
    def test_cached_encoder_not_scheduled_is_not_an_observed_call(self):
        self.scheduler.scheduled_encoder_inputs={}
        with self.assertRaisesRegex(RuntimeError,'ACTUAL_ENCODER_SCHEDULE_MISMATCH'):self.run_request()
    def test_foreign_image_reference_does_not_pass_by_equal_count(self):
        self.runner.foreign_refs=True
        with self.assertRaisesRegex(RuntimeError,'ACTUAL_MM_ITEM_REQUEST_MAPPING_REQUIRED'):self.run_request()
    def test_processed_image_bytes_are_checked_after_actual_dtype_cast(self):
        self.model.foreign_pixels=True
        with self.assertRaisesRegex(RuntimeError,'ACTUAL_PROCESSED_IMAGE_BYTES_CHANGED'):self.run_request()
    def test_missing_mm_data_cannot_fall_back_to_text_identity(self):
        self.runner.requests[self.expected].mm_features[0].data=None
        with self.assertRaisesRegex(RuntimeError,'UNCACHED_IMAGE_FEATURE'):self.run_request()
    def test_changed_bound_method_invalidates_live_runtime(self):
        import types
        def foreign(self,**kwargs):return None
        self.model.embed_multimodal=types.MethodType(foreign,self.model)
        with self.assertRaisesRegex(RuntimeError,'ACTUAL_METHOD_CHANGED'):self.context.verify_runtime()
    def test_default_source_guard_never_accepts_cpu_fixture(self):
        with self.assertRaisesRegex(RuntimeError,'UNREVIEWED_ACTUAL_METHOD_SOURCE'):
            Vision17RequestContext(self.worker)
    def test_root_queued_manifest_has_no_fault_or_cache_override(self):
        from experiments.c2c3_closure_20261117.prepare_vision17_worker_census import cases
        row,=cases('/home/frozen-source',2)
        self.assertEqual(row['review']['expected_hits'],0)
        self.assertEqual(row['review']['physical_cards'],[2])
        self.assertIn('vision17_worker_census_probe.py',row['command'])
        self.assertIn('No F',row['review']['recovery_plan'])
        self.assertIn('OBSERVATION_ONLY',row['review']['acceptance'])


if __name__=='__main__':unittest.main()
