import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from demoncat.runtime.vllm_ascend.worker_extension.continuous_model import validate_continuous_model


class ModelAdmissionTests(unittest.TestCase):
    def test_unadapted_model_never_inherits_qwen_admission(self):
        with self.assertRaisesRegex(RuntimeError,'MODEL_NOT_ADAPTED'):
            validate_continuous_model(NS(model=object()),{})

    def test_mla_source_checks_required_and_mrope_branch_refused(self):
        model_type=type('DeepseekV2ForCausalLM',(),{'__module__':'vllm.model_executor.models.deepseek_v2','forward':lambda self:None})
        runner_type=type('NPUModelRunner',(),{'__module__':'vllm_ascend.worker.model_runner_v1'})
        r=runner_type();r.model=model_type();r.device='npu:0';r.use_async_scheduling=False;r.speculative_config=None
        r.supports_mm_inputs=False;r.uses_mrope=False;r.enable_prompt_embeds=False
        r.vllm_config=NS(model_config=NS(enforce_eager=True,multimodal_config=None))
        for n in ('execute_model','sample_tokens','_preprocess','_model_forward'):setattr(r,n,lambda:None)
        base='demoncat.runtime.vllm_ascend.worker_extension.v026_text_observer.'
        with patch(base+'_verify_function') as source,patch(base+'_parallel_report') as parallel:
            self.assertEqual(validate_continuous_model(r,{}),'deepseek_v2')
            self.assertEqual(source.call_count,5);parallel.assert_called_once()
            self.assertTrue(any(row['sha256']=='f22d4458a604875fbc3fb194119519734b3f2b2460276ef2091419ed6988f1be' for row in source.call_args.args[1]))
            r.uses_mrope=True
            with self.assertRaisesRegex(RuntimeError,'TOKEN_ONLY'):validate_continuous_model(r,{})
            self.assertEqual(source.call_count,5)
