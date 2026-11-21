"""Source-pinned actual model families for the continuous request observer."""
import json
from .serving_context import require


def validate_continuous_model(runner,prompts):
    from .v026_text_observer import SourceBoundTextObserver,PROFILES,_verify_function,_parallel_report
    model=runner.model;kind=(type(model).__module__,type(model).__name__)
    if kind==('vllm.model_executor.models.qwen3_5','Qwen3_5ForConditionalGeneration'):
        SourceBoundTextObserver(runner,prompts,source_profile='v026_installed_c9c9432b').validate()
        return 'qwen35'
    require(kind==('vllm.model_executor.models.deepseek_v2','DeepseekV2ForCausalLM'),'CONTINUOUS_MODEL_NOT_ADAPTED')
    require((type(runner).__module__,type(runner).__name__)==('vllm_ascend.worker.model_runner_v1','NPUModelRunner'),
            'MLA_REVIEWED_RUNNER_REQUIRED')
    require(str(runner.device).split(':')[0]=='npu' and runner.use_async_scheduling is False
        and runner.speculative_config is None and runner.vllm_config.model_config.enforce_eager is True,
        'MLA_SYNC_EAGER_NONSPEC_REQUIRED')
    require(runner.supports_mm_inputs is False and runner.uses_mrope is False
        and runner.enable_prompt_embeds is False and runner.vllm_config.model_config.multimodal_config is None,
        'MLA_REAL_TOKEN_ONLY_BRANCH_REQUIRED')
    rows=json.loads(PROFILES['v026_installed_c9c9432b'].read_bytes())['files']
    rows=rows+[dict(path='vllm/model_executor/models/deepseek_v2.py',
        sha256='f22d4458a604875fbc3fb194119519734b3f2b2460276ef2091419ed6988f1be')]
    for function in (runner.execute_model,runner.sample_tokens,runner._preprocess,runner._model_forward,model.forward):
        _verify_function(function,rows)
    _parallel_report(runner,rows,'v026_installed_c9c9432b')
    return 'deepseek_v2'
