"""Wheel-installed positive common Worker acceptance, with genuine enqueue IDs."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import traceback
import faulthandler

EXTENSION='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension'
PROMPT='Explain in one sentence why the sky appears blue during the day.'


def one_rpc(llm, method, *args):
    rows=llm.collective_rpc(method,args=args,timeout=300)
    if len(rows)!=1:raise RuntimeError('TP1_REPLY_REQUIRED')
    return rows[0]


def prepare_output_projection_nz(worker):
    """Explicit deployment before B/F, not part of the fault transaction.

    Some pretrained models tie the output to an embedding, for which vLLM's
    NZ2 flag does not convert storage. Preserve that alias and every logical
    element, and disclose this controlled deployment instead of claiming the
    default model route reaches WeightNz.
    """
    import torch
    import torch_npu
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import discover_logits_boundary
    from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
    model = worker.model_runner.model
    config = worker.model_runner.vllm_config.model_config.hf_config
    boundary = discover_logits_boundary(model, actual_logits_processor_type=LogitsProcessor,
                                       actual_vocab_size=config.vocab_size)
    head = boundary['lm_head']
    weight = head.weight
    if weight.dtype != torch.bfloat16 or weight.ndim != 2 or not weight.is_contiguous():
        raise RuntimeError('CONTROLLED_NZ_REQUIRES_CONTIGUOUS_BF16_OUTPUT_WEIGHT')
    original_format = int(torch_npu.get_npu_format(weight))
    if original_format not in (2, 29):
        raise RuntimeError('UNREVIEWED_OUTPUT_WEIGHT_FORMAT')
    aliases = [(name, parameter) for name, parameter in model.named_parameters(remove_duplicate=False)
               if parameter is weight]
    before, before_hash, finite = complete_tensor(weight)
    if not finite or weight.untyped_storage().nbytes() > 512 * 1024**2:
        raise RuntimeError('BOUNDED_FINITE_PRETRAINED_WEIGHT_REQUIRED')
    with torch.inference_mode():
        converted = torch_npu.npu_format_cast(weight.detach(), 29)
        after, after_hash, after_finite = complete_tensor(converted)
        if not after_finite or not torch.equal(before, after) or before_hash != after_hash:
            raise RuntimeError('NZ_CONVERSION_CHANGED_LOGICAL_PRETRAINED_WEIGHT')
        weight.data = converted
    current = dict(model.named_parameters(remove_duplicate=False))
    if (head.weight is not weight or any(current.get(name) is not parameter for name, parameter in aliases)
            or int(torch_npu.get_npu_format(weight)) != 29):
        raise RuntimeError('NZ_PREPARATION_CHANGED_PARAMETER_ALIAS')
    # A tied VocabParallelEmbedding now returns private-format activations.
    # Its actual AddRmsNorm consumer requires ND. Normalize that deployment
    # boundary identically in B/F/OFF; verify every logical element, never
    # perturb an output to manufacture the fault effect.
    # A tied Parameter need not mean the embedding and output head are the
    # same Module (Llama/SmolLM2 uses two owners). Resolve actual owners by
    # Parameter identity and implementation type, never by a supplied layer.
    embedding_owners = [(name, module) for name, module in model.named_modules()
                        if isinstance(module, VocabParallelEmbedding)
                        and getattr(module, 'weight', None) is weight]
    if not embedding_owners:
        raise RuntimeError('ACTUAL_TIED_EMBEDDING_OWNER_REQUIRED')
    owner_names = {id(module): name for name, module in embedding_owners}
    normalization = []
    def embedding_output_nd(module, args, output):
        if (id(module) not in owner_names or len(args) != 1 or args[0].ndim != 1
                or args[0].dtype not in (torch.int32, torch.int64)
                or not isinstance(output, torch.Tensor) or output.ndim != 2
                or list(output.shape) != [args[0].numel(), weight.shape[1]]):
            raise RuntimeError('UNREVIEWED_TIED_EMBEDDING_CALL')
        source_format = int(torch_npu.get_npu_format(output))
        before_cpu, before_digest, before_finite = complete_tensor(output)
        value = torch_npu.npu_format_cast(output, 2)
        after_cpu, after_digest, after_finite = complete_tensor(value)
        if not before_finite or not after_finite or before_digest != after_digest or not torch.equal(before_cpu, after_cpu):
            raise RuntimeError('EMBEDDING_ND_CONVERSION_CHANGED_LOGICAL_VALUES')
        normalization.append(dict(module_path=owner_names[id(module)],
            source_format=source_format, destination_format=int(torch_npu.get_npu_format(value)),
            logical_sha256=before_digest, exact_logical_equality=True, shape=list(value.shape)))
        Path('/evidence/deployment_embedding_normalization.json').write_text(json.dumps(normalization,indent=2)+'\n')
        return value
    handles = [module.register_forward_hook(embedding_output_nd) for _, module in embedding_owners]
    worker._worker_positive_nz_deployment = (handles, normalization)
    record = dict(scope='EXPLICIT_NZ_OUTPUT_DEPLOYMENT_BEFORE_BASELINE_NOT_FAULT',
        source_format=original_format, actual_format=29, shape=list(weight.shape),
        logical_sha256_before=before_hash, logical_sha256_after=after_hash,
        complete_logical_equality=True, parameter_aliases_preserved=[name for name,_ in aliases],
        actual_embedding_owner_paths=[name for name, _ in embedding_owners],
        boundary=boundary['report'], restored_by_fault_off=False,
        activation_format_contract='Tied embedding output converted to ND with per-call exact logical equality',
        off_reference='Same configured NZ baseline; this temporary Worker is disposed after acceptance')
    Path('/evidence/deployment_preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def setup_worker(worker, tokens, collect, baseline_control, mechanism):
    from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase
    # Callable collective_rpc receives WorkerWrapperBase, while named Extension
    # RPCs bind to its actual Worker. Freeze the actual owner for both paths.
    if isinstance(worker, WorkerWrapperBase):
        worker = worker.worker
    if not isinstance(worker, WorkerBase):
        raise RuntimeError('ACTUAL_STANDARD_VLLM_WORKER_REQUIRED')
    from demoncat.runtime.vllm_ascend.worker_extension import runtime
    if mechanism == '39.matmul_range':
        from demoncat.runtime.vllm_ascend.worker_extension.serving import install_fault39_serving_binding as install
        import demoncat_fault39.serving_qualification as qualification
        extra = {}
    else:
        from demoncat.runtime.vllm_ascend.worker_extension.serving20 import install_fault20_serving_binding as install
        import fault_injection.serving_qualification as qualification
        extra = dict(reviewed_manifest_path='/native/serving_review.json', first_live_validation=True)
        from fault_injection.environment import inspect_environment
        review = json.loads(Path(extra['reviewed_manifest_path']).read_text())
        Path('/evidence/environment_before.json').write_text(json.dumps(
            inspect_environment(additional_libraries=review['required_libraries']), indent=2)+'\n')
    import fault_injection
    modules=[]
    for module in (runtime,qualification,fault_injection):
        path=Path(inspect.getfile(module)).resolve()
        if not path.is_relative_to(Path('/installed')):raise RuntimeError('SOURCE_IMPORT_NOT_INSTALLED_WHEEL')
        modules.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for obj in (worker,worker.model_runner):
        if 'execute_model' in vars(obj):raise RuntimeError('CORE_METHOD_INSTANCE_OVERRIDE')
    return dict(binding=install(worker,prompt_token_ids=tokens,
        evidence_dir='/evidence/serving',collect_statistics=collect,baseline_control=baseline_control,**extra),modules=modules,
        worker_execute_source=inspect.getsourcefile(worker.execute_model),
        runner_execute_source=inspect.getsourcefile(worker.model_runner.execute_model))


def unset_worker(worker, mechanism):
    from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase
    if isinstance(worker, WorkerWrapperBase):
        worker = worker.worker
    if not isinstance(worker, WorkerBase):
        raise RuntimeError('ACTUAL_STANDARD_VLLM_WORKER_REQUIRED')
    if mechanism == '39.matmul_range':
        from demoncat.runtime.vllm_ascend.worker_extension.serving import remove_fault39_serving_binding as remove
    else:
        from demoncat.runtime.vllm_ascend.worker_extension.serving20 import remove_fault20_serving_binding as remove
    return remove(worker)


def result_record(row):
    return dict(external_request_id=row.request_id,prompt_token_ids=row.prompt_token_ids,
                outputs=[dict(tokens=list(x.token_ids),text=x.text,finish_reason=x.finish_reason) for x in row.outputs])


def main():
    # Read-only diagnostics for an owned temporary acceptance process. Never
    # intercept or replace Worker/runner methods to recover a failed request.
    faulthandler.enable()
    p=argparse.ArgumentParser();p.add_argument('--model',default='/model');p.add_argument('--statistics',action='store_true')
    p.add_argument('--baseline-control',action='store_true')
    p.add_argument('--mechanism',choices=['39.matmul_range','20.buffer_substitute'],default='39.matmul_range')
    p.add_argument('--operator',choices=['aclnnMatmulWeightNz','aclnnQuantMatmulWeightNz'],default='aclnnMatmulWeightNz')
    p.add_argument('--memory',type=float,default=.35)
    p.add_argument('--weight-nz-output',action='store_true')
    p.add_argument('--out',type=Path,default=Path('/evidence/result.json'));a=p.parse_args()
    from vllm import LLM,SamplingParams
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core,read_enqueued_identity
    if a.mechanism == '39.matmul_range':
        from demoncat_fault39.serving_qualification import PHASES
    else:
        from fault_injection.serving_qualification import PHASES
    report=dict(status='STARTING',scope='REAL_PRETRAINED_TP1_FRESH_PREFILL_COMMON_WORKER',phases=[],statistics=a.statistics,
                baseline_control=a.baseline_control,mechanism=a.mechanism,operator=a.operator)
    def save():a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    save();llm=None;ticket=tenant=None;seen=[]
    try:
        llm=LLM(model=a.model,dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,
            max_model_len=128,max_num_seqs=1,max_num_batched_tokens=128,enable_prefix_caching=False,
            enable_chunked_prefill=False,async_scheduling=False,gpu_memory_utilization=a.memory,
            worker_extension_cls=EXTENSION,disable_log_stats=True)
        require_host_driven_core(llm)
        if a.weight_nz_output:
            if a.mechanism!='20.buffer_substitute' or a.operator!='aclnnMatmulWeightNz':
                raise RuntimeError('EXPLICIT_NZ_PREPARATION_ONLY_FOR_WEIGHTNZ')
            report['deployment_preparation']=one_rpc(llm,prepare_output_projection_nz)
            save()
        tokens=llm.get_tokenizer().encode(PROMPT,add_special_tokens=True)
        sampling=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True)
        prompt={'prompt_token_ids':tokens}
        baseline,=llm.generate([prompt],sampling,use_tqdm=False)
        report['healthy_before']=result_record(baseline)
        report['setup']=one_rpc(llm,setup_worker,tokens,a.statistics,a.baseline_control,a.mechanism)
        identity=one_rpc(llm,'demoncat_worker_identity')
        report['identity']=identity
        attached=one_rpc(llm,'demoncat_attach',dict(tenant_id='worker-positive-20261117',worker=identity['worker']))
        if attached['status']!='ATTACHED':raise RuntimeError(str(attached))
        tenant=attached['ticket']
        prepare_settings = {'operator': a.operator} if a.mechanism == '20.buffer_substitute' else {}
        prepared=one_rpc(llm,'demoncat_fault_rpc','prepare',dict(tenant,mechanism=a.mechanism,
            implementation_version='1',owner='positive-'+a.mechanism,settings=prepare_settings))
        report['prepare']=prepared
        if prepared['status']!='ACTION_COMPLETED':raise RuntimeError(str(prepared))
        ticket=prepared['ticket']
        for action in ('register','check','execute'):
            payload=(dict(ticket,settings=dict(pattern='half',amplitude_policy='auto'))
                     if action=='register' and a.mechanism=='39.matmul_range' else ticket)
            reply=one_rpc(llm,'demoncat_fault_rpc',action,payload);report[action]=reply;save()
            if reply['status']!='ACTION_COMPLETED':raise RuntimeError(str(reply))
        for phase in PHASES:
            if llm.llm_engine.has_unfinished_requests():raise RuntimeError('UNRELATED_OR_UNFINISHED_REQUEST')
            if llm.reset_prefix_cache(reset_running_requests=False) is not True:raise RuntimeError('CACHE_RESET_NOT_CONFIRMED')
            ids=llm.enqueue([prompt],sampling,use_tqdm=False)
            mapping=read_enqueued_identity(llm,ids,seen_internal_request_ids=seen);seen.append(ids[0])
            row=dict(phase=phase,mapping=mapping);report['phases'].append(row);save()
            row['advance']=one_rpc(llm,'demoncat_fault_rpc','advance',dict(ticket,settings={'request_id':ids[0]}));save()
            if row['advance']['status']!='ACTION_COMPLETED':raise RuntimeError(str(row['advance']))
            outputs=llm.wait_for_completion(use_tqdm=False)
            if len(outputs)!=1:raise RuntimeError('ONE_COMPLETED_REAL_REQUEST_REQUIRED')
            row['response']=result_record(outputs[0])
            if row['response']['external_request_id']!=mapping['external_request_id']:raise RuntimeError('ACTUAL_DRIVER_ID_MAPPING_CHANGED')
            row['query']=one_rpc(llm,'demoncat_fault_query',ticket);save()
            if row['query'].get('quarantined'):raise RuntimeError('SERVING_QUALIFICATION_FAILED')
        report['finish']=one_rpc(llm,'demoncat_fault_rpc','finish',ticket)
        if report['finish']['status']!='ACTION_COMPLETED':raise RuntimeError('FINISH_NOT_VERIFIED')
        q=report['finish']['native']['qualification']
        expected_status='BASELINE_CONTROL_COMPLETED' if a.baseline_control else 'SCOPED_ATTEMPT_COMPLETED'
        if q['status']!=expected_status or not q['recovery_verified']:raise RuntimeError('QUALIFICATION_OR_RECOVERY_INCOMPLETE')
        report['status']='PASS_COMMON_WORKER_BASELINE_CONTROL' if a.baseline_control else 'PASS_COMMON_WORKER_NATIVE_BFOFF'
    except BaseException as exc:
        report.update(status='FAILED_WITH_EVIDENCE',error=repr(exc),traceback=traceback.format_exc())
        save()
        if llm is not None and ticket is not None:
            try:report['failure_query']=one_rpc(llm,'demoncat_fault_query',ticket)
            except BaseException as query_error:report['failure_query_error']=repr(query_error)
        # The failed request may remain queued in the standard in-process
        # scheduler. Abort only IDs returned by this driver's own enqueue calls.
        if llm is not None and seen:
            try:
                llm.llm_engine.abort_request(seen, internal=True)
                report['owned_request_abort']={'actual_internal_ids':seen,
                    'unfinished_after_abort':llm.llm_engine.has_unfinished_requests()}
            except BaseException as abort_error:report['owned_request_abort_error']=repr(abort_error)
        save()
    finally:
        if llm is not None and ticket is not None:
            try:
                report['unregister']=one_rpc(llm,'demoncat_fault_rpc','unregister',ticket)
                if report['unregister']['status']!='ACTION_COMPLETED':raise RuntimeError('OWNER_REMAINS_QUARANTINED')
                report['detach']=one_rpc(llm,'demoncat_detach',tenant)
                report['unbind']=one_rpc(llm,unset_worker,a.mechanism)
                after,=llm.generate([prompt],sampling,use_tqdm=False)
                report['healthy_after']=result_record(after)
                if report['healthy_after']['outputs']!=report['healthy_before']['outputs']:raise RuntimeError('POST_CLEAN_GENERATION_CHANGED')
            except BaseException as cleanup_error:
                report['cleanup_error']=repr(cleanup_error)
                report['status']='FAILED_WITH_EVIDENCE'
        save()
    return 0 if report['status'] in ('PASS_COMMON_WORKER_NATIVE_BFOFF','PASS_COMMON_WORKER_BASELINE_CONTROL') else 2


if __name__=='__main__':raise SystemExit(main())
