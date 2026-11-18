"""Root-queued public observe protocol with a preloaded stock CANN observer.

Two ordinary 2048-token prefill requests; no private vendor or F stage.
Complete raw runtime/registered ELF files remain beside every public receipt.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import natural_attention_worker_core_probe as common

MECHANISMS=('28.fia.legal_tiling','28.fia.denominator_replay')
SUCCESS='PUBLIC_ATTENTION_STOCK_CENSUS_PROTOCOL_RECORDED_PENDING_RAW_REVIEW'


def setup(worker, expected, mechanism, prompt, provenance, observer):
    from demoncat.runtime.vllm_ascend.worker_extension.attention28_stock_census import StockTndPrefillCensus
    from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import install_attention_serving_binding
    worker=common.actual(worker)
    observed=worker.demoncat_worker_identity()
    if observed.get('status') != 'WORKER_IDENTIFIED' or observed.get('quarantined') or observed['worker'] != expected:
        raise RuntimeError('STOCK_CENSUS_DEPLOYMENT_WORKER_CHANGED')
    native=StockTndPrefillCensus(mechanism=mechanism,observer_receipt=observer['path'],
        observer_receipt_sha256=observer['sha256'],evidence_dir=observer['calls'])
    install_attention_serving_binding(worker,mechanism,prompt_token_ids=prompt,provenance=provenance,native=native)
    return dict(worker=observed['worker'],mechanism=mechanism,observer_only=True,native_loaded_by_probe=True,
        native_state=native.state(),fault_execution_ready=False,qualification_granted=False)


def bootstrap(args):
    """No framework import or device call before the actual loader preload."""
    if common.sha(args.observer_receipt) != args.observer_receipt_sha256:
        raise RuntimeError('EXACT_OBSERVER_RECEIPT_REQUIRED')
    receipt=json.loads(args.observer_receipt.read_bytes());library=receipt['observer_library']
    if (receipt.get('observer_compiled') is not True or receipt.get('actual_npu_calls') != 0
            or receipt.get('observer_memory_selftest',{}).get('exit_code') != 0
            or common.sha(library['path']) != library['sha256']):
        raise RuntimeError('UNCHANGED_CPU_COMPILED_OBSERVER_REQUIRED')
    if not args.child:
        if (os.environ.get('LD_PRELOAD') or os.environ.get('ASCEND_CUSTOM_OPP_PATH')
                or any(name.startswith(('NES_FA_','FAULT28_','DCAT33_','DCAT_F33_','DCAT_ATTN_')) for name in os.environ)
                or any(name in sys.modules for name in ('torch','torch_npu','vllm'))):
            raise RuntimeError('CLEAN_PREFRAMEWORK_STOCK_CENSUS_PARENT_REQUIRED')
        args.out.mkdir(parents=True,exist_ok=False);runtime=args.out/'runtime';runtime.mkdir()
        env=dict(os.environ,LD_PRELOAD=library['path'],DCAT_ATTN_OBSERVER_DIR=str(runtime),
            ACLNN_CACHE_LIMIT='0',TASK_QUEUE_ENABLE='0',PYTHONDONTWRITEBYTECODE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0')
        os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--child'],env)
    if Path(os.environ.get('DCAT_ATTN_OBSERVER_DIR','')).resolve() != (args.out/'runtime').resolve():
        raise RuntimeError('ACTUAL_STOCK_RUNTIME_PATH_CHANGED')
    return receipt


def run(args, observer_receipt):
    report=dict(status='STARTED',mechanism=args.mechanism,probe_sha256=common.sha(__file__),
        mode='ACTUAL_TND_PREFILL_STOCK_NATIVE_CENSUS_ONLY',observer_receipt_sha256=args.observer_receipt_sha256,
        observer_library_sha256=observer_receipt['observer_library']['sha256'],native_loaded_by_probe=True,
        forced_private_vendor=False,fault_authorized=False,public_qualification_granted=False,C1=False,C2=False,C3=False,
        E5='NOT_ASSESSED',stock_pod_decoded=False,all_runtime_entrypoints_covered=False)
    client=None
    try:
        from vllm import LLM,SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core
        weights={path.name:dict(sha256=common.sha(path),bytes=path.stat().st_size)
            for path in sorted({*args.model.glob('*.safetensors'),*args.model.glob('pytorch_model*.bin')})}
        if not weights:raise RuntimeError('ACTUAL_PRETRAINED_WEIGHTS_REQUIRED')
        config=json.loads((args.model/'config.json').read_bytes())
        if (config.get('num_hidden_layers') != 28 or config.get('num_attention_heads') != 16
                or config.get('num_key_value_heads') != 8 or config.get('hidden_size') != 1024):
            raise RuntimeError('SOURCE71_MODEL_GEOMETRY_REQUIRED_FOR_THIS_CASE')
        report['model_identity']=dict(weights=weights,config_sha256=common.sha(args.model/'config.json'))
        llm=LLM(model=str(args.model),dtype='float16',tensor_parallel_size=1,enforce_eager=True,
            max_model_len=2304,max_num_seqs=1,max_num_batched_tokens=2304,block_size=16,num_gpu_blocks_override=160,
            enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,
            gpu_memory_utilization=args.memory,seed=42,disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        words=llm.get_tokenizer().encode('Explain how attention reads a sequence of tokens. ',add_special_tokens=True)
        tokens=(words*((2048+len(words)-1)//len(words)))[:2048];prompt=dict(prompt_token_ids=tokens)
        params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True)
        healthy,=llm.generate([prompt],params,use_tqdm=False);report['healthy_before']=list(healthy.outputs[0].token_ids)
        provenance=dict(model_identity=report['model_identity'],prompt_token_ids=tokens,standard_worker=True,
            constructed_replay=False,probe_sha256=report['probe_sha256'],observer_only=True,
            observer_receipt_sha256=args.observer_receipt_sha256)
        observer=dict(path=str(args.observer_receipt),sha256=args.observer_receipt_sha256,calls=str(args.out/'native_calls'))
        def deploy(expected, config):
            rows=llm.collective_rpc(setup,args=(expected,config.mechanism,tokens,provenance,observer),timeout=300)
            if len(rows) != 1 or rows[0]['worker'] != expected:raise RuntimeError('TP1_STOCK_CENSUS_DEPLOYMENT_CHANGED')
            report['setup']=rows[0];return {}
        fault,variant=args.mechanism.split('.',1)
        client=common.WorkerFaultClient(common.HostAdapter(llm,prepare_binding=deploy,timeout=300),common.FaultConfig(int(fault),variant))
        common.drive(llm,client,prompt=prompt,params=params,out=args.out,report=report,expected_native_observer=True)
        if report.get('setup') and not client.snapshot()['ownership_retained']:
            removed=llm.collective_rpc(common.remove,args=(client.snapshot()['worker'],args.mechanism),timeout=300)
            if len(removed) != 1 or removed[0]['status'] != 'SERVING_DEPLOYMENT_REMOVED':
                raise RuntimeError('STOCK_CENSUS_BINDING_REMOVAL_UNCONFIRMED')
            report['binding_removal']=removed[0]
        if report['status'] == common.SUCCESS:
            counts=[sum(len(f['calls']) for f in row['worker']['forwards']) for row in report['requests']]
            if counts != [28,28]:raise RuntimeError('ACTUAL_2_BY_28_CALL_CENSUS_INCOMPLETE')
            report['stock_launch_census']=report['final_cleanup']['evidence']['stock_launch_census']
            report['actual_calls_per_request']=counts;report['status']=SUCCESS
    except BaseException as error:report.update(status='REFUSED_OR_FAILED',error=repr(error),traceback=traceback.format_exc())
    finally:common.save(args.out/'result.json',report)
    print(json.dumps({k:report[k] for k in ('status','error','stock_launch_census') if k in report}))
    return 0 if report['status'] == SUCCESS else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mechanism',choices=MECHANISMS,default=MECHANISMS[1])
    for name in ('model','out','observer-receipt'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--observer-receipt-sha256',required=True)
    parser.add_argument('--memory',type=float,default=.25)
    parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.out.resolve().is_relative_to('/evidence') or not 0 < args.memory <= .5:
        parser.error('Private /evidence output and bounded memory required')
    return run(args,bootstrap(args))


if __name__ == '__main__':raise SystemExit(main())
