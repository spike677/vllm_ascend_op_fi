"""Prepare one reviewed, zero-F public TND census case; never run a device."""
import argparse
import json
from pathlib import Path
import shlex
import sys

import arch22_normal_bridge_evidence as evidence
from queue_contract import validate_cases

WORKER=evidence.WORKER
EXPERIMENT=evidence.EXPERIMENT
REQUIRED=(EXPERIMENT+'prepare_attention28_stock_census.py',
    EXPERIMENT+'natural_attention28_stock_census_probe.py',EXPERIMENT+'natural_attention_worker_core_probe.py',
    EXPERIMENT+'arch22_normal_bridge_evidence.py',EXPERIMENT+'queue_contract.py',
    *(WORKER+name+'.py' for name in ('attention28_stock_census','attention_stock_launch','attention_native',
        'attention_execution','serving_attention','providers/attention_serving','attention28_prefill',
        'providers/tensor_modules','paged33_boundary_probe','paged33_audit','paged33_tnd','paged33_loading',
        'paged33_arch22_bridge','serving_context','call_binding','identity','batch_context','config','host','runtime')),
    evidence.NATIVE+'launch.py',evidence.NATIVE+'launch_observer.cpp')


def frozen_source(root, expected):
    root=Path(root).resolve(strict=True);path=root/'source_manifest.json'
    evidence.need(evidence.sha(path) == expected,'EXACT_STOCK_CENSUS_SOURCE_MANIFEST_REQUIRED')
    data=evidence.read(path);files=data['files'];index={row['path']:row for row in files}
    evidence.need(data['schema'] == 'demoncat.c2c3.source.v1' and len(index) == len(files),'SOURCE_MANIFEST_AMBIGUOUS')
    pins={}
    for name in REQUIRED:
        p=evidence.child(root,name);row=index[name]
        evidence.need(evidence.sha(p) == row['sha256'] and p.stat().st_size == row['bytes'],
                      'FROZEN_STOCK_CENSUS_SOURCE_DIFFERS:'+name)
        pins[name]=row['sha256']
    evidence.need(evidence.sha(__file__) == pins[EXPERIMENT+Path(__file__).name]
        and evidence.sha(evidence.__file__) == pins[EXPERIMENT+'arch22_normal_bridge_evidence.py'],
        'AMBIENT_STOCK_CENSUS_PREPARER_DIFFERS')
    return dict(root=root,manifest=data,manifest_sha256=expected,pins=pins)


def make_case(args, source, observer, installed):
    evidence.need(args.card in (2,5) and type(args.timeout) is int and 1 <= args.timeout <= 1800,
                  'STOCK_CENSUS_RESOURCE_BUDGET_INVALID')
    name='attention28-tnd-stock-native-census'
    argv=['python3','-B','/repo/'+EXPERIMENT+'natural_attention28_stock_census_probe.py',
        '--mechanism','28.fia.denominator_replay','--model','/model','--out','/evidence/probe',
        '--observer-receipt','/build/attention-preflight/result.json','--observer-receipt-sha256',observer['sha256']]
    mounts=[[str(args.source),'/repo','ro'],[str(args.installed),'/installed','ro'],[str(args.model),'/model','ro'],
        [str(observer['folder']),'/build/attention-preflight','ro']]
    if observer['alias'] != '/build/attention-preflight':mounts.append([str(observer['folder']),observer['alias'],'ro'])
    env=dict(PYTHONPATH='/installed:/msprobe_packages',PYTHONDONTWRITEBYTECODE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_ASCEND_ENABLE_NZ='0',VLLM_ALLOW_INSECURE_SERIALIZATION='1',
        OMP_NUM_THREADS='4',VLLM_HOST_IP='127.0.0.1',GLOO_SOCKET_IFNAME='lo',HCCL_CONNECT_TIMEOUT='120',
        HCCL_EXEC_TIMEOUT='120',ACLNN_CACHE_LIMIT='0',TASK_QUEUE_ENABLE='0')
    case=dict(id=name,image=evidence.IMAGE,timeout=args.timeout,mounts=mounts,env=env,
        command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '+shlex.join(argv),
        review=dict(experiment_id=name,provider='28.fia.denominator_replay / public observe only; no replay policy',
            model=str(args.model),cann='8.5.1',soc='Ascend910B4/A2',physical_cards=[args.card],ranks=[0],
            worker='Standard Qwen TP1 eager; two natural full 2048-token prefill requests',
            source_commit=source['manifest']['base_commit'],source_manifest_sha256=source['manifest_sha256'],
            required_sources=source['pins'],installed_required_sources_sha256=evidence.digest(installed),
            build_fingerprint=dict(observer_only=True,receipt_sha256=observer['sha256'],
                library_sha256=observer['receipt']['observer_library']['sha256'],private_native_build=None),
            expected_hits='0 faults; retain all 2 x 28 actual calls, matching stock scopes and every unobserved edge',
            resource_budget=dict(timeout_seconds=args.timeout,memory_gib=48,cpus=8),
            recovery_plan='Drain actual owned scopes, remove owned hooks/wrappers, unregister/detach; same Worker ordinary token health.',
            diagnostic_only=True,statistics=False,fault_authorized=False,C1=False,C2=False,C3=False,E5='NOT_ASSESSED',
            stock_pod_decoded=False,forced_private_vendor=False,
            limits=['Only intercepted stock launch entrypoints are observed; unknown entries remain unobserved.',
                    'No assumed stock ABI or final Native QKV pointer/POD decode.',
                    'This public route observation does not qualify either FIA fault or the distinct FA API.']))
    return validate_cases([case],[args.card])


def prepare(args):
    for key in ('source','installed','observer','model'):setattr(args,key,getattr(args,key).resolve(strict=True))
    source=frozen_source(args.source,args.source_manifest_sha256)
    installed=evidence.installed_sources(source,args.installed)
    observer=evidence.observer_bundle(args.observer,source)
    return make_case(args,source,observer,installed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','installed','observer','model','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-manifest-sha256',required=True)
    parser.add_argument('--card',type=int,choices=(2,5),default=2)
    parser.add_argument('--timeout',type=int,default=1800)
    args=parser.parse_args();sys.dont_write_bytecode=True;sys.addaudithook(evidence.prohibit_runtime)
    evidence.need(not args.out.exists(),'FRESH_STOCK_CENSUS_CASE_MANIFEST_REQUIRED')
    cases=prepare(args);args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:json.dump(cases,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='ZERO_F_STOCK_CENSUS_CASE_PREPARED',cases=1,path=str(args.out),
        sha256=evidence.sha(args.out),actual_npu_calls=0,fault_authorized=False)))
    return 0


if __name__ == '__main__':raise SystemExit(main())
