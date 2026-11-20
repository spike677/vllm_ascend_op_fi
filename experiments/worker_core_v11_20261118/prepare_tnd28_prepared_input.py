"""Prepare one bounded normal input-connection case; no process/device calls."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/c2c3_closure_20261117')]
from experiments.c2c3_closure_20261117 import arch22_normal_bridge_evidence as E
from experiments.c2c3_closure_20261117 import prepare_attention28_stock_census as stock
from experiments.c2c3_closure_20261117 import review_tnd28_prepared_input as R
from experiments.c2c3_closure_20261117.queue_contract import validate_cases

EXP='experiments/worker_core_v11_20261118/'
FIA='fault_injection/native_adapters/attention28/fia/'
EXTRA=(EXP+'tnd28_research_observation.py',EXP+'prepare_tnd28_prepared_input.py',EXP+'natural_tnd28_prepared_input_probe.py',
    E.EXPERIMENT+'review_tnd28_prepared_input.py',E.EXPERIMENT+'attention28_census_raw.py',
    E.WORKER+'attention28_prepared_input.py',E.WORKER+'attention28_replay_binding.py',
    E.WORKER+'attention28_typed_plan.py','fault_injection/fia_input_journal.py',
    'fault_injection/quant_container_native.py','fault_injection/metadata_arrays.py',
    *(FIA+n for n in ('tnd28_input_compile.py','tnd28_input_observation.h','test_tnd28_input_observation.cpp',
        'tnd28_execute_borrow.h','test_tnd28_execute_borrow.cpp','tnd28_denominator_build.py','build_fresh.py')))


def frozen(source,expected):
    result=stock.frozen_source(source,expected);pins={r['path']:r for r in result['manifest']['files']}
    for name in EXTRA:
        path=E.child(source,name);pin=pins[name]
        E.need(path.stat().st_size==pin['bytes'] and E.sha(path)==pin['sha256'],'FROZEN_INPUT_OBSERVER_SOURCE_CHANGED:'+name)
        result['pins'][name]=pin['sha256']
    E.need(E.sha(__file__)==result['pins'][EXP+Path(__file__).name],'AMBIENT_PREPARER_CHANGED')
    return result


def installed(source,root):
    found={}
    for name,digest in source['pins'].items():
        if not name.endswith('.py'):continue
        relative=name.removeprefix('src/') if name.startswith('src/demoncat/') else name if name.startswith('fault_injection/') else None
        if relative is not None:
            E.need(E.sha(E.child(root,relative))==digest,'INSTALLED_INPUT_OBSERVER_SOURCE_CHANGED:'+relative);found[relative]=digest
    return found


def libraries(path,expected):
    E.need(E.sha(path)==expected,'EXPECTED_PRODUCTION_LIBRARY_MANIFEST_CHANGED');value=E.read(path)
    E.need(value['schema']=='demoncat.tnd28.library_bytes.v1' and 1<=len(value['files'])<=24,'BOUNDED_PRODUCTION_LIBRARY_MANIFEST')
    mounts=[];seen=set()
    for row in value['files']:
        target=PurePosixPath(row['recorded_path']);source=Path(row['path']).resolve(strict=True)
        E.need(str(target).startswith(('/usr/local/Ascend/cann-8.5.1/','/vllm-workspace/vllm-ascend/'))
            and '..' not in target.parts and '\\' not in str(target) and str(target) not in seen,
            'EXACT_PRODUCTION_LIBRARY_PATH_REQUIRED')
        E.need(source.stat().st_size==row['bytes']<=256*1024**2 and E.sha(source)==row['sha256'],
            'PRODUCTION_LIBRARY_STAGED_BYTES_CHANGED:'+str(target))
        seen.add(str(target));mounts.append([str(source),str(target),'ro'])
    return value,mounts


def make_case(args,source,observer,build,library_mounts,installed_pins):
    E.need(args.card in (2,5) and type(args.timeout) is int and 1<=args.timeout<=1800,'ONE_CARD_ORIGINAL_TIMEOUT_BOUND')
    name='tnd28-one-prepared-input-normal'
    if getattr(args,'execute_borrow',False):name='tnd28-one-prepared-input-borrow-normal'
    mounts=[[str(args.source),'/repo','ro'],[str(args.installed),'/installed','ro'],[str(args.model),'/model','ro'],
        [str(args.build_root),'/native-run','ro'],[str(args.vendor),'/native','ro'],
        [str(observer['folder']),'/build/attention-preflight','ro']]
    if observer['alias']!='/build/attention-preflight':mounts.append([str(observer['folder']),observer['alias'],'ro'])
    mounts+=library_mounts
    argv=['python3','-B','/repo/'+EXP+'natural_tnd28_prepared_input_probe.py','--model','/model','--out','/evidence/probe',
        '--build','/native-run/evidence/completion.json','--build-sha256',args.build_sha256,'--vendor','/native',
        '--source','/repo','--source-manifest-sha256',args.source_manifest_sha256,
        '--observer-receipt','/build/attention-preflight/result.json','--observer-receipt-sha256',observer['sha256']]
    if getattr(args,'execute_borrow',False):argv.append('--execute-borrow')
    case=dict(id=name,image=E.IMAGE,timeout=args.timeout,mounts=mounts,
        env=dict(PYTHONPATH='/installed:/repo:/repo/src:/repo/fault39/src:/msprobe_packages',PYTHONDONTWRITEBYTECODE='1',
            VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_ASCEND_ENABLE_NZ='0',
            VLLM_ALLOW_INSECURE_SERIALIZATION='1',OMP_NUM_THREADS='4',VLLM_HOST_IP='127.0.0.1',GLOO_SOCKET_IFNAME='lo',
            HCCL_CONNECT_TIMEOUT='120',HCCL_EXEC_TIMEOUT='120',ACLNN_CACHE_LIMIT='0',TASK_QUEUE_ENABLE='0'),
        command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '+shlex.join(argv),
        review=dict(experiment_id=name,provider='28.fia.denominator_replay / one normal typed-input observation, no policy',
            model=str(args.model),cann='8.5.1',soc='Ascend910B4/A2',physical_cards=[args.card],ranks=[0],
            worker='same-process standard Qwen3 TP1 eager; one 2048-token prefill then post-unregister token health',
            source_commit=source['manifest']['base_commit'],source_manifest_sha256=source['manifest_sha256'],
            required_sources=source['pins'],installed_required_sources_sha256=E.digest(installed_pins),
            build_fingerprint=dict(completion_sha256=args.build_sha256,variant=build['variant'],
                kernel_sha256=build['kernel_binary_sha256'],compiled_input_observer=build['tnd28_compiled_input_observer'],
                observer_receipt_sha256=observer['sha256'],production_libraries_sha256=args.libraries_sha256),
            expected_hits='0 faults; exactly one actual matched input window, retain all 28 natural calls and missing edges',
            resource_budget=dict(timeout_seconds=args.timeout,memory_gib=48,cpus=8,observed_requests=1,
                ordinary_health_requests=1,compiled_input_records=24,compiled_input_bytes=1024**2,runtime_log_bytes=128*1024**2),
            recovery_plan='Unchanged original call, existing synchronization, strict owned environment readback, disable/unregister/detach, '
                'remove exact deployment binding, then ordinary same-Worker token health. Preserve first failure and owner on uncertain cleanup.',
            diagnostic_only=True,statistics=False,fault_authorized=False,C1=False,C2=False,C3=False,E5=False,
            normal_stock_private_exact_proven=False,host_tiling_precedes_prepared_input=True,
            execute_borrow_guard_requested=bool(getattr(args,'execute_borrow',False)),
            actual_cached_api_route_proven=False,slice_pod_decoded=False))
    return validate_cases([case],[args.card])


def prepare(args):
    for name in ('source','installed','observer','model','build_root','vendor','libraries'):
        setattr(args,name,getattr(args,name).resolve(strict=True))
    source=frozen(args.source,args.source_manifest_sha256);installed_pins=installed(source,args.installed)
    build,files,_,_=R.bundle(args.build_root/'evidence/completion.json',args.build_sha256,args.vendor,args.source,args.source_manifest_sha256)
    observer=E.observer_bundle(args.observer,source);expected,mounts=libraries(args.libraries,args.libraries_sha256)
    expected=dict(expected,files=list(expected['files']))
    private='op_api/lib/libcust_opapi.so';pin=files[private]
    expected['files'].append(dict(recorded_path='/native/'+private,path=str(args.vendor/private),bytes=pin['bytes'],sha256=pin['sha256']))
    return make_case(args,source,observer,build,mounts,installed_pins),expected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','installed','observer','model','build-root','vendor','libraries','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('source-manifest-sha256','build-sha256','libraries-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--card',type=int,choices=(2,5),default=5);p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--execute-borrow',action='store_true',help='Request the new compiled normal Execute borrow guard, never F')
    args=p.parse_args();sys.dont_write_bytecode=True;sys.addaudithook(E.prohibit_runtime)
    library_out=args.out.with_name(args.out.stem+'-expected-libraries.json')
    E.need(not args.out.exists() and not library_out.exists(),'FRESH_PREPARATION_ARTIFACTS_REQUIRED')
    cases,expected=prepare(args);args.out.parent.mkdir(parents=True,exist_ok=True)
    for path,value in ((library_out,expected),(args.out,cases)):
        with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='ONE_NORMAL_INPUT_OBSERVATION_PREPARED_NOT_RUN',cases=str(args.out),cases_sha256=E.sha(args.out),
        expected_libraries=str(library_out),expected_libraries_sha256=E.sha(library_out),actual_npu_calls=0,fault_authorized=False)))


if __name__=='__main__':main()
