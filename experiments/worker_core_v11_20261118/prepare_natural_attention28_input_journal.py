"""Prepare exactly one bounded zero-F case; pure file reads, no process/device."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/c2c3_closure_20261117')]
from experiments.c2c3_closure_20261117 import arch22_normal_bridge_evidence as E
from experiments.c2c3_closure_20261117 import prepare_attention28_stock_census as stock
from experiments.c2c3_closure_20261117.queue_contract import validate_cases

EXPERIMENT='experiments/worker_core_v11_20261118/'
EXTRA=(EXPERIMENT+'prepare_natural_attention28_input_journal.py',
    EXPERIMENT+'natural_attention28_input_journal_probe.py',EXPERIMENT+'fia_input_journal_review.py',
    'fault_injection/fia_input_journal.py','fault_injection/quant_container_native.py','fault_injection/metadata_arrays.py')


def validate_journal(root,expected):
    root=Path(root).resolve(strict=True);path=root/'build.json'
    E.need(E.sha(path)==expected,'EXACT_FIA_JOURNAL_BUILD_REQUIRED')
    build=E.read(path)
    E.need(build.get('schema')=='demoncat.quant.container_deployment.v1'
        and build.get('candidate')=='fia_input_journal' and build.get('actual_npu_calls')==0
        and build.get('qualified') is False and build.get('runtime_activation_default') is False
        and build.get('public_provider_registered') is False
        and build.get('status')=='SDK_TYPED_CPU_CONTRACTS_COMPLETE_NOT_NPU','CPU_ONLY_INERT_JOURNAL_REQUIRED')
    E.need(type(build.get('files')) is dict and 1<=len(build['files'])<=128,'BOUNDED_BUILD_FILE_MANIFEST_REQUIRED')
    for relative,digest in build['files'].items():
        E.need(E.sha(E.child(root,relative))==digest,'JOURNAL_BUILD_FILE_CHANGED:'+relative)
    E.need('profile.json' in build['files'] and 'receipt.json' in build['files']
        and build['library'] in build['files'],'COMPLETE_JOURNAL_BUILD_IDENTITY_REQUIRED')
    profile=E.read(root/'profile.json');canonical=dict(profile);ident=canonical.pop('profile_sha256')
    E.need(ident==build['profile_sha256']==E.digest(canonical)
        and profile.get('candidate')=='fia_input_journal' and profile.get('image')==E.IMAGE,
        'JOURNAL_PROFILE_CHANGED')
    E.need(profile['sdk_headers']['include/aclnnop/aclnn_fused_infer_attention_score_v3.h']==
        'bf73aea4146a9b12f91eda466927c23c5d43e56f78aa5039bd9644732f521550','ACTUAL_REVIEWED_FIA_V3_HEADER_REQUIRED')
    receipt=E.read(root/'receipt.json')
    E.need(receipt.get('status')=='SDK_TYPED_CPU_CONTRACTS_COMPLETE_NOT_NPU'
        and receipt.get('actual_npu_calls')==0 and receipt.get('qualification_granted') is False
        and len(receipt.get('cases',[]))==26 and len({x['case'] for x in receipt['cases']})==26
        and all(x.get('status')=='PASS' and x.get('returncode')==0 for x in receipt['cases']),
        'FRESH_26_CPU_CONTRACTS_REQUIRED_NOT_NPU')
    for row in receipt['cases']:
        log='fia_input_journal_'+row['case']+'.log'
        E.need(build['files'].get(log)==row['log_sha256'],'CPU_CASE_LOG_IDENTITY_CHANGED')
    E.need(receipt['source_sha256']==profile['sources'],'CPU_COMPILE_SOURCE_IDENTITY_CHANGED')
    return dict(root=root,build=build,profile=profile,receipt=receipt,sha256=expected)


def frozen_source(root,expected,journal):
    source=stock.frozen_source(root,expected);index={x['path']:x for x in source['manifest']['files']}
    for name in dict.fromkeys(EXTRA+tuple(journal['profile']['sources'])):
        path=E.child(source['root'],name);row=index[name]
        E.need(E.sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],'FROZEN_INPUT_JOURNAL_SOURCE_DIFFERS:'+name)
        if name in journal['profile']['sources']:
            E.need(row['sha256']==journal['profile']['sources'][name],'COMPILE_RUNTIME_SOURCE_DIFFERS:'+name)
        source['pins'][name]=row['sha256']
    E.need(E.sha(__file__)==source['pins'][EXPERIMENT+Path(__file__).name],'AMBIENT_JOURNAL_PREPARER_DIFFERS')
    return source


def installed_sources(source,root):
    root=Path(root).resolve(strict=True);result={}
    for name,digest in source['pins'].items():
        if not name.endswith('.py'):continue
        relative=name.removeprefix('src/') if name.startswith('src/demoncat/') else name if name.startswith('fault_injection/') else None
        if relative is not None:
            E.need(E.sha(E.child(root,relative))==digest,'INSTALLED_JOURNAL_SOURCE_DIFFERS:'+relative)
            result[relative]=digest
    return result


def add_observer_alias(mounts,folder,alias):
    path=PurePosixPath(alias)
    E.need(path.is_absolute() and '..' not in path.parts and '\\' not in str(path)
        and (str(path)=='/observer' or str(path).startswith(('/observer/','/build/','/output/'))),
        'UNREVIEWED_OBSERVER_DEPENDENCY_ALIAS')
    existing=[row for row in mounts if row[1]==str(path)]
    if existing:
        E.need(existing==[[str(folder),str(path),'ro']],'CONFLICTING_OBSERVER_READONLY_ALIAS')
    else:mounts.append([str(folder),str(path),'ro'])


def make_case(args,source,observer,journal,installed):
    E.need(args.card in (2,5) and type(args.timeout) is int and 1<=args.timeout<=1800,'BOUNDED_ONE_CARD_CASE_REQUIRED')
    original=journal['profile']['originals']['observer'];path=PurePosixPath(original['path'])
    actual_name=PurePosixPath(observer['receipt']['observer_library']['path']).name
    E.need(original['file_sha256']==observer['receipt']['observer_library']['sha256'] and path.name==actual_name,
        'EXACT_COMPILED_OBSERVER_DEPENDENCY_REQUIRED')
    name='natural-fia-input-journal-two-requests'
    argv=['python3','-B','/repo/'+EXPERIMENT+'natural_attention28_input_journal_probe.py',
        '--model','/model','--out','/evidence/probe','--observer-receipt','/build/attention-preflight/result.json',
        '--observer-receipt-sha256',observer['sha256'],'--journal','/fia-journal','--journal-build-sha256',journal['sha256']]
    mounts=[[str(args.source),'/repo','ro'],[str(args.installed),'/installed','ro'],[str(args.model),'/model','ro'],
        [str(journal['root']),'/fia-journal','ro']]
    for alias in ('/build/attention-preflight',observer['alias'],str(path.parent)):
        add_observer_alias(mounts,observer['folder'],alias)
    case=dict(id=name,image=E.IMAGE,timeout=args.timeout,mounts=mounts,
        env=dict(PYTHONPATH='/installed:/repo:/repo/src:/repo/fault39/src:/msprobe_packages',PYTHONDONTWRITEBYTECODE='1',
            VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_ASCEND_ENABLE_NZ='0',
            VLLM_ALLOW_INSECURE_SERIALIZATION='1',OMP_NUM_THREADS='4',VLLM_HOST_IP='127.0.0.1',GLOO_SOCKET_IFNAME='lo',
            HCCL_CONNECT_TIMEOUT='120',HCCL_EXEC_TIMEOUT='120',ACLNN_CACHE_LIMIT='0',TASK_QUEUE_ENABLE='0'),
        command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '+shlex.join(argv),
        review=dict(experiment_id=name,provider='stock FIA input observer; no policy or F',model=str(args.model),
            cann='8.5.1',soc='Ascend910B4/A2',physical_cards=[args.card],ranks=[0],
            worker='same-process standard Qwen3 TP1 eager; two natural 2048-token prefill requests, all 28 layers',
            source_commit=source['manifest']['base_commit'],source_manifest_sha256=source['manifest_sha256'],
            required_sources=source['pins'],installed_required_sources_sha256=E.digest(installed),
            build_fingerprint=dict(journal_build_sha256=journal['sha256'],journal_profile_sha256=journal['profile']['profile_sha256'],
                observer_receipt_sha256=observer['sha256'],observer_library_sha256=original['file_sha256'],
                native_compile_source_sha256=journal['profile']['sources'],observer_dependency_alias=str(path),
                private_native_operator_package=None),
            expected_hits='0 faults; retain 2 x 28 actual public calls and every missing Native/metadata edge',
            resource_budget=dict(timeout_seconds=args.timeout,memory_gib=48,cpus=8,observed_requests=2,
                ordinary_health_requests=2,public_windows=64,expected_public_windows=56,native_calls=128,
                native_events=256,native_record_bytes=16*1024*1024,pending_pairs=64),
            recovery_plan='Pre-LLM exact NOLOAD/owner resolve; same held object bound by real Worker RPC. '
                'Keep original noncontiguous inputs and mask. Enter only inside the existing launch scope, call '
                'original once, use existing completion sync then leave. Public query does not query the journal. '
                'Finish/unregister/detach and remove owned binding, stop, immutable snapshot, release receipt, '
                'then same-Worker ordinary token health. Preserve primary errors and pending owners; no reset, '
                'global wrapper unloading claim or fabricated missing lineage.',
            diagnostic_only=True,fault_authorized=False,statistics=False,C1=False,C2=False,C3=False,E5=False,
            slice_pointer_lineage_proven=False,prepared_internal_tensor_lineage_proven=False))
    return validate_cases([case],[args.card])


def prepare(args):
    for name in ('source','installed','observer','journal','model'):setattr(args,name,getattr(args,name).resolve(strict=True))
    journal=validate_journal(args.journal,args.journal_build_sha256)
    source=frozen_source(args.source,args.source_manifest_sha256,journal)
    installed=installed_sources(source,args.installed);observer=E.observer_bundle(args.observer,source)
    return make_case(args,source,observer,journal,installed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','installed','observer','journal','model','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('source-manifest-sha256','journal-build-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--card',type=int,choices=(2,5),default=5);p.add_argument('--timeout',type=int,default=1800)
    args=p.parse_args();sys.dont_write_bytecode=True;sys.addaudithook(E.prohibit_runtime)
    E.need(not args.out.exists(),'FRESH_CASE_OUTPUT_REQUIRED')
    cases=prepare(args);args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:json.dump(cases,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='READONLY_FIA_INPUT_CASE_PREPARED_NOT_EXECUTED',path=str(args.out),sha256=E.sha(args.out),
        actual_npu_calls=0,qualified=False)))


if __name__=='__main__':main()
