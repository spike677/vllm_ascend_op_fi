"""Prepare one root-queued public C1 attempt only after a fresh normal pair.

CPU file validation only. No device, operator, process or injection module is
loaded. The Worker independently reconstructs its own live normal evidence.
"""
import argparse
import json
from pathlib import Path
import shlex
from types import SimpleNamespace
import sys

import arch22_normal_bridge_evidence as E
from prepare_arch22_normal_bridge_cases import case_environment


def prepare(args,*,group=False):
    for name in ('source','installed','build_root','observer','model','stock_run','baseline_run','normal_report'):
        setattr(args,name,getattr(args,name).resolve(strict=True))
    E.need(args.card in (2,5) and 1<=args.timeout<=1800,'S2_SINGLE_CASE_BUDGET')
    source=E.frozen_source(args.source,args.source_manifest_sha256,build_variant=E.GROUP_VARIANT if group else E.S2_VARIANT)
    E.need(E.sha(__file__)==source['pins'][E.EXPERIMENT+Path(__file__).name],'S2_FROZEN_PREPARER_DIFFERS')
    helper=E.helpers(source);pins=E.installed_sources(source,args.installed)
    bundle=E.build_bundle(args.build_root,source,helper,args.completion_sha256)
    observer=E.observer_bundle(args.observer,source)
    E.need(observer['sha256']==args.observer_receipt_sha256,'S2_OBSERVER_RECEIPT_DIFFERS')
    import review_arch22_normal_bridge as reader
    if group:import review_arch22_all_normal as reader
    result=reader.review(SimpleNamespace(source=args.source,source_manifest_sha256=args.source_manifest_sha256,
        build_root=args.build_root,completion_sha256=args.completion_sha256,observer=args.observer,
        observer_receipt_sha256=args.observer_receipt_sha256,stock_run=args.stock_run,stock_sha256=args.stock_sha256,
        baseline_run=args.baseline_run,baseline_sha256=args.baseline_sha256,arch22_s2_normal=True),**({'build_variant':E.GROUP_VARIANT} if group else {}))
    E.need(not result['issues'] and result['normal_payloads_exact'] and result['complete_payload_pairs']==435,
           'S2_COMPLETE_FRESH_NORMAL_RAW_REVIEW_FAILED')
    E.need(E.sha(args.normal_report)==args.normal_report_sha256,'S2_PUBLISHED_NORMAL_REPORT_CHANGED')
    published=E.read(args.normal_report)
    E.need(published['source']['manifest_sha256']==source['manifest_sha256']
        and published['build']['completion_sha256']==args.completion_sha256
        and published['arms']['stock']['result_sha256']==args.stock_sha256
        and published['arms']['baseline']['result_sha256']==args.baseline_sha256,'S2_PUBLISHED_REPORT_INPUTS_DIFFER')
    budget=None
    if group:
        import importlib.util
        resource_path=args.source/'fault_injection/native_adapters/attention28/group_resource_budget.py'
        spec=importlib.util.spec_from_file_location('frozen_group_resource_plan',resource_path)
        resources=importlib.util.module_from_spec(spec);spec.loader.exec_module(resources)
        budget=resources.plan(result['arms']['baseline']['resource_observations'],variant=E.GROUP_VARIANT,
            source_sha256=args.source_manifest_sha256,build_sha256=args.completion_sha256,case_seconds=args.timeout)
    probe='natural33_arch22_group_worker_core_probe.py' if group else 'natural33_arch22_s2_worker_core_probe.py'
    execution=helper.contract.execution_contract(args.model,args.installed,
        args.source/E.EXPERIMENT/probe,.35,case_environment(args.card))
    original=E.read(args.baseline_run/'result.json')['execution_contract']
    E.need({k:v for k,v in original.items() if k!='driver_sha256'}==
        {k:v for k,v in execution.items() if k!='driver_sha256'},'S2_MODEL_PACKAGE_OR_RUNNER_DIFFERS_FROM_NORMAL')
    # Preserve /native for loader equivalence; the second original-path alias
    # includes the built source + patch receipt required by raw evidence review.
    build_top=args.build_root.parents[1]
    E.need(args.build_root.relative_to(build_top).as_posix()=='build/run','S2_ORIGINAL_BUILD_LAYOUT_REQUIRED')
    mounts=[[str(args.source),'/repo','ro'],[str(args.installed),'/installed','ro'],[str(args.model),'/model','ro'],
        [str(args.build_root),'/native','ro'],[str(build_top),str(build_top),'ro'],
        [str(observer['folder']),'/build/attention-preflight','ro'],
        [str(args.stock_run),'/normal-stock','ro'],[str(args.baseline_run),'/normal-private','ro'],
        [str(args.normal_report.parent),'/normal-review','ro']]
    if observer['alias']!='/build/attention-preflight':mounts.append([str(observer['folder']),observer['alias'],'ro'])
    E.need(len({x[1] for x in mounts})==len(mounts),'S2_MOUNT_COLLISION')
    argv=['python3','-B','/repo/'+E.EXPERIMENT+probe,
        '--source','/repo','--source-manifest-sha256',args.source_manifest_sha256,
        '--build-root',str(args.build_root),'--completion-sha256',args.completion_sha256,
        '--observer','/build/attention-preflight/result.json','--observer-receipt-sha256',args.observer_receipt_sha256,
        '--stock-run','/normal-stock','--stock-sha256',args.stock_sha256,
        '--baseline-run','/normal-private','--baseline-sha256',args.baseline_sha256,
        '--normal-report','/normal-review/'+args.normal_report.name,'--normal-report-sha256',args.normal_report_sha256,
        '--model','/model','--out','/evidence/probe']
    if group:argv+=['--case-budget-seconds',str(args.timeout)]
    label='attention33-arch22-group-public-c2' if group else 'attention33-arch22-s2-public-single-c1'
    return [dict(id=label,image=E.IMAGE,timeout=args.timeout,mounts=mounts,
        env=case_environment(args.card),command='source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexec '+shlex.join(argv),
        review=dict(experiment_id=label,provider='33.paged_decode / public all-target isolated C2' if group else '33.paged_decode / public single-site C1',
            phase='3_CONTROLS_28_ISOLATED_F_OFF_HEALTH' if group else 'R_L_OFF_F_OFF',source_commit=source['manifest']['base_commit'],source_manifest_sha256=source['manifest_sha256'],
            execution_contract=execution,installed_required_sources_sha256=E.digest(pins),build_fingerprint=bundle['summary'],
            normal_replayed_report_sha256=E.digest(result),normal_report_sha256=args.normal_report_sha256,
            observer_receipt_sha256=observer['sha256'],model=str(args.model),cann='8.5.1',soc='Ascend910B4/A2',
            physical_cards=[args.card],ranks=[0],worker='Same standard Qwen TP1 eager configuration as the fresh normal pair',
            expected_hits='28 isolated actual S2 selections; 60 x 56 calls and 60 x 87 complete payloads' if group else 'One actual first-layer S2 512->256 selection; five 56-call request denominators and 5 x 87 full payloads',
            measured_cost_gate=budget,
            recovery_plan='New same-owner OFF; exact full output recovery; zero owned scopes/hooks/API/input refs; unregister/detach and ordinary token/text health.',
            resource_budget=dict(timeout_seconds=args.timeout,memory_gib=48,cpus=16),
            C1=False,C2=False,C3=False,E5='NOT_ASSESSED',qualification_granted=False,
            acceptance='Raw Native POD/ELF + actual request + full output/OFF independent review required.',
            limits=(['Fresh final group-package normal per actual target; no source105/source108 authority.',
                'C3 current propagated input and stock-owner exact normal are not proven; E5 not assessed.'] if group else ['Only the source-bound first-layer experiment is admitted.',
                'Other 27 kernel-input-matching decode sites have no per-site normal/effect proof; C2 remains blocked.',
                'Mask and read-before logical KV are hashes; later matching complete logical KV bytes are saved.',
                'No repeated F after an uncertain call; OFF recovery preserves the original failure.'])))]


def main(*,group=False):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','installed','build-root','observer','model','stock-run','baseline-run','normal-report','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('source-manifest-sha256','completion-sha256','observer-receipt-sha256','stock-sha256','baseline-sha256','normal-report-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--card',type=int,choices=(2,5),default=5);parser.add_argument('--timeout',type=int,default=1800)
    args=parser.parse_args();sys.dont_write_bytecode=True
    from review_arch22_normal_bridge import initialize_cpu_byte_reader
    numeric=initialize_cpu_byte_reader();sys.addaudithook(E.prohibit_runtime)
    E.need(not args.out.exists(),'S2_FRESH_CASE_MANIFEST_REQUIRED')
    try:
        cases=prepare(args,group=group);args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x',encoding='utf-8') as stream:json.dump(cases,stream,allow_nan=False,indent=2);stream.write('\n')
    except Exception as error:
        print(json.dumps(dict(status='S2_SINGLE_PREPARATION_REFUSED',error=repr(error),actual_npu_calls=0)));return 2
    print(json.dumps(dict(status='S2_SINGLE_ROOT_CASE_PREPARED',cases=1,path=str(args.out),sha256=E.sha(args.out),
        actual_npu_calls=0,cpu_numeric_reader=numeric,C2=False,C3=False)));return 0


if __name__=='__main__':raise SystemExit(main())
