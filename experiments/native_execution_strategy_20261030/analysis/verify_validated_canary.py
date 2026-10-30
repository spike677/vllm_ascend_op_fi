"""CPU/read-only canary proof, separate from the formal20-case/56-call report."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import validate_phase1 as base

CERT_SHA='be58b4e77098a8911ab97ad8711144a5a5999dc3742080726c594ad88e8fcc5b'
RUNTIME_SHA='ca0bf188bdc031ea3e4249c03ac13a9a8f751cd5ae8485655d9048c9f6cdba3b'
FORMAL_SHA='3399b779234d84b6f58d4e23a5a8d15aa18d08fcee737f4c25ace33122e93579'
SOURCES_SHA='6ab8694faaa85e2fb15ce290e6e36bf3b6b75a892cbd08212960599a01133606'
require=base.require


def numerical_projection(row):
    fields=('shape','stride','dtype','device','storage_offset','actual_format','full_sha256','bytes',
            'byteorder','raw_order','target_sha256','per_batch_sha256')
    return {k:row[k] for k in fields if k in row}


def verify(root, output):
    root=root.resolve(strict=True); attempt=root/'outputs/validated_alias_a1'; folder=attempt/'validated_alias_b8'
    oldfolder=root/'outputs/controlled_query_tile_a1/controlled_query_tile_b8_r0'
    completion=base.read(attempt/'completion.json');base.strict_completion(completion)
    require(len(completion['cases'])==1,'Canary must be one separate four-call case')
    require(completion['owned_container_id']!=base.read(oldfolder.parent/'completion.json')['owned_container_id'],
            'Canary unexpectedly reused the formal execution container')
    formal_path=root/'outputs/independent_validation/phase1_full_v5.json';base.bound(formal_path,FORMAL_SHA)
    formal=base.read(formal_path);require(formal['status']=='PASS_NATIVE_EXECUTION_STRATEGY_FA_PHASE1','Formal report did not pass')
    require(base.sha(base.__file__)==formal['analysis_source_sha256'],'Formal validator source revision changed')
    # Reuse only stateless byte/POD/readback routines; never re-run the formal
    # generation gate against the newer catalog/current harness.
    checker=base.Validator.__new__(base.Validator);checker.root=root;checker.evidence={};checker.audit_dir=output.with_suffix('.audit')
    checker.audit_dir.mkdir(parents=True,exist_ok=False)
    checker.record(formal_path,FORMAL_SHA);checker.record(attempt/'completion.json')
    checker.record(attempt/'source_manifest.json',SOURCES_SHA);sources=base.read(attempt/'source_manifest.json')
    require(len(sources)==62,'Unexpected canary execution source inventory')
    for name,digest in sources.items():
        checker.record(base.child(attempt/'source',name),digest);checker.record(base.child(root,name),digest)
    checker.generation=sources
    spec=importlib.util.spec_from_file_location('canary_pinned_native_evidence',attempt/'source/native_evidence.py')
    checker.native_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker.native_module)
    require(sources['native_evidence.py']==formal['source_generation']['native_evidence.py'],'Canary native evidence parser changed')
    checker.record(root/'runtime_manifest.json',RUNTIME_SHA);runtime=base.read(root/'runtime_manifest.json')
    require(len(runtime['files'])==23,'Expected explicit23-file validated runtime')
    require({p.relative_to(root/'runtime').as_posix() for p in (root/'runtime').rglob('*') if p.is_file()}==set(runtime['files']),
            'Runtime contains unlisted files')
    for name,row in runtime['files'].items():checker.record(base.child(root/'runtime',name),row['sha256'],row['bytes'])
    python_pins={n:runtime['files']['native_execution_strategy/'+n]['sha256'] for n in base.SOURCE_FILES}
    require(python_pins==formal['python_source_files'],'Canary changed the validated15 Python source closure')
    package=root/'runtime/native_execution_strategy';certificate_path=package/'data/fa_cann91_s1s2_v1.json'
    checker.record(certificate_path,CERT_SHA);checker.record(package/'data/evidence/fa_phase1_validation.json',FORMAL_SHA)
    sys.path.insert(0,str(root/'runtime'))
    from native_execution_strategy.certificates import resolve_catalog
    resolved=resolve_catalog(package/'data/catalog.json',evidence_root=package,version='validated',allow_experimental=False)
    require(resolved.fingerprints.certificate_sha256==CERT_SHA and resolved.data['validation_state']=='validated'
            and resolved.data['contexts']==formal['validated_contexts'],'Unique validated alias did not resolve to bound actual plans')
    schema_path=package/'adapters/fa_cann91_schema.json';checker.record(schema_path,base.SCHEMA_SHA);checker.schema=base.read(schema_path)
    checker.input_fields={f['name'] for f in checker.schema['fields'] if f['name'].startswith('inputParams.') and f['name'] not in base.PADDING}
    checker.plan_fields={f['name'] for f in checker.schema['fields'] if not f['name'].startswith('inputParams.') and f['name'] not in base.PADDING}
    checker.verify_vendor();checker.record(root/'inputs/manifest.json',base.INPUT_SHA)
    case=base.read(attempt/'source/setup/canary/validated_alias_b8.json')
    require(case['case_id']=='validated_alias_b8' and case['batch_size']==case['request_count']==8
        and case['allow_experimental'] is False and case['execution_config']['adapter']['version']=='validated'
        and case['adapter_certificate_sha256']==CERT_SHA and case['runtime_manifest_sha256']==RUNTIME_SHA
        and case['mode_sequence']==['native','controlled','controlled','off'] and case['native_calls_per_process']==4,
        'Canary did not use the validated alias with experimental disabled')
    d=base.read(folder/'result.json');row=completion['cases'][0];checker.record(folder/'result.json',row['result_sha256'])
    require(row['rc']==0 and row['case_id']==d['case_id']==case['case_id'] and d['status']=='CAPTURE_COMPLETED'
        and d['original_api_calls']==len(d['steps'])==4 and d['model_forward'] is False
        and d['runtime_manifest_sha256']==RUNTIME_SHA and d['replay_source_sha256']==sources['replay.py']
        and d['source_case_sha256']==sources['setup/canary/validated_alias_b8.json'],'Canary completion/source binding differs')
    expected_resolution={'name':'FlashAttentionScore','requested_version':'validated','resolved_version':'cann9.1-s1s2-v1',
        'route':'S1S2','validation_state':'validated','fingerprints':{'source_sha256':base.SOURCE_SHA,
        'build_sha256':base.BUILD_SHA,'certificate_sha256':CERT_SHA}}
    require(all(d['adapter_resolution'].get(k)==v for k,v in expected_resolution.items()),'Actual adapter alias resolution differs')
    require(d['input_unchanged'] is True and d['host_scope_restored'] is True,'Actual canary integrity incomplete')
    require(d['loader']['vendor_inventory_verified'] is True and d['loader']['manifest_sha256']==base.BUILD_SHA
        and d['loader']['policy_source_manifest_sha256']==base.SOURCE_SHA,'Actual loader did not verify approved native bridge')
    launcher=attempt/'validated_alias_b8.vendor_launch.json';checker.record(launcher,d['loader']['launcher_receipt_sha256']);launch=base.read(launcher)
    require(launch['case_sha256']==d['source_case_sha256'] and launch['launcher_sha256']==sources['vendor_exec.py']
        and launch['probe_sha256']==sources['replay.py'] and launch['status']=='VERIFIED_READY_TO_EXEC','Launcher not bound to canary source/case')
    for module,ref in d['runtime_imports'].items():
        require(ref['path'].startswith('/audit/runtime/'),'Actual imported implementation outside approved runtime')
        require(ref['sha256']==runtime['files'][ref['path'][len('/audit/runtime/'):]]['sha256'],'Actual imported implementation changed')
    checker.verify_statistics(folder,d,4)
    def bind_formal(path):
        name=path.relative_to(root).as_posix();ref=formal['evidence_files'][name]
        checker.record(path,ref['sha256'],ref['bytes'])
    bind_formal(oldfolder/'result.json');old=base.read(oldfolder/'result.json')
    require(d['native_kwargs']==old['native_kwargs'] and d['invocation_context']==old['invocation_context'],
            'Validated alias changed public kwargs or actual workload interpretation')
    comparisons=[];new_rows=[];previous=None
    for i,(s,prior) in enumerate(zip(d['steps'],old['steps'])):
        sp=folder/'steps'/f'step{i}';oldsp=oldfolder/'steps'/f'step{i}';call=s['native_call']
        checker.record(sp/'result.json');require(base.read(sp/'result.json')==s,'Canary step record differs')
        require(s['step']==i and s['mode']==case['mode_sequence'][i] and call['pid']==launch['pid']
            and call['step']==i and call['mode']==s['mode'] and call['case_id']==case['case_id']
            and call['status']=='returned' and call['batch_size']==8 and call['phase']=='operator_execution','Canary invocation scope differs')
        require(call['enter_epoch_ns']>prior['native_call']['exit_epoch_ns'],'Canary does not follow formal run')
        for key in ('query','key','value'):
            base.unchanged(s['inputs_before'][key],s['inputs_after'][key])
            for side in ('inputs_before','inputs_after'):
                require(numerical_projection(s[side][key])==numerical_projection(prior[side][key]),
                        'Canary QKV values/geometry differ from formal stage')
        base.unchanged(s['mask_before'],s['mask_after'])
        for side in ('mask_before','mask_after'):
            require(numerical_projection(s[side])==numerical_projection(prior[side]),'Canary causal mask differs')
        if previous is not None:
            require(s['inputs_before']==previous['inputs_after'] and s['mask_before']==previous['mask_after'],
                'Canary reset input objects between invocations')
            require(call['pid']==previous['native_call']['pid'] and call['thread_id']==previous['native_call']['thread_id']
                and call['enter_monotonic_ns']>previous['native_call']['exit_monotonic_ns'],'Canary execution order changed')
        meta=base.child(folder,s['step_metadata_file']);checker.record(meta,s['step_metadata_sha256'])
        require(base.read(meta)['resource_meter_source_sha256']==sources['resource_meter.py'],'Resource meter source changed')
        resources,_,_=checker.native_module.bound_resources(meta,call);require(resources==s['resources'],'Resource accounting differs')
        current=checker.verify_native(attempt,folder,sp,s,case);checker.verify_session(sp,s,i,current)
        host=base.read(sp/'native_profile/verified_record.json')['host_event']
        require(host['plan']['certificate']['validation_state']=='validated'
            and host['plan']['certificate']['id']==resolved.data['id']
            and host['plan']['certificate']['validation_sha256']==FORMAL_SHA,'Actual native scope used another/experimental certificate')
        require(s['native_observation']['fingerprints']['certificate_sha256']==CERT_SHA,'Actual observation certificate differs')
        rawpath=base.child(sp,s['private_target_output']['file']);checker.record(rawpath,s['private_target_output']['sha256'],8388608)
        oldrawpath=base.child(oldsp,prior['private_target_output']['file']);bind_formal(oldrawpath)
        current.update(raw=rawpath.read_bytes(),step=s)
        oldprofile=oldsp/'native_profile/verified_record.json';bind_formal(oldprofile);oldrecord=base.read(oldprofile)
        oldrow={'raw':oldrawpath.read_bytes(),'step':prior,'pod':oldrecord['pod'],'native':oldrecord['native_launch'],
            'host_context':{k:oldrecord['host_event'][k] for k in ('schedule_mode','workspace_sizes')}}
        base.check_outputs_equal(oldrow,current,f'canary step{i}')
        require(checker.plan(oldrow)==checker.plan(current),'Canary changed complete169-field execution plan')
        input_contract={k:current['pod'][k] for k in checker.input_fields}
        require(input_contract=={k:oldrow['pod'][k] for k in checker.input_fields},'Canary changed34-field input contract')
        require(s['output']['target_sha256']==base.sha(rawpath),'Output/private bytes binding differs')
        expected_switch=None if i==0 else i in (1,2)
        require(s['assessment']['strategy_switched'] is expected_switch and s['assessment']['model_impact'] is None,'Canary switch/model result differs')
        comparisons.append({'step':i,'mode':s['mode'],'complete_batch0_output_equal':True,
            'complete169_plan_equal':True,'input34_equal':True,'primary_output_sha256':base.sha(rawpath),
            'actual_recipe':list(base.recipe(current['pod'])),'strategy_switched':expected_switch,
            'padding_differences':[k for k in sorted(base.PADDING) if oldrow['pod'][k]!=current['pod'][k]],
            'resources':resources,'model_impact':None})
        new_rows.append(current);previous=s
    base.check_outputs_equal(new_rows[1],new_rows[2],'Canary every-match repeat')
    base.check_outputs_equal(new_rows[0],new_rows[3],'Canary off recovery')
    # Deliberate harness revision, separate from the unchanged15 runtime pins.
    old_replay=root/'generations/runtime_a2/replay.py';checker.record(old_replay,formal['source_generation']['replay.py'])
    oldtext=old_replay.read_text();newtext=(root/'replay.py').read_text()
    replacements=[("and case['allow_experimental'] is True", "and type(case['allow_experimental']) is bool"),
        ("'validation_state': 'experimental'", "'vendor_inventory_verified': True"),
        ('selection_target=cfg.selection_target, allow_experimental=True,',
         "selection_target=cfg.selection_target, allow_experimental=case.get('allow_experimental', False),")]
    for a,b in replacements:
        require(oldtext.count(a)==1,'Expected reviewed harness revision is not unique');oldtext=oldtext.replace(a,b)
    require(oldtext==newtext,'Canary harness contains additional unreviewed changes')
    return {'schema':'nes.validated_alias_canary_validation.v1','status':'PASS_VALIDATED_ALIAS_CANARY',
        'formal_validation_sha256':FORMAL_SHA,'certificate_sha256':CERT_SHA,'runtime_manifest_sha256':RUNTIME_SHA,
        'source_manifest_sha256':SOURCES_SHA,'source_sha256':base.SOURCE_SHA,'build_sha256':base.BUILD_SHA,
        'python_source_files':python_pins,'python15_unchanged':True,'allow_experimental':False,
        'requested_version':'validated','resolved_version':'cann9.1-s1s2-v1','case_count':1,'native_call_count':4,
        'comparisons':comparisons,'same_process_repeat_equal':True,'off_restores_complete_output_and_plan':True,
        'harness_revision':[{'before':a,'after':b} for a,b in replacements],
        'formal56_results_rewritten':False,'certificate_issued_by_this_tool':False,'new_native_calls':0,
        'evidence_files':checker.evidence,'model_impact':None,
        'limits':['Separate single B8 query_tile alias canary; does not expand certified contexts or the formal56-call matrix.',
            'Complete output byte proof is separate from native summary statistics and auxiliary RNG variation.',
            'Timing is synchronized API/collector wall time; peak memory is allocator peak, not whole-card HBM.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sys.dont_write_bytecode=True;require(not args.output.exists(),'Output must be new');start=time.monotonic()
    try:result=verify(args.root,args.output)
    except (ValueError,KeyError,TypeError,OSError,AssertionError) as error:result={'status':'FAIL','error':str(error),'new_native_calls':0}
    result.update(analysis_source_sha256=base.sha(__file__),elapsed_cpu_wall_seconds=time.monotonic()-start)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'error':result.get('error'),'report_sha256':base.sha(args.output)}))
    return 0 if result['status']=='PASS_VALIDATED_ALIAS_CANARY' else 1


if __name__=='__main__':raise SystemExit(main())
