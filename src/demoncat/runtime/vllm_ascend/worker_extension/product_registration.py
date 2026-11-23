"""Release-reviewed product registrations. No public registration or permit bypass API."""
import hashlib, inspect, json, platform
from pathlib import Path
from .deployment_acceptance import observe_deployment, _get, _BLUE
from .deployment_authorization import authority_for

RECORDS = Path(__file__).with_name('product_registered_records.json')
IDENTITY_FIELDS = ('hardware_family','hardware_device_name','hardware_logical_device','torch','torch_npu',
 'vllm','vllm_ascend','python','cann','dtype','tp','pp','dp','ep','rank','eager','model_class',
 'model_config_sha256','quant_model_description_sha256','safetensors_index_sha256','live_config_matches_file',
 'max_model_len','max_num_batched_tokens','max_num_seqs','async_scheduling','prefix_caching',
 'mtp_disabled','kv_connector_disabled','source_fingerprints','runtime_libraries','implementation_sha256',
 'request_range','native_identity','execution_mode','variant_revision')
PARALLEL=('tp','pp','dp','ep','rank')
RANGE=('max_model_len','max_num_batched_tokens','max_num_seqs','request_range')
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(v):return type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v)
def fail(code,paths=()):raise ValueError(code+(':'+','.join(paths) if paths else ''))
def implementation_identity():
    import fault_injection
    roots={'worker':Path(__file__).parent,'fault_injection':Path(fault_injection.__file__).parent}
    return digest({label+'/'+p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for label,root in roots.items() for p in sorted(root.rglob('*.py'))})
def observe_identity(worker,mechanism,native_library=None):
    a=observe_deployment(worker,worker.model_runner.model);runner=worker.model_runner;cfg=runner.vllm_config
    from fault_injection.environment import inspect_environment
    env=inspect_environment(additional_libraries=('libascendcl.so','libruntime.so'))
    links=env.get('loaded_toolkit_links',{}).get('libraries',[]);libs={}
    for name in ('libascendcl.so','libruntime.so'):
        rows=[r for r in env.get('loaded_libraries',[]) if r['name']==name]
        match=[r for r in links if r['name']==name]
        if len(rows)==1 and rows[0]['status']=='MAPPED_FILE_VERIFIED' and len(match)==1 and match[0]['status']=='PATH_LINKED_TO_INSTALLATION' and len(match[0]['matches'])==1:
            m=match[0]['matches'][0];libs[name]=dict(sha256=rows[0]['sha256'],version=m['version'],metadata_sha256=m['metadata_sha256'])
    sources={}
    for label,obj in [('model',type(runner.model)),('runner',type(runner))]:
        path=inspect.getsourcefile(obj)
        sources[label]=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    ep=_get(cfg,'parallel_config.enable_expert_parallel')
    a.update(python=platform.python_version(),rank=worker.rank,ep={'enabled':False,'placement':[],'routing':'NOT_APPLICABLE'} if ep is False else None,
      async_scheduling=getattr(runner,'use_async_scheduling',None),prefix_caching=_get(cfg,'cache_config.enable_prefix_caching'),
      mtp_disabled=(cfg.speculative_config is None) if hasattr(cfg,'speculative_config') else None,
      kv_connector_disabled=(cfg.kv_transfer_config is None) if hasattr(cfg,'kv_transfer_config') else None,
      source_fingerprints=sources,runtime_libraries=libs,implementation_sha256=implementation_identity(),execution_mode='product',variant_revision='1',
      request_range=dict(max_context=4096 if mechanism=='33.paged_decode' else 1024,max_tokens=4,max_num_seqs=8 if mechanism=='quant.batch_shared_scale' else 1,single_chunk=True,image_grid=[1,4,4] if mechanism=='17.spatial_padding' else None),
      native_identity={'applicable':False} if native_library is None else {'applicable':True,'sha256':hashlib.sha256(Path(native_library[0]).read_bytes()).hexdigest(),'declared_sha256':native_library[1]})
    if native_library is not None:
        sidecar=Path(native_library[0]).with_name('dcat-native-build.json')
        try:
            raw=sidecar.read_bytes();a['native_identity'].update(build_metadata_sha256=hashlib.sha256(raw).hexdigest(),build_metadata=json.loads(raw))
        except (OSError,ValueError):a['native_identity'].update(build_metadata_sha256=None,build_metadata=None)
    return {k:a.get(k) for k in IDENTITY_FIELDS}
def validate_identity(a):
    missing=[k for k in IDENTITY_FIELDS if k not in a or a[k] is None]
    if missing:fail('PRODUCT_REGISTRATION_IDENTITY_MISSING',missing)
    from .deployment_identity import registration_identity_issues
    bad=registration_identity_issues(a)
    for k in ('model_config_sha256','implementation_sha256'):
        if not sha(a[k]):bad.append(k)
    for k in ('source_fingerprints',):
        if not isinstance(a[k],dict) or not a[k] or not all(sha(v) for v in a[k].values()):bad.append(k)
    if type(a['runtime_libraries']) is not dict or set(a['runtime_libraries'])!={'libascendcl.so','libruntime.so'}:bad.append('runtime_libraries')
    else:
        for n,v in a['runtime_libraries'].items():
            if type(v) is not dict or not sha(v.get('sha256')) or not sha(v.get('metadata_sha256')) or type(a['cann']) is not list or v.get('version') not in a['cann']:bad.append('runtime_libraries.'+n)
    if any(type(a[k]) is not int or a[k]<1 for k in ('tp','pp','dp')) or type(a['rank']) is not int or not 0<=a['rank']<a['tp']:bad.append('parallel')
    if a['execution_mode']!='product' or a['live_config_matches_file'] is not True:bad.append('execution_mode/live_config')
    for k in ('hardware_family','hardware_device_name','torch','torch_npu','vllm','vllm_ascend','python','dtype','model_class','variant_revision'):
        if type(a[k]) is not str or not a[k]:bad.append(k)
    if type(a['hardware_logical_device']) not in (int,str) or a['hardware_logical_device']=='':bad.append('hardware_logical_device')
    for k,want in [('eager',True),('async_scheduling',False),('prefix_caching',False),('mtp_disabled',True),('kv_connector_disabled',True)]:
        if a[k] is not want:bad.append(k)
    if a['ep']!={'enabled':False,'placement':[],'routing':'NOT_APPLICABLE'}:bad.append('ep:EP_REGISTRATION_NOT_ADAPTED')
    for k in ('max_model_len','max_num_batched_tokens','max_num_seqs'):
        if type(a[k]) is not int or a[k]<1:bad.append(k)
    rr=a['request_range']
    if type(rr) is not dict or set(rr)!={'max_context','max_tokens','max_num_seqs','single_chunk','image_grid'} or rr['single_chunk'] is not True or rr['max_tokens']!=4 or type(rr['max_context']) is not int or type(rr['max_num_seqs']) is not int:bad.append('request_range')
    ni=a['native_identity']
    if type(ni) is not dict or type(ni.get('applicable')) is not bool:bad.append('native_identity')
    elif ni['applicable'] is False:
        if set(ni)!={'applicable'}:bad.append('native_identity')
    else:
        bm=ni.get('build_metadata')
        if not all(sha(ni.get(k)) for k in ('sha256','declared_sha256','build_metadata_sha256')) or ni.get('sha256')!=ni.get('declared_sha256'):bad.append('native_identity:NATIVE_BUILD_REGISTRATION_PENDING')
        if (type(bm) is not dict or set(bm)!={'source_release','source_commit','source_files','compiler_version','cann','recipe_sha256','receipt_sha256','library_sha256'}
          or not all(type(bm[k]) is str and bm[k] for k in ('source_release','source_commit','compiler_version'))
          or bm['cann'] not in a['cann'] or bm['library_sha256']!=ni.get('sha256')
          or not all(sha(bm[k]) for k in ('recipe_sha256','receipt_sha256','library_sha256'))
          or type(bm['source_files']) is not dict or not bm['source_files'] or not all(sha(x) for x in bm['source_files'].values())):bad.append('native_identity:NATIVE_BUILD_REGISTRATION_PENDING')
    if bad:fail('PRODUCT_REGISTRATION_IDENTITY_MISSING',bad)
def validate_record(r):
    required={'schema','source','mechanism','expected_ranks','evidence','record_sha256'}
    if type(r) is not dict or set(r)!=required or r['schema']!='dcat.product.registration.v1' or r['source'] not in ('BLUE_COMBINATION','YELLOW_DEDICATED_INSTANCE'):fail('PRODUCT_REGISTRATION_RECORD_INVALID')
    if r['record_sha256']!=digest({k:v for k,v in r.items() if k!='record_sha256'}):fail('PRODUCT_REGISTRATION_RECORD_INVALID')
    from .config import FaultConfig
    number,variant=r['mechanism'].split('.',1);FaultConfig('quant' if number=='quant' else int(number),variant)
    rows=r['expected_ranks']
    if type(rows) is not list or not rows:fail('PRODUCT_REGISTRATION_RECORD_INVALID')
    for a in rows:validate_identity(a)
    if [a['rank'] for a in rows]!=list(range(len(rows))) or any(a['tp']!=len(rows) for a in rows):fail('PRODUCT_REGISTRATION_RECORD_INVALID')
    common=lambda a:{k:v for k,v in a.items() if k not in ('rank','hardware_logical_device')}
    if any(common(a)!=common(rows[0]) for a in rows):fail('PRODUCT_REGISTRATION_RECORD_INVALID')
    e=r['evidence']
    if (not isinstance(e,dict) or e.get('review_passed') is not True or e.get('mode')!='product'
      or e.get('level') not in ('BLUE_ORIGINAL_REVIEWED','YELLOW_TEXT_ATTESTED')
      or len(set(e.get('owners',[])))!=2 or not all(sha(e.get(k)) for k in ('receipt_sha256','reviewer_sha256','source_manifest_sha256'))
      or not all(e.get(k) is True for k in ('consumers','bytes_restored','resources_zero','canary_recovered','negative_controls'))):fail('PRODUCT_REGISTRATION_RECORD_INVALID')
    coverage=e.get('coverage',[])
    expected={(owner,rank) for owner in e['owners'] for rank in range(len(rows))}
    if len(coverage)!=len(expected) or {(c.get('owner'),c.get('rank')) for c in coverage}!=expected:fail('PRODUCT_REGISTRATION_RECORD_INVALID')
    for row in coverage:
        c=row.get('counts',{})
        keys=('observed','eligible','attempted','applied','changed','consumed','restored','failed')
        if set(c)!=set(keys) or any(type(c[k]) is not int or c[k]<0 for k in keys):fail('PRODUCT_REGISTRATION_RECORD_INVALID')
        if not (c['eligible']>0 and c['observed']>=c['eligible']==c['attempted']==c['applied']==c['consumed']==c['restored'] and c['changed']<=c['applied'] and c['failed']==0):fail('PRODUCT_REGISTRATION_RECORD_INVALID')
    return r
def load_records():
    obj=json.loads(RECORDS.read_bytes())
    if set(obj)!={'schema','records'} or obj['schema']!='dcat.product.registry.v1':fail('PRODUCT_REGISTRATION_RECORD_INVALID')
    rows=[validate_record(r) for r in obj['records']]
    if len({r['record_sha256'] for r in rows})!=len(rows):fail('PRODUCT_REGISTRATION_RECORD_INVALID')
    return rows
def select_registration(mechanism,source,actual,records=None):
    validate_identity(actual)
    rows=[r for r in (load_records() if records is None else [validate_record(x) for x in records]) if r['source']==source and r['mechanism']==mechanism]
    if not rows:fail('PRODUCT_COMBINATION_NOT_REGISTERED')
    mismatches=[]
    for r in rows:
        if actual['rank']>=len(r['expected_ranks']):mismatches.append(['tp','rank']);continue
        expected=r['expected_ranks'][actual['rank']]
        diff=[k for k in IDENTITY_FIELDS if actual[k]!=expected[k]]
        if not diff:return r
        mismatches.append(diff)
    best=min(mismatches,key=len)
    code='PRODUCT_REGISTRATION_PARALLEL_SCOPE_MISMATCH' if any(k in PARALLEL for k in best) else 'PRODUCT_REGISTRATION_DEPLOYMENT_RANGE_MISMATCH' if any(k in RANGE for k in best) else 'PRODUCT_REGISTRATION_IDENTITY_MISMATCH'
    fail(code,best)
def registered_for(worker,mechanism,profile,*,expected_sha=None):
    authority=authority_for(worker)
    if authority is None:fail('TRUSTED_PRODUCT_DEPLOYMENT_AUTHORITY_REQUIRED')
    if profile['mode']!='PRODUCT_REGISTERED':fail('PRODUCT_REGISTRATION_MODE_MISMATCH')
    if hasattr(worker,_BLUE):fail('PRODUCT_REGISTRATION_MODE_MIXED_WITH_PERMIT')
    actual=observe_identity(worker,mechanism,profile.get('native_library'))
    record=select_registration(mechanism,authority.source,actual)
    if expected_sha is not None and record['record_sha256']!=expected_sha:fail('PRODUCT_REGISTRATION_IDENTITY_MISMATCH',['record_sha256'])
    return dict(record_sha256=record['record_sha256'],actual=actual,source=authority.source,evidence=record['evidence'],mode='PRODUCT_REGISTERED')
