"""Read-only phase1 validation: actual outputs/launch/POD, not policy intent.

No Torch import, device execution, or certificate issuance. Private output
binaries are read only and never copied into the public statistical evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import time

SOURCE_SHA = '15bdb7b6e63ddec53f9d3d9a4c31ef4217732f7fdcf46f724cbad52099a5a3f3'
BUILD_SHA = '42173485d5d3c86b57cd8a847dbce2dfd1fc42004c1a451bb282b7aa211d9ef1'
RUNTIME_SHA = '6982e9b83717b03349fa57707f635dfcfc3bc46ae778d676442c03bc734b6e7a'
READINESS_SHA = 'fd58e7db675265448dc805f8687f808e37ec233ca147a77e98b48cd889d7458a'
GENERATION_SHA = 'a6668b40f9411bacf9d64239ce34302a801ee851a81ba5b9fb7dd0dc0ae5976c'
INPUT_SHA = '0e983abac01a72dc4f6039211c27fbce6fb573bcdcebe8f4051e1b7e8735e03d'
SCHEMA_SHA = 'f5865ad7185b7fabeb71b37aababb2b1719f79861ed70f487351dbac7248638b'
COLLECTOR_SHA = '743354feff5a5f91a17e68695bad167fa2584a48d257ec227d0973d071f185af'
PROCESSOR_SHA = '3aa7ede049bc6cb4d54a572a85f907214a98890a3c9544ed7f75e25df1306141'
TARGETS = ('query_tile', 'key_group', 'joint')
RECIPES = {'query_tile': (64, 8), 'key_group': (128, 4), 'joint': (64, 4)}
PADDING = {'inputParams.remainPH', 'inputParams.attenMaskS2SizePH',
           'inputParams.qStartIdxPH', 'inputParams.InputParamsPH', 'coreParams.s1SparseValidSizePH'}
OPAQUE = {'@raw.softmaxFlashTilingData':(1248,64), '@raw.transposeTilingData':(1312,64),
          '@raw.transposeTilingDataTailCore':(1376,64)}
SOURCE_FILES = ('__init__.py', '__main__.py', 'models.py', 'config.py', 'policies.py', 'registry.py',
    'certificates.py', 'protocol.py', 'session.py', 'discovery.py', 'certificate_tools.py',
    'source_fingerprints.py', 'adapters/__init__.py', 'adapters/fa_schema.py', 'adapters/flash_attention.py')
FORMATS = {'int64_t':'q','uint64_t':'Q','int32_t':'i','uint32_t':'I',
           'int16_t':'h','uint16_t':'H','int8_t':'b','uint8_t':'B','float':'f'}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def child(root, name):
    root = Path(root).resolve(strict=True); p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts, 'Non-relative evidence path')
    result = (root / p).resolve(strict=True)
    require(result.is_relative_to(root), 'Evidence escaped root')
    return result


def bound(path, expected, count=None):
    require(sha(path) == expected, 'SHA mismatch: ' + str(path))
    if count is not None: require(Path(path).stat().st_size == count, 'Size mismatch: ' + str(path))


def strict_completion(c):
    require(c.get('status') == 'CAPTURE_COMPLETED', 'Whole attempt failed/incomplete; partial cases cannot count')
    require(c.get('card') == 5 and c.get('cleanup_errors') == [] and c.get('owned_container_removed') is True
        and c.get('device_lock_acquired') is True and c.get('device_lock_released') is True
        and c.get('external_container_mutations') is False and c.get('owned_container_id'), 'Controller cleanup/resource scope failed')


def unchanged(before, after):
    """Only observed native format0->2 normalization; no value/identity waiver."""
    require(set(before) == set(after), 'Input observation fields changed')
    differences = [k for k in before if before[k] != after[k]]
    require(not differences or (differences == ['actual_format'] and before['actual_format'] == 0
        and after['actual_format'] == 2), 'Native call mutated input values/geometry/identity: '+str(differences))


def semantic(pod):
    require(len(pod) == 205 and PADDING <= set(pod), 'Incomplete actual public POD')
    return {k:v for k,v in pod.items() if k not in PADDING}


def recipe(pod):
    return pod['coreParams.s1BaseSize'], pod['coreParams.nRatio']


def native_kwargs_equal(actual, frozen):
    expected=dict(frozen)
    for key in ('pse','padding_mask','prefix','actual_seq_qlen','actual_seq_kvlen','sink'):
        require(expected.get(key) is None,'Frozen optional input unexpectedly present')
        expected[key]=None
    require(actual==expected,'Actual precision/mask/API scalars changed')


def check_outputs_equal(left, right, reason):
    require(left['raw'] == right['raw'], reason + ': complete private batch0 output differs')
    require(semantic(left['pod']) == semantic(right['pod']), reason + ': semantic actual POD differs')
    require(left['step']['output']['per_batch_sha256'] == right['step']['output']['per_batch_sha256'],
        reason + ': other batch slot output hash differs')


def binary_difference(a, b):
    require(len(a) == len(b) == 16*2048*128*2, 'Expected complete FP16 batch0 output')
    if a == b:
        return {'bitwise_equal':True,'changed_elements':0,'changed_queries':0,'max_abs':0.0,
                'nonfinite_pairs':0,'first_changed':None}
    def units(bits):
        sign, exponent, fraction = bits >> 15, (bits >> 10) & 31, bits & 1023
        if exponent == 31: return None
        x = fraction if exponent == 0 else (1024+fraction) << (exponent-1)
        return -x if sign else x
    count, queries, maximum, invalid, first = 0, set(), 0, 0, None
    for index, (left, right) in enumerate(zip(struct.iter_unpack('<H', a), struct.iter_unpack('<H', b))):
        x, y = left[0], right[0]
        if x == y: continue
        count += 1; queries.add((index//128) % 2048)
        if first is None: first = {'head':index//(2048*128),'query':(index//128)%2048,'channel':index%128}
        x, y = units(x), units(y)
        if x is None or y is None: invalid += 1
        else: maximum = max(maximum, abs(x-y))
    return {'bitwise_equal':False,'changed_elements':count,'changed_queries':len(queries),
            'max_abs':maximum*2.0**-24,'nonfinite_pairs':invalid,'first_changed':first}


class Validator:
    def __init__(self, root, audit_dir):
        self.root = Path(root).resolve(strict=True); self.audit_dir = Path(audit_dir)
        self.evidence = {}; self.generation = None; self.cases = {}; self.native_module = None
        self.audit_dir.mkdir(parents=True, exist_ok=False)
        self.record(self.root/'outputs/controller_plan_a3/readiness.json',READINESS_SHA)
        self.runtime = read(self.root/'runtime_manifest.json'); self.record(self.root/'runtime_manifest.json', RUNTIME_SHA)
        require(self.runtime['schema'] == 'nes.runtime_snapshot.v1', 'Unexpected runtime inventory')
        actual = {p.relative_to(self.root/'runtime').as_posix() for p in (self.root/'runtime').rglob('*') if p.is_file()}
        require(actual == set(self.runtime['files']), 'Runtime inventory changed')
        for name,row in self.runtime['files'].items():
            require(not (self.root/'runtime'/name).is_symlink(), 'Symlink in frozen runtime')
            self.record(child(self.root/'runtime',name), row['sha256'], row['bytes'])
        self.python_sources = {n:self.runtime['files']['native_execution_strategy/'+n]['sha256'] for n in SOURCE_FILES}
        schema_path = self.root/'runtime/native_execution_strategy/adapters/fa_cann91_schema.json'
        self.record(schema_path, SCHEMA_SHA); self.schema = read(schema_path)
        require(self.schema['size'] == 1464 and len(self.schema['fields']) == 205, 'Wrong public POD')
        self.input_fields = {f['name'] for f in self.schema['fields'] if f['name'].startswith('inputParams.') and f['name'] not in PADDING}
        self.plan_fields = {f['name'] for f in self.schema['fields'] if not f['name'].startswith('inputParams.') and f['name'] not in PADDING}
        require(len(self.input_fields) == 34 and len(self.plan_fields) == 166, 'Wrong semantic field split')
        self.record(self.root/'inputs/manifest.json', INPUT_SHA); self.inputs = read(self.root/'inputs/manifest.json')
        for row in self.inputs['inputs'].values():
            p = child(self.root/'inputs',row['file']); self.record(p,row['sha256'],row['bytes']); raw = p.read_bytes()
            require(all(hashlib.sha256(raw*b).hexdigest() == row['expected_full_sha256'][str(b)] for b in (1,2,4,8)), 'Input batch construction hash differs')
        for filekey,hashkey in [('source_call_file','source_call_sha256'),('parent_manifest_file','parent_manifest_sha256')]:
            self.record(child(self.root/'inputs',self.inputs[filekey]),self.inputs[hashkey])
        self.mask_sha = hashlib.sha256(b''.join(bytes(q+1)+bytes([1])*(2047-q) for q in range(2048))).hexdigest()
        self.expected_cases = {}
        for phase in ('discovery','controlled'):
            directory = self.root/'setup'/phase; self.record(directory/'manifest.json'); matrix = read(directory/'manifest.json')
            for row in matrix['cases']:
                p = child(directory,row['file']); self.record(p,row['sha256']); c=read(p)
                require(c['case_id']==row['case_id'], 'Case manifest identity differs'); self.expected_cases[c['case_id']] = c
        require(len(self.expected_cases)==20, 'Preregistered 20-case matrix missing'); self.verify_vendor()

    def record(self,path,expected=None,count=None):
        p=Path(path).resolve(strict=True); require(p.is_relative_to(self.root), 'Evidence outside task root: '+str(p)); digest=sha(p)
        if expected is not None: require(digest==expected,'SHA mismatch: '+str(p))
        if count is not None: require(p.stat().st_size==count,'Size mismatch: '+str(p))
        self.evidence[p.relative_to(self.root).as_posix()]={'sha256':digest,'bytes':p.stat().st_size}
        return digest

    def verify_vendor(self):
        p=self.root/'native_vendor_stage_a1/vendor_manifest.json'; self.record(p,BUILD_SHA); v=read(p)
        require(v['policy_source_manifest_sha256']==SOURCE_SHA and v['schema']=='nes.fa.private_vendor.v1', 'Native bridge version differs')
        vr=child(self.root,v['vendor_root']); names=set()
        for row in v['artifacts']:
            require(row['path'] not in names,'Duplicate vendor artifact'); names.add(row['path']); p=vr/row['path']
            require(p.is_symlink()==(row['symlink'] is not None),'Vendor symlink type changed')
            if p.is_symlink(): require(str(p.readlink())==row['symlink'],'Vendor symlink target changed')
            self.record(child(vr,row['path']),row['sha256'],row['bytes'])
        require({p.relative_to(vr).as_posix() for p in vr.rglob('*') if p.is_file()}==names,'Vendor inventory changed')
        self.vendor_count=len(names)

    def verify_generation(self,attempt):
        path=attempt/'source_manifest.json'; self.record(path); sources=read(path)
        canonical=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        require(len(sources)==59 and canonical==GENERATION_SHA,'Execution source generation is not the approved complete-POD generation')
        if self.generation is None: self.generation=sources
        require(sources==self.generation,'Cannot merge different frozen execution generations')
        for name,digest in sources.items():
            self.record(child(attempt/'source',name),digest); self.record(child(self.root,name),digest)
        if self.native_module is None:
            spec=importlib.util.spec_from_file_location('nes_pinned_native_evidence',attempt/'source/native_evidence.py')
            self.native_module=importlib.util.module_from_spec(spec); spec.loader.exec_module(self.native_module)

    def add_attempt(self,name):
        attempt=child(self.root/'outputs',name); completion=read(attempt/'completion.json')
        self.record(attempt/'completion.json'); strict_completion(completion); self.verify_generation(attempt)
        self.record(attempt/'before.json'); self.record(attempt/'container_identity.json')
        for row in completion['cases']:
            cid=row['case_id']; require(cid not in self.cases and cid in self.expected_cases,'Duplicate/unknown case '+cid)
            require(row.get('status')=='CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE' and row.get('rc')==0,'Incomplete case '+cid)
            folder=child(attempt,cid); self.record(folder/'result.json',row['result_sha256']); d=read(folder/'result.json')
            require(d['status']=='CAPTURE_COMPLETED' and d['case_id']==cid,'Case result incomplete')
            self.cases[cid]=self.verify_case(attempt,folder,d,self.expected_cases[cid])

    def verify_case(self,attempt,folder,d,c):
        cid=c['case_id']; modes=['native'] if c['phase']=='discovery' else ['native','controlled','controlled','off']
        require(c['mode_sequence']==modes and d['original_api_calls']==len(modes)==len(d['steps']), 'Original call count/modes differ')
        require(d['model_forward'] is False and d['execution_kind']=='original_operator_execution','Unexpected model execution')
        require(d['source_case_sha256']==self.evidence['setup/'+c['phase']+'/'+cid+'.json']['sha256'],'Actual case differs')
        require(d['runtime_manifest_sha256']==RUNTIME_SHA and d['replay_source_sha256']==self.generation['replay.py'],'Actual runtime differs')
        native_kwargs_equal(d['native_kwargs'],self.inputs['native_kwargs'])
        context=d['invocation_context']; b=c['batch_size']
        require(context['operator']=='FlashAttentionScore' and context['request_count']==b and context['token_count']==b*2048
            and context['seq_lens']==[2048]*b and context['dtype']=='float16' and context['layout']=='BNSD'
            and context['matrix_dims']=={'query_heads':16,'kv_heads':8,'query_tokens':2048,'kv_tokens':2048,'head_dim':128},'Explicit request/token/matrix semantics differ')
        device=d['physical_device_binding']
        require(device['device']==device['public_api_device_name']=='Ascend910B4' and device['physical_card']==5 and device['logical_device']==0,'Wrong physical device')
        self.record(attempt/'before.json',device['preflight_sha256']); self.record(attempt/'container_identity.json',device['container_identity_sha256'])
        require(d['input_unchanged'] is True and d['host_scope_restored'] is True,'Capture integrity not completed')
        for module,row in d['runtime_imports'].items():
            require(row['path'].startswith('/audit/runtime/'),'Actual import outside pinned runtime')
            name=row['path'][len('/audit/runtime/'):]
            require(row['sha256']==self.runtime['files'][name]['sha256'],'Actual imported module changed')
        self.verify_statistics(folder,d,len(modes)); rows=[]; previous=None
        for i,s in enumerate(d['steps']):
            require(s['step']==i and s['mode']==modes[i],'Step order differs')
            sp=folder/'steps'/f'step{i}'; self.record(sp/'result.json'); require(read(sp/'result.json')==s,'Embedded and original step result differ')
            call=s['native_call']; require(call['case_id']==cid and call['step']==i and call['mode']==s['mode']
                and call['batch_size']==b and call['api']=='torch_npu.npu_fusion_attention_v3'
                and call['phase']=='operator_execution' and call['status']=='returned','Actual invocation identity differs')
            if previous is not None:
                require(previous['native_call']['pid']==call['pid'] and previous['native_call']['thread_id']==call['thread_id'],'Case changed PID/TID')
                require(previous['native_call']['exit_monotonic_ns']<call['enter_monotonic_ns'],'Overlapping/out-of-order invocations')
                require(previous['inputs_after']==s['inputs_before'] and previous['mask_after']==s['mask_before'],'Inputs reset/replaced between invocations')
            for key,row in self.inputs['inputs'].items():
                before=s['inputs_before'][key]; after=s['inputs_after'][key]; unchanged(before,after)
                require(before['shape']==[b,*row['shape'][1:]] and before['stride']==row['source_stride']
                    and before['dtype']=='torch.float16' and before['device']=='npu:0' and before['storage_offset']==0
                    and before['full_sha256']==row['expected_full_sha256'][str(b)]
                    and before['target_sha256']==row['sha256'] and before['per_batch_sha256']==[row['sha256']]*b,
                    'Actual complete QKV values/geometry differ: '+key)
            unchanged(s['mask_before'],s['mask_after'])
            require(s['mask_before']['full_sha256']==self.mask_sha and s['mask_before']['dtype']=='torch.bool'
                and s['mask_before']['shape']==[2048,2048],'Actual legal causal key set changed')
            meta=child(folder,s['step_metadata_file']); self.record(meta,s['step_metadata_sha256']); md=read(meta)
            require(md['resource_meter_source_sha256']==self.generation['resource_meter.py'] and md['native_call']==call
                and md['resource_measurement']==s['resource_measurement'],'Resource measurement not bound to actual call')
            resources,_,_=self.native_module.bound_resources(meta,call)
            require(resources==s['resources'],'Resource scalar differs from window/allocator reconstruction')
            data=self.verify_native(attempt,folder,sp,s,c); data.update(step=s,case_id=cid,case=c,directory=folder)
            private=s['private_target_output']; rawpath=child(sp,private['file']); self.record(rawpath,private['sha256'],16*2048*128*2)
            require(private['shape']==[16,2048,128] and private['dtype']=='torch.float16' and private['batch_slot']==0
                and private['byteorder']=='little','Wrong complete target output mapping')
            out=s['output'];require(out['shape']==[b,16,2048,128] and out['dtype']=='torch.float16'
                and out['target_sha256']==private['sha256']==out['per_batch_sha256'][0]
                and len(out['per_batch_sha256'])==b,'Actual output mapping changed')
            data['raw']=rawpath.read_bytes(); data['output_file']=str(rawpath.relative_to(self.root))
            require(s['assessment']['model_impact'] is None,'Operator-only experiment cannot establish model impact')
            if c['phase']=='controlled': self.verify_session(sp,s,i,data)
            else:
                require(call['scope_token'] is None and s['typed_plan_intent'] is None and s['policy_decision'] is None
                    and s['session_execution'] is None and not list(sp.rglob('host_events*.jsonl')),'Stock must have no private control scope')
            rows.append(data); previous=s
        require(len({x['step']['native_call']['call_id'] for x in rows})==len(rows),'Invocation token reused')
        return {'result':d,'case':c,'rows':rows,'directory':folder}

    def verify_statistics(self,folder,result,steps):
        self.record(folder/'receipt.json',result['receipt_sha256']); receipt=read(folder/'receipt.json')
        self.record(folder/'config.json',receipt['config_sha256']); config=read(folder/'config.json')
        require(config['task']=='statistics' and config['level']=='mix' and config['precision']=='high'
            and config['step']==list(range(steps)) and config['rank']==[] and config['async_dump'] is False
            and config['statistics']=={'scope':[],'list':[],'data_mode':['all'],'summary_mode':'statistics','tensor_list':[]},'Actual msprobe configuration differs')
        collector=read(folder/'collector_provenance.json');self.record(folder/'collector_provenance.json',receipt['collector_provenance_sha256'])
        require(collector==receipt['statistics_collector']==result['statistics_collector']
            and collector['variant']=='task_local_fp32_high_v1' and collector['modified_official_collector'] is True
            and collector['manifest_sha256']==COLLECTOR_SHA and collector['patched_processor_sha256']==PROCESSOR_SHA
            and collector['actual_loaded']['sha256']==PROCESSOR_SHA,'Task-local collector provenance differs')
        expected={f'observation/step{i}/rank0/{name}.json' for i in range(steps) for name in ('dump','stack','construct')}
        require({r['path'] for r in receipt['native_files']}==expected and len(receipt['native_files'])==len(expected),'Incomplete raw statistics triplets')
        require(receipt['steps']==list(range(steps)) and receipt['rank']==0,'Receipt step/rank differs')
        for row in receipt['native_files']:self.record(child(folder,row['path']),row['sha256'],row['bytes'])
        require(all(p.suffix=='.json' for p in (folder/'observation').rglob('*') if p.is_file()),'Tensor payload in statistics tree')

    def verify_native(self,attempt,folder,sp,s,c):
        profile=sp/'native_profile'; private=c['phase']=='controlled'; call=s['native_call']
        file=profile/('verified_record.json' if private else 'stock_discovery.json');self.record(file); record=read(file)
        if not private:require(record==s['native_observation_only'],'Stock embedded observation differs')
        else:
            require(record['native_launch']['framing_verified_against_public_context'] is True,'Private raw not framed by public context')
            for ref in record['evidence']:
                prefix='/evidence/'+folder.name+'/'
                require(ref['path'].startswith(prefix),'Native evidence path outside invocation')
                self.record(child(folder,ref['path'][len(prefix):]),ref['sha256'])
        with tempfile.TemporaryDirectory(prefix='native-reparse-',dir=self.audit_dir) as temp:
            actual=self.native_module.actual_launch(attempt/(folder.name+'.log'),call,Path(temp)/'native_window.log')
        native=record['native_launch']; ignored={'window_file','source_log','framing_verified_against_public_context'}
        require({k:v for k,v in actual.items() if k not in ignored}=={k:v for k,v in native.items() if k not in ignored},'Original CANN log reparse differs')
        self.record(profile/'native_window.log',native['window_sha256'])
        raw=bytes.fromhex(native['raw_hex']);require(len(raw)==1464 and native['raw_size']==1464
            and hashlib.sha256(raw).hexdigest()==native['raw_sha256'] and str(native['tiling_key'])=='22619644976','Wrong matched route/raw ABI')
        pod={}
        for field in self.schema['fields']:
            values=struct.unpack_from('<'+str(field['count'])+FORMATS[field['type']],raw,field['offset'])
            pod[field['name']]=values[0] if len(values)==1 else list(values)
        require(pod==record['pod'],'205 POD fields differ from original launched bytes')
        expected={'inputParams.bSize':c['batch_size'],'inputParams.n2Size':8,'inputParams.gSize':2,
            'inputParams.s1Size':2048,'inputParams.s2Size':2048,'inputParams.dSize':128,'inputParams.keepProb':1,
            'inputParams.preTokens':2048,'inputParams.nextTokens':0,'inputParams.layoutType':3,
            'inputParams.implMode':0,'inputParams.sparseType':3,'inputParams.needDropMaskOp':0,
            'inputParams.scaleValue':struct.unpack('<f',struct.pack('<f',128**-0.5))[0]}
        require(all(pod[k]==v for k,v in expected.items()),'Native dtype/layout/precision/causal semantics changed')
        libraries=[x['path'] for x in actual['loaded_libraries_before_exit']]
        if private:
            host=record['host_event']; events=[json.loads(x) for x in (profile/'host_events.original.jsonl').read_text().splitlines() if x.strip()]
            original=sp/'host_events.jsonl';self.record(original)
            require(original.read_bytes()==(profile/'host_events.original.jsonl').read_bytes(),'Original public host event changed')
            require(events==[host] and read(profile/'host_event.json')==host,'Expected one exact public callback')
            require(host['callback_status']==0 and host['eligible'] is True and host['raw_hex']==native['raw_hex']
                and host['raw_size']==1464 and host['pid']==call['pid'] and host['tid']==call['thread_id']
                and host['token']==call['scope_token']==call['call_id'],'Host callback/raw/caller binding failed')
            for key in ('tiling_key','block_dim','workspace_count'):require(str(host[key])==str(native[key]),'Host/launch context differs: '+key)
            plan=host['plan'];require(plan==s['typed_plan_intent'] and plan['certificate']['source_manifest_sha256']==SOURCE_SHA
                and plan['certificate']['build_manifest_sha256']==BUILD_SHA and plan['scope']['request_count']==c['batch_size']
                and plan['scope']['pid']==call['pid'] and plan['scope']['native_tid']==call['thread_id']
                and plan['scope']['token']==call['call_id'],'Actual typed scope/version differs')
            require(host['native_factors']=={'m':128,'n_ratio':8},'Observed native baseline differs')
            require(host['selected_factors']=={'m':recipe(pod)[0],'n_ratio':recipe(pod)[1]},'Selector intent differs from actual POD')
            changed=s['mode']=='controlled'; wanted=RECIPES[c['selection_target']] if changed else (128,8)
            require(recipe(pod)==wanted and host['selection_applied'] is changed and host['mode']==('controlled' if changed else 'native'),'Repeated eligible/off actual selection differs')
            require(plan['requested']['selected']==({'m':wanted[0],'n_ratio':wanted[1]} if changed else None),'Typed request differs')
            require(any(c['approved_custom_opp_path'] in x and 'libcust_opmaster' in x for x in libraries),'Missing custom operator load evidence')
        else:
            require(record['host_callback_available'] is False and record['controlled_support_granted'] is False,'Stock observation improperly controlled')
            if c['batch_size'] in (1,8): require(recipe(pod)==(128,8),'Unexpected B1/B8 stock baseline for phase1 recipes')
            require(not any('/vendors/native_execution_strategy_transformer/' in x for x in libraries),'Stock loaded private bridge')
        return {'pod':pod,'native':native,'resources':s['resources'],'padding':{k:pod[k] for k in sorted(PADDING)}}

    def verify_session(self,sp,s,index,data):
        self.record(sp/'session_execution.json'); execution=read(sp/'session_execution.json')
        require(execution==s['session_execution'] and execution['entry_point']=='StrategySession.execute'
            and execution['original_return_identity'] is True and execution['observation_error'] is None
            and execution['passthrough_reason'] is None and execution['observation_present'] is True,'Unified Session failed')
        require(execution['mode']==('native' if index==0 else 'controlled') and execution['policy_enabled'] is (index!=3),'Session mode differs')
        require(s['selection_current_observation_step']==(None if index==0 else 0),'Controlled result replaced native baseline')
        decision=s['policy_decision'];require(execution['decision']==decision,'Session decision changed')
        if index==0:require(decision is None,'Native observation chose a policy')
        else:require(decision is not None and (decision['action']=='request_strategy') is (index in (1,2)),'Every-match/off decision differs')
        switch=None if index==0 else index in (1,2)
        require(execution['assessment']['strategy_switched'] is switch and s['assessment']['strategy_switched'] is switch,'Actual switch assessment differs')
        require(execution['assessment']['numerics_observed'] is None and execution['assessment']['model_impact'] is None,'Session fabricated numerical/model proof')
        observation=s['native_observation']
        require(observation['strategy']['parameters']['execution_plan']==self.plan(data),'Session actual plan differs from native POD')
        require(observation['fingerprints']['source_sha256']==SOURCE_SHA and observation['fingerprints']['build_sha256']==BUILD_SHA,'Session fingerprint differs')

    def plan(self,row):
        raw=bytes.fromhex(row['native']['raw_hex']);require(len(raw)==1464,'Missing complete raw plan')
        result={k:row['pod'][k] for k in self.plan_fields}
        result.update({k:raw[offset:offset+size].hex() for k,(offset,size) in OPAQUE.items()})
        require(len(result)==169,'Incomplete typed/opaque execution plan')
        return result

    def compare(self,full):
        expected={k for k,c in self.expected_cases.items() if full or c['phase']=='discovery'}
        require(set(self.cases)==expected,'Matrix incomplete/extra: '+str(sorted(expected-set(self.cases))))
        comparisons=[]; contexts=[]
        def compare(label,a,b,equal=True):
            if equal:
                check_outputs_equal(a,b,label)
                require(self.plan(a)==self.plan(b),label+': full dependent plan differs')
            diff=binary_difference(a['raw'],b['raw'])
            aa={x['name']:x for x in a['step']['auxiliary_outputs']};bb={x['name']:x for x in b['step']['auxiliary_outputs']}
            require(set(aa)==set(bb)=={'softmax_max','softmax_sum','softmax_out','seed','offset'},'Incomplete native six-return accounting')
            aux=[n for n in sorted(aa) if aa[n].get('full_sha256')!=bb[n].get('full_sha256') or aa[n].get('value')!=bb[n].get('value')]
            comparisons.append({'id':label,'B_case':a['case_id'],'B_step':a['step']['step'],'F_case':b['case_id'],
                'F_step':b['step']['step'],'required_equal':equal,'primary_output':diff,'auxiliary_differences':aux,
                'semantic_strategy_equal':self.plan(a)==self.plan(b),
                'raw_padding_differences':[k for k in sorted(PADDING) if a['pod'][k]!=b['pod'][k]],'model_impact':None})
        for batch in (1,2,4,8):
            compare('discovery_repeat_b'+str(batch),self.cases[f'discovery_b{batch}_r0']['rows'][0],self.cases[f'discovery_b{batch}_r1']['rows'][0])
        if full:
            for batch in (1,8):
                recipes={};input_values=None
                for target in TARGETS:
                    for repeat in (0,1):
                        case=self.cases[f'controlled_{target}_b{batch}_r{repeat}'];rows=case['rows']
                        stock=self.cases[f'discovery_b{batch}_r{repeat}']['rows'][0]
                        compare(case['case']['case_id']+'_stock_native',stock,rows[0])
                        compare(case['case']['case_id']+'_every_match_repeat',rows[1],rows[2])
                        compare(case['case']['case_id']+'_off_recovery',rows[0],rows[3])
                        for index,row in enumerate(rows):
                            values={k:row['pod'][k] for k in self.input_fields}
                            if input_values is None:input_values=values
                            require(values==input_values,'Same-batch native input/precision contract changed')
                            key=recipe(row['pod']);plan=self.plan(row)
                            require(key not in recipes or recipes[key]==plan,'Recipe had inconsistent dependent execution plan')
                            recipes[key]=plan
                            if index:
                                require(row['step']['assessment']['numerics_observed'] is (row['raw']!=rows[0]['raw']),
                                    'Runtime numerical assessment differs from complete output bytes')
                            if index in (1,2):compare(case['case']['case_id']+'_effect_s'+str(index),rows[0],row,False)
                    a=self.cases[f'controlled_{target}_b{batch}_r0']['rows'];b=self.cases[f'controlled_{target}_b{batch}_r1']['rows']
                    for index in range(4):compare(f'{target}_b{batch}_independent_repeat_s{index}',a[index],b[index])
                require(set(recipes)=={(128,8),(64,8),(128,4),(64,4)},'Actual four-strategy coverage incomplete')
                contexts.append({'request_count':batch,'input_contract':input_values,'strategies':[
                    {'strategy_id':f'fa.s1s2.m{m}.r{r}','parameters':{'query_tile':m,'key_group_ratio':r,'key_tile':128,'execution_plan':recipes[(m,r)]}}
                    for m,r in ((128,8),(64,8),(128,4),(64,4))]})
        discovery=[]
        for batch in (1,2,4,8):
            x=self.cases[f'discovery_b{batch}_r0']['rows'][0]
            discovery.append({'request_count':batch,'recipe':list(recipe(x['pod'])),'actual_execution_plan':self.plan(x),
                'output_target_sha256':x['step']['output']['target_sha256'],'per_batch_output_sha256':x['step']['output']['per_batch_sha256'],
                'tiling_key':x['native']['tiling_key'],'block_dim':x['native']['block_dim'],'controlled_support_granted':False})
        resources=[{'case_id':cid,'step':r['step']['step'],**r['resources']} for cid,c in self.cases.items() for r in c['rows']]
        return {'validated_contexts':contexts,'comparisons':comparisons,'native_discovery':discovery,'resources':resources,
            'resource_limits':['device_seconds is the synchronized API/collector wall window, not pure kernel time.',
                'peak_memory_bytes is the reset allocator peak including live inputs and capture overhead, not whole-card HBM.',
                'Selected-card checks permit explicitly reviewed idle external containers; whole-machine exclusivity is not claimed.'],
            'unknown_resource_measurements':sum(any(r[k] is None for k in ('device_seconds','peak_memory_bytes')) for r in resources)}


def verify_statistics_matrix(path,cases):
    p=Path(path); summary=read(p)
    require(summary['schema']=='nes.independent_statistics_matrix.v1' and summary['group']=='all'
        and summary['status']=='COMPARISONS_COMPLETE_NOT_MECHANISM_ACCEPTANCE'
        and summary['completed_comparisons']==summary['planned_comparisons']==88,'Full independent88-pair statistics matrix required')
    rows=summary['comparisons'];require(len(rows)==88 and len({x['comparison_id'] for x in rows})==88,'Statistics pair coverage differs')
    counts=Counter(); details=[]
    for row in rows:
        report_path=child(p.parent,row['report_file']);bound(report_path,row['report_sha256']); detail=read(report_path)
        require(row['status']==detail['status'] and row['status']!='INVALID_CAPTURE_EVIDENCE','Invalid independent statistics capture')
        for side in ('B','F'):
            data=row[side]; cid=Path(data['case_directory']).name
            require(cid in cases and Path(data['case_directory']).resolve()==cases[cid]['directory'].resolve(),'Statistics refers to other capture generation')
            require(0<=data['step']<len(cases[cid]['rows']),'Statistics step absent')
        if row['kind']!='stock_vs_private_native':
            require(row['alignment_complete'] is True and row['native_input_observed_difference'] is False,'Common Session statistics input/order differs')
            if row['kind'] in ('off_restore','same_process_matching_repeat','controlled_independent_repeat','stock_independent_repeat'):
                require(row['primary_attention_output_observed_difference'] is False,'Primary statistics instability/restoration failure')
        counts[row['observed_class']]+=1
        details.append({'comparison_id':row['comparison_id'],'kind':row['kind'],'status':row['status'],
            'alignment_complete':row['alignment_complete'],'first_observed_api':row['first_observed_api'],
            'first_difference_boundary':row['first_difference_boundary'],
            'primary_attention_output_observed_difference':row['primary_attention_output_observed_difference'],
            'unobservable_statistic_fields':row['unobservable_statistic_fields']})
    require(Counter(x['kind'] for x in rows)=={'stock_independent_repeat':4,'native_vs_controlled':24,
        'off_restore':12,'same_process_matching_repeat':12,'controlled_independent_repeat':24,'stock_vs_private_native':12},'Pair kinds missing')
    return {'summary_sha256':sha(p),'summary_file':str(p),'outcomes':dict(counts),'comparisons':details,
        'tensor_equality_proven_by_statistics':False,'physical_instruction_localized':False,
        'null_fields_are_unobservable':True,'collector_variant':'task_local_fp32_high_v1'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--attempt',action='append',required=True)
    parser.add_argument('--phase',choices=('discovery','full'),required=True);parser.add_argument('--statistics-report',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.dont_write_bytecode=True;require(not args.output.exists(),'Report must be new')
    start=time.monotonic();report={'schema':'nes.phase1.validation.v1','status':'FAIL','phase':args.phase,
        'analysis_source_sha256':sha(__file__),'source_sha256':SOURCE_SHA,'build_sha256':BUILD_SHA,
        'requested_attempts':args.attempt,'new_native_calls':0,'model_impact':None,'historical_defect_reproduced':False}
    validator=None
    try:
        validator=Validator(args.root,args.output.with_suffix('.audit'))
        for attempt in args.attempt:validator.add_attempt(attempt)
        full=args.phase=='full';report.update(validator.compare(full))
        if full:
            require(args.statistics_report is not None,'Full gate requires independent88-pair statistics report')
            report['independent_msprobe']=verify_statistics_matrix(args.statistics_report,validator.cases)
        report.update(status='PASS_NATIVE_EXECUTION_STRATEGY_FA_PHASE1' if full else 'PASS_NATIVE_DISCOVERY_OBSERVATIONS_ONLY',
            python_source_files=validator.python_sources,source_generation=validator.generation,
            runtime_manifest_sha256=RUNTIME_SHA,input_manifest_sha256=INPUT_SHA,public_pod_schema_sha256=SCHEMA_SHA,
            case_count=len(validator.cases),native_call_count=sum(len(c['rows']) for c in validator.cases.values()),
            source_audited_padding_fields_only=sorted(PADDING),controlled_support_granted=full,
            limitations=['Only explicit fixed FP16 BNSD S1S2 FA contexts and exact source/build/runtime are covered.',
                'Stock discovery B2/B4 does not grant controlled support for those workloads.',
                'Full-byte equality is independently checked; auxiliary seed/offset is separated from primary attention output.',
                'Strategy switching, observed numerics, and model impact are independent. Numerical equality is allowed.',
                'No historical FIA defect, unique Reduce instruction, new model forward, or unmodified official collector claim.'])
    except (ValueError,KeyError,TypeError,OSError,AssertionError) as error:
        report['error']=str(error)
    if validator is not None:report['evidence_files']=validator.evidence
    report['elapsed_cpu_wall_seconds']=time.monotonic()-start;args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':report['status'],'error':report.get('error'),'report_sha256':sha(args.output)}))
    return 0 if report['status'].startswith('PASS_') else 1


if __name__=='__main__':raise SystemExit(main())
