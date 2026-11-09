"""Read-only SDPA/native context binding for the fixed model coverage experiment.

This module never calls an operator, creates a tensor, or modifies the certified
adapter. Calibration templates are UNVERIFIED intents. Supported native context
is established only from the public native IR descriptors and original raw POD.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from native_execution_strategy.models import InvocationContext
from native_execution_strategy.adapters.fa_schema import decode, input_contract, execution_plan
from native_execution_strategy.adapters.flash_attention import EFFECTIVE_SCALE, TILING_KEY


class UnknownNativeContext(ValueError):
    """Evidence is absent/inconsistent: not a known unsupported invocation."""


class UnsupportedModelContext(ValueError):
    """An observed invocation lies outside the fixed certified prefill scope."""


def need(condition, message, error=UnknownNativeContext):
    if not condition:
        raise error(message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def bind_sdpa(args, kwargs):
    """Bind the original functional SDPA signature without invoking anything."""
    names=('query','key','value','attn_mask','dropout_p','is_causal')
    need(isinstance(args,(tuple,list)) and isinstance(kwargs,dict),'Malformed SDPA invocation')
    need(len(args)<=len(names),'Unexpected positional SDPA arguments')
    need(set(kwargs)<=set(names)|{'scale','enable_gqa'},'Unknown SDPA keyword arguments')
    result={'attn_mask':None,'dropout_p':0.0,'is_causal':False,'scale':None,'enable_gqa':False}
    for name,value in zip(names,args):
        need(name not in kwargs,'Duplicate SDPA argument: '+name)
        result[name]=value
    result.update(kwargs)
    need(all(name in result for name in names[:3]),'Missing required SDPA Q/K/V')
    return result


def validate_request_rows(rows, *, prefill=True):
    """Explicit scheduler/model slot mapping; never infer request count from Q.shape[0]."""
    need(type(rows) is list and bool(rows),'Missing independently recorded request rows')
    ids=[]
    for slot,row in enumerate(rows):
        need(type(row) is dict and type(row.get('request_id')) is str and row['request_id'],'Missing stable request ID')
        need(type(row.get('batch_slot')) is int and row['batch_slot']==slot,'Request slot mapping is not one-to-one/in order')
        need(re.fullmatch('[0-9a-f]{64}',row.get('prompt_token_ids_sha256','')) is not None,'Missing frozen request token fingerprint')
        for name in ('prompt_length','query_length','kv_length'):
            need(type(row.get(name)) is int and row[name]>0,'Missing explicit request sequence metadata: '+name)
        positions=row.get('query_positions')
        need(type(positions) is list and len(positions)==row['query_length']
             and all(type(x) is int and 0<=x<row['kv_length'] for x in positions),'Invalid actual query positions')
        need(len(set(positions))==len(positions),'Repeated query positions')
        if prefill:
            need(row['prompt_length']==row['query_length']==row['kv_length']==2048
                 and positions==list(range(2048)), 'Only full2048-token prefill is certified',UnsupportedModelContext)
        ids.append(row['request_id'])
    need(len(set(ids))==len(ids),'Duplicate logical request IDs')
    return len(rows)


def _attributes():
    return {'scale':EFFECTIVE_SCALE,'keep_prob':1,'pre_tokens':2147483647,'next_tokens':0,
            'heads':16,'inner_precise':0,'sparse_mode':2,'optional_inputs_absent':True}


def probe_template(adapter,batch):
    """UNVERIFIED native-only scope intent; never use it as the actual baseline cache.

After calibration, inspect(actual_native_call(...)[0]) and parse native evidence
with that real context. Parsing/caching this template would retain its deliberate
UNVERIFIED marker and would not establish the later real-context observation.
    """
    need(type(batch) is int and batch in (1,8),'Template batch outside certificate',UnsupportedModelContext)
    cert=adapter.certificate
    need(cert.get('validation_state')=='validated' and any(x['request_count']==batch for x in cert['contexts']),
         'A real validated context certificate is required')
    attrs=_attributes()
    attrs.update(device=cert['runtime']['device'],
        tensor_shapes={'query':[batch,16,2048,128],'key':[batch,8,2048,128],
                       'value':[batch,8,2048,128],'atten_mask':[2048,2048]},
        tensor_dtypes={'query':'float16','key':'float16','value':'float16','atten_mask':'bool'},
        context_evidence_kind='UNVERIFIED_NATIVE_SCOPE_INTENT')
    return InvocationContext('FlashAttentionScore',request_count=batch,token_count=batch*2048,
        seq_lens=(2048,)*batch,dtype='float16',layout='BNSD',
        matrix_dims={'query_heads':16,'kv_heads':8,'query_length':2048,'key_length':2048,'head_dim':128,'value_dim':128},
        attributes=attrs)


def _descriptor(row):
    shape=row.get('shape')
    if shape is None:
        need(row.get('dtype') is None,'Absent native shape has a present dtype')
        return None
    need(type(shape) is dict and set(shape)=={'rank','elements','storage','origin'},'Incomplete native shape descriptor')
    for key in ('storage','origin'):
        need(type(shape[key]) is list and all(type(x) is int and x>=0 for x in shape[key]),'Invalid native dimensions')
    need(type(shape['rank']) is int and shape['rank']==len(shape['storage']) and type(shape['elements']) is int
         and shape['elements']==math.prod(shape['storage']),'Native rank/element count inconsistent')
    need(type(row.get('dtype')) is int,'Native dtype is not observable')
    return shape


def _native_dtype(code):
    # Required types from the pinned GE DataType enum; other observed codes are
    # preserved as unsupported codes rather than guessed semantic type names.
    return {1:'float16',12:'bool'}.get(code,'ge_dtype_'+str(code))


def _original_host(record,host):
    refs=[x for x in record.get('evidence',[]) if Path(x.get('path','')).name=='host_events.original.jsonl']
    need(len(refs)==1,'UNKNOWN: missing/ambiguous original host-event evidence')
    path=Path(refs[0]['path']);need(path.is_file(),'UNKNOWN: original host-event file unavailable')
    raw=path.read_bytes();need(hashlib.sha256(raw).hexdigest()==refs[0]['sha256'],'Original host-event SHA mismatch')
    try:events=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    except (UnicodeError,ValueError) as exc:raise UnknownNativeContext('Malformed original host evidence') from exc
    need(events==[host],'Expected exactly the same single original public host event')
    return refs[0]['sha256']


def actual_native_call(record,request_rows,device):
    """Return actual native descriptors and provenance, not an injection-hit verdict.

Known unsupported descriptor values remain visible in returned attributes and
the independent guard reasons. Missing public context is UNKNOWN, never folded
into a benign unsupported count. This function performs filesystem reads only.
    """
    batch=validate_request_rows(request_rows)
    need(type(record) is dict,'UNKNOWN: missing native observation')
    host=record.get('host_event');native=record.get('native_launch')
    need(type(host) is dict and type(native) is dict,'UNKNOWN: missing host/native observation')
    need(host.get('schema')=='nes.fa.host_observation.v1' and host.get('callback_status')==0,
         'UNKNOWN: native callback unavailable or failed')
    original_sha=_original_host(record,host)
    reports=host.get('guard_reports')
    need(type(reports) is list and reports,'UNKNOWN: actual selector/public context was not observed')
    need(host.get('eligibility_checks')==len(reports),'Public guard count inconsistent')
    public=None
    for index,report in enumerate(reports,1):
        need(type(report) is dict and report.get('check')==index,'Public guard sequence inconsistent')
        current=report.get('public_context')
        need(type(current) is dict and type(current.get('attrs')) is dict,'UNKNOWN: public context/attributes missing')
        if public is None:public=current
        need(current==public,'One native call exposed inconsistent public contexts')
    ir=public.get('ir_inputs');need(type(ir) is list and len(ir)==19,'UNKNOWN: expected all19 actual IR descriptors')
    need([row.get('ir_index') for row in ir]==list(range(19)),'Native IR indices missing/duplicated/out of order')
    shapes=[_descriptor(row) for row in ir]
    attrs=public['attrs'];need(set(attrs)=={'scale','keep_prob','pre_tokens','next_tokens','heads','layout','precision','sparse_mode'},
                             'UNKNOWN: native attribute coverage differs')
    need(all(attrs[k] is not None for k in attrs),'UNKNOWN: actual native attribute is absent')
    need(all(type(attrs[k]) in (int,float) and math.isfinite(attrs[k]) for k in ('scale','keep_prob'))
         and all(type(attrs[k]) is int for k in ('pre_tokens','next_tokens','heads','precision','sparse_mode'))
         and type(attrs['layout']) is str,'UNKNOWN: actual native attribute type is invalid')
    plan=host.get('plan');need(type(plan) is dict and type(plan.get('scope')) is dict,'UNKNOWN: native scope binding missing')
    scope=plan['scope']
    need(host['pid']==native.get('pid')==scope.get('pid') and host['tid']==scope.get('native_tid')
         and host['token']==scope.get('token') and scope.get('request_count')==batch,'Host/native scope or request mapping differs')
    need(host.get('raw_hex')==native.get('raw_hex') and host.get('raw_size')==native.get('raw_size')==1464,
         'Host and actual launched raw POD differ')
    try:raw=bytes.fromhex(native['raw_hex'])
    except (ValueError,TypeError) as exc:raise UnknownNativeContext('Malformed actual raw POD') from exc
    need(len(raw)==1464 and hashlib.sha256(raw).hexdigest()==native.get('raw_sha256'),'Actual raw POD SHA/size differs')
    for key in ('tiling_key','block_dim','workspace_count'):
        need(str(host.get(key))==str(native.get(key)),'Host/actual launch differs: '+key)
    need(str(native['tiling_key'])==TILING_KEY,'UNKNOWN: native route has no certified POD interpretation')
    pod=decode(raw);need(pod==record.get('pod'),'All205 original POD fields must match independent raw decode')
    inputs={}
    for name,index in [('query',0),('key',1),('value',2),('atten_mask',6)]:
        need(shapes[index] is not None,'UNKNOWN: required native descriptor missing: '+name)
        need(shapes[index]['origin']==shapes[index]['storage'],'UNKNOWN: origin/storage logical mapping differs: '+name)
        inputs[name]={'shape':list(shapes[index]['storage']),'dtype':_native_dtype(ir[index]['dtype'])}
    optional_indices=(3,4,5,7,8,9,10,11,12,13,14,15,16,17,18)
    optional_absent=all(shapes[i] is None or shapes[i]['rank']==0 or shapes[i]['elements']==0 for i in optional_indices)
    # Fixed FP16 FA source semantics only: a rank0 placeholder is not generally
    # an absent tensor in arbitrary operators or quantized execution paths.
    actual_attrs={'scale':attrs['scale'],'keep_prob':attrs['keep_prob'],'pre_tokens':attrs['pre_tokens'],
        'next_tokens':attrs['next_tokens'],'heads':attrs['heads'],'inner_precise':attrs['precision'],
        'sparse_mode':attrs['sparse_mode'],'optional_inputs_absent':optional_absent}
    expected_shapes={'query':[batch,16,2048,128],'key':[batch,8,2048,128],
                     'value':[batch,8,2048,128],'atten_mask':[2048,2048]}
    failures=[]
    for name,value in inputs.items():
        if value['shape']!=expected_shapes[name]:failures.append(name+'_shape')
        if value['dtype']!=('bool' if name=='atten_mask' else 'float16'):failures.append(name+'_dtype')
    if attrs['layout']!='BNSD':failures.append('layout')
    for key,value in _attributes().items():
        if actual_attrs[key]!=value:failures.append(key)
    # Never derive eligibility from the injected 'eligible'/'passed' flags.
    supported=not failures
    need(type(host.get('eligible')) is bool and host['eligible'] is supported
         and all(type(x.get('passed')) is bool and x['passed'] is supported for x in reports),
         'Host flags disagree with independently reconstructed descriptor eligibility')
    if supported:
        expected_raw={'inputParams.bSize':batch,'inputParams.n2Size':8,'inputParams.gSize':2,
            'inputParams.s1Size':2048,'inputParams.s2Size':2048,'inputParams.dSize':128,
            'inputParams.scaleValue':EFFECTIVE_SCALE,'inputParams.keepProb':1.,
            'inputParams.preTokens':2048,'inputParams.nextTokens':0,'inputParams.layoutType':3,
            'inputParams.implMode':0,'inputParams.sparseType':3,'inputParams.needDropMaskOp':0}
        need(all(pod[k]==v for k,v in expected_raw.items()),'Actual public descriptors and semantic raw POD disagree')
    actual={'operator':'FlashAttentionScore','request_count':batch,'token_count':sum(x['query_length'] for x in request_rows),
        'seq_lens':[x['kv_length'] for x in request_rows],'layout':attrs['layout'],'device':device,
        'inputs':inputs,'attributes':actual_attrs}
    provenance={'kind':'ACTUAL_NATIVE_PUBLIC_DESCRIPTOR_AND_LAUNCHED_POD',
        'host_original_sha256':original_sha,'native_raw_sha256':hashlib.sha256(raw).hexdigest(),
        'public_context_sha256':canonical_sha(public),'public_context':public,
        'input_contract':input_contract(pod),'execution_plan':execution_plan(pod,raw=raw),
        'independent_descriptor_supported':supported,'unsupported_reasons':failures,
        'controlled_certificate_support_not_decided_by_this_helper':True,
        'request_mapping_source':'Explicit frozen request IDs and actual model batch slots, not inferred from shape[0]',
        'request_rows_sha256':canonical_sha(request_rows),'device_source':'Caller-observed public device identity',
        'mask_observation':'Actual native shape/dtype descriptor only; no Python mask tensor or internal mask values captured',
        'tensor_values_observed_by_this_helper':False,'native_calls_executed':0}
    actual['_native_descriptor_provenance']={k:provenance[k] for k in ('kind','host_original_sha256','native_raw_sha256','public_context_sha256')}
    return actual,provenance


def _python_tensor(tensor,name):
    need(tensor is not None and hasattr(tensor,'shape') and hasattr(tensor,'dtype') and hasattr(tensor,'device'),
         'Missing actual Python tensor metadata: '+name,UnsupportedModelContext)
    shape=list(tensor.shape);dtype=str(tensor.dtype).removeprefix('torch.')
    need(all(type(x) is int and x>=0 for x in shape),'Symbolic/invalid SDPA geometry',UnsupportedModelContext)
    need(str(tensor.device).split(':')[0]=='npu','Current tensor is not NPU',UnsupportedModelContext)
    return {'shape':shape,'dtype':dtype}


def prefill_context(adapter,values,request_rows,baseline_actual_call):
    """Bind current real SDPA tensors/scalars to a previously observed native ABI.

Internal causal mask/FA defaults are derived from this fixed source conversion
and the native baseline descriptor, then must be checked against the real host
context after *each* original call. No mask tensor is generated or claimed read.
    """
    batch=validate_request_rows(request_rows)
    need(type(values) is dict and set(values)=={'query','key','value','attn_mask','dropout_p','is_causal','scale','enable_gqa'},
         'SDPA arguments were not bound with the fixed signature')
    need(values['attn_mask'] is None and values['is_causal'] is True,'Only backend-generated causal prefill mask is supported',UnsupportedModelContext)
    need(type(values['dropout_p']) in (int,float) and values['dropout_p']==0,'Dropout is outside inference contract',UnsupportedModelContext)
    need(values['enable_gqa'] is True,'Actual16Q/8KV call must request native GQA',UnsupportedModelContext)
    scale=128**-0.5 if values['scale'] is None else values['scale']
    need(type(scale) in (int,float) and math.isfinite(scale),'Invalid SDPA scale',UnsupportedModelContext)
    try:effective_scale=struct.unpack('<f',struct.pack('<f',scale))[0]
    except (OverflowError,struct.error) as exc:raise UnsupportedModelContext('SDPA scale exceeds native FP32 range') from exc
    need(effective_scale==EFFECTIVE_SCALE,'SDPA effective native scale differs',UnsupportedModelContext)
    need(type(baseline_actual_call) is dict and baseline_actual_call.get('_native_descriptor_provenance',{}).get('kind')
         =='ACTUAL_NATIVE_PUBLIC_DESCRIPTOR_AND_LAUNCHED_POD','UNKNOWN: no genuine native baseline descriptors')
    actual=copy.deepcopy(baseline_actual_call)
    need(actual['request_count']==batch,'Native baseline belongs to another explicit request count')
    need(actual['attributes']==_attributes(),'Baseline native attributes are not the certified causal source contract',UnsupportedModelContext)
    for name in ('query','key','value'):
        meta=_python_tensor(values[name],name)
        need(meta==actual['inputs'][name],'Current SDPA tensor differs from real native descriptor: '+name,UnsupportedModelContext)
        actual['inputs'][name]=meta
    need(actual['inputs']['query']['shape']==[batch,16,2048,128]
         and actual['inputs']['key']['shape']==actual['inputs']['value']['shape']==[batch,8,2048,128],
         'Expanded KV cannot be implicitly collapsed into certified GQA',UnsupportedModelContext)
    actual['token_count']=sum(x['query_length'] for x in request_rows);actual['seq_lens']=[x['kv_length'] for x in request_rows]
    context=adapter.inspect(actual)
    reason=adapter.unsupported_reason(context,controlled=True)
    need(reason is None,'Actual prefill context is unsupported: '+str(reason),UnsupportedModelContext)
    return context
