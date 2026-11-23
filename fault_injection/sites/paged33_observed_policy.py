"""Observed split-fuse candidate preparation; no admission from user parameters."""
import hashlib,json
from .paged33_splitfuse import NativeBoundarySplitFuse33Site

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def validate_spans(spans,length):
    start=0
    for row in spans:
        if len(row)!=2 or any(type(x) is not int for x in row) or row[0]!=start or not start<row[1]<=length:raise ValueError('PAGED33_KEY_COVERAGE')
        start=row[1]
    if start!=length:raise ValueError('PAGED33_KEY_COVERAGE')
def prepare_policy(cases,*,fault_block):
    from tools.review_paged33_scope import review  # offline preparation only
    if type(fault_block) is not int or fault_block not in (32,64,128,256,512):raise ValueError('PAGED33_FAULT_PARTITION_UNSUPPORTED')
    if not cases:raise ValueError('PAGED33_OBSERVED_CASES_REQUIRED')
    rows=[];seen=set()
    for case in cases:
        scope=case['scope'];raw=json.dumps(case['launch'],sort_keys=True,separators=(',',':')).encode()
        review(case['checkpoint'],scope,case['expected_scope'],raw)
        if scope['tp']!=2 or scope['hardware_family']!='A3':raise ValueError('PAGED33_A3_TP2_CANDIDATE_ONLY')
        launch=case['launch'];pod=case['pod']
        if set(pod)!={'tiling_key','block_dim','sche_mode','scale','host_source_sha256','kernel_source_sha256','serializer_sha256'}:raise ValueError('PAGED33_ACTUAL_POD_REQUIRED')
        if type(pod['tiling_key']) is not int or type(pod['block_dim']) is not int or pod['block_dim']<=0 or type(pod['sche_mode']) is not int:raise ValueError('PAGED33_ACTUAL_POD_REQUIRED')
        for k in ('host_source_sha256','kernel_source_sha256','serializer_sha256'):
            if type(pod[k]) is not str or len(pod[k])!=64 or any(c not in '0123456789abcdef' for c in pod[k]):raise ValueError('PAGED33_SOURCE_REQUIRED')
        if pod['serializer_sha256']!=launch['serializer_sha256'] or type(pod['scale']) not in (float,int) or not 0<pod['scale']<=1:raise ValueError('PAGED33_POD_JOIN_FAILED')
        key=(scope['rank'],scope['kv_length'])
        if key in seen:raise ValueError('PAGED33_DUPLICATE_SCOPE')
        seen.add(key);rows.append(dict(scope=scope,control_spans=launch['spans'],pod=pod,qkv_sha256=launch['qkv_sha256'],checkpoint_sha256=digest(case['checkpoint'])))
    lengths={r['scope']['kv_length'] for r in rows}
    if seen!={(rank,length) for rank in range(2) for length in lengths}:raise ValueError('PAGED33_COMPLETE_RANK_LENGTH_MATRIX_REQUIRED')
    policy=dict(schema='dcat.paged33.observed_policy.v1',arithmetic='BF16_MM_FP32_OUT_PBF16_LSE_FP32',fault_block=fault_block,rows=rows,qualification_granted=False)
    policy['policy_sha256']=digest(policy);return policy

class ObservedSplitFuse33Site(NativeBoundarySplitFuse33Site):
    """Not installed by the public factory until a reviewed policy is in a release."""
    def __init__(self,*args,policy,rank,**kwargs):
        if policy.get('policy_sha256')!=digest({k:v for k,v in policy.items() if k!='policy_sha256'}):raise ValueError('PAGED33_POLICY_HASH_MISMATCH')
        if policy.get('arithmetic')!='BF16_MM_FP32_OUT_PBF16_LSE_FP32':raise ValueError('PAGED33_ARITHMETIC_UNADAPTED')
        self.observed_rows={r['scope']['kv_length']:r for r in policy['rows'] if r['scope']['rank']==rank}
        if not self.observed_rows:raise ValueError('PAGED33_RANK_UNOBSERVED')
        self.policy=policy;super().__init__(*args,**kwargs)
    def eligible_call(self,count,length):
        if count!=1:return False,'PREFILL_NOT_PAGED_DECODE'
        if length not in self.observed_rows:raise ValueError('PAGED33_LENGTH_UNOBSERVED')
        return True,None
    def geometry_and_spans(self,q,k,v,impl,event):
        row=self.observed_rows.get(len(k))
        if row is None:raise ValueError('PAGED33_LENGTH_UNOBSERVED')
        s=row['scope'];pod=row['pod']
        if tuple(q.shape)!=(s['q_heads'],s['head_dim']) or tuple(k.shape[1:])!=(s['kv_heads'],s['head_dim']) or k.shape!=v.shape or impl.scale!=pod['scale'] or event.get('page_mapping',{}).get('block_size')!=s['block_size']:raise ValueError('PAGED33_OBSERVED_GEOMETRY_MISMATCH')
        control=row['control_spans'];fault=[(i,min(i+self.policy['fault_block'],len(k))) for i in range(0,len(k),self.policy['fault_block'])]
        validate_spans(control,len(k));validate_spans(fault,len(k))
        event['observed_policy_sha256']=self.policy['policy_sha256'];return control,fault
