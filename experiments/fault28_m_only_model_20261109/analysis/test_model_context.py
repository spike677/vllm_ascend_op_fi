"""Synthetic CPU-only context contract tests; no framework/operator import."""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

EXPERIMENT=Path(__file__).resolve().parents[1]
REPOSITORY=EXPERIMENT.parents[1]
sys.path.insert(0,str(EXPERIMENT));sys.path.insert(0,str(REPOSITORY))
import model_context as m
from native_execution_strategy.adapters.flash_attention import FlashAttentionAdapter
from native_execution_strategy.adapters.fa_schema import FIELDS,FORMATS,decode,OPAQUE_SECTIONS


def workspace_temporary():
    temporary=tempfile.TemporaryDirectory(prefix=".context_cpu_",dir=EXPERIMENT/"analysis")
    assert Path(temporary.name).resolve().is_relative_to(EXPERIMENT.resolve())
    return temporary


class TensorMetadata:
    def __init__(self,shape,dtype='torch.float16',device='npu:0'):
        self.shape=shape;self.dtype=dtype;self.device=device


def rows(batch=1,decode_step=False):
    return [{'request_id':f'P{i}','batch_slot':i,'prompt_token_ids_sha256':hashlib.sha256(str(i).encode()).hexdigest(),
        'prompt_length':2048,'query_length':1 if decode_step else 2048,'kv_length':2049 if decode_step else 2048,
        'query_positions':[2048] if decode_step else list(range(2048))} for i in range(batch)]


def sdpa(batch=1):
    q,k,v=(TensorMetadata([batch,n,2048,128]) for n in (16,8,8))
    return m.bind_sdpa((q,k,v),{'is_causal':True,'enable_gqa':True})


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package=EXPERIMENT/'runtime/native_execution_strategy'
        cls.adapter=FlashAttentionAdapter(package/'data/fa_cann91_s1s2_v1.json',evidence_root=package)

    def fixture(self,directory,batch=1):
        cert=next(x for x in self.adapter.certificate['contexts'] if x['request_count']==batch)
        plan=cert['strategies'][0]['parameters']['execution_plan'];values={**cert['input_contract'],**plan}
        raw=bytearray(1464)
        for name,f in FIELDS.items():
            value=values.get(name,0 if f['count']==1 else [0]*f['count'])
            vector=[value] if f['count']==1 else value
            struct.pack_into('<'+str(f['count'])+FORMATS[f['type']],raw,f['offset'],*vector)
        for name,(offset,size) in OPAQUE_SECTIONS.items():raw[offset:offset+size]=bytes.fromhex(plan[name])
        raw=bytes(raw);ir=[]
        for i in range(19):
            shape=([batch,16 if i==0 else 8,2048,128] if i<3 else [2048,2048] if i==6 else [])
            ir.append({'ir_index':i,'dtype':12 if i==6 else 1,
                'shape':{'rank':len(shape),'elements':math.prod(shape),'storage':shape,'origin':list(shape)}})
        ir[7]['shape']={'rank':1,'elements':0,'storage':[0],'origin':[0]}
        for i in (15,16,17):ir[i].update(shape=None,dtype=None)
        attrs={'scale':m.EFFECTIVE_SCALE,'keep_prob':1,'pre_tokens':2147483647,'next_tokens':0,
            'heads':16,'layout':'BNSD','precision':0,'sparse_mode':2}
        host={'schema':'nes.fa.host_observation.v1','callback_status':0,'pid':100,'tid':101,'token':'fixture',
            'eligible':True,'eligibility_checks':1,'guard_reports':[{'check':1,'passed':True,'failed_guards':[],
                'public_context':{'ir_inputs':ir,'attrs':attrs}}],
            'plan':{'scope':{'pid':100,'native_tid':101,'token':'fixture','request_count':batch}},
            'raw_hex':raw.hex(),'raw_size':1464,'tiling_key':m.TILING_KEY,'block_dim':20,'workspace_count':1}
        record={'host_event':host,'native_launch':{'pid':100,'tid':111,'raw_hex':raw.hex(),'raw_size':1464,
            'raw_sha256':hashlib.sha256(raw).hexdigest(),'tiling_key':m.TILING_KEY,'block_dim':20,'workspace_count':1},
            'pod':decode(raw),'evidence':[]}
        self.reseal(record,directory)
        return record

    def reseal(self,record,directory):
        file=Path(directory)/'host_events.original.jsonl'
        file.write_text(json.dumps(record['host_event'])+'\n',encoding='utf-8')
        record['evidence']=[{'path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}]

    def test_bind_exact_original_signature(self):
        value=sdpa();self.assertTrue(value['is_causal']);self.assertTrue(value['enable_gqa'])
        with self.assertRaises(m.UnknownNativeContext):m.bind_sdpa((1,2,3),{'query':4})
        with self.assertRaises(m.UnknownNativeContext):m.bind_sdpa((1,2,3),{'invented':True})
        with self.assertRaises(m.UnknownNativeContext):m.bind_sdpa((1,2),{})
        with self.assertRaises(m.UnknownNativeContext):m.bind_sdpa(tuple(range(7)),{})

    def test_template_is_unverified_and_does_not_cache(self):
        template=m.probe_template(self.adapter,8)
        self.assertEqual(template.attributes['context_evidence_kind'],'UNVERIFIED_NATIVE_SCOPE_INTENT')
        self.assertEqual(template.request_count,8)
        with self.assertRaises(ValueError):self.adapter.observe_current(template)
        with self.assertRaises(m.UnsupportedModelContext):m.probe_template(self.adapter,4)

    def test_native_actual_context_and_matching_prefill(self):
        for batch in (1,8):
            with self.subTest(batch=batch),workspace_temporary() as directory:
                record=self.fixture(directory,batch);actual,proof=m.actual_native_call(record,rows(batch),'Ascend910B4')
                self.assertTrue(proof['independent_descriptor_supported']);self.assertEqual(len(proof['input_contract']),34)
                self.assertEqual(len(proof['execution_plan']),169);self.assertFalse(proof['tensor_values_observed_by_this_helper'])
                ctx=m.prefill_context(self.adapter,sdpa(batch),rows(batch),actual)
                self.assertEqual(ctx.as_dict(),self.adapter.inspect(actual).as_dict())
                self.assertNotIn('context_evidence_kind',ctx.attributes)

    def test_missing_host_or_public_context_is_unknown(self):
        with workspace_temporary() as directory:
            record=self.fixture(directory);record.pop('host_event')
            with self.assertRaisesRegex(m.UnknownNativeContext,'missing host'):m.actual_native_call(record,rows(),'Ascend910B4')
            record=self.fixture(directory);record['host_event']['guard_reports']=[];self.reseal(record,directory)
            with self.assertRaisesRegex(m.UnknownNativeContext,'selector/public context'):m.actual_native_call(record,rows(),'Ascend910B4')

    def test_original_host_bytes_must_match(self):
        with workspace_temporary() as directory:
            record=self.fixture(directory);Path(record['evidence'][0]['path']).write_text('{}\n')
            with self.assertRaisesRegex(m.UnknownNativeContext,'SHA mismatch'):m.actual_native_call(record,rows(),'Ascend910B4')

    def test_all19_ir_indices_required(self):
        with workspace_temporary() as directory:
            record=self.fixture(directory);record['host_event']['guard_reports'][0]['public_context']['ir_inputs'][18]['ir_index']=17
            self.reseal(record,directory)
            with self.assertRaisesRegex(m.UnknownNativeContext,'indices'):m.actual_native_call(record,rows(),'Ascend910B4')

    def test_repeated_native_guards_must_have_same_context(self):
        with workspace_temporary() as directory:
            record=self.fixture(directory);host=record['host_event'];second=copy.deepcopy(host['guard_reports'][0]);second['check']=2
            second['public_context']['attrs']['precision']=1;host['guard_reports'].append(second);host['eligibility_checks']=2
            self.reseal(record,directory)
            with self.assertRaisesRegex(m.UnknownNativeContext,'inconsistent public contexts'):m.actual_native_call(record,rows(),'Ascend910B4')

    def test_raw_pod_and_actual_context_are_bound(self):
        with workspace_temporary() as directory:
            record=self.fixture(directory);record['pod']['inputParams.bSize']=8
            with self.assertRaisesRegex(m.UnknownNativeContext,'205'):m.actual_native_call(record,rows(),'Ascend910B4')
            record=self.fixture(directory);record['native_launch']['pid']=999
            with self.assertRaisesRegex(m.UnknownNativeContext,'scope'):m.actual_native_call(record,rows(),'Ascend910B4')

    def test_flags_cannot_define_eligibility(self):
        with workspace_temporary() as directory:
            record=self.fixture(directory);record['host_event']['eligible']=False;self.reseal(record,directory)
            with self.assertRaisesRegex(m.UnknownNativeContext,'independently reconstructed'):m.actual_native_call(record,rows(),'Ascend910B4')

    def test_observed_unsupported_optional_input_is_not_missing_evidence(self):
        with workspace_temporary() as directory:
            record=self.fixture(directory);host=record['host_event'];guard=host['guard_reports'][0]
            guard['public_context']['ir_inputs'][3]['shape']={'rank':1,'elements':1,'storage':[1],'origin':[1]}
            host['eligible']=False;guard['passed']=False;guard['failed_guards']=['optional_ir_3_absent'];self.reseal(record,directory)
            actual,proof=m.actual_native_call(record,rows(),'Ascend910B4')
            self.assertFalse(proof['independent_descriptor_supported']);self.assertIn('optional_inputs_absent',proof['unsupported_reasons'])
            with self.assertRaises(m.UnsupportedModelContext):m.prefill_context(self.adapter,sdpa(),rows(),actual)

    def test_current_sdpa_cannot_fold_expanded_kv_or_change_source_semantics(self):
        with workspace_temporary() as directory:
            actual,_=m.actual_native_call(self.fixture(directory),rows(),'Ascend910B4')
            mutations=[{'key':TensorMetadata([1,16,2048,128])},{'enable_gqa':False},{'dropout_p':0.1},
                {'is_causal':False},{'attn_mask':TensorMetadata([2048,2048],'torch.bool')},
                {'query':TensorMetadata([1,16,2048,128],'torch.float32')},{'scale':1.},{'scale':1e100}]
            for changed in mutations:
                with self.subTest(changed=list(changed)),self.assertRaises(m.UnsupportedModelContext):
                    m.prefill_context(self.adapter,{**sdpa(),**changed},rows(),actual)

    def test_request_mapping_and_decode_boundary(self):
        bad=rows(8);bad[7]['batch_slot']=0
        with self.assertRaises(m.UnknownNativeContext):m.validate_request_rows(bad)
        bad=rows(8);bad[7]['request_id']='P0'
        with self.assertRaises(m.UnknownNativeContext):m.validate_request_rows(bad)
        bad=rows();bad[0]['query_positions'][-1]=0
        with self.assertRaises(m.UnknownNativeContext):m.validate_request_rows(bad)
        self.assertEqual(m.validate_request_rows(rows(8,True),prefill=False),8)
        with self.assertRaises(m.UnsupportedModelContext):m.validate_request_rows(rows(8,True))


if __name__=='__main__':unittest.main()
