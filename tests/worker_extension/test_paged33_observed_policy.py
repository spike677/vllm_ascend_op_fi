import copy,hashlib,json,unittest
from fault_injection.sites.paged33_observed_policy import prepare_policy
class Observed33Tests(unittest.TestCase):
 def cases(self):
  rows=[]
  for rank in (0,1):
   scope=dict(hardware_name='Ascend910_9382',hardware_family='A3',tp=2,rank=rank,q_heads=12,kv_heads=2,head_dim=256,kv_length=2305,block_size=128,runtime_source_sha256='a'*64)
   launch=dict(schema='dcat.paged33.observed_spans.v1',operator='FIA_paged_decode',arithmetic='BF16_MM_FP32_OUT_PBF16_LSE_FP32',qkv_sha256='a'*64,launch_sha256='b'*64,serializer_sha256='c'*64,spans=[[i,min(i+128,2305)] for i in range(0,2305,128)])
   raw=json.dumps(launch,sort_keys=True,separators=(',',':')).encode()
   check=dict(schema='dcat.paged33.checkpoint.v1',passed=True,device=scope['hardware_name'],qkv_sha256=launch['qkv_sha256'],launch_review_sha256=hashlib.sha256(raw).hexdigest(),spans=launch['spans'],native_checks=[dict(elements=3072,bitwise_equal=3070,checkpoint_99pct_passed=True)]*2,repeat=dict(elements=3072,bitwise_equal=3072))
   pod=dict(tiling_key=123456789,block_dim=42,sche_mode=1,scale=.0625,host_source_sha256='d'*64,kernel_source_sha256='e'*64,serializer_sha256='c'*64)
   rows.append(dict(scope=scope,expected_scope=copy.deepcopy(scope),launch=launch,checkpoint=check,pod=pod))
  return rows
 def test_different_observed_groups_preserved_no_grant(self):
  p=prepare_policy(self.cases(),fault_block=256)
  self.assertEqual(p['rows'][0]['pod']['tiling_key'],123456789);self.assertEqual(p['rows'][0]['control_spans'][0],[0,128]);self.assertFalse(p['qualification_granted'])
 def test_missing_rank_low_control_or_changed_source_rejected(self):
  with self.assertRaisesRegex(ValueError,'COMPLETE_RANK'):prepare_policy(self.cases()[:1],fault_block=256)
  rows=self.cases();rows[1]['checkpoint']['native_checks'][0]['bitwise_equal']=3000
  with self.assertRaisesRegex(ValueError,'THRESHOLD'):prepare_policy(rows,fault_block=256)
  rows=self.cases();rows[0]['pod']['serializer_sha256']='f'*64
  with self.assertRaisesRegex(ValueError,'JOIN'):prepare_policy(rows,fault_block=256)
if __name__=='__main__':unittest.main()
