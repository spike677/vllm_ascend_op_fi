import copy,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from demoncat.runtime.vllm_ascend.worker_extension import product_registration as pr
from demoncat.runtime.vllm_ascend.worker_extension.product_cli import choices_from_clients,select_product_client

def identity(rank=0,tp=2):
 a={k:'observed' for k in pr.IDENTITY_FIELDS}
 a.update(hardware_family='A2',hardware_device_name='Ascend910B4',hardware_logical_device=rank,tp=tp,pp=1,dp=1,rank=rank,
 ep={'enabled':False,'placement':[],'routing':'NOT_APPLICABLE'},eager=True,live_config_matches_file=True,async_scheduling=False,prefix_caching=False,mtp_disabled=True,kv_connector_disabled=True,
 model_config_sha256='a'*64,quant_model_description_sha256='b'*64,safetensors_index_sha256={'model.safetensors.index.json':'c'*64},implementation_sha256='d'*64,
 source_fingerprints={'model':'e'*64,'runner':'f'*64},runtime_libraries={n:dict(sha256='a'*64,version='9.1.0',metadata_sha256='b'*64) for n in ('libascendcl.so','libruntime.so')},cann=['9.1.0'],
 max_model_len=1024,max_num_batched_tokens=1024,max_num_seqs=1,execution_mode='product',variant_revision='1',native_identity={'applicable':False},
 request_range=dict(max_context=1024,max_tokens=4,max_num_seqs=1,single_chunk=True,image_grid=None))
 return a

def record(tp=2):
 e=dict(mode='product',level='BLUE_ORIGINAL_REVIEWED',owners=['one','two'],review_passed=True,receipt_sha256='1'*64,reviewer_sha256='2'*64,source_manifest_sha256='3'*64,consumers=True,bytes_restored=True,resources_zero=True,canary_recovered=True,negative_controls=True,
 coverage=[dict(owner=o,rank=r,counts=dict(observed=5,eligible=5,attempted=5,applied=5,changed=4,consumed=5,restored=5,failed=0)) for o in ('one','two') for r in range(tp)])
 r=dict(schema='dcat.product.registration.v1',source='BLUE_COMBINATION',mechanism='18.api_half',expected_ranks=[identity(i,tp) for i in range(tp)],evidence=e)
 r['record_sha256']=pr.digest(r);return r

def sign(r):r['record_sha256']=pr.digest({k:v for k,v in r.items() if k!='record_sha256'});return r
class RegistrationTests(unittest.TestCase):
 def test_positive_world_one_and_two_same_path(self):
  for tp in (1,2):
   r=record(tp)
   for rank in range(tp):self.assertEqual(pr.select_registration('18.api_half','BLUE_COMBINATION',identity(rank,tp),[r]),r)
 def test_every_required_field_missing_refused_image_optional(self):
  r=record()
  for field in pr.IDENTITY_FIELDS:
   with self.subTest(field=field):
    a=identity();a[field]=None
    with self.assertRaisesRegex(ValueError,'IDENTITY_MISSING'):pr.select_registration('18.api_half','BLUE_COMBINATION',a,[r])
  a=identity();a['image_digest']=None;pr.select_registration('18.api_half','BLUE_COMBINATION',a,[r])
 def test_mutation_unregistered_parallel_ranges_source_weights(self):
  with self.assertRaisesRegex(ValueError,'NOT_REGISTERED'):pr.select_registration('18.api_half','BLUE_COMBINATION',identity(),[])
  for field,value,code in [('tp',1,'PARALLEL'),('pp',2,'PARALLEL'),('max_num_seqs',8,'RANGE'),('model_config_sha256','f'*64,'IDENTITY_MISMATCH'),('quant_model_description_sha256','f'*64,'IDENTITY_MISMATCH'),('implementation_sha256','f'*64,'IDENTITY_MISMATCH'),('safetensors_index_sha256',{'other.safetensors.index.json':'f'*64},'IDENTITY_MISMATCH'),('source_fingerprints',{'model':'f'*64,'runner':'a'*64},'IDENTITY_MISMATCH')]:
   with self.subTest(field=field):
    a=identity();a[field]=value
    with self.assertRaisesRegex(ValueError,code):pr.select_registration('18.api_half','BLUE_COMBINATION',a,[record()])
 def test_damaged_or_inconsistent_record_and_effect_counts(self):
  r=record();r['expected_ranks'][1]['max_model_len']=2048
  with self.assertRaisesRegex(ValueError,'RECORD_INVALID'):pr.validate_record(r)
  sign(r)
  with self.assertRaisesRegex(ValueError,'RECORD_INVALID'):pr.validate_record(r)
  for key,val in [('changed',6),('failed',1),('consumed',4)]:
   r=record();r['evidence']['coverage'][0]['counts'][key]=val;sign(r)
   with self.assertRaisesRegex(ValueError,'RECORD_INVALID'):pr.validate_record(r)
 def test_live_registered_never_installs_or_accepts_permit(self):
  w=types.SimpleNamespace();auth=types.SimpleNamespace(source='BLUE_COMBINATION');profile={'mode':'PRODUCT_REGISTERED'}
  with patch.object(pr,'authority_for',return_value=auth),patch.object(pr,'observe_identity',return_value=identity()),patch.object(pr,'load_records',return_value=[record()]):
   self.assertEqual(pr.registered_for(w,'18.api_half',profile)['mode'],'PRODUCT_REGISTERED')
   setattr(w,pr._BLUE,object())
   with self.assertRaisesRegex(ValueError,'MODE_MIXED'):pr.registered_for(w,'18.api_half',profile)
 def test_native_and_ep_no_implicit_grant(self):
  for field,value in [('native_identity',{'applicable':True,'sha256':'a'*64}),('ep',{'enabled':True})]:
   a=identity();a[field]=value
   with self.assertRaisesRegex(ValueError,'REGISTRATION_PENDING|NOT_ADAPTED'):pr.validate_identity(a)

class CliTests(unittest.TestCase):
 def clients(self):
  row=dict(rank=0,status='PRODUCT_CANDIDATE',mechanism='15.alias_detach',mode='PRODUCT_ACCEPTANCE_ONLY',product_qualified=False,range={'max_tokens':4},named_variant='alias_detach')
  c=types.SimpleNamespace(capabilities=lambda:[row,dict(row,rank=1)])
  return {'new-profile':c},row
 def test_new_mechanism_needs_no_whitelist(self):
  c,r=self.clients();self.assertEqual(choices_from_clients(c)[0]['mechanism'],'15.alias_detach');self.assertIs(select_product_client(c,'15.alias_detach'),c['new-profile'])
  for bad in ('99',99,'15',None):
   with self.assertRaisesRegex(ValueError,'explicit mechanism'):select_product_client(c,bad)
 def test_rank_disagreement_and_duplicate_rejected(self):
  c,r=self.clients();c['new-profile'].capabilities=lambda:[r,dict(r,rank=1,mode='PRODUCT_REGISTERED')]
  with self.assertRaisesRegex(ValueError,'RANK_CAPABILITY'):choices_from_clients(c)
  c,r=self.clients();c['another']=c['new-profile']
  with self.assertRaisesRegex(ValueError,'AMBIGUOUS'):choices_from_clients(c)
if __name__=='__main__':unittest.main()
