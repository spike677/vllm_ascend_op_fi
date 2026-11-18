"""Independent stdlib/CPU byte reader contracts; no Native execution proof."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];EXPERIMENT=ROOT/'experiments/c2c3_closure_20261117'
sys.path.insert(0,str(EXPERIMENT))
import review_arch22_s2_single as review
import test_arch22_normal_bridge_review as fixtures


class SingleS2ReaderContracts(unittest.TestCase):
    def test_fault_raw_pod_requires_explicit_new_variant_and_actual_tile_bytes(self):
        helper=fixtures.evidence.helpers(fixtures.source())
        event,resolved,host,bundle=fixtures.pod_fixture(helper)
        bundle['completion']={'variant':review.E.S2_VARIANT}
        raw=bytearray.fromhex(host['raw_hex'])
        field=next(x for x in bundle['schema']['fields'] if x['name']=='innerSplitParams.s2BaseSize')
        struct.pack_into('<Q',raw,field['offset'],256)
        host.update(schema='demoncat.paged33.arch22_s2_host.v1',mode='arch22_s2_256',policy_applied=True,
                    selected_s2=256,raw_hex=raw.hex())
        typed=helper.abi.decode_raw(raw,bundle['schema'])
        host['typed']={k:v for k,v in typed.items() if not k.endswith('ParamsPH')}
        args=bytearray.fromhex(event['args_hex']);args[64:64+len(raw)]=raw;event['args_hex']=args.hex()
        result=review.N.final_private_pod(event,resolved,host,bundle,helper.abi,selected_s2=256)
        self.assertEqual(result['typed']['innerSplitParams.s2BaseSize'],256)
        with self.assertRaises(ValueError):review.N.final_private_pod(event,resolved,host,bundle,helper.abi)
        bad=deepcopy(bundle);bad['completion']['variant']=review.E.VARIANT
        with self.assertRaises(ValueError):review.N.final_private_pod(event,resolved,host,bad,helper.abi,selected_s2=256)
        for change in ('host_only','array','slot'):
            e,h=deepcopy(event),deepcopy(host)
            if change=='host_only':h['typed']['innerSplitParams.s2BaseSize']=512
            elif change=='array':h['typed']['actual_fixture.full_array'][27]+=1
            else:e['placeholders_hex']=struct.pack('<IIII',16,64,8,64+1576).hex()
            with self.subTest(change=change),self.assertRaises((ValueError,AssertionError)):
                review.N.final_private_pod(e,resolved,h,bundle,helper.abi,selected_s2=256)

    def test_logical_normalization_preserves_unused_pages_and_actual_kv_and_scalar_values(self):
        table=list(range(65))+[0]*15
        value=dict(block_table_values=[table],cache_geometry={k:{'shape':[128,16,8,128]} for k in ('key','value')},
            writer_consumer_slot=[1024],tensors={'block_table':dict(bytes=320,sha256=hashlib.sha256(struct.pack('<80i',*table)).hexdigest())},
            logical_kv={'key':{'sha256':'1'*64},'value':{'sha256':'2'*64}},scalars={'input_layout':'TND','sparse_mode':3})
        moved=deepcopy(value);moved['block_table_values'][0][:65]=reversed(range(65));moved['writer_consumer_slot']=[0]
        moved['tensors']['block_table']['sha256']=hashlib.sha256(struct.pack('<80i',*moved['block_table_values'][0])).hexdigest()
        self.assertNotEqual(value,moved);self.assertEqual(review.normalized(value),review.normalized(moved))
        for change in ('duplicate','outside','slot','stale_hash'):
            bad=deepcopy(moved)
            if change=='duplicate':bad['block_table_values'][0][1]=64
            elif change=='outside':bad['block_table_values'][0][-1]=128
            elif change=='slot':bad['writer_consumer_slot']=[1]
            else:bad['tensors']['block_table']['sha256']='0'*64
            with self.subTest(change=change),self.assertRaises(ValueError):review.normalized(bad)
        for change in ('unused','kv','scalar'):
            bad=deepcopy(moved)
            if change=='unused':
                bad['block_table_values'][0][-1]=1
                bad['tensors']['block_table']['sha256']=hashlib.sha256(struct.pack('<80i',*bad['block_table_values'][0])).hexdigest()
            elif change=='kv':bad['logical_kv']['key']['sha256']='3'*64
            else:bad['scalars']['pre_tokens']=2147483647
            self.assertNotEqual(review.normalized(value),review.normalized(bad))

    def test_raw_outputs_use_actual_dtype_bytes_and_reject_changed_and_nonfinite_data(self):
        helper=fixtures.evidence.helpers(fixtures.source())
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);path=run/'logits.bin';path.write_bytes(struct.pack('<2f',1,2))
            row=dict(path='/evidence/probe/logits.bin',metadata={'shape':[1,2],'dtype':'torch.float32'},
                bytes=8,sha256=review.E.sha(path),finite=True)
            result=review.raw_tensor(run,row,[1,2],'torch.float32',helper,row['path'])
            self.assertEqual(result['bytes'],8)
            with self.assertRaises(ValueError):review.raw_tensor(run,row,[1,2],'torch.float16',helper,row['path'])
            path.write_bytes(struct.pack('<2f',1,3))
            with self.assertRaises(ValueError):review.raw_tensor(run,row,[1,2],'torch.float32',helper,row['path'])
            path.write_bytes(struct.pack('<2f',1,float('nan')));row['sha256']=review.E.sha(path)
            with self.assertRaises(ValueError):review.raw_tensor(run,row,[1,2],'torch.float32',helper,row['path'])

    def test_new_entrypoints_and_full_closure_cold_import_without_framework_or_dso(self):
        code=r'''
import runpy,sys
from pathlib import Path
root=Path(sys.argv[1]);folder=root/'experiments/c2c3_closure_20261117'
sys.path[:0]=[str(folder),str(root/'src'),str(root)]
import arch22_normal_bridge_evidence as e
sys.addaudithook(e.prohibit_runtime)
for name in e.REQUIRED+e.S2_REQUIRED:assert (root/name).is_file(),name
for name in ('prepare_arch22_s2_single_cases','review_arch22_s2_single','natural33_arch22_s2_worker_core_probe'):
    runpy.run_path(str(folder/(name+'.py')),run_name='cold_import_contract')
assert not any(x.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for x in sys.modules)
print('cold-cpu-source-closure-only')
'''
        result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT)],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(result.stdout.strip(),'cold-cpu-source-closure-only')


if __name__=='__main__':unittest.main()
