import hashlib,unittest
from types import SimpleNamespace as NS
from demoncat.runtime.vllm_ascend.worker_extension.attention28_value_plan import verify_live_values,SIZES

class PreparedValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.value=b''.join(b'\x91'*6144+bytes([row%256])*2048 for row in range(2048))
        cls.logical=b''.join(bytes([row%256])*2048 for row in range(2048))
        cls.parts=(b'\x31'*SIZES[0],b'\x42'*SIZES[1],cls.value,b'\x55'*SIZES[3],cls.logical)
    def plan(self):
        raw=dict(query=self.parts[0],key=self.parts[1],value=self.logical,atten_mask=self.parts[3])
        return NS(public_metadata={'value':{'storage_bytes':SIZES[2]}},before={'inputs':{
            name:{'snapshot':{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}} for name,data in raw.items()}})
    def test_current_public_and_prepared_complete_values(self):
        calls=[]
        def read(index,size):calls.append((index,size));return self.parts[index]
        result=verify_live_values(self.plan(),read)
        self.assertEqual(calls,list(enumerate(SIZES)));self.assertEqual(len(result),5)
        self.assertEqual(result[2]['sha256'],result[4]['sha256'])
        self.assertNotEqual(result[2]['sha256'],hashlib.sha256(self.value[:len(self.logical)]).hexdigest())
    def test_changed_public_query_refused(self):
        def read(index,size):return b'\x00'+self.parts[index][1:] if index==0 else self.parts[index]
        with self.assertRaisesRegex(RuntimeError,'VALUE_CHANGED:query'):verify_live_values(self.plan(),read)
    def test_wrong_prepared_value_refused(self):
        def read(index,size):return self.parts[index][::-1] if index==4 else self.parts[index]
        with self.assertRaisesRegex(RuntimeError,'VALUE_CHANGED:prepared_value'):verify_live_values(self.plan(),read)
    def test_strided_source_change_refused(self):
        def read(index,size):
            if index==2:return self.value[:6144]+b'\xff'+self.value[6145:]
            return self.parts[index]
        with self.assertRaisesRegex(RuntimeError,'VALUE_CHANGED:value'):verify_live_values(self.plan(),read)
    def test_truncated_read_refused(self):
        with self.assertRaisesRegex(RuntimeError,'READ_SIZE'):verify_live_values(self.plan(),lambda i,n:self.parts[i][:-1])
    def test_storage_span_not_inferred_from_shape(self):
        plan=self.plan();plan.public_metadata['value']['storage_bytes']-=1
        with self.assertRaisesRegex(RuntimeError,'EXACT_STORAGE'):verify_live_values(plan,lambda i,n:self.parts[i])

if __name__=='__main__':unittest.main()
