"""Actual DSO control isolation, no device handles or NPU operations."""
import ctypes as C
import json,os,sys,threading
lib=C.CDLL(sys.argv[1]);os.environ['DCAT_DQ_V2_ENABLE']='1'
lib.dcat_dq_arm.argtypes=[C.c_void_p,C.c_int64,C.c_int64,C.c_int]
lib.dcat_dq_finish.argtypes=[C.POINTER(C.c_uint64),C.c_size_t]
assert lib.dcat_dq_thread_scope_v1()==1
assert lib.dcat_dq_arm(4096,8,128,0)==0
outcomes=[]
def foreign():
    counters=(C.c_uint64*6)()
    outcomes.append(lib.dcat_dq_finish(counters,6))
    outcomes.append(lib.dcat_dq_arm(8192,8,128,1))
    outcomes.append(lib.dcat_dq_finish(counters,6))
t=threading.Thread(target=foreign);t.start();t.join()
assert outcomes==[-1,0,-1],outcomes
# The foreign finish must not disarm the original owner's selection.
assert lib.dcat_dq_arm(4096,8,128,0)==-1
counters=(C.c_uint64*6)();assert lib.dcat_dq_finish(counters,6)==-1
assert lib.dcat_dq_arm(4096,8,128,0)==0
assert lib.dcat_dq_finish(counters,6)==-1
print(json.dumps(dict(status='PASS',actual_npu_calls=0,foreign_finish_preserves_owner=True,thread_scope=1)))
