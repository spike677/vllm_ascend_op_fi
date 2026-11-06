"""CPU sensitivity discovery, not an msprobe capture or acceptance."""
import json,numpy as np
from pathlib import Path
from compare_targets import load
ROOT=Path('/home/fault28-fia-graph-20261106/outputs')
for m in (128,256):
 names=[f'g_m{m}_b1_r1',f'g_causal_m{m}_b1_r1']
 values=[load(ROOT/(n+'_final_a1')/n)[0].astype('float64') for n in names]
 stats=[{'Max':float(v.max()),'Min':float(v.min()),'Mean':float(v.mean()),'Norm':float(np.linalg.norm(v.reshape(-1)))} for v in values]
 print(json.dumps({'M':m,'cpu_numpy_fp64_sensitivity_only':stats,'not_msprobe_evidence':True}))
