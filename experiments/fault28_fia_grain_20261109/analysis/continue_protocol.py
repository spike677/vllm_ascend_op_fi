"""Finish preregistered bounded NPU windows and CPU gates; no publication.

Run only after all eight grain_a2 stock/OFF cases complete. Each new NPU case
uses the existing lock and resource gates; any failure stops this protocol.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/fault28-fia-grain-20261109')
HERE=Path(__file__).resolve().parent
SUFFIX='grain_a2'


def main():
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    def call(name,*arguments):
        print(json.dumps({'protocol_step':name,'arguments':list(map(str,arguments))}),flush=True)
        subprocess.run([sys.executable,'-B',str(HERE/name),*map(str,arguments)],check=True,env=env)
    for repeat in (1,2):
        for batch in (1,8):
            for mode in ('stock','off'):
                path=ROOT/'outputs'/(f'g_{mode}_b{batch}_r{repeat}_'+SUFFIX)/'completion.json'
                assert json.loads(path.read_bytes())['status']=='CAPTURE_COMPLETED'
    call('check_bridge.py','--repetitions','1','2','--suffix',SUFFIX,
         '--output',ROOT/'outputs/bridge_gate_grain_a2.json')
    call('run_formal.py','--collection','fp64','--cases','g_causal_base512_b1_r1','g_causal_base512_b8_r1',
         '--suffix',SUFFIX)
    call('check_base_variant.py','--suffix',SUFFIX,'--output',ROOT/'outputs/base_variant_gate_grain_a2.json')
    call('run_formal.py','--collection','fp64','--cases','g_m128_b1_r1','g_m128_b8_r1','g_causal_m128_b1_r1',
         'g_m256_b1_r1','g_m256_b8_r1','g_causal_m256_b1_r1','--suffix',SUFFIX)
    call('run_formal.py','--collection','fp64','--cases','g_causal_m256_b8_r1',
         'g_batch_b1_r1','g_batch_b8_r1','g_batch_b1_r2','g_batch_b8_r2','g_restore_b8_r1','--suffix',SUFFIX)
    report=ROOT/'outputs/final_acceptance_grain_a2'
    call('accept_matrix.py','--collection','fp64','--suffix',SUFFIX,'--output',report)
    call('observer_neutrality.py','--suffix',SUFFIX,'--output',ROOT/'outputs/device_observer_neutrality_a1.json')
    call('run_device_audits.py','--reports',report,'--suffix',SUFFIX,'--output',ROOT/'outputs/device_audits_a1')
    print(json.dumps({'status':'REGISTERED_NPU_MATRIX_AND_CPU_GATES_COMPLETED','cases':23,'actual_graph_calls':69}),flush=True)


if __name__=='__main__':main()
