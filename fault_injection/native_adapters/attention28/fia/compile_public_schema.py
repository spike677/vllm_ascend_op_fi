"""Derive a new public serializer layout from the fresh FIA host build.

CPU-only qualification tooling. It does not load an operator or grant a runtime
certificate. The decoder source is a byte-preserved historical source import;
all headers, compile flags and the executable are from this new build.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from base_policy import typed_materials

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    header=a.source/'attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.h'
    _,projection,_=typed_materials(header.read_text())
    projected=a.out/'fia_schema_projection.h';projected.write_text(projection)
    entries=json.loads((a.source/'build/compile_commands.json').read_text())
    candidates=[e for e in entries if Path(e['file']).name=='fia_tiling_nonquant.cpp' and '/arch22/' in e['file']]
    if len(candidates)!=1:raise RuntimeError('EXACT_NEW_FIA_COMPILE_ENTRY_REQUIRED')
    entry=candidates[0];original=entry.get('arguments') or shlex.split(entry['command'])
    flags=[];i=1
    while i<len(original):
        item=original[i]
        if item in ('-o','-MF','-MT','-MQ'):i+=2;continue
        if item in ('-c','-MD','-MMD',entry['file']):i+=1;continue
        flags.append(item);i+=1
    cpp=Path(__file__).with_name('fia_abi_decoder.cpp')
    binary=a.out/'fia_abi_decoder'
    argv=[original[0],*flags,'-I'+str(a.out),str(cpp),'-L'+str(a.sdk/'lib64'),'-lregister','-lc_sec','-o',str(binary)]
    subprocess.run(argv,cwd=entry['directory'],check=True,timeout=180)
    text=subprocess.check_output([str(binary)],text=True,timeout=30)
    result=json.loads(text)
    if result['status']!='PASS_CPU_DERIVED_LAYOUT_SENTINEL' or result['serialized_bytes']!=1576:
        raise RuntimeError('PUBLIC_SERIALIZER_SENTINEL_FAILED')
    (a.out/'fia_abi_selftest.json').write_text(text)
    schema=dict(schema='fault28.fia_graph.public_serializer_layout.v1',type='FusedInferAttentionScoreTilingData',
        size=result['serialized_bytes'],fields=[dict(name=name,**row) for name,row in result['fields'].items()],
        projection_sha256=sha(projected),public_source_sha256=sha(header),
        layout_kind='public_host_serializer_projection',kernel_argument_raw_observed=False)
    out=a.out/'fia_abi_schema.json';out.write_text(json.dumps(schema,indent=2)+'\n')
    receipt=dict(schema='demoncat.fia.fresh_abi_compile.v1',compiler_argv=argv,actual_new_fia_compile_entry=entry,
        compiler_source_sha256=sha(cpp),projection_sha256=sha(projected),public_source_sha256=sha(header),
        decoder_sha256=sha(binary),schema_sha256=sha(out),npu_executed=False,loading_allowed=False)
    (a.out/'compile_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='CPU_PUBLIC_LAYOUT_DERIVED',schema=str(out),sha256=sha(out))))
if __name__=='__main__':main()

