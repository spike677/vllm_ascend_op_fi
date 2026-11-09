"""Compile serializer checks with an actual same-generation FIA host command."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from generation import layout


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--generation',required=True);a=p.parse_args()
    paths=layout(Path('/audit'),a.generation);src=paths['source'];ev=paths['evidence']
    entries=json.loads((src/'build/compile_commands.json').read_text())
    candidates=[e for e in entries if Path(e['file']).name=='fia_tiling_nonquant.cpp' and '/arch22/' in e['file']]
    if len(candidates)!=1:raise ValueError('Requires exactly one actual matching FIA host compile entry')
    entry=candidates[0];original=entry.get('arguments') or shlex.split(entry['command'])
    flags=[];i=1
    while i<len(original):
        v=original[i]
        if v in ('-o','-MF','-MT','-MQ'):i+=2;continue
        if v in ('-c','-MD','-MMD',entry['file']):i+=1;continue
        flags.append(v);i+=1
    cpp=Path(__file__).with_name('fia_abi_decoder.cpp');binary=ev/'fia_abi_decoder'
    command=[original[0],*flags,'-I'+str(ev),str(cpp),'-L/usr/local/Ascend/cann-9.1.0/lib64','-lregister','-lc_sec','-o',str(binary)]
    subprocess.run(command,cwd=entry['directory'],check=True)
    value=subprocess.check_output([str(binary)],text=True);r=json.loads(value)
    if r['status']!='PASS_CPU_DERIVED_LAYOUT_SENTINEL' or r['serialized_bytes']!=1576:raise ValueError('FIA serializer failed CPU checks')
    (ev/'fia_abi_selftest.json').write_text(value)
    fields=[dict(name=name,**item) for name,item in r['fields'].items()]
    schema={'schema':'fault28.fia_graph.public_serializer_layout.v1','type':'FusedInferAttentionScoreTilingData',
            'size':r['serialized_bytes'],'fields':fields,'projection_sha256':sha(ev/'fia_schema_projection.h'),
            'layout_kind':'public_host_serializer_projection','kernel_argument_raw_observed':False}
    (ev/'fia_abi_schema.json').write_text(json.dumps(schema,indent=2)+'\n')
    (ev/'fia_abi_compile.json').write_text(json.dumps({'schema':'fault28.fia_graph.abi_compile.v1',
        'compiler_argv':command,'actual_fia_compile_entry':entry,'source_sha256':sha(cpp),
        'projection_sha256':sha(ev/'fia_schema_projection.h'),'decoder_sha256':sha(binary),'npu_executed':False},indent=2)+'\n')


if __name__=='__main__':main()
