"""CPU-only private host relink from the pinned, byte-matched original build.

The State ABI is unchanged. Only fia_tiling_nonquant.cpp is recompiled; all
other original object files are read-only and recorded by SHA256. The inherited private kernel is retained during host staging; a separate build
then adds the denominator replay template before execution is allowed. New M/N combinations remain unvalidated
until separate NPU legality and numerical checks succeed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

ROOT=Path('/home/fault28-reduce-replay-20261113')
OLD=Path('/home/fault28-fia-graph-20261106')
BASE=Path('/home/fault28-eager-20261110/vendor_eager_a5')
IMAGE='sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9'
LIB='op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so'
PROTO='op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def once(s,old,new):
    if s.count(old)!=1:raise ValueError('Missing or ambiguous source anchor: '+old[:100])
    return s.replace(old,new)


def inside():
    source=Path('/audit/source_a4/ops-transformer-v9.1.0')
    build=source/'build';ev=Path('/new/build/host_a1')
    commands=json.loads((build/'compile_commands.json').read_text())
    rows=[r for r in commands if r['file'].endswith('/fia_tiling_nonquant.cpp')]
    assert len(rows)==1
    row=rows[0];args=shlex.split(row['command'])
    args=[a for a in args if not a.startswith('-D__FILE__=')]
    args.insert(1,'-D__FILE__="fia_tiling_nonquant.cpp"')
    args.insert(1,'-I'+str(source/'attention/common/op_host/arch22'))
    original_object=args[args.index('-o')+1]
    args[args.index('-o')+1]=str(ev/'fia_tiling_nonquant.cpp.o')
    args[args.index('-c')+1]=str(ev/'source/fia_tiling_nonquant.cpp')
    consumers=[];original_objects={}
    for target in ('cust_proto','cust_opmaster'):
        link_file=build/f'CMakeFiles/{target}.dir/link.txt';link=shlex.split(link_file.read_text())
        objects=[a for a in link if a.endswith('fia_tiling_nonquant.cpp.o')]
        assert len(objects)==1
        assert (build/objects[0]).resolve()==(Path(row['directory'])/original_object).resolve()
        original_objects.update({a:sha(build/a) for a in link if a.endswith('.o')})
        output_name=link[link.index('-o')+1]
        link[link.index(objects[0])]=str(ev/'fia_tiling_nonquant.cpp.o')
        link[link.index('-o')+1]=str(ev/output_name)
        consumers.append({'target':target,'link':link,'link_cwd':str(build),
                          'output':output_name,'original_link_file_sha256':sha(link_file)})
    save(ev/'exact_commands.json',{'compile':args,'compile_cwd':row['directory'],
          'consumers':consumers,'original_objects':original_objects,
          'original_compile_commands_sha256':sha(build/'compile_commands.json'),
          'state_layout_changed':False,
          'devices':[],'npu_executed':False})
    subprocess.run(args,cwd=row['directory'],check=True,timeout=240)
    for consumer in consumers:subprocess.run(consumer['link'],cwd=build,check=True,timeout=120)
    assert original_objects=={name:sha(build/name) for name in original_objects}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inside',action='store_true');a=p.parse_args()
    if a.inside:return inside()
    ev=ROOT/'build/host_a1';(ev/'source').mkdir(parents=True,exist_ok=False)
    source=OLD/'source_a4/ops-transformer-v9.1.0'
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()=='c77c0829f1f0f0b8f882db1d9b051a758b605472'
    original=source/'attention/common/op_host/arch22'
    old_policy=(original/'fault28_fia_policy.h').read_text()
    policy=(ROOT/'source_policy/fault28_fia_policy.h').read_text()
    state=lambda t:re.search(r'struct State \{.*?\n\};',t,re.S).group()
    assert state(old_policy)==state(policy),'State ABI change would require rebuilding all dependents'
    for name in ('fia_tiling_nonquant.h','fault28_fia_typed.h'):
        shutil.copyfile(original/name,ev/'source'/name)
    shutil.copyfile(ROOT/'source_policy/fault28_fia_policy.h',ev/'source/fault28_fia_policy.h')
    cpp=(original/'fia_tiling_nonquant.cpp').read_text()
    old='    CalcMBaseSize();\n}'
    new='''    CalcMBaseSize();
    // Pre-registered fault28 extension: M selects the in-core KV grouping.
    // Keep the State layout, precision types and all later native derivations.
    if (fault28_fia::Coupled(fault28State_)) {
        sInnerSize_ = fault28State_.selectedM;
        sInnerLoopTimes_ = (s2Size + sInnerSize_ - 1U) / sInnerSize_;
        sInnerSizeTail_ = s2Size - (sInnerLoopTimes_ - 1U) * sInnerSize_;
        sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK);
    }
}'''
    cpp=once(cpp,old,new)
    cpp=once(cpp,'    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) ||','''    // A distinct compiled template changes only the online denominator update.
    // All M/S2/workspace/partition fields were generated by native host above.
    if (fault28_fia::Replay(fault28State_)) {
        if (tilingKey_ != 109000000000400000ULL || mBaseSize_ != 128 || sInnerSize_ != 128) {
            OP_LOGE(fiaInfo_->opName, "Denominator replay requires the validated M128/S2=128 BRC template.");
            return ge::GRAPH_FAILED;
        }
        tilingKey_ = 119000000000400000ULL;
    }
    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) ||''')
    (ev/'source/fia_tiling_nonquant.cpp').write_text(cpp,encoding='utf-8',newline='\n')
    manifest=json.loads((BASE/'vendor_manifest.json').read_bytes())
    assert sha(source/'build/libcust_opmaster_rt2.0.so')==next(r['sha256'] for r in manifest['artifacts'] if r['path']==LIB)
    assert sha(source/'build/libcust_opsproto_rt2.0.so')==next(r['sha256'] for r in manifest['artifacts'] if r['path']==PROTO)
    save(ev/'source_identity.json',{'base_vendor_manifest_sha256':sha(BASE/'vendor_manifest.json'),
         'source_commit':manifest['source_commit'],'old_host_sha256':sha(source/'build/libcust_opmaster_rt2.0.so'),
         'files':{f.name:sha(f) for f in sorted((ev/'source').iterdir())},'state_abi_unchanged':True})
    argv=['docker','run','--rm','--network','none','--cpus','4','--memory','8g','--pids-limit','512',
          '--name','dcat-f28-replay-host-a1','--mount','type=bind,src='+str(OLD)+',dst=/audit,readonly',
          '--mount','type=bind,src='+str(OLD/'dependencies_a4')+',dst=/deps,readonly',
          '--mount','type=bind,src='+str(ROOT)+',dst=/new',
          '--entrypoint','bash',IMAGE,'-c',
          'source /usr/local/Ascend/cann-9.1.0/set_env.sh && exec python3 -B /new/build_host.py --inside']
    save(ev/'start.json',{'command':argv,'devices':[],'npu_executed':False,'started_at':time.time()})
    with (ev/'build.log').open('x') as log:
        rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=480).returncode
    save(ev/'completion.json',{'returncode':rc,'npu_executed':False,'finished_at':time.time(),
         'status':'CPU_HOST_BUILT_NOT_NPU_VALIDATED' if rc==0 else 'FAIL_CPU_HOST_BUILD','log_sha256':sha(ev/'build.log')})
    if rc:return rc
    stage=ROOT/'vendor_visible_a1';shutil.copytree(BASE,stage,symlinks=True)
    vendor=stage/'vendors/fault28_fia_graph_transformer'
    for r in manifest['artifacts']:assert sha(vendor/r['path'])==r['sha256']
    for name in (LIB,'op_impl/ai_core/tbe/op_tiling/liboptiling.so',PROTO):
        target=vendor/name
        assert target.resolve().is_relative_to(vendor.resolve())
        binary='libcust_opsproto_rt2.0.so' if name==PROTO else 'libcust_opmaster_rt2.0.so'
        shutil.copyfile(ev/binary,target)
        strings=subprocess.check_output(['strings',str(target)],text=True)
        assert 'batch_causal_m256' in strings and 'batch_coupled_m32' in strings
    manifest.update(vendor_root='vendor_visible_a1/vendors/fault28_fia_graph_transformer',
                    policy_kind='fault28_M_selected_causal_grain_and_optional_M_coupled_incore_grouping',
                    policy_source_manifest_sha256=sha(ev/'source_identity.json'),
                    status='CPU_HOST_BUILT_NPU_COMBINATIONS_NOT_YET_VALIDATED',generation='visible_a1',
                    host_extension={'state_abi_unchanged':True,'source_identity_sha256':sha(ev/'source_identity.json'),
                                    'host_sha256':sha(ev/'libcust_opmaster_rt2.0.so'),
                                    'proto_sha256':sha(ev/'libcust_opsproto_rt2.0.so'),
                                    'all_loaded_host_consumers_relinked':['cust_proto','cust_opmaster'],
                                    'device_tiling_library_extended_modes_supported':False,
                                    'M_candidates':[512,256,128,64,32],'coupled_S2_candidates':[512,256,128,64,32],
                                    'kernel_binary_unchanged':True,'precision_types_unchanged':True})
    manifest['artifacts']=[{'path':f.relative_to(vendor).as_posix(),'bytes':f.stat().st_size,'sha256':sha(f)}
                           for f in sorted(vendor.rglob('*')) if f.is_file()]
    (stage/'vendor_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    save(ev/'stage_completion.json',{'status':'CPU_STAGE_READY_REQUIRES_NPU_LEGALITY_CHECKS',
         'manifest_sha256':sha(stage/'vendor_manifest.json'),'kernel_binary_unchanged':True,'npu_executed':False})
    print(json.dumps({'status':'CPU_STAGE_READY_REQUIRES_NPU_LEGALITY_CHECKS','manifest_sha256':sha(stage/'vendor_manifest.json')}))


if __name__=='__main__':raise SystemExit(main() or 0)
