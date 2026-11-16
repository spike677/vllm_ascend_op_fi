"""Read installed headers/ELF exports and compile ABI facts; never load CANN.

An image tag or version directory is not evidence of the installed package
version. This records installation metadata separately from path/tag hints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

APIS = ('aclnnMatmul', 'aclnnMatmulWeightNz', 'aclnnQuantMatmulWeightNz',
        'aclnnDynamicQuant', 'aclnnDynamicQuantV2', 'aclnnFusedInferAttentionScore',
        'aclnnFusedInferAttentionScoreV2', 'aclnnFusedInferAttentionScoreV3',
        'aclnnIncreFlashAttention', 'aclnnAddRmsNormBias', 'aclnnAddRmsNorm')
PRIVATE = ('_ZN2op8internal11OpExecCache3RunEPvS2_RKSt6vectorIS2_SaIS2_EE',
           '_ZN2op8internal14AddOpExecCacheEPNS0_11OpExecCacheE',
           '_ZN2op8internal13AddParamToBufEPK9aclTensor',
           '_ZN2op8internal13AddParamToBufEP9aclTensor',
           '_ZN2op8internal13GetLogApiInfoEv')
PROBE = r'''
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn/opdev/op_cache.h>
#include <aclnnop/aclnn_matmul.h>
#include <iostream>
#include <type_traits>
using Expected = aclnnStatus(*)(const aclTensor*,const aclTensor*,aclTensor*,int8_t,uint64_t*,aclOpExecutor**);
static_assert(std::is_same<decltype(&aclnnMatmulGetWorkspaceSize),Expected>::value,"Matmul GWS signature changed");
int main(){std::cout << "{\"aclTensor_size\":" << sizeof(aclTensor)
 << ",\"OpExecCache_size\":" << sizeof(op::internal::OpExecCache)
 << ",\"ACL_BF16\":" << int(ACL_BF16) << ",\"ACL_FORMAT_ND\":" << int(ACL_FORMAT_ND) << "}";}
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(root):
    root = root.resolve(strict=True)
    result = dict(toolkit=str(root), installation={}, headers={}, exports={}, libraries={}, signatures={}, actual_npu_calls=0)
    for p in sorted(root.glob('*install.info')):
        result['installation'][p.name] = dict(re.findall(r'^([^=\n]+)=(.*)$', p.read_text(errors='replace'), re.M))
    names = ('acl_meta.h','op_cache.h','common_types.h','aclnn_matmul.h','aclnn_quant_matmul_weight_nz.h',
             'aclnn_matmul_weight_nz.h','aclnn_dynamic_quant_v2.h','aclnn_add_rms_norm_bias.h',
             'aclnn_fused_infer_attention_score_v3.h','aclnn_incre_flash_attention.h')
    for p in sorted((root/'include').rglob('*.h')):
        if p.name in names:
            result['headers'][p.relative_to(root).as_posix()] = sha(p)
        if p.name.startswith('aclnn_'):
            source=re.sub(r'/\*.*?\*/|//[^\n]*',' ',p.read_text(errors='replace'),flags=re.S)
            for api in APIS:
                for symbol in (api,api+'GetWorkspaceSize'):
                    found=re.search(r'\b'+symbol+r'\s*\((.*?)\)\s*;',source,re.S)
                    if found:
                        result['signatures'].setdefault(symbol,[]).append(dict(
                            header=p.relative_to(root).as_posix(),
                            parameters=' '.join(found[1].split()),
                            parameter_count=len(found[1].split(','))))
    libs = sorted((root/'lib64').glob('libopapi*.so')) + [root/'lib64/libnnopbase.so', root/'lib64/libascendcl.so']
    for p in libs:
        if not p.is_file(): continue
        result['libraries'][p.name] = {'resolved': str(p.resolve()), 'sha256': sha(p)}
        r = subprocess.run(['nm','-D','--defined-only',str(p)], text=True, capture_output=True)
        if r.returncode: result['libraries'][p.name]['nm_error'] = r.stderr; continue
        exports = {line.split()[-1].split('@')[0] for line in r.stdout.splitlines() if line.split()}
        for symbol in (*PRIVATE, *(x+s for x in APIS for s in ('','GetWorkspaceSize'))):
            if symbol in exports: result['exports'].setdefault(symbol,[]).append(p.name)
    with tempfile.TemporaryDirectory() as d:
        source, exe = Path(d)/'probe.cpp', Path(d)/'probe'
        source.write_text(PROBE)
        command = ['g++','-std=c++17','-D_GLIBCXX_USE_CXX11_ABI=0',
                   '-I'+str(root/'include'),'-I'+str(root/'include/aclnn'),'-I'+str(root/'pkg_inc'), str(source),'-o',str(exe)]
        r = subprocess.run(command, text=True, capture_output=True)
        result['compile_probe'] = {'returncode':r.returncode,'stderr':r.stderr}
        if not r.returncode:
            # This executable contains sizeof/enum facts only, no CANN linkage.
            result['abi_facts'] = json.loads(subprocess.check_output([str(exe)], text=True))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,action='append')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image-tag',required=True)
    a=p.parse_args()
    ascend=Path('/usr/local/Ascend')
    roots=a.root or sorted({p.resolve() for pattern in ('cann-*/aarch64-linux','ascend-toolkit/*/aarch64-linux')
                           for p in ascend.glob(pattern) if (p/'ascend_toolkit_install.info').is_file()})
    data={'schema':'demoncat.cann.interface_inventory.v1','image_tag_hint':a.image_tag,
          'toolkits':[inspect(r) for r in roots], 'actual_npu_calls':0,
          'acceptance':'CPU_INTERFACE_FACTS_ONLY'}
    with a.output.open('x') as f:json.dump(data,f,indent=2)


if __name__=='__main__':main()
