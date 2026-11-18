"""Compile only the pinned, original CPU SplitCore; no SDK/DSO/device loading."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

if __package__:
    from .apply_tnd33_arch22_s2 import SOURCE_SHA
    from .arch22_s2_geometry import ARRAYS,validate_matrix
else:
    from apply_tnd33_arch22_s2 import SOURCE_SHA
    from arch22_s2_geometry import ARRAYS,validate_matrix

SPLIT='attention/common/op_host/'
PROGRAM=r'''
#include <cstddef>
#include <climits>
#include <iostream>
#include "split_core.h"
using namespace optiling;
void array(const char* name,const std::vector<uint32_t>& value){
 std::cout<<",\""<<name<<"\":[";
 for(size_t i=0;i<value.size();++i){if(i)std::cout<<',';std::cout<<value[i];}std::cout<<']';
}
int main(){
 for(unsigned cores=16;cores<=26;++cores)for(int64_t length=1025;length<=2048;++length)for(unsigned tile:{512U,256U}){
  BaseInfo b;b.bSize=1;b.n2Size=8;b.gSize=2;b.s1Size=1;b.s2Size=(length+15)/16*16;
  b.isS1G=true;b.isAccumSeqS1=true;b.isAccumSeqS2=false;
  b.actualSeqS1Size={1};b.actualSeqS2Size={length};b.actualLenQDims=b.actualLenKvDims=1;
  b.attenMaskFlag=true;b.sparseMode=3;b.preToken=INT_MAX;b.nextToken=0;
  SplitParam p;p.mBaseSize=512;p.s2BaseSize=tile;p.gS1BaseSizeOfFd=8;
  SplitResult r(cores,2);SplitCore(cores,b,p,r);
  std::cout<<"{\"aic\":"<<cores<<",\"kv_length\":"<<length<<",\"s2\":"<<tile
   <<",\"usedCoreNum\":"<<r.usedCoreNum<<",\"numOfFdHead\":"<<r.numOfFdHead
   <<",\"usedVecNumOfFd\":"<<r.usedVecNumOfFd<<",\"maxS2SplitNum\":"<<r.maxS2SplitNum;
  ARRAY_LINES
  std::cout<<"}\n";
 }
}
'''


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--compiler',default='g++');args=parser.parse_args()
    source=args.source.resolve(strict=True);out=args.out.resolve()
    if out.exists() or source==out or source in out.parents:raise ValueError('FRESH_EXTERNAL_CPU_MATRIX_DIRECTORY')
    pins={}
    for name in ('split_core.h','split_core.cpp'):
        path=source/SPLIT/name;expected=SOURCE_SHA[SPLIT+name]
        if path.is_symlink() or sha(path)!=expected:raise ValueError('ACTUAL_SPLIT_CORE_SOURCE_DIFFERS:'+name)
        pins[SPLIT+name]=expected
    out.mkdir(parents=True);shim=out/'log';shim.mkdir()
    # These are the source's only external interfaces. Disable diagnostic
    # logging, preserving every arithmetic/loop/array line byte for byte.
    (shim/'log.h').write_text('#pragma once\n#define OP_LOGI(...) ((void)0)\n#define OP_LOGD(...) ((void)0)\n')
    lines='\n'.join('array("'+name+'",r.'+('' if name in ('bN2End','gS1End','s2End') else 'fdRes.')+name+');' for name in ARRAYS)
    program=out/'matrix.cpp';program.write_text(PROGRAM.replace('ARRAY_LINES',lines),encoding='utf-8')
    executable=out/'matrix'
    argv=[args.compiler,'-std=c++17','-O2','-include','cstddef','-include','climits','-I',str(out),
          '-I',str(source/SPLIT),str(program),str(source/SPLIT/'split_core.cpp'),'-o',str(executable)]
    subprocess.run(argv,check=True,timeout=120)
    matrix=out/'split_matrix.jsonl'
    with matrix.open('xb') as stream:subprocess.run([str(executable)],stdout=stream,check=True,timeout=120)
    with matrix.open(encoding='utf-8') as stream:receipt=validate_matrix(json.loads(line) for line in stream)
    receipt.update(matrix_sha256=sha(matrix),source_sha256=pins,compiler_command=argv,
        source_execution='PINNED_ORIGINAL_SPLIT_CORE_IN_CPU_SUBPROCESS',
        logging_only_shim_sha256=sha(shim/'log.h'),program_sha256=sha(program),validator_sha256=sha(Path(__file__).with_name('arch22_s2_geometry.py')))
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
