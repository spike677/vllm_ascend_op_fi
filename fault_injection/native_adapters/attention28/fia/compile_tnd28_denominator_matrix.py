"""Execute the pinned original SplitCore on CPU, for the new TND M128 scope."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

if __package__:
    from .apply_tnd28_denominator import SOURCE_SHA
    from .tnd28_denominator_geometry import ARRAYS,validate_matrix
else:
    from apply_tnd28_denominator import SOURCE_SHA
    from tnd28_denominator_geometry import ARRAYS,validate_matrix

SPLIT='attention/common/op_host/'
PROGRAM=r'''
#include <cstddef>
#include <climits>
#include <iostream>
#include "split_core.h"
using namespace optiling;
void array(const char* n,const std::vector<uint32_t>& v){
 std::cout<<",\""<<n<<"\":[";
 for(size_t i=0;i<v.size();++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']';
}
int main(){for(unsigned cores=16;cores<=26;++cores){for(unsigned splitCores: {cores,8U}){
 BaseInfo b;b.bSize=1;b.n2Size=8;b.gSize=2;b.s1Size=b.s2Size=2048;
 b.isS1G=true;b.isAccumSeqS1=b.isAccumSeqS2=true;
 b.actualSeqS1Size={2048};b.actualSeqS2Size={2048};b.actualLenQDims=b.actualLenKvDims=1;
 b.attenMaskFlag=true;b.sparseMode=3;b.preToken=INT_MAX;b.nextToken=0;
 SplitParam p;p.mBaseSize=p.s2BaseSize=128;p.gS1BaseSizeOfFd=8;
 SplitResult r(cores,2);SplitCore(splitCores,b,p,r);
 std::cout<<"{\"aic\":"<<cores<<",\"scheduling_cores\":"<<splitCores
  <<",\"m\":128,\"s2\":128,\"usedCoreNum\":"<<r.usedCoreNum
  <<",\"numOfFdHead\":"<<r.numOfFdHead<<",\"usedVecNumOfFd\":"<<r.usedVecNumOfFd
  <<",\"maxS2SplitNum\":"<<r.maxS2SplitNum;
 ARRAY_LINES
 std::cout<<"}\n";
}}}
'''


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--compiler',default='g++');args=p.parse_args()
    source=args.source.resolve(strict=True);out=args.out.resolve()
    if out.exists() or source==out or source in out.parents: raise ValueError('TND28_FRESH_MATRIX_DIRECTORY_REQUIRED')
    for name in ('split_core.h','split_core.cpp'):
        file=source/SPLIT/name
        if file.is_symlink() or sha(file)!=SOURCE_SHA[SPLIT+name]: raise ValueError('TND28_ACTUAL_SPLIT_SOURCE_DIFFERS')
    out.mkdir(parents=True);(out/'log').mkdir()
    (out/'log/log.h').write_text('#pragma once\n#define OP_LOGI(...) ((void)0)\n#define OP_LOGD(...) ((void)0)\n')
    lines='\n'.join('array("'+n+'",r.'+('' if n in ARRAYS[:3] else 'fdRes.')+n+');' for n in ARRAYS)
    program=out/'matrix.cpp';program.write_text(PROGRAM.replace('ARRAY_LINES',lines),encoding='utf-8')
    binary=out/'matrix'
    argv=[args.compiler,'-std=c++17','-O2','-include','cstddef','-include','climits','-I',str(out),
          '-I',str(source/SPLIT),str(program),str(source/SPLIT/'split_core.cpp'),'-o',str(binary)]
    subprocess.run(argv,check=True,timeout=120)
    matrix=out/'matrix.jsonl'
    with matrix.open('xb') as stream: subprocess.run([str(binary)],stdout=stream,check=True,timeout=120)
    report=validate_matrix(json.loads(line) for line in matrix.read_text().splitlines())
    report.update(matrix_sha256=sha(matrix),program_sha256=sha(program),compiler_argv=argv,
        source_sha256={SPLIT+n:sha(source/SPLIT/n) for n in ('split_core.cpp','split_core.h')},
        validator_sha256=sha(Path(__file__).with_name('tnd28_denominator_geometry.py')),
        executable_sha256=sha(binary),source_execution='PINNED_ORIGINAL_SPLIT_CORE_CPU',
        logging_only_shim_sha256=sha(out/'log/log.h'))
    (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))
    if not report['supported_core_profiles']: raise SystemExit('NO_COMPLETE_IN_CORE_TND_PROFILE_STOP_BEFORE_SDK')


if __name__=='__main__':main()
