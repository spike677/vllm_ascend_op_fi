"""Exact-SDK host FD decisions, complete public arrays and workspace matrix."""
import argparse
import hashlib
import json
from pathlib import Path

import compile_tnd33_fai_schema as common
from apply_tnd33_fai_fd_policy import FIA, native_decode_decision
from fai_fd_geometry import validate_fd_geometry


def geometry_source(decision):
    return r'''
#include "flash_attention_infer_tiling.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <climits>
using namespace optiling;
namespace paged33_fai { inline std::string Quote(const std::string& x){return "\""+x+"\"";} }
#include "paged33_fai_typed.h"
''' + decision + r'''
int main(){try{
  int64_t q[1]={1},kv[1]={1025};FAInferContext c;
  c.numHeads=16;c.kvHeads=8;c.batch=1;c.embeddingSize=c.embeddingSizeV=128;
  c.blockSize=16;c.numBlocks=256;c.maxNumBlocksPerBatch=128;c.pagedCacheFlag=true;
  c.scaleValue=0.08838834613561630249f;c.layout="TND";c.sparseMode=3;
  ComputeSeqLenAndDecodeFlagsForFAI(c,16,q,kv,1);
  if(c.flashDecodeFlag||c.decodingFlag||c.maxQSeqlen||c.maxKvSeqlen)
    throw std::runtime_error("native 16-core decode decision or max context changed");
  // Every healthy key is derived through the original condition, not assigned.
  unsigned healthy=0;
  for(bool paged:{false,true})for(auto mask:{MaskType::NO_MASK,MaskType::MASK_SPEC}){
    c.pagedCacheFlag=paged;c.maskType=mask;c.sparseMode=mask==MaskType::MASK_SPEC?3:0;
    c.flashDecodeFlag=false;c.decodingFlag=false;
    ComputeSeqLenAndDecodeFlagsForFAI(c,16,q,kv,1);
    FAInferTiling t(c);t.SetCoreNum(16);FAInferTilingData d;
    if(t.DoTiling(d)!=ge::GRAPH_SUCCESS||c.flashDecodeFlag||
       t.GetTilingKey()!=5000000000000200100ULL+(paged?10000000ULL:0ULL)+(mask==MaskType::MASK_SPEC?3:0))
      throw std::runtime_error("native healthy branch changed");
    healthy++;
  }
  std::cout<<std::boolalpha<<std::setprecision(17);
  std::cout<<"{\"status\":\"ACTUAL_NATIVE_FD_MATRIX\",\"aic16_fd_false_verified\":true,\"healthy_non_fd_keys\":"<<healthy<<",\"rows\":[";
  bool comma=false;c.pagedCacheFlag=true;
  for(unsigned aic=20;aic<=25;++aic)for(int64_t length=1025;length<=2048;++length)
    for(auto mask:{MaskType::NO_MASK,MaskType::MASK_SPEC})for(unsigned tile:{512U,256U}){
      kv[0]=length;c.maskType=mask;c.sparseMode=mask==MaskType::MASK_SPEC?3:0;
      c.flashDecodeFlag=false;c.decodingFlag=false;
      ComputeSeqLenAndDecodeFlagsForFAI(c,aic,q,kv,1);
      if(!c.flashDecodeFlag||c.decodingFlag||c.maxQSeqlen||c.maxKvSeqlen)
        throw std::runtime_error("native FD scope differs");
      FAInferTiling t(c);t.SetCoreNum(aic);t.DcatSetKvTile(tile);FAInferTilingData d;
      if(t.DoTiling(d)!=ge::GRAPH_SUCCESS)throw std::runtime_error("native FD DoTiling failed");
      if(comma)std::cout<<',';comma=true;
      const uint64_t workspace=16ULL*1024*1024+uint64_t(aic)*t.DcatWorkspaceBlockSize()*4*3*4+
         d.get_splitLseTotalSize()+d.get_splitOTotalSize();
      std::cout<<"{\"aic\":"<<aic<<",\"length\":"<<length<<",\"tile\":"<<tile
         <<",\"key\":"<<t.GetTilingKey()<<",\"block_dim\":"<<(d.get_needCoreNum()?d.get_needCoreNum():aic)
         <<",\"workspace\":"<<workspace<<",\"typed\":";
      paged33_fai::AppendTyped(std::cout,d);std::cout<<'}';
    }
  std::cout<<"]}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
'''


def validate_matrix(result):
    if result.get('status') != 'ACTUAL_NATIVE_FD_MATRIX' or result.get('aic16_fd_false_verified') is not True \
            or result.get('healthy_non_fd_keys') != 4 or len(result.get('rows', [])) != 24576:
        raise RuntimeError('ACTUAL_FD_CPU_MATRIX_INCOMPLETE')
    seen, keys, active = set(), set(), set()
    for row in result['rows']:
        aic, length, tile = (row[name] for name in ('aic', 'length', 'tile'))
        host = dict(route='FAInferTiling', aic=aic, native_route_or_decode_flags_changed=False,
            flash_decode=True, decoding=False, tiling_sink=False, query_length=1, kv_length=length,
            context_max_q=0, context_max_kv=0, workspace=row['workspace'])
        checked = validate_fd_geometry(row['typed'], host, key=row['key'], block_dim=row['block_dim'],
            schedule_mode=1, arm='fault' if tile == 256 else 'baseline', kv_length=length)
        identity = (aic, length, row['typed']['maskType'], tile)
        if identity in seen:raise RuntimeError('ACTUAL_FD_CPU_MATRIX_DUPLICATE')
        seen.add(identity);keys.add(row['key']);active.add(checked['actual_active_cores'])
    expected = {(aic, length, mask, tile) for aic in range(20, 26) for length in range(1025, 2049)
                for mask in (0, 1) for tile in (512, 256)}
    if seen != expected:raise RuntimeError('ACTUAL_FD_CPU_MATRIX_SCOPE_DIFFERS')
    return dict(status='PASS_CPU_ACTUAL_FAI_FD_COVERAGE_AND_REDUCTION', cases=len(seen),
        aic_supported_bounds=[20, 25], actual_worker_aic_observed=False,
        native_aic16_fd_false_verified=True, healthy_non_fd_keys=4,
        actual_compiled_host_keys=sorted(keys), active_core_counts=sorted(active),
        all_public_arrays_checked=True, npu_executed=False)


def main():
    p = argparse.ArgumentParser(add_help=False);p.add_argument('--source', type=Path);p.add_argument('--out', type=Path)
    args, _ = p.parse_known_args()
    text = (args.source / FIA / 'op_host/fused_infer_attention_score_tiling.cpp').read_text()
    decision = native_decode_decision(text)
    def validator(result):
        checked = validate_matrix(result)
        raw = json.dumps(result, separators=(',', ':'), allow_nan=False).encode()
        path = args.out / 'fai_fd_geometry_matrix.json'
        with path.open('xb') as stream:stream.write(raw)
        return dict(checked, matrix_sha256=hashlib.sha256(raw).hexdigest(),
                    native_decision_sha256=hashlib.sha256(decision.encode()).hexdigest())
    common.main(geometry_source=geometry_source(decision), geometry_validator=validator)


if __name__ == '__main__':main()
