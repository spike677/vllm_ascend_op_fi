"""Compile public FAI serializer sentinels and actual host geometry, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from apply_tnd33_fai_policy import typed_materials, FIA


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decoder_source(fields):
    common = Path(__file__).with_name('fia_abi_decoder.cpp').read_text()
    common = common[:common.index('int main()')]
    common = common.replace('fia_schema_projection.h', 'fai_schema_projection.h').replace('FusedInferAttentionScoreTilingData', 'FAInferTilingData')
    begin = common.index(' Layout out;fields(')
    end = common.index(';return out;', begin)
    common = common[:begin]+' Layout out;fields(v,"",0,out,{{"coreInfo",&v.coreInfo},{"splitInfo",&v.splitInfo}})'+common[end:]
    body = ['int main(){try{', 'FAInferTilingData v,again;const auto size=v.GetDataSize();const auto l=layout(v);',
            'if(!size||size%8)throw std::runtime_error("unexpected public size");']
    for index, field in enumerate(fields):
        name, dtype, count = field['name'], field['dtype'], field['count']
        prefix, _, leaf = name.rpartition('.')
        owner = (prefix+'.') if prefix else ''
        sentinel = str(index+17) if dtype != 'float' else '0.125f'
        if dtype.startswith('int'):
            sentinel = '-'+sentinel
        if count == 1:
            body += [f'v.{owner}set_{leaf}(static_cast<{dtype}>({sentinel}));',
                     f'again.{owner}set_{leaf}(static_cast<{dtype}>({sentinel}));']
        else:
            body += [f'{dtype} a{index}[26];for(unsigned j=0;j<26;++j)a{index}[j]=static_cast<{dtype}>({sentinel}+j);',
                     f'v.{owner}set_{leaf}(a{index});again.{owner}set_{leaf}(a{index});']
    body += ['std::vector<std::uint64_t> raw(size/8),serialized(size/8);v.SaveToBuffer(raw.data(),size);']
    for index, field in enumerate(fields):
        name, dtype, count = field['name'], field['dtype'], field['count']
        expected = field['getter'].replace('d.', 'v.')
        body += [f'for(unsigned j=0;j<{count};++j)if(read<{dtype}>(raw,l,"{name}",j)!='+
                 expected+('[j]' if count != 1 else '')+')throw std::runtime_error("public field sentinel differs:'+name+'");']
    body += ['again.SaveToBuffer(serialized.data(),size);if(raw!=serialized)throw std::runtime_error("roundtrip differs");',
             'std::cout<<"{\\"status\\":\\"PASS_CPU_FAI_PUBLIC_SENTINELS\\",\\"serialized_bytes\\":"<<size<<",\\"fields\\":{";bool first=true;',
             'for(const auto& p:l){if(!first)std::cout<<\',\';first=false;std::cout<<\'"\'<<p.first<<"\\":{\\"offset\\":"<<p.second.offset<<",\\"width\\":"<<p.second.width<<",\\"count\\":"<<p.second.count<<",\\"dtype\\":\\""<<p.second.dtype<<"\\"}";}',
             'std::cout<<"}}\\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<\'\\n\';return 1;}}']
    return common+'\n'+'\n'.join(body)+'\n'


GEOMETRY = r'''
#include "flash_attention_infer_tiling.h"
#include <iostream>
#include <stdexcept>
int main(){try{
using namespace optiling;
int64_t q[1]={1},kv[1]={1025};FAInferContext c;
c.numHeads=16;c.kvHeads=8;c.batch=1;c.embeddingSize=c.embeddingSizeV=128;
c.blockSize=16;c.numBlocks=128;c.maxNumBlocksPerBatch=65;c.pagedCacheFlag=true;
c.maxQSeqlen=1;c.maxKvSeqlen=1025;c.qSeqlenList=q;c.kvSeqlenList=kv;
c.scaleValue=0.08838834613561630249f;c.layout="TND";c.sparseMode=3;
for(auto mask:{MaskType::NO_MASK,MaskType::MASK_SPEC}){
 c.maskType=mask;
 for(unsigned tile:{512U,256U}){
  FAInferTiling t(c);t.SetCoreNum(16);t.DcatSetKvTile(tile);FAInferTilingData d;
  if(t.DoTiling(d)!=ge::GRAPH_SUCCESS)throw std::runtime_error("actual host tiling failed");
  const uint64_t unit=uint64_t(16)*128*tile*3;
  if(d.get_mm1OutSize()!=unit*4||d.get_smOnlineOutSize()!=unit*2||
     d.get_mm2OutSize()!=unit*4||d.get_UpdateSize()!=unit*4||d.get_workSpaceSize()!=unit*14||
     d.get_splitLseTotalSize()!=0||d.get_splitOTotalSize()!=0||d.get_totalTaskNum()!=8||
     d.get_firstBatchTaskNum()!=8||d.get_blockSize()!=16||d.get_maxKvSeqlen()!=1025||
     t.DcatWorkspaceBlockSize()!=128*tile)throw std::runtime_error("actual dependent geometry differs");
  const uint64_t expected=5000000000010200100ULL+(mask==MaskType::MASK_SPEC?3:0)+
      (tile==256?900000000000000000ULL:0);
  if(t.GetTilingKey()!=expected)throw std::runtime_error("actual key differs");
  // D128 output/update storage must still fit in each selected allocation.
  if(tile%16||tile%128||128*tile<128*128)throw std::runtime_error("tile buffer lower bound");
 }
}
std::cout<<"{\"status\":\"PASS_CPU_ACTUAL_FAI_HOST_GEOMETRY\",\"cases\":4,\"npu_executed\":false}\n";
return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
'''


def main(*, geometry_source=None, geometry_validator=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    host = args.source/FIA/'op_host'
    header = host/'flash_attention_infer_tiling.h'
    _, projection, fields = typed_materials(header.read_text())
    projected = args.out/'fai_schema_projection.h';projected.write_text(projection)
    entries = json.loads((args.source/'build/compile_commands.json').read_text())
    candidates = [e for e in entries if Path(e['file']).name == 'fused_infer_attention_score_tiling.cpp']
    if len(candidates) != 1:
        raise RuntimeError('EXACT_ACTUAL_FAI_HOST_COMPILE_ENTRY_REQUIRED')
    entry = candidates[0]
    original = entry.get('arguments') or shlex.split(entry['command'])
    flags, index = [], 1
    while index < len(original):
        item = original[index]
        if item in ('-o', '-MF', '-MT', '-MQ'):
            index += 2;continue
        if item in ('-c', '-MD', '-MMD', entry['file']):
            index += 1;continue
        flags.append(item);index += 1
    receipts = {}
    for name, source in (('fai_abi_decoder', decoder_source(fields)),
                         ('fai_host_geometry', GEOMETRY if geometry_source is None else geometry_source)):
        cpp, binary = args.out/(name+'.cpp'), args.out/name
        cpp.write_text(source)
        argv = [original[0], *flags, '-I'+str(args.out), '-I'+str(host), str(cpp),
                '-L'+str(args.sdk/'lib64'), '-lregister', '-lc_sec', '-o', str(binary)]
        subprocess.run(argv, cwd=entry['directory'], check=True, timeout=180)
        result = json.loads(subprocess.check_output([str(binary)], text=True, timeout=30))
        if name == 'fai_host_geometry' and geometry_validator is not None:
            result = geometry_validator(result)
        (args.out/(name+'.json')).write_text(json.dumps(result, indent=2)+'\n')
        receipts[name] = dict(argv=argv, cpp_sha256=sha(cpp), binary_sha256=sha(binary), result=result)
    result = receipts['fai_abi_decoder']['result']
    if result['status'] != 'PASS_CPU_FAI_PUBLIC_SENTINELS' or len(result['fields']) < len(fields):
        raise RuntimeError('ACTUAL_PUBLIC_FAI_SENTINEL_FAILED')
    names = {field['name'] for field in fields}
    if not names <= set(result['fields']):
        raise RuntimeError('DECLARED_FAI_FIELD_MISSING_FROM_ACTUAL_PUBLIC_LAYOUT')
    schema = dict(schema='demoncat.paged33.fai_public_serializer.v1', type='FAInferTilingData',
        size=result['serialized_bytes'], fields=[dict(name=name, **row) for name, row in result['fields'].items() if name in names],
        projection_sha256=sha(projected), public_source_sha256=sha(header),
        layout_kind='public_host_serializer_projection', kernel_argument_raw_observed=False)
    target = args.out/'fai_abi_schema.json';target.write_text(json.dumps(schema, indent=2)+'\n')
    receipt = dict(schema='demoncat.paged33.fai_cpu_schema_compile.v1', actual_compile_entry=entry,
        tools=receipts, public_source_sha256=sha(header), schema_sha256=sha(target), npu_executed=False,
        actual_launch_verified=False, loading_allowed=False)
    (args.out/'compile_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(status='CPU_FAI_LAYOUT_AND_HOST_GEOMETRY_CHECKED', schema=str(target), sha256=sha(target))))


if __name__ == '__main__':
    main()
