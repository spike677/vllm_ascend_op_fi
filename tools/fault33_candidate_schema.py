"""Generate an SDK layout probe from the pinned PUBLIC candidate declarations.

This tool does not establish correspondence to an installed operator binary.
It copies only the three declaration blocks into an SDK-compiled probe; it does
not replace the vendor serializer macro implementations with guessed packing.
"""
import argparse,hashlib,json,re,subprocess
from pathlib import Path

COMMIT='c77c0829f1f0f0b8f882db1d9b051a758b605472'
HEADER='attention/fused_infer_attention_score/op_host/flash_attention_infer_tiling.h'
CLASSES=('coreNode','splitNode','FAInferTilingData')


def generate(root,out):
    root=Path(root);out=Path(out)
    actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if actual!=COMMIT:raise ValueError('CANDIDATE_COMMIT_MISMATCH')
    raw=(root/HEADER).read_bytes()
    expected=subprocess.check_output(['git','-C',str(root),'show','HEAD:'+HEADER])
    if raw.replace(b'\r\n',b'\n')!=expected.replace(b'\r\n',b'\n'):
        raise ValueError('CANDIDATE_WORKTREE_BYTES_CHANGED')
    # Git blob bytes are authoritative; Windows checkout may have CRLF.
    source=expected.decode('utf-8');blocks=[]
    for name in CLASSES:
        matches=re.findall(r'BEGIN_TILING_DATA_DEF\('+name+r'\).*?END_TILING_DATA_DEF',source,re.S)
        if len(matches)!=1:raise ValueError('DECLARATION_NOT_UNIQUE:'+name)
        blocks.append(matches[0])
    constant=re.search(r'constexpr int32_t MAX_CORE_NUM_FD = (\d+);',source)
    if constant is None:raise ValueError('ARRAY_EXTENT_UNKNOWN')
    cpp='#include <iostream>\n#include <stdexcept>\n#include <vector>\n#include <cstring>\n#include "register/tilingdata_base.h"\nnamespace optiling {\n'
    cpp+=constant.group(0)+'\n'+';\n'.join(blocks)+';\n}\n'
    cpp+=r'''
using namespace optiling;
size_t width(const char* s){
    std::string t(s);
    if(t=="uint8_t"||t=="int8_t"||t=="char")return 1;
    if(t=="uint16_t"||t=="int16_t")return 2;
    if(t=="uint32_t"||t=="int32_t"||t=="float")return 4;
    if(t=="uint64_t"||t=="int64_t"||t=="double")return 8;
    throw std::runtime_error("UNRECOGNIZED_SERIALIZER_TYPE:"+t);
}
void emit(TilingDef& d){
    size_t offset=0;bool first=true;
    std::cout<<"{\"name\":\""<<d.GetTilingClassName()<<"\",\"size\":"<<d.GetDataSize()<<",\"fields\":[";
    for(auto f:d.GetFieldInfo()){
        if(!first)std::cout<<",";first=false;
        std::string kind(f.classType_);size_t count=kind=="1"?f.arrSize_:1;
        size_t bytes=kind=="2"?f.structSize_:width(f.dtype_)*count;
        std::cout<<"{\"name\":\""<<f.name_<<"\",\"type\":\""<<(kind=="2"?f.structType_:f.dtype_)
                 <<"\",\"kind\":\""<<kind<<"\",\"offset\":"<<offset<<",\"count\":"<<count<<",\"bytes\":"<<bytes<<"}";
        offset+=bytes;
    }
    if(offset!=d.GetDataSize())throw std::runtime_error("SERIALIZER_FIELD_BOUNDARY_MISMATCH");
    std::cout<<"]}";
}
size_t offset_of(TilingDef& d,const char* name){
    size_t offset=0;
    for(auto f:d.GetFieldInfo()){
        if(std::string(f.name_)==name)return offset;
        std::string kind(f.classType_);
        offset+=kind=="2"?f.structSize_:width(f.dtype_)*(kind=="1"?f.arrSize_:1);
    }
    throw std::runtime_error("MARKER_FIELD_MISSING");
}
'''
    # Exercise the SDK setters AND SaveToBuffer, independently of merely
    # summing FieldInfo. Every logical scalar and every array element is marked;
    # nested offsets and both buffer guards are checked against actual bytes.
    declarations=dict(zip(CLASSES,blocks));setters=[];checks=[]
    def mark(name,prefix,offset):
        pattern=r'TILING_DATA_FIELD_DEF(_ARR|_STRUCT)?\(([^\n]+)\)'
        for match in re.finditer(pattern,declarations[name]):
            kind=match[1];args=[x.strip() for x in match[2].split(',')]
            if kind=='_STRUCT':
                typename,field=args
                mark(typename,prefix+'.'+field,offset+'+offset_of('+prefix+',"'+field+'")');continue
            typename,field=args[0],args[-1];n=len(checks)+1;var='marker_'+str(n)
            count=args[1] if kind=='_ARR' else '1'
            setters.append(f'    std::vector<{typename}> {var}({count});')
            setters.append(f'    for(size_t i=0;i<{var}.size();++i){var}[i]=static_cast<{typename}>({n}*1000+i+1);')
            setters.append(f'    {prefix}.set_{field}({var}.data());' if kind=='_ARR'
                           else f'    {prefix}.set_{field}({var}[0]);')
            field_offset=offset+'+offset_of('+prefix+',"'+field+'")'
            checks.append(f'    check({field_offset},{var}.data(),{var}.size()*sizeof({typename}));')
    mark('FAInferTilingData','d','0')
    cpp+='\nsize_t marker_roundtrip(){\n    FAInferTilingData d;\n'+'\n'.join(setters)+r'''
    const size_t n=d.GetDataSize();std::vector<uint8_t> raw(n+32,0xa5);
    d.SaveToBuffer(raw.data()+16,n);
    for(size_t i=0;i<16;++i)if(raw[i]!=0xa5||raw[n+16+i]!=0xa5)throw std::runtime_error("SERIALIZER_BUFFER_OVERRUN");
    auto check=[&](size_t offset,const void* value,size_t bytes){
        if(offset>n||bytes>n-offset||std::memcmp(raw.data()+16+offset,value,bytes)!=0)
            throw std::runtime_error("SERIALIZER_MARKER_MISMATCH_AT_"+std::to_string(offset));
    };
'''+ '\n'.join(checks)+f'\n    return {len(checks)};\n}}\n'
    cpp+=r'''
int main(){
    size_t marked=marker_roundtrip();coreNode c;splitNode s;FAInferTilingData d;
    std::cout<<"{\"schema\":\"dcat.fault33.sdk_layout.v1\",\"qualification_granted\":false,\"marker_roundtrip\":{\"passed\":true,\"fields\":"<<marked<<",\"buffer_guards\":true},\"classes\":[";
    emit(c);std::cout<<",";emit(s);std::cout<<",";emit(d);std::cout<<"]}\n";
}
'''
    out.mkdir(parents=True,exist_ok=True)
    (out/'layout_probe.cpp').write_text(cpp,encoding='utf-8')
    files=[]
    for rel in [HEADER,'attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp',
                'attention/fused_infer_attention_score/op_host/arch22/fused_infer_attention_score_tiling_v3.cpp',
                'attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score_tilingkey.h','version.cmake']:
        data=subprocess.check_output(['git','-C',str(root),'show','HEAD:'+rel])
        files.append(dict(path=rel,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    manifest=dict(schema='dcat.fault33.public_source_candidate.v1',repository='https://gitcode.com/cann/ops-transformer',
        tag='v9.1.0',commit=actual,files=files,probe_sha256=hashlib.sha256((out/'layout_probe.cpp').read_bytes()).hexdigest(),
        installed_binary_correspondence='NOT_PROVEN',qualification_granted=False)
    (out/'source-candidate.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source_root');p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(generate(a.source_root,a.output),indent=2))
