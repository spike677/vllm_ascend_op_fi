"""Bounded decoder for an SDK-probed PUBLIC FAInfer candidate, not admission.

Capture completeness, installed-library identity, and full-output control remain
independent gates. Caller supplies actual API metadata, never an expected profile.
"""
import argparse,hashlib,json,math,struct
from pathlib import Path

TYPES={'uint8_t':'B','int8_t':'b','uint16_t':'H','int16_t':'h','uint32_t':'I','int32_t':'i',
       'uint64_t':'Q','int64_t':'q','float':'f','double':'d'}
REQUIRED=('numHeads','kvHeads','embeddingSize','embeddingSizeV','batch','blockSize',
          'maxQSeqlen','maxKvSeqlen','sparseMode')


def check_binding(data, binding):
    if len(data)<80:raise ValueError('TRUNCATED_CAPTURE')
    _,pid,serial,key,*_=struct.unpack_from('<10Q',data)
    if (binding.get('schema')!='dcat.launch.binding.v1'
            or binding.get('pid')!=pid or binding.get('serial')!=serial
            or binding.get('key_source')!='entry'
            or type(binding.get('entry')) is not int or binding['entry']!=key):
        raise ValueError('REAL_ENTRY_BINDING_REQUIRED')


def fields(layout):
    if layout.get('schema')!='dcat.fault33.sdk_layout.v1':raise ValueError('SDK_LAYOUT_SCHEMA_REQUIRED')
    classes={c['name']:c for c in layout['classes']}
    if len(classes)!=len(layout['classes']):raise ValueError('DUPLICATE_CLASS')
    result=[]
    def visit(name,prefix,start,ancestors):
        if name in ancestors:raise ValueError('RECURSIVE_LAYOUT')
        c=classes[name];cursor=0;names=set()
        if type(c['size']) is not int or not 0<c['size']<=65536:raise ValueError('CLASS_SIZE_INVALID')
        for f in c['fields']:
            if f['name'] in names:raise ValueError('DUPLICATE_FIELD')
            names.add(f['name'])
            if (any(type(f[k]) is not int for k in ('offset','count','bytes'))
                    or f['offset']!=cursor or f['count']<1 or f['bytes']<1
                    or cursor+f['bytes']>c['size']):raise ValueError('FIELD_BOUNDARY_INVALID')
            path=prefix+f['name']
            if f['kind']=='2':
                if f['count']!=1 or f['bytes']!=classes[f['type']]['size']:raise ValueError('STRUCT_SIZE_INVALID')
                visit(f['type'],path+'.',start+cursor,ancestors+(name,))
            else:
                if f['kind'] not in ('0','1') or f['type'] not in TYPES:raise ValueError('FIELD_TYPE_INVALID')
                if f['kind']=='0' and f['count']!=1:raise ValueError('SCALAR_COUNT_INVALID')
                if f['bytes']!=struct.calcsize('<'+TYPES[f['type']])*f['count']:raise ValueError('FIELD_SIZE_INVALID')
                result.append(dict(f,path=path,offset=start+cursor))
            cursor+=f['bytes']
        if cursor!=c['size']:raise ValueError('CLASS_END_INVALID')
    visit('FAInferTilingData','',0,())
    return classes['FAInferTilingData']['size'],result


def decode(data,layout,metadata,*,holder_index,allowed_keys):
    if len(data)<80:raise ValueError('TRUNCATED_CAPTURE')
    magic,pid,serial,key,blocks,size,count,width,rc,phbytes=struct.unpack_from('<10Q',data)
    if (magic!=0x444341544c415531 or not 0<size<=65536 or count>128 or width!=8
            or phbytes!=count*width or len(data)!=80+size+phbytes or rc!=0 or not 0<blocks<=26):
        raise ValueError('LAUNCH_HEADER_INVALID')
    if key not in allowed_keys:raise ValueError('KEY_NOT_REVIEWED_IN_CANDIDATE')
    if type(holder_index) is not int or not 0<=holder_index<count:raise ValueError('HOLDER_INDEX_INVALID')
    pointer_offset,tiling_offset=struct.unpack_from('<II',data,80+size+holder_index*width)
    total,flat=fields(layout)
    marker=layout.get('marker_roundtrip',{})
    if (marker.get('passed') is not True or marker.get('buffer_guards') is not True
            or type(marker.get('fields')) is not int
            or marker['fields']!=sum(not f['name'].endswith('PH') for f in flat)):
        raise ValueError('SDK_SERIALIZER_MARKER_CHECK_REQUIRED')
    if pointer_offset+8>size or tiling_offset+total>size:raise ValueError('TILING_FIELD_OUTSIDE_CAPTURE')
    if max(pointer_offset,tiling_offset)<min(pointer_offset+8,tiling_offset+total):
        raise ValueError('POINTER_OVERLAPS_TILING')
    decoded={}
    for f in flat:
        values=struct.unpack_from('<'+TYPES[f['type']]*f['count'],data,80+tiling_offset+f['offset'])
        if any(isinstance(v,float) and not math.isfinite(v) for v in values):raise ValueError('NONFINITE_TILING_FIELD')
        if f['name'].endswith('PH'):continue # alignment bytes are not logical fields
        decoded[f['path']]=values[0] if f['kind']=='0' else list(values)
    missing=[k for k in REQUIRED if type(metadata.get(k)) is not int]
    if missing:raise ValueError('ACTUAL_API_METADATA_MISSING:'+','.join(missing))
    mismatched=[k for k in REQUIRED if decoded.get(k)!=metadata[k]]
    if decoded['numHeads']<1 or decoded['kvHeads']<1 or decoded['numHeads']%decoded['kvHeads']:
        mismatched.append('HEAD_RATIO')
    if decoded['needCoreNum']!=blocks:mismatched.append('ACTUAL_BLOCKDIM')
    if not 0<=decoded['totalSplitNodeNum']<=26:mismatched.append('SPLIT_NODE_BOUND')
    return dict(schema='dcat.fault33.candidate_decode.v1',capture_sha256=hashlib.sha256(data).hexdigest(),
        key=key,block_dim=blocks,holder_index=holder_index,tiling_offset=tiling_offset,tiling_bytes=total,
        fields=decoded,shape_consistent=not mismatched,issues=mismatched,
        serializer_validated=False,qualification_granted=False,
        remaining_gates=['INSTALLED_SDK_LAYOUT_CHECK','THREE_ACTUAL_SHAPES','EACH_FULL_CONTROL_GE_99_PERCENT','DIFFERENCE_LOCALIZATION'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('--layout',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--holder-index',type=int,required=True)
    p.add_argument('--reviewed-key',type=int,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=a.capture.read_bytes()
    check_binding(raw,json.loads(Path(str(a.capture)+'.binding.json').read_bytes()))
    r=decode(raw,json.loads(a.layout.read_text()),json.loads(a.metadata.read_text()),
        holder_index=a.holder_index,allowed_keys=a.reviewed_key)
    a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if r['shape_consistent'] else 1)
