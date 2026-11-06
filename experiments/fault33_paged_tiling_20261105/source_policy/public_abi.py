"""Bind actual same-generation public-serializer ABI artifacts into a vendor."""
import hashlib
import json
from generation import layout


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(root,generation,policy):
    ev=layout(root,generation)['evidence']
    schema_path=ev/'ifa_abi_schema.json';compile_path=ev/'ifa_abi_compile.json';test_path=ev/'ifa_abi_selftest.json'
    schema=json.loads(schema_path.read_text());compiled=json.loads(compile_path.read_text());test=json.loads(test_path.read_text())
    if schema['schema']!='fault33.pa.public_serializer_layout.v1' or schema['type']!='IncreFlashAttentionTilingDataV2' or schema['size']!=3000 or len(schema['fields'])!=72:
        raise ValueError('Unexpected public IFA serializer schema')
    if test['status']!='PASS_PUBLIC_SERIALIZER_CPU_ROUNDTRIP' or test['size']!=3000 or test['projected_fields']!=72:
        raise ValueError('Public ABI roundtrip gate failed')
    if compiled['source_sha256']!=sha(policy/'ifa_abi_decoder.cpp') or compiled['decoder_sha256']!=sha(ev/'ifa_abi_decoder'):
        raise ValueError('ABI decoder/source differs from its compile receipt')
    if f'/source_{generation}/ops-transformer-v9.1.0/' not in compiled['actual_ifa_compile_entry']['file']:
        raise ValueError('ABI compile entry belongs to another generation')
    used=set();names=set()
    for f in schema['fields']:
        if f['name'] in names or f['width']!=4 or f['count']<1:
            raise ValueError('Invalid public ABI field')
        names.add(f['name'])
        for offset in range(f['offset'],f['offset']+f['width']*f['count']):
            if offset<0 or offset>=3000 or offset in used:
                raise ValueError('Overlapping or invalid public ABI field range')
            used.add(offset)
    return {'generation':generation,'schema_file':schema_path.relative_to(root).as_posix(),'schema_sha256':sha(schema_path),
        'schema_bytes':schema_path.stat().st_size,'compile_file':compile_path.relative_to(root).as_posix(),'compile_sha256':sha(compile_path),
        'selftest_file':test_path.relative_to(root).as_posix(),'selftest_sha256':sha(test_path),'raw_size':3000,'projected_fields':72,
        'layout_kind':'public_host_serializer_projection','kernel_argument_raw_observed':False}
