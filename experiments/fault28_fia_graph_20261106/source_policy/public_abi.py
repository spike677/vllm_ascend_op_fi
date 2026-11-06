"""Bind same-generation measured public serializer layout; no device ABI claim."""
import hashlib
import json
from generation import layout


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def bind(root,generation,policy):
    ev=layout(root,generation)['evidence']
    schema_file=ev/'fia_abi_schema.json';compiled_file=ev/'fia_abi_compile.json';test_file=ev/'fia_abi_selftest.json'
    schema=json.loads(schema_file.read_bytes());compiled=json.loads(compiled_file.read_bytes());test=json.loads(test_file.read_bytes())
    if schema['schema']!='fault28.fia_graph.public_serializer_layout.v1' or schema['type']!='FusedInferAttentionScoreTilingData' or schema['size']!=1576:
        raise ValueError('Unexpected FIA serializer schema')
    if test['status']!='PASS_CPU_DERIVED_LAYOUT_SENTINEL' or test['serialized_bytes']!=1576 or test['array_entries_checked']!=78 or test['invalid_lengths_rejected']!=2:
        raise ValueError('CPU serializer checks failed')
    if compiled['source_sha256']!=sha(policy/'fia_abi_decoder.cpp') or compiled['decoder_sha256']!=sha(ev/'fia_abi_decoder') or compiled['projection_sha256']!=sha(ev/'fia_schema_projection.h'):
        raise ValueError('ABI material differs from compiled receipt')
    if f'/source_{generation}/ops-transformer-v9.1.0/' not in compiled['actual_fia_compile_entry']['file']:
        raise ValueError('Wrong generation compiler entry')
    used=set();names=set()
    for f in schema['fields']:
        if f['name'] in names or f['width'] not in (1,4,8) or f['count']<1:raise ValueError('Invalid serializer field')
        names.add(f['name'])
        for j in range(f['offset'],f['offset']+f['width']*f['count']):
            if not 0<=j<1576 or j in used:raise ValueError('Invalid overlapping range')
            used.add(j)
    typed=json.loads((ev/'typed_fields.json').read_bytes())
    if len(typed)!=63 or not {f['name'] for f in typed}<=names or used!=set(range(1576)):
        raise ValueError('Public SDK metadata and typed declarations disagree')
    return {'generation':generation,'schema_file':schema_file.relative_to(root).as_posix(),'schema_sha256':sha(schema_file),
        'compile_file':compiled_file.relative_to(root).as_posix(),'compile_sha256':sha(compiled_file),
        'selftest_file':test_file.relative_to(root).as_posix(),'selftest_sha256':sha(test_file),
        'projection_file':(ev/'fia_schema_projection.h').relative_to(root).as_posix(),'projection_sha256':sha(ev/'fia_schema_projection.h'),
        'raw_size':1576,'projected_fields':len(schema['fields']),'typed_fields':63,
        'cpu_scalar_and_outer_array_sentinel_checks_only':True,
        'layout_kind':'public_host_serializer_projection','kernel_argument_raw_observed':False}
