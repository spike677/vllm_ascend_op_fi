"""Stdlib-only checks of new S2 build inputs; never load its library or decoder."""
import ast
import hashlib
import json
from pathlib import Path

if __package__:
    from .arch22_s2_geometry import ARRAYS,validate_matrix,validate_split
else:
    from arch22_s2_geometry import ARRAYS,validate_matrix,validate_split

PREFIX='fault_injection/native_adapters/attention28/fia/'
VARIANT='tnd_page16_arch22_s2_256'
VENDOR='dcat_f33_arch22_s2_transformer'


def need(value,reason):
    if not value:raise ValueError(reason)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(raw):
    def pairs(items):
        result={}
        for key,value in items:
            need(key not in result,'S2_DUPLICATE_JSON_KEY:'+key);result[key]=value
        return result
    def invalid(value):raise ValueError('S2_NONFINITE_JSON:'+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)


def read(path):
    path=Path(path)
    need(not path.is_symlink() and 0<path.stat().st_size<64*1024**2,'S2_BUILD_JSON_BOUND_OR_LINK')
    return parse(path.read_bytes())


def pristine_split_pins(recipe_sha256):
    """Read only literal source pins; do not import or execute the build recipe."""
    path=Path(__file__).with_name('apply_tnd33_arch22_s2.py')
    need(sha(path)==recipe_sha256,'S2_CPU_RECIPE_SOURCE_DIFFERS')
    nodes=[n.value for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.Assign)
           and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SOURCE_SHA']
    need(len(nodes)==1 and isinstance(nodes[0],ast.Call) and isinstance(nodes[0].func,ast.Name)
        and nodes[0].func.id=='dict' and len(nodes[0].keywords)==1 and nodes[0].keywords[0].arg is None,
        'S2_PRISTINE_SPLIT_SOURCE_DECLARATION')
    declared=ast.literal_eval(nodes[0].keywords[0].value)
    names=('attention/common/op_host/split_core.h','attention/common/op_host/split_core.cpp')
    return {name:declared[name] for name in names}


def verify_inputs(root,data,pins):
    root=Path(root).resolve(strict=True)
    need(data.get('variant')==VARIANT and data.get('arch22_s2_recipe_sha256')==pins[PREFIX+'apply_tnd33_arch22_s2.py']
        and data.get('arch22_s2_normal_route_recipe_sha256')==pins[PREFIX+'apply_tnd33_arch22_bridge.py']
        and data.get('arch22_s2_headers')=={name:pins[PREFIX+name] for name in
            ('paged33_arch22_s2.h','paged33_arch22_bridge.h')}
        and all(data.get(k) is False for k in ('old_normal_certificate_reused','fault_execution_ready','stock_private_exact_proven')),
        'S2_NEW_RECIPE_HEADERS_AND_NO_INHERITED_AUTHORITY_REQUIRED')
    folder=root/'evidence/arch22_s2_matrix';receipt=read(folder/'receipt.json');matrix=folder/'split_matrix.jsonl'
    need(sha(folder/'receipt.json')==data.get('arch22_s2_cpu_receipt_sha256')
        and not matrix.is_symlink() and 0<matrix.stat().st_size<64*1024**2
        and sha(matrix)==data.get('arch22_s2_cpu_matrix_sha256')==receipt.get('matrix_sha256')
        and receipt.get('validator_sha256')==data.get('arch22_s2_geometry_validator_sha256')==pins[PREFIX+'arch22_s2_geometry.py'],
        'S2_ACTUAL_CPU_MATRIX_OR_VALIDATOR_CHANGED')
    need(receipt.get('source_execution')=='PINNED_ORIGINAL_SPLIT_CORE_IN_CPU_SUBPROCESS'
        and receipt.get('npu_executed') is False and receipt.get('qualification_granted') is False
        and receipt.get('actual_worker_platform_observed') is False,
        'S2_CPU_MATRIX_IS_NOT_DEVICE_OR_PLATFORM_QUALIFICATION')
    source_pins=pristine_split_pins(pins[PREFIX+'apply_tnd33_arch22_s2.py'])
    need(receipt.get('source_sha256')==source_pins
        and all(not (root/'source'/name).is_symlink() and sha(root/'source'/name)==digest
                for name,digest in source_pins.items()),
        'S2_ORIGINAL_SPLIT_SOURCE_BYTES_CHANGED')
    need(sha(folder/'matrix.cpp')==receipt.get('program_sha256')
        and sha(folder/'log/log.h')==receipt.get('logging_only_shim_sha256')
        and (folder/'log/log.h').read_text()=='#pragma once\n#define OP_LOGI(...) ((void)0)\n#define OP_LOGD(...) ((void)0)\n',
        'S2_MATRIX_PROGRAM_OR_LOGGING_ONLY_SHIM_CHANGED')
    driver=Path(__file__).with_name('compile_arch22_s2_matrix.py')
    need(sha(driver)==pins[PREFIX+'compile_arch22_s2_matrix.py'],'S2_CPU_MATRIX_DRIVER_SOURCE_DIFFERS')
    assignments=[n for n in ast.parse(driver.read_bytes()).body if isinstance(n,ast.Assign)
                 and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='PROGRAM']
    need(len(assignments)==1,'S2_CPU_MATRIX_PROGRAM_DECLARATION')
    lines='\n'.join('array("'+name+'",r.'+('' if name in ('bN2End','gS1End','s2End') else 'fdRes.')+name+');' for name in ARRAYS)
    need((folder/'matrix.cpp').read_text()==ast.literal_eval(assignments[0].value).replace('ARRAY_LINES',lines),
         'S2_CPU_MATRIX_PROGRAM_NOT_FROZEN_DRIVER')
    selected={}
    def rows():
        with matrix.open(encoding='utf-8') as stream:
            for line in stream:
                need(len(line)<128*1024,'S2_MATRIX_ROW_BOUND')
                row=parse(line)
                if row['kv_length']==1025 and row['s2']==512:selected[row['aic']]=row
                yield row
    checked=validate_matrix(rows())
    need(all(receipt.get(k)==v for k,v in checked.items()) and set(selected)==set(range(16,27)),
        'S2_CPU_MATRIX_RECOMPUTATION_DIFFERS')
    need(sha(matrix)==receipt['matrix_sha256'] and sha(folder/'receipt.json')==data['arch22_s2_cpu_receipt_sha256'],
         'S2_MATRIX_CHANGED_DURING_REVIEW')
    return dict(normal_first_decode_rows=selected,receipt_sha256=sha(folder/'receipt.json'),
                matrix_sha256=sha(matrix),cases=checked['cases'],qualification_granted=False)


def verify_normal_geometry(typed,host,row):
    checked=validate_split(row)
    need(row['kv_length']==host['actual_kv_length']==1025 and row['s2']==host['native_s2']==host['selected_s2']==512
        and row['aic']==host['aic'] and host['aiv']==2*host['aic']
        and host['real_s2_capacity']==typed['baseParams.s2Size']==1040
        and host['s2_loops']==3 and host['s2_tail']==16 and host['s2_aligned']==512
        and typed['innerSplitParams.mBaseSize']==512 and typed['innerSplitParams.s2BaseSize']==512
        and typed['baseParams.usedCoreNum']==host['block_dim']==row['usedCoreNum']
        and typed['fdParams.numOfFdHead']==row['numOfFdHead']
        and typed['fdParams.usedVecNumOfFd']==row['usedVecNumOfFd']
        and typed['fdParams.gS1BaseSizeOfFd']==8,'S2_NORMAL_ACTUAL_SPLIT_GEOMETRY_DIFFERS')
    for name in ARRAYS:
        group='outerSplitParams.' if name in ('bN2End','gS1End','s2End') else 'fdParams.'
        value=typed[group+name];width=52 if name in ARRAYS[-2:] else 26
        need(type(value) is list and len(value)==width and value[:len(row[name])]==row[name]
            and all(type(n) is int and n==0 for n in value[len(row[name]):]),'S2_NORMAL_COMPLETE_SPLIT_ARRAY_DIFFERS:'+name)
    need(all(typed['workspaceParams.'+key]==value for key,value in checked['workspace_elements'].items())
        and type(host['libapi_workspace']) is int and host['libapi_workspace']>=0
        and host['workspace']==host['libapi_workspace']+checked['workspace_excluding_libapi'],
        'S2_NORMAL_ACTUAL_WORKSPACE_DIFFERS')
    return checked
