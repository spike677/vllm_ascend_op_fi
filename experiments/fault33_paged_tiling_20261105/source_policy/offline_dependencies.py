"""Validate every enabled third-party acquisition branch before the CPU container.

All input directories are new generation-owned copies. Sources are not patched
to hide download failures, and network remains disabled during the build.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    """Inventory declarations/calls; records are evidence, not a CMake evaluator."""
    expressions = re.compile(r'\b(FetchContent_Declare|FetchContent_Populate|FetchContent_MakeAvailable|ExternalProject_Add|add_cann_third_party)\s*\(\s*([^\s()]+)', re.I)
    result=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or (p.suffix!='.cmake' and p.name!='CMakeLists.txt'):
            continue
        if any(part in ('.git','build') for part in p.relative_to(root).parts):
            continue
        text=p.read_text(encoding='utf-8',errors='strict')
        for match in expressions.finditer(text):
            line_start=text.rfind('\n',0,match.start())+1
            if text[line_start:match.start()].lstrip().startswith('#'):
                continue
            result.append({'path':p.relative_to(root).as_posix(),'line':text.count('\n',0,match.start())+1,
                'command':match.group(1),'name':match.group(2).strip('"')})
    return result


def validate(source,deps,evidence):
    here=Path(__file__).resolve().parent
    contract=json.loads((here/'offline_cmake_contract.json').read_text())
    for item in contract['reviewed_files']:
        p=(source if item['repository']=='ops-transformer' else deps/'cann-cmake')/item['path']
        if not p.is_file() or sha(p)!=item['sha256']:
            raise RuntimeError('Reviewed CMake interface differs: '+str(p))
    op_calls=inventory(source)
    root_calls=[item for item in op_calls if item['path']=='CMakeLists.txt' or item['path'].startswith('cmake/')]
    fetched={item['name'].lower() for item in root_calls if item['command'].lower()=='fetchcontent_declare'}
    if fetched!={'cann-cmake','opbase','ops-tensor'}:
        raise RuntimeError('Unreviewed ops-transformer FetchContent declaration: '+str(sorted(fetched)))
    called={item['name'].lower() for item in root_calls if item['command'].lower()=='add_cann_third_party'}
    if not called.issubset({'json','gtest','eigen','abseil-cpp','protobuf','makeself-fetch'}):
        raise RuntimeError('Unreviewed third-party call: '+str(sorted(called)))
    expected=json.loads((here/'expected_dependencies.json').read_text())
    pins={item['path']:item for item in expected['files'] if item['kind']=='source_dependency'}
    assets=[]
    for actual,pinned in [
        ('json/single_include/nlohmann/json.hpp','third_party/json/single_include/nlohmann/json.hpp'),
        ('protobuf/protobuf-25.1.tar.gz','third_party/protobuf/protobuf-25.1.tar.gz'),
        ('abseil-cpp/abseil-cpp-20230802.1.tar.gz','third_party/pkg/abseil-cpp-20230802.1.tar.gz')]:
        p=deps/actual;pin=pins[pinned]
        if not p.is_file() or p.stat().st_size!=pin['bytes'] or sha(p)!=pin['sha256']:
            raise RuntimeError('Offline archive/header pin failed: '+str(p))
        assets.append({'path':actual,'bytes':p.stat().st_size,'sha256':sha(p)})
    for p in (deps/'eigen/Eigen/Core',source/'third_party/makeself/makeself.sh',source/'third_party/makeself/makeself-header.sh'):
        if not p.is_file() or p.is_symlink():
            raise RuntimeError('Required offline source or packager missing: '+str(p))
        assets.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
    sibling=source.parent/'ops-tensor'
    if not sibling.is_symlink() or sibling.resolve()!=(deps/'ops-tensor').resolve():
        raise RuntimeError('Native ops-tensor first-branch sibling is not generation-private')
    if (source.parent.parent/'ops-base').exists():
        raise RuntimeError('Unexpected opbase ancestor would override the reviewed /deps branch')
    result={'schema':'fault33.pa.offline_dependencies.v1','status':'PASS_ENABLED_OFFLINE_INPUT_PREFLIGHT',
        'review_contract_sha256':sha(here/'offline_cmake_contract.json'),'reviewed_interfaces':contract['reviewed_files'],
        'ops_declaration_inventory':op_calls,'cann_cmake_declaration_inventory':inventory(deps/'cann-cmake'),
        'enabled_branches':contract['enabled_branches'],'disabled_branches':contract['disabled_branches'],
        'assets':assets,'network_enabled':False,'npu_executed':False}
    (evidence/'offline_dependencies.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
