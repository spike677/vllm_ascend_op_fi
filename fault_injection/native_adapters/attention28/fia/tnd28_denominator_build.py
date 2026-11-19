"""Exact new variant hooks for the existing fresh FIA builder (CPU/SDK only)."""
import hashlib
import importlib.util
import json
from pathlib import Path


def recipe(here):
    spec=importlib.util.spec_from_file_location('tnd28_replay_recipe',here/'apply_tnd28_denominator.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def configure(builder,source_in,base):
    r=recipe(base.HERE);require=base.require;sha=base.sha
    require(builder.sdk.name=='cann-8.5.1'
        and getattr(builder.args,'opdef_compat',None)=='ascend910b-cann851'
        and getattr(builder.args,'sdk_interface_manifest',None)
        and getattr(builder.args,'sdk_interface_sha256',None)==base.TND28_SDK_INTERFACE_SHA256
        and sha(builder.args.sdk_interface_manifest)==base.TND28_SDK_INTERFACE_SHA256,
        'TND28_REPLAY_EXACT_CANN851_INTERFACE_REQUIRED')
    raw={p:base.git_blob(source_in,p) for p in (*r.SOURCE_SHA,*r.EXTRA_UNCHANGED)}
    final,_,projection,fields=r.materials(raw)
    require(builder.receipt==r.receipt(raw,final,projection,fields),'TND28_REPLAY_RECONSTRUCTED_RECEIPT_DIFFERS')
    for name,value in final.items():
        p=source_in/name
        require(not p.is_symlink() and p.read_bytes()==value,'TND28_REPLAY_RECONSTRUCTED_BYTES_DIFFER:'+name)
    builder.variant=r.VARIANT;builder.vendor_name=r.VENDOR
    builder.report.update(variant=r.VARIANT,normal_only=False,**{name:False for name in base.TND28_DENIED})
    builder.report.update(tnd28_source_closure_reconstructed=True,
        tnd28_source_closure={p:hashlib.sha256(b).hexdigest() for p,b in final.items()},
        tnd28_projection_sha256=hashlib.sha256(projection.encode()).hexdigest(),
        tnd28_typed_fields=fields,tnd28_typed_field_count=len(fields),
        tnd28_replay_recipe_sha256=sha(base.HERE/'apply_tnd28_denominator.py'),
        tnd28_replay_build_helper_sha256=sha(Path(__file__)),
        tnd28_sdk_interface_sha256=base.TND28_SDK_INTERFACE_SHA256,
        tnd28_builder_sha256=sha(Path(base.__file__)),
        tnd28_binary_registration_sha256=sha(base.HERE/'binary_registration.py'))


def cpu_checks(builder,base):
    ev=builder.ev;here=base.HERE
    binary=ev/'tnd28_replay_contract';program=here/'test_tnd28_denominator_contract.cpp'
    builder.run('tnd28_replay_contract_compile',['g++','-std=c++17','-Wall','-Wextra','-Werror','-pedantic',
        '-pthread',program,'-o',binary],env=builder.env)
    builder.run('tnd28_replay_contract',[binary],env=builder.env)
    base.require((ev/'tnd28_replay_contract.log').read_text().strip()=='PASS_CPU_TND28_DENOMINATOR_CONTRACT_NOT_NPU',
                 'TND28_REPLAY_CPU_CONTRACT_REQUIRED')
    missing=ev/'tnd28_missing_input_contract'
    builder.run('tnd28_missing_input_compile',['g++','-std=c++17','-Wall','-Wextra','-Werror','-pedantic',
        '-pthread','-DDCAT_TND28_NO_INPUT_GATE',program,'-o',missing],env=builder.env)
    builder.run('tnd28_missing_input_contract',[missing],env=builder.env)
    base.require((ev/'tnd28_missing_input_contract.log').read_text().strip()=='PASS_CPU_TND28_MISSING_INPUT_SYMBOL_REFUSED',
                 'TND28_REPLAY_MISSING_INPUT_SYMBOL_MUST_REFUSE')
    builder.run('actual_tnd28_replay_split_matrix',[builder.args.python,here/'compile_tnd28_denominator_matrix.py',
        '--source',builder.source,'--out',ev/'tnd28_replay_matrix'],env=builder.env)
    matrix=json.loads((ev/'tnd28_replay_matrix/receipt.json').read_bytes())
    base.require(matrix.get('supported_core_profiles') and matrix.get('cases')==11
        and matrix.get('source_execution')=='PINNED_ORIGINAL_SPLIT_CORE_CPU'
        and matrix.get('npu_executed') is False and matrix.get('qualification_granted') is False,
        'TND28_REPLAY_NO_COMPLETE_NATIVE_SPLIT_PROFILE')
    builder.report.update(tnd28_replay_matrix_sha256=base.sha(ev/'tnd28_replay_matrix/matrix.jsonl'),
        tnd28_replay_matrix_receipt_sha256=base.sha(ev/'tnd28_replay_matrix/receipt.json'),
        tnd28_replay_contract_source_sha256=base.sha(program),tnd28_replay_contract_executable_sha256=base.sha(binary),
        tnd28_replay_contract_log_sha256=base.sha(ev/'tnd28_replay_contract.log'),
        tnd28_missing_input_contract_log_sha256=base.sha(ev/'tnd28_missing_input_contract.log'),
        tnd28_missing_input_executable_sha256=base.sha(missing),
        supported_cpu_core_profiles=matrix['supported_core_profiles'],actual_worker_platform_observed=False)


def finish_check(builder,base):
    ev=builder.ev;report=builder.report
    base.require(all(report.get(name) is False for name in base.TND28_DENIED)
        and report.get('tnd28_source_closure_reconstructed') is True
        and report.get('tnd28_host_header_compiled_with_actual_sdk') is True
        and report.get('compiled_keys')==base.TND28_REPLAY_KEYS
        and report.get('tnd28_replay_matrix_receipt_sha256')==base.sha(ev/'tnd28_replay_matrix/receipt.json')
        and report.get('tnd28_replay_matrix_sha256')==base.sha(ev/'tnd28_replay_matrix/matrix.jsonl')
        and report.get('tnd28_replay_contract_log_sha256')==base.sha(ev/'tnd28_replay_contract.log')
        and report.get('tnd28_missing_input_contract_log_sha256')==base.sha(ev/'tnd28_missing_input_contract.log')
        and report.get('tnd28_public_schema_sha256')==base.sha(ev/'tnd28_schema/fia_abi_schema.json')
        and report.get('tnd28_schema_compile_sha256')==base.sha(ev/'tnd28_schema/compile_receipt.json')
        and report.get('tnd28_schema_selftest_sha256')==base.sha(ev/'tnd28_schema/fia_abi_selftest.json')
        and report.get('tnd28_macro_dependency_sha256')==base.sha(ev/'tnd28_schema/macro_dependency.json')
        and report.get('tnd28_macro_header_sha256')==base.TND28_MACRO_HEADER_SHA256
        and base.sha(report['tnd28_macro_header_path'])==base.TND28_MACRO_HEADER_SHA256,
        'TND28_REPLAY_COMPLETE_CPU_BUILD_EVIDENCE_REQUIRED')
    report.update(qualification_missing=['new package actual loaded-function and registered ELF identity',
        'current public/typed prepare/internal Slice/kernel input binding; source113 journal had zero events',
        'fresh stock/private native-normal AND M128 control normal comparison on the same actual operands',
        'actual host/POD/key/scoped launch equality and actual core profile',
        'single actual denominator replay numerical effect; OFF and same Worker health',
        'main-project active-request source integration'],
        numerical_effect_observed=False,actual_worker_platform_observed=False)
