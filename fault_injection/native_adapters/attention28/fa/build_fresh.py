"""Fresh CANN 9.1 FA host, public POD decoder and matching kernel; CPU only.

The imported selector source is reusable. No historical certificate, vendor,
decoder binary, kernel object, lib_cache or validation result is a build input.
The shared FIA builder supplies only dependency acquisition and host packaging.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
NATIVE = REPO / 'native_execution_strategy/native'
FIA = HERE.parent / 'fia'


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_file('dcat_fa_shared_build', FIA / 'build_fresh.py')
registration = load_file('dcat_fa_registration', HERE / 'binary_registration.py')
require, sha, save = common.require, common.sha, common.save
COMMIT = common.COMMIT
BINARY = registration.BINARY
KEY = registration.KEY
HOST = 'attention/flash_attention_score/op_host/'
PATCHED = {
    HOST + 'flash_attention_score_tiling.cpp': '84968d70057472b43494686daed258f34d32a81db6894775eb7193633c556812',
    HOST + 'arch22/flash_attention_score_tiling_general.cpp': '7feed9373f6aef6d57972c6e212641058af33f600cd987d8be9c3c4efe330854',
    HOST + 'fa_bridge.h': 'a3e7e72ce19c970f34348c79ee7767533301b4b47bb5a5a74d8a03782239f807',
    HOST + 'fa_plan.h': '48c006dfc5ca6088c4a9f06d8e9be4d12f087354e7cd25b1c28b9c5c6362c5dc',
}
IMPORTED = ('apply_fa_bridge.py', 'fa_bridge.h', 'fa_plan.h', 'fa_pod_decoder.cpp',
            'test_fa_plan.cpp', 'test_fa_bridge_scope.cpp', 'cpu_fixture_plan.json')


def validate_receipt(receipt):
    require(receipt.get('schema') == 'nes.fa.policy_source.v1'
            and receipt.get('source_commit') == COMMIT
            and receipt.get('status') == 'PATCHED_NOT_BUILT_NOT_NPU_VALIDATED'
            and receipt.get('kernel_source_modified') is False, 'EXACT_FA_SOURCE_RECEIPT_REQUIRED')
    rows = receipt.get('files', [])
    require(len(rows) == len(PATCHED)
            and {r['path']: r['after_sha256'] for r in rows} == PATCHED,
            'FA_REVIEWED_PATCH_BYTES_OR_FILE_SET_DIFFER')


def verify_imported():
    imported = json.loads((HERE.parent / 'source_import.json').read_bytes())
    rows = {r['destination']: r for r in imported['files']}
    result = []
    for name in IMPORTED:
        relative = 'native_execution_strategy/native/' + name
        row = rows[relative]
        require(row['source_commit'] == '6dad85c480255829b0a4e74657a7e2ff2ce390b3'
                and sha(REPO / relative) == row['sha256'], 'FROZEN_FA_SOURCE_IMPORT_CHANGED:' + name)
        result.append(dict(path=relative, sha256=row['sha256'], bytes=(REPO / relative).stat().st_size))
    return result


class Builder(common.Builder):
    operator = 'flash_attention_score'
    include_components = ('common', 'flash_attention_score')

    def __init__(self, args):
        super().__init__(args)
        self.report['schema'] = 'demoncat.fa.fresh_build.v1'
        self.report['mechanism'] = '28.fa.legal_tiling'
        self.report['historical_certificate_inherited'] = False

    def configure_policy(self, source_in):
        validate_receipt(self.receipt)
        require(not getattr(self.args, 'sdk_interface_manifest', None)
                and not getattr(self.args, 'opdef_compat', None), 'FA_EXACT_PINNED_CANN91_REQUIRED')
        self.report['frozen_imported_sources'] = verify_imported()
        self.variant, self.vendor_name = 'fa_s1s2_m_nratio', 'dcat_f28_fa'

    def cpu_checks(self):
        includes = ['-I', self.deps / 'json/single_include', '-I', NATIVE,
                    '-I', self.sdk / 'include']
        self.run('fa_bridge_syntax', ['g++', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0',
                 *includes, '-x', 'c++', '-fsyntax-only', self.source / (HOST + 'fa_bridge.h')], env=self.env)
        for name, extra in [('test_fa_plan', []), ('test_fa_bridge_scope',
                ['-L', self.sdk / 'lib64', '-lregister'])]:
            self.run(name + '_compile', ['g++', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0',
                *includes, NATIVE / (name + '.cpp'), *extra, '-o', self.ev / name], env=self.env)
            argv = [self.ev / name]
            if name == 'test_fa_bridge_scope':argv.append(NATIVE / 'cpu_fixture_plan.json')
            self.run(name, argv, env=self.env)
        decoder = self.ev / 'fa_pod_decoder'
        self.run('fa_public_pod_compile', ['g++', '-std=c++17', '-MD', '-MF', self.ev / 'decoder.d',
            '-I', self.source / 'attention/flash_attention_score/op_kernel/arch22',
            '-I', self.sdk / 'aarch64-linux/ascendc/include/highlevel_api', '-I', self.sdk / 'include',
            NATIVE / 'fa_pod_decoder.cpp', '-o', decoder], env=self.env)
        self.run('fa_public_schema', [decoder, '--schema'], env=self.env)
        self.run('fa_public_roundtrip', [decoder, '--self-test'], env=self.env)
        schema = json.loads((self.ev / 'fa_public_schema.log').read_bytes())
        test = json.loads((self.ev / 'fa_public_roundtrip.log').read_bytes())
        require(schema.get('schema') == 'nes.fa.public_pod_layout.v1'
                and schema.get('type') == 'FlashAttentionScoreGeneralTilingData'
                and schema.get('size') == 1464 and len(schema.get('fields', [])) == 205,
                'FRESH_FA_PUBLIC_POD_LAYOUT_DIFFERS')
        require(test == dict(status='PASS_PUBLIC_POD_CPU_ROUNDTRIP', size=1464, fields=205,
                            array_elements=65, stock_abi_validated=False), 'FA_CPU_ROUNDTRIP_FAILED')
        save(self.ev / 'fa_public_schema.json', schema)
        dependencies = (self.ev / 'decoder.d').read_text().replace('\\\n', ' ').split(':', 1)[1].split()
        headers = []
        for name in sorted(set(dependencies)):
            path = Path(name).resolve(strict=True)
            headers.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
        save(self.ev / 'fa_decoder_headers.json', headers)
        self.report.update(public_pod_schema_sha256=sha(self.ev / 'fa_public_schema.json'),
            public_pod_decoder_sha256=sha(decoder), decoder_headers_sha256=sha(self.ev / 'fa_decoder_headers.json'),
            public_pod_cpu_roundtrip=test, actual_serialized_pod_verified=False)

    def kernel(self):
        params = list((self.source / 'build').rglob(BINARY + '_param.json'))
        require(len(params) == 1, 'UNIQUE_NEW_FA_GENERATED_KERNEL_INPUT_REQUIRED')
        payload = json.loads(params[0].read_bytes())
        require(payload.get('op_type') == 'FlashAttentionScore' and len(payload.get('op_list', [])) == 1
                and payload['op_list'][0]['bin_filename'] == BINARY
                and [x['dtype'] for x in payload['op_list'][0]['inputs'][:3]] == ['float16'] * 3,
                'NEW_FA_GENERATED_INPUT_IDENTITY_DIFFERS')
        root = self.root / 'kernel';root.mkdir()
        parameter = root / 'input_param.json';shutil.copyfile(params[0], parameter)
        output = root / 'output';output.mkdir()
        dynamic = self.vendor / ('op_impl/ai_core/tbe/' + self.vendor_name + '_transformer_impl/dynamic/flash_attention_score.py')
        opc = self.sdk / 'python/site-packages/opc_tool/opc.py'
        require(dynamic.is_file() and opc.is_file(), 'FRESH_FA_JIT_SOURCE_OR_MATCHED_OPC_MISSING')
        env = dict(self.env, ASCEND_CUSTOM_OPP_PATH=str(self.vendor), OMP_NUM_THREADS='2')
        self.run('new_matching_kernel', [self.args.python, opc, dynamic,
            '--main_func=flash_attention_score', '--input_param=' + str(parameter),
            '--soc_version=Ascend910B4', '--output=' + str(output), '--impl_mode=high_performance,optional',
            '--simplified_key_mode=0', '--op_mode=dynamic', '--tiling_key=' + str(KEY),
            '--op_debug_config=dump_cce'], env=env, timeout=1800)
        matches = list(output.rglob(BINARY + '.json'))
        require(len(matches) == 1, 'UNIQUE_FRESH_FA_KERNEL_METADATA_REQUIRED')
        metadata = matches[0];binary = metadata.with_suffix('.o')
        compiled = json.loads(metadata.read_bytes())
        registration.validate_metadata(compiled, binary)
        sdk_matcher = self.sdk / 'opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b/ops_transformer/flash_attention_score.json'
        rows = [r for r in json.loads(sdk_matcher.read_bytes())['binList']
                if Path(r['binInfo']['jsonFilePath']).stem == BINARY]
        require(len(rows) == 1 and len(rows[0]['outputs']) == 4
                and rows[0]['outputs'][3]['name'] == 'attention_out'
                and rows[0]['outputs'][3]['dtype'] == 'float16', 'MATCHED_FA_FP16_SDK_MATCHER_REQUIRED')
        target_root = self.vendor / 'op_impl/ai_core/tbe/kernel'
        target = target_root / 'ascend910b/flash_attention_score';target.mkdir(parents=True, exist_ok=True)
        for path in (metadata, binary):shutil.copyfile(path, target / path.name)
        rows[0]['binInfo']['jsonFilePath'] = 'ascend910b/flash_attention_score/' + metadata.name
        registry = target_root / 'config/ascend910b/flash_attention_score.json'
        registry.parent.mkdir(parents=True, exist_ok=True);save(registry, dict(binList=rows))
        receipt = registration.register_index(self.source, self.vendor)
        save(self.ev / 'binary_summary_registration.json', receipt)
        self.report.update(compiled_keys=[KEY], kernel_binary_sha256=sha(binary), kernel_metadata_sha256=sha(metadata),
            generated_input_sha256=sha(parameter), sdk_matcher_sha256=sha(sdk_matcher),
            binary_summary_registration_sha256=sha(self.ev / 'binary_summary_registration.json'),
            actual_compiled_core_type=compiled.get('coreType'), actual_compiled_task_ration=compiled.get('taskRation'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'policy-receipt', 'dependencies', 'work'):p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sdk', type=Path, default=Path('/usr/local/Ascend/cann-9.1.0'))
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--jobs', type=int, choices=range(1, 5), default=4)
    args = p.parse_args();builder = Builder(args)
    try:
        builder.prepare();builder.host();builder.kernel();builder.finish()
    except BaseException as error:
        builder.report.update(status='CPU_BUILD_FAILED', error=repr(error), traceback=traceback.format_exc(), finished_unix=time.time())
        save(builder.ev / 'completion.json', builder.report)
        print(json.dumps(dict(status=builder.report['status'], error=repr(error))))
        return 2
    print(json.dumps({k: builder.report[k] for k in ('status', 'vendor_root', 'variant')}))
    return 0


if __name__ == '__main__':raise SystemExit(main())
