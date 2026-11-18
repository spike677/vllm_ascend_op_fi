"""Independent CPU raw-event/ELF contracts; these bytes are not NPU evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import test_attention33_first_boundary_review as raw_fixtures
import test_arch22_normal_bridge_review as elf_fixtures
from demoncat.runtime.vllm_ascend.worker_extension import attention_stock_launch as reader
from experiments.c2c3_closure_20261117 import prepare_attention28_stock_census as prepare


class StockLaunchContracts(unittest.TestCase):
    def setUp(self):
        fixture=raw_fixtures.RuntimeResolutionTests('test_missing_completion_sync_refused')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.run,self.rows,self.token=fixture.run,fixture.rows,fixture.token
        self.identity=dict(pid=12,python_thread_ident=281463120013952,native_tid=34,
            source='CURRENT_THREAD_GET_IDENT_AND_GET_NATIVE_ID')
        binary=self.run/'runtime/registered-12-1.bin';binary.write_bytes(elf_fixtures.elf_fixture())
        self.rows[0]['binary_bytes']=binary.stat().st_size

    def read(self):
        (self.run/'runtime/runtime.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        return reader.observe_scope(self.run,self.token,self.identity)

    def test_actual_current_scope_resolves_repeated_bind_history_and_rehashes_elf(self):
        value=self.read();launch,=value['launches']
        self.assertEqual(value['status'],'OBSERVED_INTERCEPTED_STOCK_LAUNCHES')
        self.assertEqual(launch['function_binding']['repeated_identical_history'],28)
        self.assertEqual(launch['registered_binary']['sha256'],reader.sha(self.run/'runtime/registered-12-1.bin'))
        self.assertEqual(len(launch['registered_entry_symbols']['actual_selected_entry_symbols']),2)
        self.assertEqual(value['kernel_argument_abi'],'NOT_DECODED')
        self.assertFalse(value['native_fault_qualified'] or value['fault_authorized'] or value['all_runtime_entrypoints_covered'])

    def test_missing_end_foreign_thread_failed_sync_or_overflow_cannot_be_observed(self):
        original=deepcopy(self.rows)
        cases=(lambda:self.rows.pop(),lambda:self.rows[-1].__setitem__('tid',99),
            lambda:self.rows[-2].__setitem__('result',7),lambda:self.rows[-2].__setitem__('event','capture_overflow'))
        for edit in cases:
            self.rows=deepcopy(original);edit()
            with self.subTest(edit=edit),self.assertRaises(ValueError):self.read()

    def test_cross_handle_key_or_generation_history_is_not_filtered_away(self):
        for key,value in (('handle',81),('generation',2),('tiling_key','5100000000010200103')):
            old=self.rows[2][key];self.rows[2][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'CONFLICTING_FUNCTION'):
                self.read()
            self.rows[2][key]=old

    def test_python_ident_cannot_substitute_for_actual_native_tid(self):
        self.identity['native_tid']=self.identity['python_thread_ident']
        with self.assertRaisesRegex(ValueError,'ACTUAL_SCOPE_NATIVE_THREAD'):self.read()

    def test_unknown_or_missing_stock_symbol_bytes_stay_unobserved(self):
        path=self.run/'runtime/registered-12-1.bin';data=path.read_bytes()
        data=data.replace(b'103000000010300003',b'103000000030000003');path.write_bytes(data)
        with self.assertRaisesRegex(ValueError,'ENTRY_SYMBOLS_UNOBSERVED'):self.read()
        path.unlink()
        with self.assertRaises(FileNotFoundError):self.read()

    def test_actual_entry_may_end_in_key_without_private_mix_suffix(self):
        path=self.run/'runtime/registered-12-1.bin';data=path.read_bytes()
        for suffix in (b'_mix_aic',b'_mix_aiv'):data=data.replace(suffix,b'\0'+b'x'*(len(suffix)-1))
        path.write_bytes(data)
        entries=self.read()['launches'][0]['registered_entry_symbols']['actual_selected_entry_symbols']
        self.assertTrue(all(row['name'].endswith('103000000010300003') for row in entries))

    def test_stdlib_reader_import_cannot_load_a_runtime(self):
        code="""import sys
def guard(event,args):
 if event == 'ctypes.dlopen' or event == 'import' and args[0].split('.')[0] in ('torch','torch_npu','vllm'):
  raise RuntimeError('NO_RUNTIME')
sys.addaudithook(guard)
from demoncat.runtime.vllm_ascend.worker_extension import attention_stock_launch
assert not any(name in sys.modules for name in ('torch','torch_npu','vllm'))
print('CPU_STDLIB_ONLY')
"""
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,check=False)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('CPU_STDLIB_ONLY',result.stdout)


class StockCensusPreparationContracts(unittest.TestCase):
    def test_probe_bootstrap_import_precedes_any_device_framework(self):
        result=subprocess.run([sys.executable,'-B','-c',
            "import natural_attention28_stock_census_probe,sys; assert not any(x in sys.modules for x in ('torch','torch_npu','vllm')); print('NO_FRAMEWORK')"],
            capture_output=True,text=True,check=False)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('NO_FRAMEWORK',result.stdout)

    def test_one_card_eight_cpu_case_only_observes_and_preserves_original_observer_alias(self):
        args=SimpleNamespace(source=PurePosixPath('/source'),installed=PurePosixPath('/installed-host'),model=PurePosixPath('/model-host'),card=5,timeout=1800)
        source=dict(manifest={'base_commit':'fixture'},manifest_sha256='1'*64,pins={'source':'2'*64})
        observer=dict(folder=PurePosixPath('/observer/attention-preflight'),alias='/output/cann851/attention-preflight',
            sha256='3'*64,receipt={'observer_library':{'sha256':'4'*64}})
        cases=prepare.make_case(args,source,observer,{'installed':'5'*64})
        self.assertEqual(len(cases),1);case=cases[0]
        self.assertEqual(case['review']['resource_budget']['cpus'],8)
        self.assertEqual(case['review']['physical_cards'],[5])
        self.assertIn('soc',case['review'])
        self.assertFalse(case['review']['fault_authorized'] or case['review']['forced_private_vendor'])
        self.assertNotIn('ASCEND_CUSTOM_OPP_PATH',case['env'])
        self.assertTrue(all(row[2] == 'ro' for row in case['mounts']))
        self.assertIn(observer['alias'],[row[1] for row in case['mounts']])
        self.assertNotIn('--phase fault',case['command'])
        args.timeout=1801
        with self.assertRaisesRegex(ValueError,'RESOURCE_BUDGET'):prepare.make_case(args,source,observer,{})


if __name__ == '__main__':unittest.main()
