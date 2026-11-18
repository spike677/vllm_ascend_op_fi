"""Pure-file/prestart CPU contracts; no SDK load, Worker or NPU authority."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from experiments.worker_core_v11_20261118 import natural_attention28_input_journal_probe as probe
from experiments.worker_core_v11_20261118 import prepare_natural_attention28_input_journal as P
from fault_injection import fia_input_journal as J


class PreparationContracts(unittest.TestCase):
    def fixture(self):
        args=NS(source=PurePosixPath('/frozen-source'),installed=PurePosixPath('/installed-host'),
            model=PurePosixPath('/model-host'),card=5,timeout=1800)
        source=dict(manifest={'base_commit':'explicit-cpu-source'},manifest_sha256='1'*64,pins={'source':'2'*64})
        observer=dict(folder=PurePosixPath('/observer-host/attention-preflight'),alias='/output/frozen/attention-preflight',
            sha256='3'*64,receipt={'observer_library':{'sha256':'4'*64,'path':'/output/frozen/attention-preflight/libdcat_attention_observer.so'}})
        journal=dict(root=PurePosixPath('/journal-host'),sha256='5'*64,
            profile=dict(profile_sha256='6'*64,sources={'native.cpp':'7'*64},
                originals=dict(observer=dict(path='/observer/libdcat_attention_observer.so',file_sha256='4'*64))))
        return args,source,observer,journal

    def test_one_case_has_exact_dependency_alias_and_fixed_budget(self):
        args,source,observer,journal=self.fixture()
        case,=P.make_case(args,source,observer,journal,{})
        self.assertEqual(case['review']['resource_budget']['expected_public_windows'],56)
        self.assertEqual(case['review']['resource_budget']['public_windows'],64)
        self.assertEqual(case['review']['resource_budget']['native_record_bytes'],16*1024*1024)
        self.assertEqual(case['review']['resource_budget']['ordinary_health_requests'],2)
        self.assertEqual(case['review']['physical_cards'],[5])
        self.assertEqual([x for x in case['mounts'] if x[1]=='/observer'],
            [['/observer-host/attention-preflight','/observer','ro']])
        self.assertNotIn('LD_PRELOAD',case['env'])
        self.assertNotIn('ASCEND_CUSTOM_OPP_PATH',case['env'])
        self.assertFalse(case['review']['fault_authorized'] or case['review']['C2'] or case['review']['E5'])
        self.assertEqual(len({x[1] for x in case['mounts']}),len(case['mounts']))
        self.assertNotIn('/msprobe_packages',[x[1] for x in case['mounts']])

    def test_foreign_observer_sha_system_alias_or_duplicate_mount_refuse(self):
        for change in (lambda j:j['profile']['originals']['observer'].update(file_sha256='9'*64),
                lambda j:j['profile']['originals']['observer'].update(path='/usr/local/libdcat_attention_observer.so')):
            a,s,o,j=self.fixture();change(j)
            with self.assertRaises(ValueError):P.make_case(a,s,o,j,{})
        with self.assertRaisesRegex(ValueError,'CONFLICTING_OBSERVER'):
            P.add_observer_alias([['/other','/observer','ro']],'/actual','/observer')

    def test_no_timeout_or_card_scope_extension(self):
        for key,value in (('timeout',1801),('card',0),('timeout',0)):
            a,s,o,j=self.fixture();setattr(a,key,value)
            with self.assertRaises(ValueError):P.make_case(a,s,o,j,{})

    def test_import_and_help_remain_preframework_without_library_load(self):
        code="""import sys, ctypes
# Windows ctypes performs its OS kernel32 bootstrap at import. Complete that
# standard-library bootstrap first; the task imports may not load any DSO.
def guard(event,args):
 if event=='ctypes.dlopen' or event=='import' and args[0].split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):
  raise RuntimeError('NO_RUNTIME_ALLOWED')
sys.addaudithook(guard)
from experiments.worker_core_v11_20261118 import natural_attention28_input_journal_probe
from experiments.worker_core_v11_20261118 import prepare_natural_attention28_input_journal
print('PREFRAMEWORK_CPU_ONLY')
"""
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('PREFRAMEWORK_CPU_ONLY',result.stdout)

    def test_validator_rehashes_all_artifacts_and_requires_all_26_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);files={}
            def save(name,value):
                raw=json.dumps(value,sort_keys=True).encode() if not isinstance(value,bytes) else value
                (root/name).write_bytes(raw);files[name]=hashlib.sha256(raw).hexdigest()
            save('not-a-real-library.so',b'CPU fixture; never loaded')
            cases=[]
            for i in range(26):
                case='cpu'+str(i);log='fia_input_journal_'+case+'.log';save(log,b'explicit synthetic CPU log')
                cases.append(dict(case=case,status='PASS',returncode=0,log_sha256=files[log]))
            profile=dict(candidate='fia_input_journal',image=P.E.IMAGE,sources={'native.cpp':'7'*64},
                sdk_headers={'include/aclnnop/aclnn_fused_infer_attention_score_v3.h':
                    'bf73aea4146a9b12f91eda466927c23c5d43e56f78aa5039bd9644732f521550'})
            profile['profile_sha256']=P.E.digest(profile);save('profile.json',profile)
            receipt=dict(status='SDK_TYPED_CPU_CONTRACTS_COMPLETE_NOT_NPU',actual_npu_calls=0,qualification_granted=False,
                cases=cases,source_sha256=profile['sources']);save('receipt.json',receipt)
            build=dict(schema='demoncat.quant.container_deployment.v1',candidate='fia_input_journal',actual_npu_calls=0,
                qualified=False,runtime_activation_default=False,public_provider_registered=False,
                status='SDK_TYPED_CPU_CONTRACTS_COMPLETE_NOT_NPU',files=dict(files),
                profile_sha256=profile['profile_sha256'],library='not-a-real-library.so')
            save('build.json',build);pin=files['build.json']
            self.assertEqual(P.validate_journal(root,pin)['sha256'],pin)
            (root/'fia_input_journal_cpu0.log').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'JOURNAL_BUILD_FILE_CHANGED'):P.validate_journal(root,pin)


class PrestartContracts(unittest.TestCase):
    def test_failed_preflight_never_imports_or_constructs_llm(self):
        with tempfile.TemporaryDirectory() as folder:
            args=NS(out=Path(folder),journal_build_sha256='1'*64,observer_receipt_sha256='2'*64)
            with (patch.object(probe,'prepare_startup',side_effect=RuntimeError('actual resolver refused')),
                    patch.dict(sys.modules,{'vllm':NS(LLM=lambda **kwargs:(_ for _ in ()).throw(AssertionError('must not construct LLM')))})):
                self.assertEqual(probe.run(args,{},{}),2)
            report=json.loads((args.out/'result.json').read_bytes())
            self.assertIn('actual resolver refused',report['error']);self.assertNotIn('worker_binding',report)
            self.assertFalse(report['C1'] or report['C2'] or report['C3'] or report['E5'])

    def test_canonical_held_object_cannot_be_reconstructed_or_foreign_process(self):
        source=dict(path=str(Path(probe.__file__).resolve()),sha256=probe.common.sha(probe.__file__))
        identity=dict(pid=123,native_tid=456,process_start_identity='CPU explicit fixture')
        journal=J.NativeFiaInputJournal.__new__(J.NativeFiaInputJournal);journal.startup=dict(identity=identity)
        with patch.object(probe,'_SOURCE',source),patch.object(probe,'_HELD',journal),patch.object(J,'process_identity',return_value=identity):
            self.assertIs(probe.held_source(source)[1],journal)
            with patch.object(J,'process_identity',return_value={**identity,'pid':124}):
                with self.assertRaisesRegex(RuntimeError,'SAME_PRE_LLM_PROCESS'):probe.held_source(source)
            with patch.object(probe,'_HELD',None):
                with self.assertRaisesRegex(RuntimeError,'SAME_PRE_LLM_PROCESS'):probe.held_source(source)
            with self.assertRaisesRegex(RuntimeError,'EXACT_HELD_PROBE_SOURCE'):probe.held_source({**source,'sha256':'0'*64})

    def test_immutable_failure_original_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);first=probe.immutable(path,'failure.json',{'error':'first'})
            with self.assertRaises(FileExistsError):probe.immutable(path,'failure.json',{'error':'second'})
            self.assertEqual(probe.common.sha(path/'failure.json'),first['sha256'])


if __name__=='__main__':unittest.main()
