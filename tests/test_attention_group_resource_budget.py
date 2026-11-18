"""CPU resource-policy tests. A capacity object never authorizes a fault."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import threading
import unittest
from unittest.mock import patch

from fault_injection.native_adapters.attention28 import group_resource_budget as R,runtime_log_index as I
from test_attention_launch_observer import fixture
from test_attention_runtime_log_index import publish
from demoncat.runtime.vllm_ascend.worker_extension.paged33_group_evidence import GroupNormalEvidence,_MINT


def observations(*,increment=3*1024**2,seconds=10.):
    initial=1024**2
    return dict(phases=[dict(runtime_start_bytes=initial+i*increment,runtime_end_bytes=initial+(i+1)*increment,
        elapsed_seconds=seconds,indexed_history_seconds=.01,artifact_bytes=128*1024**2,capture_json_bytes=24*1024**2) for i in range(5)],
        runtime_final_bytes=initial+5*increment+1024,registered_binary_bytes=20*1024**2,normal_result_bytes=20000,
        initialization_seconds=90.,ordinary_health_seconds=10.)


def scope(attempt,*,simulation=False,variant=R.VARIANT):
    # Explicit CPU construction seam for a resource-only identity, never a
    # production loader, Native package, actual normal result, or F permit.
    normal=GroupNormalEvidence(_MINT,dict(variant=R.VARIANT,targets={str(i):{} for i in range(28)},
        source_manifest_sha256='a'*64,completion_sha256='b'*64),simulation=simulation)
    if variant!=R.VARIANT:
        class Foreign:
            def data(self):return dict(variant=variant)
        normal=Foreign()
    return R.GroupRuntimeLogScope.from_verified_normal(normal,build_sha256='b'*64,attempt=attempt)


class GroupResourceContracts(unittest.TestCase):
    def test_fixed_new_scope_512_cap_and_conservative_complete_disk_memory_time_formula(self):
        inputs=observations();value=R.plan(inputs,variant=R.VARIANT,source_sha256='a'*64,build_sha256='b'*64)
        self.assertEqual(value['maximum_runtime_bytes'],512*1024**2);self.assertEqual(value['growth_factor'],2)
        self.assertEqual(value['projected_runtime_bytes'],1024**2+3*1024**2*2*63)
        self.assertGreater(value['estimated_disk_bytes'],128*1024**2*2*63)
        self.assertGreater(value['estimated_additional_memory_bytes'],value['projected_runtime_bytes']*8)
        self.assertLess(value['estimated_seconds'],1800);self.assertGreater(value['cumulative_prefix_factors'],60)
        self.assertFalse(value['qualification_granted'] or value['budget_extension'] or value['uncertain_fault_retry'])
        inputs['phases'][0]['artifact_bytes']=1;self.assertNotEqual(value['inputs'],inputs)

    def test_wrong_variant_incomplete_or_nonfinite_measurements_and_log_overflow_refuse_before_f(self):
        for case in ('variant','phase','time','log','nonfinite','tail','cap'):
            with self.subTest(case=case):
                inputs=observations();variant=R.VARIANT;seconds=1800
                if case=='variant':variant='tnd_page16_arch22_all_normal'
                if case=='phase':inputs['phases'].pop()
                if case=='time':inputs=observations(seconds=30.)
                if case=='log':inputs=observations(increment=5*1024**2)
                if case=='nonfinite':inputs['phases'][0]['elapsed_seconds']=float('nan')
                if case=='tail':inputs['runtime_final_bytes']=1
                if case=='cap':seconds=1801
                with self.assertRaises(RuntimeError):R.plan(inputs,variant=variant,source_sha256='a'*64,build_sha256='b'*64,case_seconds=seconds)

    def test_log_exact_boundary_and_one_byte_over_are_distinct(self):
        inputs=observations(increment=4*1024**2)
        # Both non-window growth and three recovery requests are included.
        rows=inputs['phases'];initial=R.GROUP_LOG_BYTES-4*1024**2*2*63
        for index,row in enumerate(rows):row.update(runtime_start_bytes=initial+index*4*1024**2,runtime_end_bytes=initial+(index+1)*4*1024**2)
        inputs['runtime_final_bytes']=rows[-1]['runtime_end_bytes']
        value=R.plan(inputs,variant=R.VARIANT,source_sha256='a'*64,build_sha256='b'*64)
        self.assertEqual(value['projected_runtime_bytes'],R.GROUP_LOG_BYTES)
        for row in inputs['phases']:row['runtime_start_bytes']+=1;row['runtime_end_bytes']+=1
        inputs['runtime_final_bytes']+=1
        with self.assertRaisesRegex(RuntimeError,'NO_F_CAPACITY_REFUSED'):R.plan(inputs,variant=R.VARIANT,source_sha256='a'*64,build_sha256='b'*64)

    def test_every_f_rechecks_elapsed_growth_and_free_disk_memory_with_recovery_reserve(self):
        value=R.plan(observations(),variant=R.VARIANT,source_sha256='a'*64,build_sha256='b'*64)
        args=dict(phase_index=3,started_monotonic=100.,runtime_bytes=10*1024**2,now=220.,
            resources=dict(free_disk_bytes=64*1024**3,available_memory_bytes=16*1024**3,pid=os.getpid(),
                python_thread_ident=threading.get_ident(),native_tid=threading.get_native_id(),sampled_monotonic=220.))
        receipt=R.before_fault(value,**args);self.assertFalse(receipt['fault_authorized'])
        self.assertGreater(receipt['remaining_seconds'],57*value['constant_phase_seconds'])
        for change in ('elapsed','growth','disk','memory','off_phase'):
            bad=deepcopy(args)
            if change=='elapsed':bad['now']=1700.
            if change=='growth':bad['runtime_bytes']=R.GROUP_LOG_BYTES
            if change=='disk':bad['resources']['free_disk_bytes']=1
            if change=='memory':bad['resources']['available_memory_bytes']=1
            if change=='off_phase':bad['phase_index']=4
            with self.subTest(change=change),self.assertRaises(RuntimeError):R.before_fault(value,**bad)

    def test_larger_live_prefix_recomputes_memory_even_below_runtime_cap(self):
        inputs=observations(increment=1024**2)
        inputs['phases']=[dict(row,capture_json_bytes=1024**2) for row in inputs['phases']]
        value=R.plan(inputs,variant=R.VARIANT,source_sha256='a'*64,build_sha256='b'*64)
        self.assertEqual(value['projected_runtime_bytes'],127*1024**2)
        self.assertEqual(value['estimated_additional_memory_bytes'],1540*1024**2)
        args=dict(phase_index=57,started_monotonic=100.,runtime_bytes=450*1024**2,now=220.,
            resources=dict(free_disk_bytes=64*1024**3,available_memory_bytes=1540*1024**2,pid=os.getpid(),
                python_thread_ident=threading.get_ident(),native_tid=threading.get_native_id(),sampled_monotonic=220.))
        with self.assertRaisesRegex(RuntimeError,'LIVE_DISK_OR_MEMORY'):R.before_fault(value,**args)
        args['resources']['available_memory_bytes']=4220*1024**2
        receipt=R.before_fault(value,**args)
        self.assertEqual(receipt['projected_runtime_bytes'],462*1024**2)
        self.assertEqual(receipt['required_available_memory_bytes'],4220*1024**2)

    def test_exact_nonserializable_normal_source_build_attempt_scope_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runtime=root/'runtime';runtime.mkdir();value=scope(root)
            self.assertEqual(value.check(runtime)['maximum_runtime_bytes'],R.GROUP_LOG_BYTES)
            with self.assertRaises(TypeError):pickle.dumps(value)
            with self.assertRaisesRegex(RuntimeError,'ATTEMPT_CHANGED'):value.check(root)
            with self.assertRaises(RuntimeError):scope(root,simulation=True)
            with self.assertRaises(RuntimeError):scope(root,variant='old')
            with self.assertRaises(RuntimeError):R.GroupRuntimeLogScope.from_verified_normal(value.normal,build_sha256='c'*64,attempt=root)
            errors=[]
            def foreign():
                try:value.check(runtime)
                except RuntimeError as error:errors.append(str(error))
            thread=threading.Thread(target=foreign);thread.start();thread.join();self.assertEqual(len(errors),1)
            value._receipt['source_manifest_sha256']='d'*64
            with self.assertRaises(RuntimeError):value.check(runtime)

    def test_default_small_cap_regression_and_explicit_group_bound_do_not_change_line_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runtime=root/'runtime';runtime.mkdir();cid,_,_,_,rows=fixture(runtime);publish(runtime/'runtime.jsonl',rows)
            with patch.object(I,'MAX_BYTES',32):
                with self.assertRaisesRegex(RuntimeError,'OVERFLOW'):I.RuntimeLogIndex(runtime).scope(runtime,cid)
                index=I.RuntimeLogIndex(runtime,group_scope=scope(root));self.assertEqual(index.maximum_bytes,R.GROUP_LOG_BYTES)
                self.assertEqual(len(index.scope(runtime,cid)['rows']),len(rows))
                self.assertEqual(Path(index.report()['group_resource_scope']['attempt_root']),root)
                with patch.object(I,'MAX_LINE',32),self.assertRaisesRegex(RuntimeError,'LINE_BOUND'):
                    I.RuntimeLogIndex(runtime,group_scope=scope(root)).scope(runtime,cid)
            with self.assertRaisesRegex(RuntimeError,'EXACT_GROUP_CAPACITY_SCOPE_REQUIRED'):I.RuntimeLogIndex(runtime,group_scope={'maximum_runtime_bytes':R.GROUP_LOG_BYTES})

    def test_host_available_memory_is_bounded_by_actual_cgroup_and_missing_counter_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            values={'/proc/meminfo':'MemAvailable: 67108864 kB\n',
                '/sys/fs/cgroup/memory.max':str(4*1024**3),'/sys/fs/cgroup/memory.current':str(3*1024**3)}
            def name(path):return str(path).replace('\\','/')
            with patch.object(Path,'is_file',lambda p:name(p) in values),patch.object(Path,'read_text',lambda p:values[name(p)]):
                sampled=R.host_resources(tmp)
                self.assertEqual(sampled['available_memory_bytes'],1024**3)
                self.assertEqual(sampled['cgroup_memory'][0]['current_bytes'],3*1024**3)
                del values['/sys/fs/cgroup/memory.current']
                with self.assertRaisesRegex(RuntimeError,'INCOMPLETE_CGROUP'):R.host_resources(tmp)


if __name__=='__main__':unittest.main()
