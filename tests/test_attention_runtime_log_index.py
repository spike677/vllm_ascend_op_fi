"""Append-cache safety and equivalence to the existing complete raw readers."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from fault_injection.native_adapters.attention28.runtime_log_index import RuntimeLogIndex
from fault_injection.native_adapters.attention28.launch import verify_launch
from demoncat.runtime.vllm_ascend.worker_extension.paged33_boundary_probe import read_scoped_launches
from test_attention_launch_observer import fixture
import test_paged33_boundary_probe as boundary_fixtures


def publish(path,rows,mode='w'):
    with path.open(mode) as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')


class RuntimeIndexContracts(unittest.TestCase):
    def test_native_full_reader_matches_indexed_raw_result_and_appends_parse_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cid,binary,schema,host,rows=fixture(root);path=root/'runtime.jsonl';publish(path,rows)
            index=RuntimeLogIndex(root)
            kwargs=dict(kernel_sha256=binary,schema=schema,host=host)
            self.assertEqual(verify_launch(root,cid,**kwargs),verify_launch(root,cid,runtime_index=index,**kwargs))
            first=index.parsed_events
            view=index.scope(root,cid);view['rows'][0]['generation']=999;view['line_numbers'].clear()
            self.assertEqual(index.parsed_events,first)
            self.assertEqual(verify_launch(root,cid,**kwargs),verify_launch(root,cid,runtime_index=index,**kwargs))
            more=[dict(r,call_id='b'*32,monotonic_ns=r['monotonic_ns']+10,**(
                dict(launch_begin_ns=r['launch_begin_ns']+10,launch_end_ns=r['launch_end_ns']+10)
                if r['event']=='launch' else {})) for r in rows[1:]]
            publish(path,more,'a');kwargs['host']=dict(host,token='b'*32)
            self.assertEqual(verify_launch(root,'b'*32,**kwargs),verify_launch(root,'b'*32,runtime_index=index,**kwargs))
            self.assertEqual(index.parsed_events,len(rows)+len(more))
            self.assertEqual(index.scope(root,'b'*32)['line_numbers'],[1,6,7,8,9])

    def test_native_refusals_remain_refusals_and_auxiliary_launches_are_not_dropped(self):
        for case in ('duplicate','owner','sync_error','handle','generation','raw','overflow'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cid,binary,schema,host,rows=fixture(root)
                if case=='duplicate':rows.insert(3,deepcopy(rows[2]))
                if case=='owner':rows[2]['tid']+=1
                if case=='sync_error':rows[3]['result']=1
                if case=='handle':rows[2]['handle']+=1
                if case=='generation':rows[2]['generation']+=1
                if case=='raw':rows[2]['args_hex']='0'*48
                if case=='overflow':rows.append(dict(rows[-1],event='capture_overflow',call_id='',monotonic_ns=8))
                publish(root/'runtime.jsonl',rows);reasons=[]
                for index in (None,RuntimeLogIndex(root)):
                    with self.assertRaises(RuntimeError) as error:
                        verify_launch(root,cid,kernel_sha256=binary,schema=schema,host=host,runtime_index=index)
                    reasons.append(str(error.exception))
                self.assertEqual(*reasons)
        case=boundary_fixtures.StockRawTraceTests();case.setUp();self.addCleanup(case.doCleanups)
        full=case.read();indexed=read_scoped_launches(case.folder,case.token,runtime_index=RuntimeLogIndex(case.folder))
        self.assertEqual(full,indexed);self.assertEqual(indexed['intercepted_launch_count'],2)

    def test_all_conflicting_function_identities_and_registration_generations_keep_full_reader_rejection(self):
        for change in ('handle','generation','key','new_register','late_bind','outside_scope'):
            with self.subTest(change=change):
                case=boundary_fixtures.StockRawTraceTests();case.setUp()
                try:
                    old=next(r for r in case.rows if r['event']=='function_bind' and not r['call_id'])
                    current=next(r for r in case.rows if r['event']=='function_bind' and r['call_id']==case.token)
                    if change=='handle':old['handle']+=100
                    if change=='generation':old['generation']+=1
                    if change=='key':old['tiling_key']='111'
                    if change=='new_register':case.rows.insert(1,dict(case.rows[0],generation=2,monotonic_ns=9))
                    if change=='late_bind':case.rows.insert(case.rows.index(current)+1,dict(current,call_id='foreign',tid=99,monotonic_ns=current['monotonic_ns']+1))
                    if change=='outside_scope':case.rows.remove(current);case.rows.append(current)
                    publish(case.folder/'runtime.jsonl',case.rows);reasons=[]
                    for index in (None,RuntimeLogIndex(case.folder)):
                        with self.assertRaises(RuntimeError) as error:read_scoped_launches(case.folder,case.token,runtime_index=index)
                        reasons.append(str(error.exception))
                    self.assertEqual(*reasons)
                finally:case.doCleanups()

    def test_prefix_tamper_truncation_same_size_replacement_and_partial_append_refuse_without_cache_commit(self):
        for change in ('tamper','truncate','replace','half_line'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cid,_,_,_,rows=fixture(root);path=root/'runtime.jsonl';publish(path,rows)
                index=RuntimeLogIndex(root);index.scope(root,cid);saved=index.report()
                if change=='tamper':path.write_bytes(path.read_bytes().replace(b'512',b'511',1) if b'512' in path.read_bytes()
                    else path.read_bytes().replace(b'"pid": 41',b'"pid": 40',1))
                if change=='truncate':path.write_bytes(path.read_bytes()[:-1])
                if change=='replace':
                    replacement=root/'replacement';replacement.write_bytes(path.read_bytes());os.replace(replacement,path)
                if change=='half_line':
                    with path.open('ab') as stream:stream.write(b'{"schema":')
                with self.assertRaises(RuntimeError):index.scope(root,cid)
                self.assertEqual(index.size,saved['prefix_bytes']);self.assertEqual(index.parsed_events,saved['parsed_events'])

    def test_growth_during_read_foreign_thread_and_duplicate_json_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cid,_,_,_,rows=fixture(root);path=root/'runtime.jsonl';publish(path,rows)
            index=RuntimeLogIndex(root);index.scope(root,cid);original=Path.open;grown=[]
            class Growing:
                def __init__(self,stream):self.stream=stream
                def __enter__(self):return self
                def __exit__(self,*args):self.stream.close()
                def fileno(self):return self.stream.fileno()
                def read(self,size=-1):
                    data=self.stream.read(size)
                    if not grown:
                        grown.append(1)
                        with original(path,'ab') as writer:writer.write((json.dumps(dict(rows[-1],call_id='foreign'))+'\n').encode())
                    return data
            def opened(file,*args,**kwargs):
                stream=original(file,*args,**kwargs)
                return Growing(stream) if file==path and args==('rb',) else stream
            with patch.object(Path,'open',opened),self.assertRaisesRegex(RuntimeError,'CHANGED_DURING_READ'):index.scope(root,cid)
            errors=[]
            def foreign():
                try:index.scope(root,cid)
                except RuntimeError as error:errors.append(str(error))
            thread=threading.Thread(target=foreign);thread.start();thread.join();self.assertEqual(errors,['INDEX_OWNER_OR_DIRECTORY_CHANGED'])
            with path.open('ab') as stream:stream.write(b'{"schema":1,"schema":2}\n')
            with self.assertRaisesRegex(RuntimeError,'DUPLICATE_JSON_KEY'):index.scope(root,cid)

    def test_invalid_append_is_transactional_even_after_a_valid_new_event(self):
        for change in ('call_list','launch_handle','function_list','nonfinite'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cid,_,_,_,rows=fixture(root);path=root/'runtime.jsonl';publish(path,rows)
                index=RuntimeLogIndex(root);old=index.scope(root,cid);saved=path.read_bytes()
                invalid=deepcopy(rows[2])
                if change=='call_list':invalid['call_id']=[]
                if change=='launch_handle':invalid['handle']={}
                if change=='function_list':invalid['function_handle']=[99]
                if change=='nonfinite':invalid['stream']=float('nan')
                publish(path,[dict(rows[-1],call_id='unrelated'),invalid],'a')
                with self.assertRaises(RuntimeError):index.scope(root,cid)
                self.assertEqual(index.parsed_events,len(rows));self.assertEqual(index.size,len(saved))
                path.write_bytes(saved)
                self.assertEqual(index.scope(root,cid),old)

    def test_registered_binary_is_rechecked_after_cached_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cid,binary,schema,host,rows=fixture(root);rows[2]['api']='rtKernelLaunchWithHandleV2'
            publish(root/'runtime.jsonl',rows)
            index=RuntimeLogIndex(root);kwargs=dict(kernel_sha256=binary,schema=schema,host=host)
            verify_launch(root,cid,runtime_index=index,**kwargs)
            path=root/rows[0]['binary_file'];raw=path.read_bytes();path.write_bytes(b'X'+raw[1:])
            reasons=[]
            for cached in (None,index):
                with self.assertRaises(RuntimeError) as error:verify_launch(root,cid,runtime_index=cached,**kwargs)
                reasons.append(str(error.exception))
            self.assertEqual(*reasons)

    def test_owner_directory_and_exact_index_type_are_required(self):
        class Derived(RuntimeLogIndex):pass
        with tempfile.TemporaryDirectory() as tmp,tempfile.TemporaryDirectory() as other:
            root=Path(tmp);cid,binary,schema,host,rows=fixture(root);publish(root/'runtime.jsonl',rows)
            index=RuntimeLogIndex(root)
            with self.assertRaisesRegex(RuntimeError,'OWNER_OR_DIRECTORY_CHANGED'):index.scope(other,cid)
            with self.assertRaisesRegex(RuntimeError,'EXACT_OWNED_RUNTIME_INDEX_REQUIRED'):
                verify_launch(root,cid,kernel_sha256=binary,schema=schema,host=host,runtime_index=Derived(root))


if __name__=='__main__':unittest.main()
