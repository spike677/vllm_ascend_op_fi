"""CPU-only maps/stat fixtures; failed mappings never become accepted evidence."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
import test_paged33_boundary_probe as fixtures
from demoncat.runtime.vllm_ascend.worker_extension import paged33_boundary_probe as probe
from demoncat.runtime.vllm_ascend.worker_extension.paged33_tnd_probe import TNDWorkerProbeSession


class MappedIdentityContracts(unittest.TestCase):
    def setUp(self):
        self.folder=Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.file=self.folder/'liboptiling.so';self.file.write_bytes(b'CPU mapping fixture, not a DSO')
        self.stat=self.file.stat()
        if not hasattr(os,'major'):self.enterContext(patch.object(probe.os,'major',lambda value:0,create=True))
        if not hasattr(os,'minor'):self.enterContext(patch.object(probe.os,'minor',lambda value:value,create=True))
        self.line=f'1000-2000 r-xp 00000000 {os.major(self.stat.st_dev):02x}:{os.minor(self.stat.st_dev):02x} {self.stat.st_ino} {self.file}'
        self.maps=self.line+'\n';read=Path.read_bytes
        self.enterContext(patch.object(Path,'read_bytes',lambda path: self.maps.encode() if str(path)=='\\proc\\self\\maps'
            or str(path)=='/proc/self/maps' else read(path)))

    def failed(self, code='IDENTITY_CHANGED'):
        with self.assertRaisesRegex(probe.MappedLibraryObservationError,code) as error:probe.mapped_api_libraries()
        record=error.exception.evidence
        self.assertFalse(record['complete'] or record['selected_api_owner_observed'])
        self.assertEqual(json.loads(json.dumps(record)),record)
        return record

    def test_exact_inode_device_and_bounded_size_keep_successful_hash_and_map_rows(self):
        report=probe.mapped_api_libraries()
        self.assertTrue(report['complete'])
        file,=report['files']
        self.assertEqual(file['sha256'],probe.sha(self.file))
        self.assertEqual(file['path'],str(self.file.resolve()))
        self.assertEqual(file['map_rows'][0]['raw_line'],self.line)
        self.assertEqual(file['stat_before'],file['stat_after'])
        self.assertEqual(file['failed_checks'],[])

    def test_inode_and_device_failures_record_exact_file_and_actual_numbers(self):
        parts=self.line.split(None,5)
        for index,value,check in ((4,str(self.stat.st_ino+1),'inode_matches'),(3,'ff:ff','device_matches')):
            altered=parts[:];altered[index]=value;self.maps=' '.join(altered)+'\n'
            with self.subTest(check=check):
                record=self.failed();current=record['current_file']
                self.assertEqual(current['failed_checks'],[check])
                self.assertEqual(current['path'],str(self.file.resolve()))
                self.assertEqual(current['stat_before']['inode'],self.stat.st_ino)
                self.assertEqual(current['map_rows'][0]['raw_line'],self.maps.strip())
                self.assertNotIn('sha256',current)

    def test_size_limit_is_not_misreported_as_an_inode_or_device_difference(self):
        stat=Path.stat
        larger=NS(st_dev=self.stat.st_dev,st_ino=self.stat.st_ino,st_size=512*1024**2+1,st_mtime_ns=self.stat.st_mtime_ns)
        with patch.object(Path,'stat',lambda path,*a,**kw:larger if path==self.file else stat(path,*a,**kw)):
            record=self.failed()
        self.assertEqual(record['current_file']['failed_checks'],['size_within_bound'])
        self.assertEqual(record['current_file']['stat_before']['bytes'],512*1024**2+1)

    def test_conflicting_map_rows_for_one_path_cannot_be_overwritten(self):
        parts=self.line.split(None,5);parts[4]=str(self.stat.st_ino+1)
        self.maps=' '.join(parts)+'\n'+self.line+'\n'
        record=self.failed()
        self.assertEqual(len(record['current_file']['map_rows']),2)
        self.assertEqual(record['current_file']['failed_checks'],['inode_matches'])

    def test_deleted_and_unavailable_paths_preserve_raw_evidence_before_refusal(self):
        self.maps=self.line+' (deleted)\n'
        record=self.failed('LIBRARY_DELETED')
        self.assertTrue(record['current_map_row']['raw_path'].endswith(' (deleted)'))
        self.maps=self.line+'\n';self.file.unlink()
        record=self.failed('WinError|Errno')
        self.assertEqual(record['current_map_row']['raw_path'],str(self.file))

    def test_a_post_hash_change_retains_before_and_after_stats_and_rejects(self):
        original_sha=probe.sha
        def changed(path):
            value=original_sha(path);path.write_bytes(b'changed after hash');return value
        with patch.object(probe,'sha',side_effect=changed):record=self.failed('CHANGED_DURING_HASH')
        self.assertNotEqual(record['current_file']['stat_before']['bytes'],record['current_file']['stat_after']['bytes'])
        self.assertIn('sha256',record['current_file'])

    def test_earlier_successful_file_and_later_failure_are_both_retained(self):
        other=self.folder/'libopapi.so';other.write_bytes(b'earlier CPU mapping fixture')
        before=other.stat()
        line=f'3000-4000 r-xp 00000000 {os.major(before.st_dev):02x}:{os.minor(before.st_dev):02x} {before.st_ino} {other}'
        parts=self.line.split(None,5);parts[4]=str(self.stat.st_ino+1)
        self.maps=line+'\n'+' '.join(parts)+'\n'
        record=self.failed()
        self.assertEqual(record['files'][0]['path'],str(other.resolve()))
        self.assertEqual(record['files'][0]['sha256'],probe.sha(other))
        self.assertEqual(record['current_file']['path'],str(self.file.resolve()))
        self.assertEqual(len(record['map_rows']),2)

    def test_session_saves_failure_evidence_without_rewriting_original_error_or_return(self):
        worker,module,actual,boundary=fixtures.fixture()
        capture=probe.FirstDecodeBoundaryCapture(worker,self.folder/'capture',lambda:None,allow_cpu_test=True)
        session=TNDWorkerProbeSession.__new__(TNDWorkerProbeSession)
        session.stack=[(probe.FIRST_LAYER,module)];session.rows=[boundary];session.tensor_refs={}
        session.boundary_capture=capture;session.adapter=None
        session.boundary_stock_scope=probe.StockLaunchScope(fixtures.FakeObserver(),lambda:None,self.folder)
        parts=self.line.split(None,5);parts[4]=str(self.stat.st_ino+1);self.maps=' '.join(parts)+'\n'
        returned=(actual['query']+1,torch.empty(0));calls=[]
        def original(**kwargs):calls.append(1);return returned
        with patch.object(probe,'read_scoped_launches',return_value={'source_bound_fixture':True}):
            result=session._invoke_api('npu_fused_infer_attention_score',original,(),actual,boundary)
        self.assertIs(result,returned);self.assertEqual(calls,[1])
        saved=json.loads((self.folder/'capture/boundary.json').read_text())['mapped_api_libraries']
        self.assertEqual(saved['status'],'UNOBSERVED_OR_INCOMPLETE')
        self.assertEqual(saved['error'],"ContextRefused('BOUNDARY_MAPPED_LIBRARY_IDENTITY_CHANGED')")
        self.assertEqual(saved['diagnostic']['current_file']['path'],str(self.file.resolve()))
        self.assertFalse(saved['diagnostic']['complete'])
        capture.finalize();capture.release()


if __name__ == '__main__':unittest.main()
