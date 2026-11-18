"""Bounded current-request originals before the live C2 controller sees them."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from demoncat.runtime.vllm_ascend.worker_extension import paged33_group_execution as G


class GroupArtifactContracts(unittest.TestCase):
    def setUp(self):
        self.root=Path(self.enterContext(tempfile.TemporaryDirectory()))/'group_requests';self.root.mkdir()
        self.folder=self.root/'000-reference';self.folder.mkdir();(self.folder/'boundaries').mkdir()
        self.path=self.folder/'runtime_result.json'
        self.capture=dict(natural_request=dict(phase='reference',request_id='actual-request'),native_calls=[])
        self.path.write_text(json.dumps(self.capture))
        self.payload=self.folder/'boundaries'/'query.bin';self.payload.write_bytes(b'actual original bytes')
        self.session=G.GroupExecutionAdapter.__new__(G.GroupExecutionAdapter)
        self.session.root_folder=self.root;self.session.folder=self.folder
        self.session.artifact_root_identity=G._directory_identity(self.root)
        self.session.artifact_folder_identity=G._directory_identity(self.folder)
        self.session.deployment=SimpleNamespace(simulation=True)
        self.session.control=Mock();self.session._fail=Mock()
        self.session.request_artifacts=[];self.session.rows=[{}];self.session.request_rows=[{}];self.session.payload_refs={'kept':object()}
        self.base=self.enterContext(patch.object(G.AllNormalWorkerSession,'collect',return_value=self.capture))

    def inventory(self):
        return G._request_inventory(self.root,self.folder,self.session.artifact_root_identity,self.session.artifact_folder_identity)

    def changed_sizes(self,values):
        original=G._plain_stat
        def fake(path):
            value=original(path)
            if path.name not in values:return value
            return SimpleNamespace(st_mode=value.st_mode,st_dev=value.st_dev,st_ino=value.st_ino,
                st_size=values[path.name],st_mtime_ns=value.st_mtime_ns,st_ctime_ns=value.st_ctime_ns)
        return patch.object(G,'_plain_stat',side_effect=fake)

    def refused(self,reason):
        with self.assertRaisesRegex(RuntimeError,reason):self.session.collect()
        self.session._fail.assert_called_once();self.session.control.complete_request.assert_not_called()
        self.assertTrue(self.path.exists());self.assertTrue(self.session.payload_refs)

    def test_original_readback_and_bounded_inventory_precede_completion(self):
        result=self.session.collect();self.session.control.complete_request.assert_called_once_with(self.capture,result)
        self.assertEqual(result['owned_artifacts']['files_excluding_capture'],1)
        self.assertEqual(result['owned_artifacts']['bytes_excluding_capture'],len(self.payload.read_bytes()))
        self.assertEqual(result,json.loads(json.dumps(result)))
        self.assertEqual(self.path.read_text(),json.dumps(self.capture))

    def test_capture_size_refused_before_open_without_creating_large_file(self):
        with self.changed_sizes({'runtime_result.json':G.MAX_CAPTURE_BYTES+1}),patch.object(G,'_read_capture') as read:
            self.refused('GROUP_CAPTURE_FILE_BOUND');read.assert_not_called()

    def test_zero_capture_refused_before_open(self):
        self.path.write_bytes(b'')
        with patch.object(G,'_read_capture') as read:
            self.refused('GROUP_CAPTURE_FILE_BOUND');read.assert_not_called()

    def test_more_than_1024_other_files_refused_before_read(self):
        for index in range(G.MAX_PHASE_FILES):(self.folder/f'additional-{index}').write_bytes(b'')
        with patch.object(G,'_read_capture') as read:
            self.refused('GROUP_PHASE_ARTIFACT_RESOURCE_BOUND');read.assert_not_called()

    def test_other_artifact_bytes_refused_before_read_without_large_files(self):
        with self.changed_sizes({'query.bin':G.MAX_PHASE_ARTIFACT_BYTES+1}),patch.object(G,'_read_capture') as read:
            self.refused('GROUP_PHASE_ARTIFACT_RESOURCE_BOUND');read.assert_not_called()

    def test_exact_limits_keep_capture_separate_from_other_artifacts(self):
        with self.changed_sizes({'query.bin':G.MAX_PHASE_ARTIFACT_BYTES,'runtime_result.json':G.MAX_CAPTURE_BYTES}):
            value=self.inventory()
        self.assertEqual(value['artifact_bytes'],G.MAX_PHASE_ARTIFACT_BYTES)
        self.assertEqual(value['files']['runtime_result.json'][2],G.MAX_CAPTURE_BYTES)

    def test_request_directory_replaced_before_collection_preserves_originals(self):
        retained=self.root/'retained-original';self.folder.rename(retained);self.folder.mkdir()
        self.path.write_text(json.dumps(self.capture))
        self.refused('GROUP_OWNED_REQUEST_DIRECTORY_CHANGED');self.base.assert_not_called()
        self.assertEqual((retained/'boundaries/query.bin').read_bytes(),b'actual original bytes')

    def test_root_directory_replaced_before_collection_preserves_originals(self):
        retained=self.root.with_name('retained-root');self.root.rename(retained)
        self.root.mkdir();self.folder.mkdir();self.path.write_text(json.dumps(self.capture))
        self.refused('GROUP_OWNED_REQUEST_DIRECTORY_CHANGED');self.base.assert_not_called()
        self.assertTrue((retained/'000-reference/runtime_result.json').is_file())

    def test_same_bytes_foreign_capture_inode_after_read_refused(self):
        read=G._read_capture
        def replace(path,identity):
            raw=read(path,identity);old=path.with_name('retained-capture.json')
            path.rename(old);path.write_bytes(raw);return raw
        with patch.object(G,'_read_capture',side_effect=replace):self.refused('GROUP_REQUEST_ARTIFACT_IDENTITY_CHANGED')
        self.assertEqual(self.path.read_bytes(),self.path.with_name('retained-capture.json').read_bytes())

    def test_artifact_growth_during_capture_read_refused(self):
        read=G._read_capture
        def grow(path,identity):
            raw=read(path,identity);self.payload.write_bytes(b'changed after snapshot');return raw
        with patch.object(G,'_read_capture',side_effect=grow):self.refused('GROUP_REQUEST_ARTIFACT_IDENTITY_CHANGED')

    def test_new_over_capacity_file_before_completion_refused(self):
        read=G._read_capture
        def grow(path,identity):
            raw=read(path,identity);(self.folder/'new.bin').write_bytes(b'new');return raw
        with self.changed_sizes({'new.bin':G.MAX_PHASE_ARTIFACT_BYTES}),patch.object(G,'_read_capture',side_effect=grow):
            self.refused('GROUP_PHASE_ARTIFACT_RESOURCE_BOUND')

    def test_saved_capture_cannot_disagree_with_actual_in_memory_request(self):
        foreign=deepcopy(self.capture);foreign['natural_request']['request_id']='foreign-request'
        self.path.write_text(json.dumps(foreign));self.refused('GROUP_CAPTURE_ORIGINAL_CONTENT_DIFFERS')

    def test_stable_handle_ctime_domain_is_independently_checked(self):
        original=G.os.fstat
        def handle_stat(fd):
            value=original(fd)
            return SimpleNamespace(st_mode=value.st_mode,st_dev=value.st_dev,st_ino=value.st_ino,
                st_size=value.st_size,st_mtime_ns=value.st_mtime_ns,st_ctime_ns=value.st_ctime_ns+1)
        with patch.object(G.os,'fstat',side_effect=handle_stat):self.session.collect()
        self.session.control.complete_request.assert_called_once()

    def test_changing_handle_ctime_during_read_refuses(self):
        original=G.os.fstat;calls=[]
        def handle_stat(fd):
            value=original(fd);calls.append(fd)
            return SimpleNamespace(st_mode=value.st_mode,st_dev=value.st_dev,st_ino=value.st_ino,
                st_size=value.st_size,st_mtime_ns=value.st_mtime_ns,st_ctime_ns=value.st_ctime_ns+len(calls))
        with patch.object(G.os,'fstat',side_effect=handle_stat):self.refused('GROUP_CAPTURE_FILE_CHANGED_DURING_READ')

    def test_symlink_to_foreign_original_rejected_without_reading_target(self):
        foreign=self.root/'foreign-original.bin';foreign.write_bytes(b'foreign')
        link=self.folder/'foreign-link.bin'
        try:link.symlink_to(foreign)
        except OSError as error:self.skipTest('symlink unavailable: '+str(error))
        with patch.object(G,'_read_capture') as read:
            self.refused('GROUP_OWNED_ARTIFACT_PATH_CHANGED');read.assert_not_called()
        self.assertEqual(foreign.read_bytes(),b'foreign')


if __name__=='__main__':unittest.main()
