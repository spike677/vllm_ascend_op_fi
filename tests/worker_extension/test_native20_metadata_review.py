"""Counter audit regressions; synthetic receipts grant no NPU acceptance."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'experiments/c2c3_closure_20261117'
sys.path.insert(0, str(DIRECTORY))
from review_native20_metadata_arrays import inspect_snapshot, review_group
from review_native39_metadata_arrays import COUNTERS


def snapshot(index):
    images = []
    for number, name in enumerate(('libopapi.so', 'libopapi_nn.so'), 1):
        state = dict.fromkeys(COUNTERS, 0)
        state.update(version=1, size=120, instance=number,
                     capture_calls=index, getter_calls=index * 3,
                     acquired_arrays=index * 3, released_arrays=index * 3,
                     revision=index, peak_live_arrays=3)
        images.append(dict(name=name, path='/native/' + name, sha256='a'*64,
                           symbol='dcat20_metadata_arrays_v1', mapped_paths=['/native/' + name],
                           symbol_providers=['/native/' + name], state=state))
    return dict(schema='demoncat.metadata_arrays.v1', source='ACTUAL_LOADED_PROXY_COUNTERS',
                scope='SDK_GETTER_ARRAYS_ONLY', reset_supported=False, pid=23,
                build_profile_sha256='b'*64, build_manifest_sha256='c'*64, images=images)


class Native20MetadataReviewTests(unittest.TestCase):
    def group(self):
        names = ['discovery_1','discovery_2','reference','lossless','qualification_off',
                 'baseline','fault','off_observed','off']
        phases = [dict(phase=name, model_output_sha256='d'*64, metadata_arrays=snapshot(i+1),
            metadata_array_activity=dict(process_lifetime_nonzero_alone_accepted=False, pid=23,
                build_profile_sha256='b'*64, group_observed_delta=dict(capture_calls=2*(i+1),
                getter_calls=6*(i+1), acquired_arrays=6*(i+1), released_arrays=6*(i+1))))
            for i,name in enumerate(names)]
        return dict(native_phases=phases, qualification=dict(trigger='once', completed_phases=names,
                    recovery_verified=True, failure=None))

    def execute(self, data):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'runtime').mkdir();(root/'group').mkdir()
            (root/'runtime/serving_receipt.json').write_text(json.dumps(data))
            (root/'group/00001-worker-health.json').write_text(json.dumps(dict(
                metadata_arrays=snapshot(10), owned_hooks=0, baseline='d'*64, off='d'*64)))
            return review_group(root)[0]

    def test_current_phase_activity_and_released_arrays(self):
        result=self.execute(self.group())
        self.assertTrue(result['checks_passed'])
        self.assertEqual(result['positive_within_group_intervals'],8)

    def test_lifetime_totals_cannot_replace_phase_activity(self):
        group=self.group()
        for phase in group['native_phases']:
            phase['metadata_arrays']=snapshot(10)
        with self.assertRaisesRegex(ValueError,'DELTA_MISMATCH'):
            self.execute(group)

    def test_live_arrays_or_counter_identity_refuse(self):
        for kind in ('live','identity'):
            group=self.group();state=group['native_phases'][4]['metadata_arrays']['images'][1]['state']
            if kind=='live':state['live_arrays']=1
            else:state['instance']=91
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.execute(group)

    def test_foreign_symbol_and_shared_counter_refuse(self):
        for kind in ('symbol','instance'):
            value=snapshot(1)
            if kind=='symbol':value['images'][0]['symbol']='fault39_metadata_arrays_v1'
            else:value['images'][1]['state']['instance']=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):inspect_snapshot(value,'test')
