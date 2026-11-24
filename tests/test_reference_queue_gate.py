import json
import unittest
from reference_adapters.queue_gate import normalize_empty_entries


class GateTests(unittest.TestCase):
    def test_explicit_null_is_empty_without_mutating_source(self):
        d={'Data':{'PodDeviceEntries':None,'RegisteredDevices':{'huawei.com/Ascend910':['Ascend910-2']}}}
        self.assertEqual(normalize_empty_entries(d)['Data']['PodDeviceEntries'],[])
        self.assertIsNone(d['Data']['PodDeviceEntries'])
        e={'Data':json.dumps(d['Data'])}
        self.assertEqual(json.loads(normalize_empty_entries(e)['Data'])['PodDeviceEntries'],[])

    def test_missing_unknown_and_real_allocations_are_not_bypassed(self):
        for d in ({'Data':{'RegisteredDevices':{}}},
                  {'Data':{'PodDeviceEntries':None,'RegisteredDevices':None}},
                  {'Data':{'PodDeviceEntries':[{'PodUID':'live'}],'RegisteredDevices':{}}}):
            self.assertEqual(normalize_empty_entries(d),d)
