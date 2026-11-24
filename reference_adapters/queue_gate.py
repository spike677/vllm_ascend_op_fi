"""Test-controller compatibility for kubelet's explicitly empty Go slice.

Never modify the checkpoint or discard unknown fields. Missing entries remain
unknown; a populated list still goes through the original allocation parser.
This is not a product admission rule.
"""
import copy
import json


def normalize_empty_entries(document):
    result = copy.deepcopy(document)
    if not isinstance(result, dict):
        return result
    data = result.get('Data')
    encoded = isinstance(data, str)
    if encoded:
        data = json.loads(data)
    if (isinstance(data, dict) and 'PodDeviceEntries' in data
            and data['PodDeviceEntries'] is None
            and isinstance(data.get('RegisteredDevices'), dict)):
        data['PodDeviceEntries'] = []
        result['Data'] = json.dumps(data) if encoded else data
    return result


def install(module):
    original = module.checkpoint_allocations
    def parse(document, cards):
        return original(normalize_empty_entries(document), cards)
    module.checkpoint_allocations = parse
