"""Generated-unit boundaries; actual compilation stays in the root CPU build."""
import hashlib
from pathlib import Path
import unittest

from fault39.native.test_metadata_ownership import inputs, between


ROOT = Path(__file__).resolve().parents[2]


class Writer26MetadataSourceClosureTests(unittest.TestCase):
    def test_all_three_actual_capture_bodies_are_present_without_event_runtime(self):
        unit, sources = inputs(ROOT, writer26=True)
        group = (ROOT/'fault_injection/native_adapters/writer26_group_runtime.cpp').read_text()
        trace = (ROOT/'fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_trace_preload.cpp').read_text()
        for text, start, end in (
                (trace, 'bool checked_product(', 'dcat26_fia_typed::Descriptor fia_typed_descriptor('),
                (group, 'bool descriptor_matches(', 'dcat26_fia_typed::Descriptor capture_fia_descriptor('),
                (group, 'dcat26_fia_typed::Descriptor capture_fia_descriptor(', 'bool destroy_descriptor(')):
            self.assertIn(start + between(text, start, end), unit)
        self.assertIn('return group26::capture_fia_descriptor(tensor).complete', unit)
        self.assertIn('route<7', unit)
        for unrelated in ('bool complete_event_geometry(', 'bool fia_typed_tail(', 'bool typed_tail(',
                          'bool destroy_descriptor('):
            self.assertNotIn(unrelated, unit)
        header = 'fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_fia_typed.h'
        self.assertEqual(sources[header], hashlib.sha256((ROOT/header).read_bytes()).hexdigest())

    def test_nonwriter_unit_keeps_optional_route_unexpanded(self):
        unit, sources = inputs(ROOT, writer26=False)
        self.assertIn('// @F26_GROUP_TYPED_CAPTURE@', unit)
        self.assertNotIn('dcat26_fia_typed::Descriptor capture_fia_descriptor(', unit)
        self.assertFalse(any('writer26_group' in name or 'fault26_fia_typed.h' in name for name in sources))


if __name__ == '__main__':
    unittest.main()
