"""Physical geometry is availability evidence, never target qualification."""
import unittest
from experiments.worker_core_v11_20261118.allocator26_availability import geometry, compare_sets


def segment(start, states):
    return dict(address=start,stream=7,blocks=[dict(address=start+64*i,size=64,
        requested_size=32,state=state) for i,state in enumerate(states)])


def call(residual=1000, output=1064, residual_bytes=32, output_bytes=32):
    return dict(residual=residual,output=output,residual_bytes=residual_bytes,output_bytes=output_bytes)


class AvailabilityTests(unittest.TestCase):
    def test_adjacent_has_real_padding(self):
        value=geometry([segment(1000,['active_allocated']*2)],call())
        self.assertEqual(value['status'],'ELIGIBLE_GEOMETRY')
        self.assertEqual(value['allocation']['shift'],64)
        self.assertEqual((value['relation'],value['signed_block_gap']),('ADJACENT',0))

    def test_intervening_free_and_active_blocks_remain_visible(self):
        value=geometry([segment(1000,['active_allocated','inactive','active_allocated','active_allocated'])],call(output=1192))
        self.assertEqual(value['status'],'PROVEN_NONADJACENT')
        self.assertEqual(value['signed_block_gap'],128)
        self.assertEqual(value['intervening_blocks_by_state'],{'inactive':1,'active_allocated':1})

    def test_reverse_order_is_not_adjacency(self):
        value=geometry([segment(1000,['active_allocated']*2)],call(residual=1064,output=1000))
        self.assertEqual(value['status'],'PROVEN_NONADJACENT')
        self.assertEqual(value['relation'],'OUTPUT_BEFORE_RESIDUAL')
        self.assertEqual(value['signed_block_gap'],-128)

    def test_different_segments_even_touching_do_not_admit(self):
        value=geometry([segment(1000,['active_allocated']),segment(1064,['active_allocated'])],call())
        self.assertEqual(value['status'],'UNKNOWN_GEOMETRY')
        self.assertEqual(value['relation'],'DIFFERENT_SEGMENTS')

    def test_shared_block_and_span_overrun_remain_unknown(self):
        value=geometry([segment(1000,['active_allocated']*2)],call(output=1032,residual_bytes=16,output_bytes=16))
        self.assertEqual((value['status'],value['relation']),('UNKNOWN_GEOMETRY','SAME_BLOCK'))
        value=geometry([segment(1000,['active_allocated']*2)],call(output_bytes=96))
        self.assertEqual(value['status'],'UNKNOWN_GEOMETRY')

    def test_unobserved_and_fault_geometry_cannot_enlarge_control_intersection(self):
        def row(values):return dict(calls=[dict(geometry=dict(status=v)) for v in values])
        matrix=[row(['ELIGIBLE_GEOMETRY','NO_ALLOCATOR_OBSERVATION']),
                row(['PROVEN_NONADJACENT','ELIGIBLE_GEOMETRY']),
                row(['ELIGIBLE_GEOMETRY','ELIGIBLE_GEOMETRY'])]
        result=compare_sets(matrix,[0,1])
        self.assertEqual(result['union'],[0,1])
        self.assertEqual(result['intersection'],[])
        self.assertEqual(result['variable'],[0,1])


if __name__=='__main__':unittest.main()
