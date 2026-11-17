"""Synthetic CPU metadata tests; source approval is mocked, never NPU proof."""
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension import serving_context as context_reader
try:
    from .test_serving_context import fixture, MetadataOnlyTensor
except ImportError:
    from test_serving_context import fixture, MetadataOnlyTensor


class AscendMLAMetadata(NS):
    pass


class AscendMLAPrefillMetadata(NS):
    pass


AscendMLAMetadata.__module__ = context_reader._MLA_MODULE
AscendMLAPrefillMetadata.__module__ = context_reader._MLA_MODULE


def mla_fixture():
    values = fixture()
    runner, _, _, context, _, _ = values
    table = runner.input_batch.block_table.block_tables[0]
    runner.query_start_loc.gpu = MetadataOnlyTensor(10, [2])
    mask = MetadataOnlyTensor(11, [512, 512], 'torch.bfloat16')
    prefill = AscendMLAPrefillMetadata(
        attn_mask=mask, query_lens=[4], seq_lens=[4], context_lens=[4],
        input_positions=runner.positions.gpu,
        query_start_loc=MetadataOnlyTensor(12, [2]), block_table=table.block_table.gpu,
        max_query_len=4, max_seq_lens=4, chunked_context=None, pcp_metadata=None)
    metadata = AscendMLAMetadata(num_actual_tokens=4, num_prefills=1, num_decodes=0,
        num_decode_tokens=0, num_input_tokens=4, num_actual_tokens_pcp_padded=4,
        slot_mapping=table.slot_mapping.gpu, block_tables=table.block_table.gpu,
        query_start_loc=runner.query_start_loc.gpu, seq_lens=[4], seq_lens_cpu=[4],
        query_lens=[4], attn_mask=mask, attn_state=NS(name='PrefillNoCache'),
        prefill=prefill, decode=None)
    context.attn_metadata = {'actual.cpu.mla.fixture': metadata}
    return values


class MlaPrefillContextTests(unittest.TestCase):
    def setUp(self):
        module = ModuleType(context_reader._MLA_MODULE)
        module.AscendMLAMetadata = AscendMLAMetadata
        module.AscendMLAPrefillMetadata = AscendMLAPrefillMetadata
        self.module = module
        self.enterContext(patch.dict(sys.modules, {context_reader._MLA_MODULE: module}))

    def snapshot(self, values):
        runner, model, actual, context, prompt, request_id = values
        with patch.object(context_reader, '_reviewed_mla_prefill_contract', return_value={
                'status': 'CPU_SOURCE_CONTRACT_MOCK_NOT_NATIVE', 'mask_tensor_values_verified': False}):
            return context_reader.snapshot_fresh_prefill(runner, model, actual, context,
                expected_prompt_token_ids=prompt, expected_internal_request_id=request_id)

    def test_actual_mla_fields_are_kept_without_fabricating_dense_fields(self):
        values = mla_fixture()
        metadata = next(iter(values[3].attn_metadata.values()))
        self.assertFalse(hasattr(metadata, 'causal'))
        self.assertFalse(hasattr(metadata, 'seq_lens_list'))
        self.assertFalse(hasattr(metadata, 'actual_seq_lengths_q'))
        result = self.snapshot(values)
        evidence = result['attention_metadata_evidence']['actual.cpu.mla.fixture']
        self.assertEqual(evidence['adapter'], 'ASCEND_MLA_V1_FRESH_PREFILL')
        self.assertEqual(evidence['observed_lengths']['prefill.context_lens'], [4])
        self.assertEqual(evidence['context_lens_field_role'], 'BUILDER_STORES_FULL_SEQ_LENS_NOT_COMPUTED_CACHE_PREFIX')
        self.assertFalse(evidence['derived_query_values_read'])
        self.assertFalse(result['tensor_values_copied_from_npu'])
        self.assertEqual(result['internal_request_id'], values[-1])
        self.assertEqual(result['slot_mapping'], [8, 9, 14, 15])

    def test_every_actual_length_field_must_match_the_scheduler(self):
        for field in ('seq_lens_cpu', 'seq_lens', 'query_lens', 'prefill.seq_lens',
                      'prefill.query_lens', 'prefill.context_lens'):
            values = mla_fixture()
            metadata = next(iter(values[3].attn_metadata.values()))
            owner, name = (metadata.prefill, field.split('.')[1]) if field.startswith('prefill.') else (metadata, field)
            setattr(owner, name, [3])
            with self.subTest(field=field), self.assertRaisesRegex(
                    context_reader.ContextRefused, 'MLA_PREFILL_LENGTHS_MISMATCH'):
                self.snapshot(values)

    def test_actual_mla_aliases_do_not_accept_foreign_buffers(self):
        for target, reason in (('query', 'ACTUAL_INPUT_STORAGE_MISMATCH'),
                               ('positions', 'ACTUAL_INPUT_STORAGE_MISMATCH'),
                               ('prefill_blocks', 'MLA_PREFILL_BLOCK_BUFFER_UNALIGNED'),
                               ('mask', 'MLA_PREFILL_MASK_GEOMETRY_OR_ALIAS_UNPROVEN'),
                               ('slots', 'ATTENTION_SLOT_BUFFER_UNALIGNED')):
            values = mla_fixture()
            metadata = next(iter(values[3].attn_metadata.values()))
            if target == 'query':
                metadata.query_start_loc = MetadataOnlyTensor(999, [2])
            elif target == 'positions':
                metadata.prefill.input_positions = MetadataOnlyTensor(999, [4], 'torch.int64')
            elif target == 'prefill_blocks':
                metadata.prefill.block_table = MetadataOnlyTensor(999, [1, 2])
            elif target == 'mask':
                metadata.prefill.attn_mask = MetadataOnlyTensor(999, [512, 512], 'torch.bfloat16')
            else:
                metadata.slot_mapping = MetadataOnlyTensor(999, [4])
            with self.subTest(target=target), self.assertRaisesRegex(context_reader.ContextRefused, reason):
                self.snapshot(values)

    def test_mla_decode_chunk_context_parallel_and_padding_stay_closed(self):
        for target in ('decode', 'chunked_context', 'pcp_metadata', 'num_input_tokens', 'attn_state'):
            values = mla_fixture()
            metadata = next(iter(values[3].attn_metadata.values()))
            if target in ('chunked_context', 'pcp_metadata'):
                setattr(metadata.prefill, target, object())
            elif target == 'num_input_tokens':
                metadata.num_input_tokens = 8
            elif target == 'attn_state':
                metadata.attn_state = NS(name='PrefillCacheHit')
            else:
                metadata.decode = object()
            with self.subTest(target=target), self.assertRaises(context_reader.ContextRefused):
                self.snapshot(values)

    def test_length_reader_never_copies_npu_metadata_values_to_cpu(self):
        values = mla_fixture()
        next(iter(values[3].attn_metadata.values())).seq_lens_cpu = MetadataOnlyTensor(8, [1])
        with self.assertRaisesRegex(context_reader.ContextRefused, 'UNSUPPORTED_CPU_BUFFER'):
            self.snapshot(values)

    def test_matching_class_name_is_not_implementation_review(self):
        metadata = next(iter(mla_fixture()[3].attn_metadata.values()))
        # The production check has not been patched here. Its actual reviewed
        # builder/consumer classes must exist; a metadata name is insufficient.
        with self.assertRaisesRegex(context_reader.ContextRefused, 'MISSING_FIELD'):
            context_reader._reviewed_mla_prefill_contract(metadata)
        self.module.AscendMLAMetadata = type('OtherMetadata', (), {})
        with self.assertRaisesRegex(context_reader.ContextRefused, 'UNREVIEWED_MLA_METADATA_TYPE'):
            context_reader._reviewed_mla_prefill_contract(metadata)


if __name__ == '__main__':
    unittest.main(verbosity=2)
