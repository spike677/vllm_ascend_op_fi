"""CPU-only synthetic refusal tests. These fixtures are never serving evidence."""
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension.serving_context import (
    ContextRefused, discover_logits_boundary, read_enqueued_identity, snapshot_fresh_prefill,
)


class MetadataOnlyTensor:
    """Host fake for unit tests; any attempt to read/compute values fails."""
    def __init__(self, storage, shape, dtype="torch.int32"):
        self.storage = storage
        self.shape = shape
        self.dtype = dtype
        self.device = "npu:0"

    def untyped_storage(self):
        return NS(_cdata=self.storage)

    def storage_offset(self):
        return 0

    def stride(self):
        return [1] if len(self.shape) == 1 else [self.shape[1], 1]

    def cpu(self):
        raise AssertionError("device access forbidden in draft")

    def tolist(self):
        raise AssertionError("device access forbidden in draft")


def fixture():
    model = object()
    req_id = "CPU_FIXTURE_ONLY-not-a-real-request"
    prompt = [31, 44, 9, 2]
    request = NS(req_id=req_id, num_computed_tokens=0, output_token_ids=[], mm_features=[],
                 prompt_embeds=None, lora_request=None, sampling_params=NS(n=1),
                 prompt_token_ids=prompt, num_prompt_tokens=4, num_tokens=4, block_ids=([4, 7],))
    table = NS(use_hybrid_blocks=False, blocks_per_phys_block=1, block_size=2,
               physical_block_size=2, num_blocks_per_row=[2],
               block_table=NS(cpu=[[4, 7]], gpu=MetadataOnlyTensor(3, [1, 2])),
               slot_mapping=NS(cpu=[8, 9, 14, 15], gpu=MetadataOnlyTensor(4, [4])))
    runner = NS(model=model, pcp_size=1, dcp_size=1, use_cp=False, speculative_config=None,
                use_async_scheduling=False, is_multimodal_model=False,
                vllm_config=NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                    data_parallel_size=1, enable_dbo=False),
                    scheduler_config=NS(max_num_seqs=1, enable_chunked_prefill=False),
                    cache_config=NS(enable_prefix_caching=False, kv_sharing_fast_prefill=False),
                    model_config=NS(enforce_eager=True, is_encoder_decoder=False)),
                input_batch=NS(req_ids=[req_id], num_reqs=1, req_id_to_index={req_id:0},
                    num_computed_tokens_cpu=[0], num_prompt_tokens=[4], num_tokens_no_spec=[4],
                    num_tokens=[0], token_ids_cpu=[prompt.copy()], block_table=NS(block_tables=[table])),
                requests={req_id:request},
                input_ids=NS(cpu=prompt.copy(), gpu=MetadataOnlyTensor(1, [4])),
                positions=NS(cpu=[0,1,2,3], gpu=MetadataOnlyTensor(2, [4], "torch.int64")),
                query_start_loc=NS(cpu=[0,4]), seq_lens=NS(cpu=[4]), query_lens=[4])
    metadata = NS(num_actual_tokens=4, num_prefills=1, num_decodes=0, seq_lens_list=[4],
                  actual_seq_lengths_q=[4], causal=True, slot_mapping=table.slot_mapping.gpu,
                  block_tables=table.block_table.gpu)
    context = NS(model_instance=model, in_profile_run=False, capturing=False, is_draft_model=False,
                 cudagraph_runtime_mode=NS(name="NONE"), ubatch_slices=None, dp_metadata=None,
                 flash_comm_v1_enabled=False, flashcomm_v2_enabled=False,
                 num_tokens=4, pad_size=0, attn_metadata={"actual.fixture.attention":metadata})
    kwargs = {"input_ids": runner.input_ids.gpu, "positions": runner.positions.gpu,
              "inputs_embeds":None, "intermediate_tensors":None}
    return runner, model, kwargs, context, prompt, req_id


class QuantMethod:
    def apply(self, layer, hidden_states, bias=None):
        raise AssertionError("No logits computation allowed in structural discovery")


class LogitsProcessor:
    logits_as_input = False
    org_vocab_size = vocab_size = 19

    def _get_logits(self, hidden_states, lm_head, embedding_bias=None):
        return lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)


LogitsProcessor.__module__ = "vllm.model_executor.layers.logits_processor"


_fixture_lmhead_tp = False


def lmhead_tp_enable():
    return _fixture_lmhead_tp


class AscendLogitsProcessor(LogitsProcessor):
    # Synthetic CPU fixture of the inspected Ascend implementation's actual
    # normal/LMHead-TP dispatcher. This class is not a runtime-model identity.
    def _get_logits(self, hidden_states, lm_head, embedding_bias=None):
        if lmhead_tp_enable():
            return self._get_logits_lmheadtp(hidden_states, lm_head, embedding_bias)
        else:
            return self._get_logits_normal(hidden_states, lm_head, embedding_bias)

    def _get_logits_normal(self, hidden_states, lm_head, embedding_bias=None):
        local_logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        logits = self._gather_logits(local_logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    def _get_logits_lmheadtp(self, hidden_states, lm_head, embedding_bias=None):
        raise AssertionError('TP logits must never be executed by structural discovery')

    def _gather_logits(self, logits):
        raise AssertionError('Gather must never be executed by structural discovery')


AscendLogitsProcessor.__module__ = 'vllm_ascend.ops.vocab_parallel_embedding'


def changed_dispatch(self, hidden_states, lm_head, embedding_bias=None):
    return self._get_logits_lmheadtp(hidden_states, lm_head, embedding_bias)


def changed_normal(self, hidden_states, lm_head, embedding_bias=None):
    return None


class ModelDifferentNames:
    def __init__(self):
        self.final_scores = LogitsProcessor()
        self.output_vocab = NS(num_embeddings=32, quant_method=QuantMethod())

    def compute_logits(self, hidden_states):
        return self.final_scores(self.output_vocab, hidden_states)

    def named_modules(self):
        yield "", self
        yield "final_scores", self.final_scores
        yield "output_vocab", self.output_vocab


class AmbiguousModel(ModelDifferentNames):
    def compute_logits(self, hidden_states):
        first = self.final_scores(self.output_vocab, hidden_states)
        second = self.final_scores(self.output_vocab, hidden_states)
        return first, second


class DraftTests(unittest.TestCase):
    def snapshot(self, values, **extra):
        runner, model, kwargs, context, prompt, req_id = values
        return snapshot_fresh_prefill(runner, model, kwargs, context,
            expected_prompt_token_ids=prompt, expected_internal_request_id=extra.pop("expected_id", req_id), **extra)

    def test_cpu_fixture_preserves_id_and_never_reads_device_values(self):
        values = fixture()
        result = self.snapshot(values)
        self.assertEqual(result["internal_request_id"], values[-1])
        self.assertEqual(result["slot_mapping"], [8,9,14,15])
        self.assertFalse(result["tensor_values_copied_from_npu"])
        self.assertEqual(result["device_input_bytes_equality"], "NOT_ESTABLISHED_BY_CPU_MIRROR")

    def test_uninitialized_or_previous_sampling_counter_is_not_current_input_length(self):
        # Actual NPUInputBatch starts this legacy array at zero; inherited
        # add_request does not refresh it when a batch row is reused.
        for legacy_count in (0, 5, 128):
            values = fixture()
            values[0].input_batch.num_tokens[0] = legacy_count
            with self.subTest(legacy_count=legacy_count):
                evidence = self.snapshot(values)['token_count_evidence']
                self.assertEqual(evidence['request_num_tokens'], 4)
                self.assertEqual(evidence['batch_num_tokens_no_spec'], 4)
                self.assertEqual(evidence['batch_num_tokens_legacy'], [legacy_count])
                self.assertEqual(evidence['legacy_field_role'], 'POST_SAMPLING_COUNTER_NOT_PREFILL_AUTHORITY')

    def test_authoritative_request_and_no_spec_counts_still_must_match_prompt(self):
        for target in ('request', 'no_spec'):
            values = fixture()
            runner, _, _, _, _, req_id = values
            runner.input_batch.num_tokens[0] = 4
            if target == 'request':
                runner.requests[req_id].num_tokens = 5
                reason = 'REQUEST_TOTAL_TOKEN_COUNT_MISMATCH'
            else:
                runner.input_batch.num_tokens_no_spec[0] = 5
                reason = 'SPEC_OR_DECODE_TOKEN_PRESENT'
            with self.subTest(target=target), self.assertRaisesRegex(ContextRefused, reason):
                self.snapshot(values)

    def test_scheduled_lengths_and_previous_output_are_not_hidden_by_legacy_counter(self):
        for target in ('query', 'seq', 'computed', 'output'):
            values = fixture()
            runner, _, _, _, _, req_id = values
            if target == 'query':
                runner.query_lens = [3]
                reason = 'SCHEDULED_TOKEN_COUNT_MISMATCH'
            elif target == 'seq':
                runner.seq_lens.cpu = [3]
                reason = 'SEQ_LEN_MISMATCH'
            elif target == 'computed':
                runner.input_batch.num_computed_tokens_cpu = [1]
                reason = 'BATCH_REQUEST_NOT_FRESH'
            else:
                runner.requests[req_id].output_token_ids = [99]
                reason = 'PREVIOUS_GENERATION_ALREADY_PRESENT'
            with self.subTest(target=target), self.assertRaisesRegex(ContextRefused, reason):
                self.snapshot(values)

    def test_attention_metadata_must_describe_the_current_prefill(self):
        for attribute, value, reason in (
                ('num_actual_tokens', 3, 'ATTENTION_NOT_SINGLE_FRESH_PREFILL'),
                ('num_prefills', 0, 'ATTENTION_NOT_SINGLE_FRESH_PREFILL'),
                ('num_decodes', 1, 'ATTENTION_NOT_SINGLE_FRESH_PREFILL'),
                ('seq_lens_list', [5], 'ATTENTION_LENGTHS_MISMATCH'),
                ('actual_seq_lengths_q', [3], 'ATTENTION_LENGTHS_MISMATCH'),
                ('causal', False, 'NON_CAUSAL_ATTENTION_UNADAPTED')):
            values = fixture()
            metadata = next(iter(values[3].attn_metadata.values()))
            setattr(metadata, attribute, value)
            with self.subTest(attribute=attribute), self.assertRaisesRegex(ContextRefused, reason):
                self.snapshot(values)

    def test_attention_kv_metadata_cannot_point_at_foreign_buffers(self):
        for attribute, shape, reason in (
                ('slot_mapping', [4], 'ATTENTION_SLOT_BUFFER_UNALIGNED'),
                ('block_tables', [1, 2], 'ATTENTION_BLOCK_BUFFER_UNALIGNED')):
            values = fixture()
            metadata = next(iter(values[3].attn_metadata.values()))
            setattr(metadata, attribute, MetadataOnlyTensor(999, shape))
            with self.subTest(attribute=attribute), self.assertRaisesRegex(ContextRefused, reason):
                self.snapshot(values)

    def test_missing_or_wrong_advance_id_is_refused(self):
        for req_id in (None, "", "another-fixture-id"):
            with self.subTest(req_id=req_id), self.assertRaisesRegex(ContextRefused,"ADVANCE_EXPECTED_ID"):
                self.snapshot(fixture(), expected_id=req_id)

    def test_phase_request_reuse_is_refused(self):
        values = fixture()
        with self.assertRaisesRegex(ContextRefused,"REQUEST_ID_REUSED"):
            self.snapshot(values, seen_internal_request_ids=[values[-1]])

    def test_wrong_position_or_cpu_input_mirror_is_refused(self):
        for which in ("positions", "input_ids"):
            values = fixture()
            getattr(values[0], which).cpu[1] = 999
            with self.subTest(which=which), self.assertRaises(ContextRefused):
                self.snapshot(values)

    def test_wrong_slot_or_request_blocks_are_refused(self):
        values = fixture()
        values[0].input_batch.block_table.block_tables[0].slot_mapping.cpu[0] = 999
        with self.assertRaisesRegex(ContextRefused,"ACTUAL_SLOT_MAPPING"):
            self.snapshot(values)

    def test_graph_prefix_chunk_or_nonfresh_is_refused(self):
        for branch in ("graph", "prefix", "chunk", "computed"):
            values = fixture()
            runner, _, _, context, _, req_id = values
            if branch == "graph": context.cudagraph_runtime_mode.name = "FULL"
            if branch == "prefix": runner.vllm_config.cache_config.enable_prefix_caching = True
            if branch == "chunk": runner.vllm_config.scheduler_config.enable_chunked_prefill = True
            if branch == "computed": runner.requests[req_id].num_computed_tokens = 1
            with self.subTest(branch=branch), self.assertRaises(ContextRefused):
                self.snapshot(values)

    def test_actual_input_must_share_runner_storage(self):
        values = fixture()
        values[2]["input_ids"] = MetadataOnlyTensor(99,[4])
        with self.assertRaisesRegex(ContextRefused,"INPUT_STORAGE"):
            self.snapshot(values)

    def test_logits_boundary_uses_contract_not_layer_name(self):
        model = ModelDifferentNames()
        boundary = discover_logits_boundary(model, actual_logits_processor_type=LogitsProcessor, actual_vocab_size=19)
        self.assertIs(boundary["processor"], model.final_scores)
        self.assertEqual(boundary["report"]["lm_head_path"], "output_vocab")
        self.assertEqual(boundary["report"]["output_contract"], [1,19])

    def test_logits_ambiguity_or_vocab_mismatch_is_refused(self):
        with self.assertRaisesRegex(ContextRefused,"AMBIGUOUS"):
            discover_logits_boundary(AmbiguousModel(), actual_logits_processor_type=LogitsProcessor, actual_vocab_size=19)
        with self.assertRaisesRegex(ContextRefused,"VOCAB"):
            discover_logits_boundary(ModelDifferentNames(), actual_logits_processor_type=LogitsProcessor, actual_vocab_size=20)

    def test_ascend_normal_dispatch_is_inspected_without_running_model_or_quant_method(self):
        model = ModelDifferentNames()
        model.final_scores = AscendLogitsProcessor()
        boundary = discover_logits_boundary(model, actual_logits_processor_type=LogitsProcessor, actual_vocab_size=19)
        self.assertIs(boundary['processor'], model.final_scores)
        self.assertIs(boundary['lm_head'], model.output_vocab)
        self.assertIsNotNone(boundary['report']['ascend_dispatch_source'])
        self.assertNotEqual(boundary['report']['ascend_dispatch_source']['source_sha256'],
                            boundary['report']['get_logits_source']['source_sha256'])
        self.assertTrue(boundary['branch_guard']())

    def test_ascend_lmhead_tp_is_rejected_at_discovery_and_when_branch_changes(self):
        model = ModelDifferentNames()
        model.final_scores = AscendLogitsProcessor()
        with patch(__name__ + '._fixture_lmhead_tp', True):
            with self.assertRaisesRegex(ContextRefused, 'ASCEND_LMHEAD_TP_OR_DISPATCH_CHANGED'):
                discover_logits_boundary(model, actual_logits_processor_type=LogitsProcessor, actual_vocab_size=19)
        boundary = discover_logits_boundary(model, actual_logits_processor_type=LogitsProcessor, actual_vocab_size=19)
        with patch(__name__ + '._fixture_lmhead_tp', True):
            with self.assertRaisesRegex(ContextRefused, 'ASCEND_LMHEAD_TP_OR_DISPATCH_CHANGED'):
                boundary['branch_guard']()

    def test_ascend_replaced_gate_or_dispatch_cannot_reuse_old_boundary_proof(self):
        for target in ('gate', 'dispatcher', 'normal'):
            model = ModelDifferentNames()
            model.final_scores = AscendLogitsProcessor()
            boundary = discover_logits_boundary(model, actual_logits_processor_type=LogitsProcessor, actual_vocab_size=19)
            if target == 'gate':
                replacement = patch(__name__ + '.lmhead_tp_enable', lambda: False)
            elif target == 'dispatcher':
                replacement = patch.object(model.final_scores, '_get_logits', MethodType(changed_dispatch, model.final_scores))
            else:
                replacement = patch.object(model.final_scores, '_get_logits_normal', MethodType(changed_normal, model.final_scores))
            with self.subTest(target=target), replacement:
                with self.assertRaisesRegex(ContextRefused, 'ASCEND_LMHEAD_TP_OR_DISPATCH_CHANGED'):
                    boundary['branch_guard']()

    def test_ascend_unrecognized_dispatch_cannot_be_assumed_to_take_normal_path(self):
        model = ModelDifferentNames()
        model.final_scores = AscendLogitsProcessor()
        with patch.object(model.final_scores, '_get_logits', MethodType(changed_dispatch, model.final_scores)):
            with self.assertRaisesRegex(ContextRefused, 'ASCEND_LOGITS_DISPATCH_UNADAPTED'):
                discover_logits_boundary(model, actual_logits_processor_type=LogitsProcessor, actual_vocab_size=19)

    def test_read_enqueued_id_requires_real_map_and_inproc_gate(self):
        # Synthetic class identity only tests the gate; this is never actual-serving evidence.
        core_class = type("InprocClient", (), {"__module__":"vllm.v1.engine.core_client"})
        state = NS(request_id="fixture-internal", external_req_id="fixture-external")
        llm = NS(llm_engine=NS(engine_core=core_class(), output_processor=NS(request_states={"fixture-internal":state})))
        result = read_enqueued_identity(llm, ["fixture-internal"])
        self.assertEqual(result["external_request_id"], "fixture-external")
        with self.assertRaisesRegex(ContextRefused,"MAP_MISMATCH"):
            read_enqueued_identity(llm, ["wrong"])
        llm.llm_engine.engine_core = object()
        with self.assertRaisesRegex(ContextRefused,"INPROC_ENGINE"):
            read_enqueued_identity(llm, ["fixture-internal"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
