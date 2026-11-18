"""Actual CPU frames/dispatch; synthetic weights, no Native row qualification."""
import dataclasses as dc
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

import test_quant_token_routes as base
from demoncat.runtime.vllm_ascend.worker_extension.quant_row_provenance import RequestRowProvenance
from demoncat.runtime.vllm_ascend.worker_extension.quant_source_rows import SourceBoundRows, source_bindings
from demoncat.runtime.vllm_ascend.worker_extension.quant_token_routes import IntegerRouteObserver, SCHEMAS
from demoncat.runtime.vllm_ascend.worker_extension.quant_route_frames import RouteFrameObserver


def payload(name, fields):
    return dc.make_dataclass(name, [(key, object) for key in fields.split()], frozen=True, slots=True)


TYPES = {
    name:payload(name, fields) for name,fields in {
        'MoEPrepareOutput':'hidden_states router_logits',
        'MoEWeights':'w1 w2 w1_scale w2_scale w1_scale_bias w2_scale_bias w1_offset w2_offset',
        'MoEFusedExpertsInput':'hidden_states topk_ids topk_weights weights routing quant',
        'MoETokenDispatchInput':'hidden_states topk_ids topk_weights routing quant',
        'MoETokenDispatchOutput':'hidden_states group_list group_list_type dynamic_scale topk_scales combine_metadata',
        'MoEAllGatherCombineMetadata':'expanded_row_idx topk_weights',
        'MoEMlpComputeInput':'hidden_states group_list group_list_type dynamic_scale topk_scales weights quant fusion',
        'MoERoutingParams':'apply_router_weight_on_input expert_map pertoken_scale global_redundant_expert_num',
        'MoEQuantParams':'is_quant is_mxfp',
    }.items()}


def prepare(hidden_states, router_logits):
    return TYPES['MoEPrepareOutput'](hidden_states.clone() if model.replace_prepared else hidden_states, router_logits)


def finalize(hidden_states):
    return hidden_states.clone() if model.replace_finalized else hidden_states


def build_mlp_compute_input(fused_experts_input, token_dispatch_output, use_fusion_ops):
    d = token_dispatch_output
    result = TYPES['MoEMlpComputeInput'](d.hidden_states, d.group_list, d.group_list_type,
        d.dynamic_scale, d.topk_scales, fused_experts_input.weights, fused_experts_input.quant, use_fusion_ops)
    if model.replace_payload_type:
        return SimpleNamespace(**{field.name:getattr(result,field.name) for field in dc.fields(result)})
    return dc.replace(result, group_list=d.group_list.clone()) if model.replace_builder_group else result


def cumsum_group_list(group_list, src_list_type, dst_list_type):
    return group_list.cumsum(dim=0)


class Device:
    @staticmethod
    def first(*, x, weight, group_list, weight_scale, x_scale, bias=None, use_mxfp_quant=False):
        return base._op('gmm1')(x, weight, group_list, weight_scale, x_scale, bias=bias)

    @classmethod
    def second(cls, *, hidden_states, weight, weight_scale, per_token_scale, group_list,
               group_list_type, bias=None, use_mxfp_quant=False):
        return base._op('gmm2')([hidden_states], weight, scale=weight_scale, per_token_scale=[per_token_scale],
            group_list=group_list, group_list_type=group_list_type, split_item=2, group_type=0)[0]


def quant_apply_mlp(hidden_states, w1, w2, w1_scale, w2_scale, group_list, dynamic_scale,
                    group_list_type=1, w1_scale_bias=None, w2_scale_bias=None, w1_offset=None,
                    w2_offset=None, fusion=True, dynamic_eplb=False, use_mxfp_quant=False):
    first, scale, aux = Device.first(x=hidden_states, weight=w1[0], group_list=cumsum_group_list(group_list,1,0),
        weight_scale=w1_scale[0], x_scale=dynamic_scale)
    if model.change_mlp_epoch:
        first.add_(1)
    if model.change_alias_epoch:
        first[:,1:].add_(1)
    if model.replace_mlp_output:
        first = first.clone()
    if model.retire_routed:
        hidden_states.set_(torch.empty(0, dtype=torch.int8))
    return Device.second(hidden_states=first, weight=w2, weight_scale=w2_scale, per_token_scale=scale,
        group_list=group_list, group_list_type=1)


def unified_apply_mlp(mlp_compute_input):
    value = mlp_compute_input
    return quant_apply_mlp(value.hidden_states, value.weights.w1, value.weights.w2,
        value.weights.w1_scale, value.weights.w2_scale, value.group_list, value.dynamic_scale)


class Dispatcher:
    def token_dispatch(self, token_dispatch_input):
        value = token_dispatch_input
        out, raw, counts, scale = base._op('routing')(value.hidden_states, value.topk_ids,
            active_num=48, expert_num=64, expert_tokens_num_type=1, expert_tokens_num_flag=True,
            active_expert_range=[0,64], quant_mode=1)
        return TYPES['MoETokenDispatchOutput'](out, counts, 1, scale, None,
            TYPES['MoEAllGatherCombineMetadata'](raw, value.topk_weights))

    def token_combine(self, hidden_states, combine_metadata):
        return base._op('unpermute')(hidden_states, combine_metadata.expanded_row_idx.abs(), combine_metadata.topk_weights)


class Pipeline:
    def __init__(self):
        self.dispatcher = Dispatcher()

    def fused_experts(self, fused_experts_input):
        f = fused_experts_input
        dispatch = self.dispatcher.token_dispatch(TYPES['MoETokenDispatchInput'](
            f.hidden_states, f.topk_ids, f.topk_weights, f.routing, f.quant))
        built = build_mlp_compute_input(f, dispatch, True)
        output = self._apply_mlp(built)
        return self.dispatcher.token_combine(output, dispatch.combine_metadata)

    def _apply_mlp(self, mlp_compute_input):
        return unified_apply_mlp(mlp_compute_input)


def shared(hidden_states, logits, shared_input, name):
    prepared = prepare(hidden_states, logits)
    probs, ids, aux = base.select_experts(prepared.hidden_states, logits)
    w = TYPES['MoEWeights']([torch.ones((64,4,8),dtype=torch.int8)], [torch.ones((64,4,4),dtype=torch.int8)],
        [torch.ones((64,8))], [torch.ones((64,4))], None,None,None,None)
    fused = TYPES['MoEFusedExpertsInput'](prepared.hidden_states, ids, probs, w,
        TYPES['MoERoutingParams'](False,None,None,0), TYPES['MoEQuantParams'](True,False))
    routed = finalize(model.pipeline.fused_experts(fused))
    return hidden_states.clone(), routed


LIB = torch.library.Library('cpu_quant_route_frames', 'DEF')
LIB.define('shared(Tensor(a0!) hidden_states, Tensor logits, Tensor shared_input, str name) -> (Tensor, Tensor)')
LIB.impl('shared', shared, 'CPU')


class Model(base.CpuModel):
    def __init__(self):
        super().__init__()
        self.pipeline = Pipeline()
        self.routed_scaling_factor = 2.0
        self.replace_prepared = self.replace_finalized = self.replace_builder_group = False
        self.change_mlp_epoch = self.replace_mlp_output = self.wrong_scalar = self.tensor_scalar = False
        self.wrong_shared = self.change_scalar_owner = self.extra_call = False
        self.replace_payload_type = self.change_alias_epoch = self.wrong_final_view = False

    def forward(self, hidden_states, logits):
        left, right = torch.ops.cpu_quant_route_frames.shared(hidden_states, logits, hidden_states, 'cpu-layer')
        left, right = torch.ops.aten.alias.default(left), torch.ops.aten.alias.default(right)
        if self.change_scalar_owner:
            self.routed_scaling_factor = 3.0
        scalar = torch.tensor(2.0) if self.tensor_scalar else (3.0 if self.wrong_scalar else self.routed_scaling_factor)
        right.mul_(scalar)
        right.add_(left.clone() if self.wrong_shared else left)
        return right.clone().view(8,4) if self.wrong_final_view else right.view(8,4)


class RouteFrameTests(unittest.TestCase):
    def setUp(self):
        global model
        self.model = model = base._model = Model().eval()
        roles = {'Model.forward':'rchain_model', 'shared':'moe_opaque', 'prepare':'rchain_prepare',
            'finalize':'rchain_finalize', 'Pipeline.fused_experts':'route_pipeline', 'Pipeline._apply_mlp':'route_mlp',
            'Dispatcher.token_dispatch':'route_dispatch', 'Dispatcher.token_combine':'route_combine',
            'build_mlp_compute_input':'rchain_builder', 'unified_apply_mlp':'rchain_entry',
            'quant_apply_mlp':'rchain_quant', 'cumsum_group_list':'rchain_cumsum',
            'Device.first':'rchain_gmm1','Device.second':'rchain_gmm2'}
        bindings,sources=source_bindings({__name__:(hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),roles),
            base.__name__:(hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),{'select_experts':'route_selector'})})
        aliases={base._op(role):SCHEMAS[role].split('(')[0].replace('::','.')+'.default'
                 for role in ('topk','routing','gmm1','gmm2','unpermute')}
        aliases[torch.ops.cpu_quant_route_frames.shared.default]='vllm.moe_forward_shared.default'
        self.graph=RequestRowProvenance(max_calls=2048)
        self.source=SourceBoundRows(self.graph,self.model,bindings,sources,simulation=True,operator_aliases=aliases)
        self.source.routes=IntegerRouteObserver(self.source)
        self.source.route_frames=self.observer=RouteFrameObserver(self.source,type_bindings=TYPES)
        self.graph.boundaries=self.source
        self.rows={'request_ids':['request-'+str(i) for i in range(8)],'row_token_positions':[12]*8}
        self.addCleanup(self.cleanup)

    def cleanup(self):
        if self.graph.active:self.graph.end()
        self.assertIsNone(sys.getprofile())
        self.assertEqual(_get_current_dispatch_mode_stack(),[])

    def invoke(self):
        x,logits=torch.ones((8,4)),torch.ones((8,64))
        self.graph.begin('cpu-forward');self.source.begin(self.rows)
        self.graph.seed(x,dict(verified=True,actual_rows=self.rows,source='CPU_FIXTURE'))
        return self.model(x,logits)

    def record(self):
        return self.observer.report()['records'][0]

    def test_complete_actual_frames_and_scalar_epochs_never_bind_routed_rows(self):
        with torch.inference_mode():
            result=self.invoke();r=self.record()
            self.assertIsNone(r['first_failure'],r)
            self.assertTrue(r['python_chain_complete'],r)
            self.assertFalse(self.graph.query(result)['verified'])
            self.assertEqual([e['target_after']['write_epoch']-e['target_before']['write_epoch'] for e in r['consumers']],[1,1])
            self.assertEqual(self.model.actual,dict.fromkeys(self.model.actual,1))
            self.assertEqual(self.observer.report()['live_tensor_holds'],0)
            self.assertEqual(self.observer.report()['tensor_readbacks'],0)
            self.assertTrue(all(r[key] is False for key in ('native_registration_proven','routed_output_row_proof','native_qualified')))

    def refused(self,flag,pattern):
        setattr(self.model,flag,True)
        with torch.inference_mode():
            self.invoke();r=self.record()
            self.assertIn(pattern,r['first_failure'],r)
            self.assertFalse(r['python_chain_complete'])
            self.assertEqual(self.observer.report()['live_payload_holds'],0)
            self.assertEqual(self.model.actual,dict.fromkeys(self.model.actual,1))

    def test_nonidentity_prepare_remains_unknown(self):self.refused('replace_prepared','NONIDENTITY_PREPARATION')
    def test_equal_group_values_cannot_replace_builder_object(self):self.refused('replace_builder_group','BUILDER_OUTPUT_CHANGED')
    def test_matching_payload_fields_do_not_replace_current_class(self):self.refused('replace_payload_type','ACTUAL_PAYLOAD_TYPE_CHANGED')
    def test_no_version_gmm_write_epoch_is_not_reused(self):self.refused('change_mlp_epoch','OBJECT_OR_EPOCH_CHANGED:gmm1')
    def test_partial_alias_write_invalidates_parent_without_version_counter(self):self.refused('change_alias_epoch','OBJECT_OR_EPOCH_CHANGED:gmm1')
    def test_equal_values_gmm_result_is_not_original_producer(self):self.refused('replace_mlp_output','OBJECT_OR_EPOCH_CHANGED:gmm1')
    def test_finalizer_cannot_replace_return_by_same_values(self):self.refused('replace_finalized','OBJECT_OR_EPOCH_CHANGED:combined')
    def test_wrong_actual_scalar_cannot_use_source_default(self):self.refused('wrong_scalar','ACTUAL_SCALAR_ARGUMENT_UNPROVEN')
    def test_tensor_scalar_is_not_read_as_configuration(self):self.refused('tensor_scalar','ACTUAL_SCALAR_ARGUMENT_UNPROVEN')
    def test_same_shape_shared_operand_is_not_actual_return(self):self.refused('wrong_shared','OBJECT_OR_EPOCH_CHANGED:post_shared')
    def test_changed_config_owner_is_rejected(self):self.refused('change_scalar_owner','MODEL_OR_SCALAR_OWNER_CHANGED')
    def test_final_equal_values_return_cannot_replace_observed_view(self):self.refused('wrong_final_view','CURRENT_PRODUCER_MISSING:model_return')

    def test_original_exception_cleans_all_owned_frame_refs(self):
        self.model.original_failure=True
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError,'CPU_ACTUAL_GROUPED_MLP_FAILURE'):self.invoke()
            self.assertIsNotNone(self.record()['first_failure'])
            self.assertEqual(self.observer.report()['live_parents'],0)
            self.assertEqual(self.observer.report()['live_payload_holds'],0)

    def test_budget_refusal_does_not_repeat_or_skip_model(self):
        self.observer.max_frames=2
        with torch.inference_mode():
            self.invoke()
            self.assertIn('CALL_BUDGET_EXHAUSTED',self.record()['first_failure'])
            self.assertEqual(self.model.actual,dict.fromkeys(self.model.actual,1))

    def test_metadata_byte_budget_is_charged_before_retention(self):
        self.observer.max_bytes=128
        with torch.inference_mode():
            self.invoke()
            self.assertIn('METADATA_BUDGET_EXHAUSTED',self.record()['first_failure'])
            self.assertLessEqual(self.record()['metadata_bytes'],128)
            self.assertEqual(self.model.actual,dict.fromkeys(self.model.actual,1))

    def test_parent_budget_retains_overflow_and_all_original_calls(self):
        self.observer.max_parents=1
        with torch.inference_mode():
            self.invoke()
            self.model(torch.ones((8,4)),torch.ones((8,64)))
            report=self.observer.report()
            self.assertEqual((report['total_parents'],report['overflow'],len(report['records'])),(2,1,1))
            self.assertEqual(self.model.actual,dict.fromkeys(self.model.actual,2))
            self.assertEqual(report['live_parents'],0)

    def test_missing_gmm_frame_cannot_be_replaced_by_matching_raw_operator(self):
        self.source.bindings.pop(Device.second.__func__.__code__)
        with torch.inference_mode():
            self.invoke()
            self.assertFalse(self.record()['python_chain_complete'])
            self.assertIsNotNone(self.record()['first_failure'])
            self.assertEqual(self.model.actual,dict.fromkeys(self.model.actual,1))

    def test_training_context_records_refusal_without_readback_or_row_grant(self):
        self.invoke()
        self.assertIn('INFERENCE_REQUIRED',self.record()['first_failure'])
        self.assertEqual(self.source.routes.readback_bytes,0)
        self.assertEqual(self.observer.report()['live_payload_holds'],0)

    def test_loaded_classmethod_owner_change_is_detected(self):
        original=Device.second.__func__
        def replacement(cls,**kwargs):return original(cls,**kwargs)
        with torch.inference_mode():
            x,logits=torch.ones((8,4)),torch.ones((8,64))
            self.graph.begin('cpu-forward');self.source.begin(self.rows)
            with patch.object(Device,'second',classmethod(replacement)):
                self.model(x,logits)
                self.assertIn('FROZEN_FUNCTION_CHANGED',self.record()['first_failure'])
                self.assertEqual(self.model.actual,dict.fromkeys(self.model.actual,1))

    def test_report_is_detached_and_two_inference_forwards_release(self):
        with torch.inference_mode():
            for _ in range(2):
                self.invoke();r=self.record()
                copy=self.observer.report();copy['records'][0]['consumers'][0]['formals'].clear()
                self.assertEqual(self.record(),r)
                self.graph.end()
                self.assertEqual(self.observer.report()['live_parents'],0)


if __name__=='__main__':unittest.main()
