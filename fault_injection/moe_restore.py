"""Source-bound entry to the actual opaque MoE combine, with independent census.

The ordinary model dispatch scope cannot see inside vllm.moe_forward. The
already reviewed real Python combine frame supplies a nested mode. Original
arguments are observed and ticketed before the mechanism's paired I/P remap.
No model method, custom-op registration or scheduler buffer is replaced.
"""
from .core import UnsupportedCall
from .moe_census import NaturalMoETrace, describe
from .tensor_observer import NaturalTensorObserver, metadata, tensors


def require(value, reason):
    if not value:
        raise UnsupportedCall(reason)


class MoERestoreTrace(NaturalMoETrace):
    def __init__(self, model, template, provider, adapter, live_stack, structural_paths):
        super().__init__(model, list(template.bindings.values()), template.sources,
                         simulation=provider.simulation)
        self.provider, self.adapter, self.live_stack = provider, adapter, live_stack
        self.structural_paths = tuple(structural_paths)
        self.request_id = provider.current_request_ids[0]
        self.producer_joins = []

    def _join(self, combine, arguments):
        path = combine['model_path']
        stack = self.live_stack()
        require(stack and stack[-1][0] == path, 'ACTUAL_MOE_MODULE_STACK_REQUIRED')
        require(path in self.structural_paths, 'ACTUAL_STRUCTURAL_MOE_OWNER_REQUIRED')
        rows = [r for r in self.rows if r['model_path'] == path and r['request_id'] == self.request_id]
        roles = {role: [r for r in rows if r['role'] == role] for role in
                 ('topk_producer', 'expert_pipeline', 'routing_producer', 'expert_mlp', 'native_combine')}
        require(all(len(value) == 1 for value in roles.values()), 'ONE_COMPLETE_MOE_PIPELINE_PER_MODULE_REQUIRED')
        topk, pipeline, routing, mlp, actual_combine = [roles[key][0] for key in roles]
        require(topk['returned'] and routing['returned'] and mlp['returned'] and actual_combine is combine,
                'ACTUAL_MOE_PRODUCER_RETURN_ORDER_REQUIRED')
        h, i, p = [metadata(arguments[name]) for name in ('permuted_tokens', 'sorted_indices', 'probs')]
        routed = routing['output']; cm = combine['input']['combine_metadata']
        route_input = routing['input']['token_dispatch_input']
        expert_input = pipeline['input']['fused_experts_input']
        require(topk['output'][0]['object_id'] == p['object_id'] == cm['topk_weights']['object_id']
            == routed['combine_metadata']['topk_weights']['object_id']
            == route_input['topk_weights']['object_id'] == expert_input['topk_weights']['object_id'],
            'ORIGINAL_TOPK_PROBABILITY_PRODUCER_IDENTITY_REQUIRED')
        require(topk['output'][1]['object_id'] == route_input['topk_ids']['object_id']
                == expert_input['topk_ids']['object_id'], 'ORIGINAL_EXPERT_ID_PRODUCER_REQUIRED')
        require(h['object_id'] == mlp['output']['object_id'] == combine['input']['hidden_states']['object_id']
            and mlp['input']['mlp_compute_input']['hidden_states']['object_id'] == routed['hidden_states']['object_id'],
            'ACTUAL_QUANTIZED_EXPERT_MLP_OUTPUT_REQUIRED')
        abs_rows = [row for row in self.native_rows if row['combine_sequence'] == combine['sequence']
                    and row['operator'] == 'aten.abs.default']
        require(len(abs_rows) == 1 and abs_rows[0]['input'][0]['object_id'] == cm['expanded_row_idx']['object_id']
            == routed['combine_metadata']['expanded_row_idx']['object_id']
            and abs_rows[0]['output'][0]['object_id'] == i['object_id'],
            'ACTUAL_ROUTING_INDEX_ABS_PRODUCER_REQUIRED')
        require(expert_input['routing']['apply_router_weight_on_input'] is False
                and route_input['routing']['apply_router_weight_on_input'] is False,
                'PROBABILITY_ALREADY_APPLIED_TO_HIDDEN_IS_UNADAPTED')
        require(topk['input']['top_k'] == p['shape'][1], 'REAL_TOPK_GEOMETRY_CHANGED')
        proof = dict(path=path, request_id=self.request_id, combine_sequence=combine['sequence'],
            pipeline_sequence=pipeline['sequence'], source_sequences=[r['sequence'] for r in
                (topk, pipeline, routing, mlp, combine)], original_probabilities_same_object=True,
            quantized_mlp_output_same_object=True, routing_abs_index_same_object=True,
            actual_topk=topk['input']['top_k'], input_metadata=[h, i, p])
        self.producer_joins.append(proof)
        return proof

    def _open_native(self, combine):
        from torch.utils._python_dispatch import TorchDispatchMode
        require(self.simulation or combine['function'].endswith('TokenDispatcherWithAllGather.token_combine'),
                'UNREVIEWED_NATURAL_MOE_COMMUNICATION_PATH')
        parent = self
        class Observe(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                schema = func._schema
                require(schema.name in ('aten::abs', 'npu::npu_moe_token_unpermute'),
                        'UNREVIEWED_ACTUAL_COMBINE_OPERATOR')
                bound = {arg.name: arg.default_value for arg in schema.arguments}
                bound.update(zip([arg.name for arg in schema.arguments], args)); bound.update(kwargs or {})
                native = dict(sequence=len(parent.native_rows), request_id=parent.request_id,
                    combine_sequence=combine['sequence'], model_path=combine['model_path'],
                    operator=str(func), schema=str(schema), arguments={k: describe(v) for k, v in bound.items()},
                    input=[metadata(t) for t in tensors((args, kwargs or {}))], original_calls=0)
                parent.native_rows.append(native)
                selected = schema.name == 'npu::npu_moe_token_unpermute'
                if selected:
                    native['producer_proof'] = parent._join(combine, bound)
                    # The denominator and group admission precede all I/P writes.
                    parent.provider._event(combine['model_path'], str(func), native['input'])
                    if parent.adapter is not None:
                        require(parent.adapter.stack and parent.adapter.stack[-1][0] == combine['model_path'],
                                'MECHANISM_AND_INDEPENDENT_MODEL_STACK_DISAGREE')
                class Original:
                    _schema = schema
                    def __str__(self):
                        return str(func)
                    def __call__(self, *call_args, **call_kwargs):
                        native['original_calls'] += 1
                        require(native['original_calls'] == 1, 'MOE_ORIGINAL_OPERATOR_REEXECUTION_REFUSED')
                        native['consumed_input'] = [metadata(t) for t in tensors((call_args, call_kwargs))]
                        consumed = {arg.name: arg.default_value for arg in schema.arguments}
                        consumed.update(zip([arg.name for arg in schema.arguments], call_args))
                        consumed.update(call_kwargs)
                        native['consumed_arguments'] = {name: describe(value) for name, value in consumed.items()}
                        # A visible outer mode must not intercept this same
                        # admitted native call again. Public collection stays on.
                        diagnostic = parent.adapter.diagnostic if parent.adapter else None
                        if parent.adapter:
                            parent.adapter.diagnostic = True
                        try:
                            native_result = func(*call_args, **call_kwargs)
                            native.update(output=[metadata(t) for t in tensors(native_result)], original_returned=True)
                            return native_result
                        finally:
                            if parent.adapter:
                                parent.adapter.diagnostic = diagnostic
                original = Original()
                previous_events = len(parent.adapter.events) if parent.adapter else 0
                result = (parent.adapter.dispatch(original, args, kwargs or {})
                          if selected and parent.adapter is not None else original(*args, **(kwargs or {})))
                require(native['original_calls'] == 1, 'ACTUAL_NATIVE_CONSUMER_REQUIRED')
                native.update(output=[metadata(t) for t in tensors(result)], original_returned=True)
                if selected and parent.adapter is not None and len(parent.adapter.events) > previous_events:
                    event = parent.adapter.events[-1]
                    event['actual_producer_join'] = native['producer_proof']
                    event['observed_original_native_calls'] = native['original_calls']
                return result
        return self._enter_native_mode(Observe())

    def report(self):
        result = super().report()
        for proof in self.producer_joins:
            combine = self.rows[proof['combine_sequence']]
            pipeline = self.rows[proof['pipeline_sequence']]
            native = [row for row in self.native_rows if row['combine_sequence'] == combine['sequence']
                      and row['schema'].startswith('npu::npu_moe_token_unpermute(')]
            good = (len(native) == 1 and combine['returned'] and pipeline['returned']
                and native[0]['output'][0]['object_id'] == combine['output']['object_id']
                == pipeline['output']['routed_out']['object_id'])
            proof['actual_routed_return_same_object'] = good
            result['complete'] = result['complete'] and good
        result.update(structural_moe_paths=list(self.structural_paths), producer_joins=self.producer_joins,
            mutations=sum(bool(e.get('applied')) for e in self.adapter.events) if self.adapter else 0,
            tensor_values_read=self.adapter is not None,
            complete_permutation_value_proven=bool(self.adapter) and all(
                e.get('complete_permutation_value_verified') for e in self.adapter.events),
            notes='Independent pre-remap observation and actual native consumer; mutation semantics in mechanism report')
        return result


class MoEPhaseObserver:
    def __init__(self, model, template, provider, adapter, structural_paths):
        self.observer = NaturalTensorObserver(model)
        self.trace = MoERestoreTrace(model, template, provider, adapter,
                                    lambda: self.observer.stack, structural_paths)

    def __enter__(self):
        self.observer.__enter__()
        try:
            self.trace.__enter__()
        except BaseException:
            self.observer.close()
            raise
        return self

    def close(self):
        try:
            self.trace.close()
        finally:
            self.observer.close()

    def report(self):
        row = self.observer.report(); trace = self.trace.report()
        row['moe_trace'] = trace; row['complete'] = row['complete'] and trace['complete']
        return row
