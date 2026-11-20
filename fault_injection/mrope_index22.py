"""Private Fault22 boundary for the three-axis mRoPE cache read.

The trusted source-bound attention adapter must supply the actual cache,
positions and current-request identity. This component grants no deployment
permission and is not registered as a public mechanism/provider.
"""
from .core import UnsupportedCall


def call_mrope_index22(original, cache, positions, *, request_positions,
                       expected_positions, arm, admit, simulation=False):
    import torch
    def need(ok, reason):
        if not ok:raise UnsupportedCall(reason)
    need(arm in ('baseline','fault','off') and type(simulation) is bool,
         'MROPE22_EXPLICIT_ARM_REQUIRED')
    need(callable(original) and callable(admit),'MROPE22_CURRENT_CALL_BOUNDARY_REQUIRED')
    need(positions is request_positions and positions.ndim==2 and positions.shape[0]==3
         and positions.shape[1]>=2 and positions.stride(1)==1
         and positions.stride(0)>=positions.shape[1]
         and positions.dtype in (torch.int32,torch.int64),
         'MROPE22_ACTUAL_THREE_AXIS_REQUEST_POSITIONS_REQUIRED')
    need(cache.ndim==2 and cache.shape[1]>0 and cache.shape[1]%2==0 and cache.is_contiguous()
         and cache.dtype in (torch.float16,torch.bfloat16,torch.float32) and cache.device==positions.device
         and (simulation or cache.device.type=='npu'),'MROPE22_ACTUAL_CACHE_GEOMETRY_REQUIRED')
    p=positions.detach().cpu().clone();before=cache.detach().cpu().clone()
    need(p.tolist()==expected_positions and int(p.min())>=0 and int(p.max())<cache.shape[0],
         'MROPE22_REQUEST_MAPPING_OR_LEGAL_RANGE')
    need(all(int(p[axis,0])!=int(p[axis,-1]) for axis in range(3)),
         'MROPE22_DISTINCT_POSITION_ROWS_REQUIRED')
    correct=before[p.long()]
    need(not torch.equal(correct[:,-1],correct[:,0]),'MROPE22_DISTINCT_CACHE_ROWS_REQUIRED')
    selected=arm=='fault'
    if selected:need(admit() is True,'MROPE22_LIVE_OWNER_ADMISSION_REQUIRED')
    native_positions=positions
    expected=p.clone()
    if selected:
        native_positions=positions.clone()
        native_positions[:,-1].copy_(positions[:,0])
        expected[:,-1]=expected[:,0]
        need(native_positions.untyped_storage()._cdata!=positions.untyped_storage()._cdata
             and torch.equal(native_positions.detach().cpu(),expected),
             'MROPE22_INDEPENDENT_INDEX_READBACK_REQUIRED')
    # This is the actual cache-index operation, exactly once; no return rewrite.
    output=original(cache,[native_positions])
    after=output.detach().cpu()
    need(torch.equal(positions.detach().cpu(),p) and torch.equal(cache.detach().cpu(),before),
         'MROPE22_ORIGINAL_POSITION_OR_CACHE_CHANGED')
    need(output.dtype==cache.dtype and output.device==cache.device
         and tuple(output.shape)==(3,positions.shape[1],cache.shape[1])
         and torch.equal(after,before[expected.long()]),'MROPE22_ACTUAL_INDEX_RESULT_DIFFERS')
    need(torch.equal(after[:,:-1],correct[:,:-1]),'MROPE22_NON_TARGET_CACHE_ROWS_CHANGED')
    changed=int(torch.count_nonzero(after[:,-1]!=correct[:,-1]).item())
    need((changed>0)==selected,'MROPE22_CACHE_ROW_EFFECT_NOT_OBSERVED')
    return output,dict(boundary='aten.index.Tensor / actual mRoPE cache',
        variant='THREE_AXIS_LEGAL_POSITION_LAST_FROM_FIRST',native_calls=1,
        original_positions_unchanged=True,original_cache_unchanged=True,
        non_target_rows_unchanged=True,changed_cache_elements=changed,
        original_position=p[:,-1].tolist(),replacement_position=expected[:,-1].tolist(),
        applied=selected,simulation=simulation,qualification_granted=False,
        fused_qk_effect_verified=False,request_mapping_requires_source_bound_observer=True,
        output_rewritten=False)
