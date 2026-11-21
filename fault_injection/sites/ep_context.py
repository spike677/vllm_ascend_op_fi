"""Pure EP call context validation at an existing common control boundary.

No communication or second lifecycle. Runtime adapters must supply actual
placement and dynamic dispatch rows independently for every invocation.
"""
from copy import deepcopy


def validate_ep_boundary(*, members, owner, generation, request_ids, rank_calls, rank_placement):
    members=tuple(members);requests=set(request_ids)
    if (not members or len(set(members))!=len(members) or not owner or type(generation)!=int or generation<1
        or not requests or len(requests)!=len(request_ids)
        or set(rank_calls)!=set(members) or set(rank_placement)!=set(members)):
        raise ValueError('COMPLETE_EP_CONTROL_BOUNDARY_REQUIRED')
    errors=[];seen_routes=set();counts={}
    for rank in members:
        # Physical slots are per rank; logical experts may have replicas (EPLB).
        placement=rank_placement[rank]
        slots={}
        for row in placement:
            slot=row['physical_slot'];logical=row['logical_expert']
            if type(slot)!=int or type(logical)!=int or min(slot,logical)<0 or slot in slots:
                raise ValueError('INVALID_ACTUAL_EXPERT_PLACEMENT')
            slots[slot]=logical
        counts[rank]=len(rank_calls[rank])
        for call in rank_calls[rank]:
            if call.get('owner')!=owner or call.get('generation')!=generation:
                errors.append(f'rank{rank}:STALE_SCOPE');continue
            if call.get('request_id') not in requests:
                errors.append(f'rank{rank}:UNKNOWN_REQUEST');continue
            if slots.get(call.get('physical_slot'))!=call.get('logical_expert'):
                errors.append(f'rank{rank}:EXPERT_PLACEMENT_MISMATCH');continue
            token=call.get('token_position');route=call.get('route_ordinal');invocation=call.get('invocation')
            if any(type(v)!=int or v<0 for v in (token,route,invocation)):
                errors.append(f'rank{rank}:INVALID_DYNAMIC_ROUTE');continue
            # The source EP dispatch identity, not a rank-local sequential ID.
            identity=(call['request_id'],token,invocation,route)
            if identity in seen_routes:errors.append(f'rank{rank}:DUPLICATE_DISPATCH_ROUTE')
            seen_routes.add(identity)
    return deepcopy(dict(passed=not errors,failures=errors,actual_calls_per_rank=counts,
                         no_call_ranks=[r for r in members if not rank_calls[r]],
                         qualification_granted=False,placement_source='RUNTIME_SUPPLIED_NOT_CONFIG_INFERRED'))
