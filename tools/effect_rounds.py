"""Compare actual schedule composition, not HTTP batch size or random IDs."""
from tools.product_effect_probe import require, canonical


def signature(trace, tp):
    require(isinstance(trace,list) and len(trace)==tp, 'BATCH_ALL_RANK_TRACE_REQUIRED')
    require([r.get('rank') for r in trace]==list(range(tp)), 'BATCH_TRACE_RANK_MISMATCH')
    groups=[]
    for rank in trace:
        require(rank.get('methods_restored') is True and rank.get('truncated') is False,
                'BATCH_TRACE_INCOMPLETE')
        require(rank.get('source')=='ACTUAL_RUNNER_PREPROCESS_AND_LOGITS', 'BATCH_TRACE_SOURCE')
        require(rank.get('steps'), 'BATCH_TRACE_STEPS_REQUIRED')
        steps=[]
        for step in rank['steps']:
            rows=[]
            for row in step['rows']:
                require(row.get('request_id') and len(row.get('prompt_sha256',''))==64,
                        'BATCH_TRACE_REQUEST_IDENTITY_REQUIRED')
                for k in ('position','prompt_length','computed','scheduled'):
                    require(type(row.get(k)) is int and row[k]>=0,'BATCH_TRACE_ROW_INVALID')
                rows.append({k:row[k] for k in ('position','prompt_sha256','prompt_length','computed','scheduled')})
            require(rows,'BATCH_TRACE_EMPTY_STEP')
            steps.append(rows)
        groups.append(steps)
    require(all(canonical(g)==canonical(groups[0]) for g in groups),'BATCH_TRACE_RANKS_DISAGREE')
    return groups


def same_composition(a,b,tp):
    return canonical(signature(a,tp))==canonical(signature(b,tp))
