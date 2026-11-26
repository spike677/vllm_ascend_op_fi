"""One service, two owners, baseline/fault/off. Trusted local adapter, no grant.

The adapter owns service authentication and maps public/external/HCCL controls.
It must not manufacture evidence for a control path it cannot observe.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

COUNTERS = ('observed', 'eligible', 'attempted', 'applied', 'changed',
            'consumed', 'restored', 'failed')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, code):
    if not condition:
        raise ValueError(code)


def api_values(reply):
    """Preserve returned tokens/logprobs exactly; never label top5 as full logits."""
    out = []
    for choice in sorted(reply['choices'], key=lambda c: c['index']):
        lp = choice.get('logprobs') or {}
        if 'content' in lp:  # OpenAI chat shape
            tokens = [r['token'] for r in lp['content']]
            values = [r['logprob'] for r in lp['content']]
            top = [{x['token']: x['logprob'] for x in r['top_logprobs']} for r in lp['content']]
        else:
            tokens, values, top = lp.get('tokens'), lp.get('token_logprobs'), lp.get('top_logprobs')
        require(isinstance(tokens, list) and tokens and len(tokens) == len(values or []) == len(top or []),
                'API_TOKEN_LOGPROBS_REQUIRED')
        for position,t in enumerate(top):
            require(isinstance(t,dict) and len(t)>=5,
                    'API_TOP5_REQUIRED:choice='+str(choice['index'])+':position='+str(position)
                    +':actual='+str(len(t) if isinstance(t,dict) else None))
        numbers = values + [v for t in top for v in t.values()]
        require(all(type(v) in (int, float) and math.isfinite(v) for v in numbers), 'API_LOGPROBS_NONFINITE')
        out.append(dict(tokens=tokens, token_logprobs=values, top_logprobs=top,
                        text=choice.get('text', choice.get('message', {}).get('content'))))
    require(out, 'API_CHOICES_REQUIRED')
    return out


def compare(before, after):
    b, a = api_values(before), api_values(after)
    require(len(b) == len(a), 'API_REQUEST_COUNT_CHANGED')
    return dict(tokens_changed=any(x['tokens'] != y['tokens'] for x, y in zip(b, a)),
                text_changed=any(x['text'] != y['text'] for x, y in zip(b, a)),
                top5_changed=any(canonical(x['top_logprobs']) != canonical(y['top_logprobs']) for x, y in zip(b, a)),
                exact=canonical(b) == canonical(a), baseline_sha256=sha(canonical(b)), after_sha256=sha(canonical(a)))


def run(adapter, prompts, *, temperature=0, max_tokens=4, max_off_attempts=4, checkpoint=lambda raw: None):
    """Factory returns a trusted deployment adapter (see product_effect_test.md).

    Each new_session creates a new owner on the SAME service/Worker group.
    No per-user request registration, process creation or permission installation.
    """
    require(prompts and 0 <= temperature and max_tokens > 0, 'INVALID_INPUT')
    # Completion top_logprobs is a dictionary. Distinct vocabulary IDs can
    # decode to the same string (including incomplete byte tokens), collapsing
    # its keys. Ask vLLM for its lossless token-ID representation in every round;
    # do not relax the top-5 or finite-value checks when a fault exposes this.
    sampling = dict(temperature=temperature, seed=42, logprobs=5, max_tokens=max_tokens,
                    return_tokens_as_token_ids=True)
    raw = dict(schema='dcat.effect.raw.v1', identity=adapter.identity(), prompts=prompts,
               sampling=sampling, scopes=[], control_sequence=[], qualification_granted=False)
    raw['tool_sha256'] = sha(Path(__file__).read_bytes())
    batch=raw['identity'].get('mechanism')=='quant.batch_shared_scale'
    require(type(max_off_attempts) is int and 1<=max_off_attempts<=8,'OFF_ATTEMPT_BUDGET_INVALID')
    require(not batch or callable(getattr(adapter,'infer_observed',None)), 'BATCH_ROUND_OBSERVER_REQUIRED')
    for index in range(2):
        session = adapter.new_session()
        scope = dict(index=index, owner=session.owner, worker_identity=session.worker_identity,
                     controls=[], outputs={}, attempts=[])
        raw['scopes'].append(scope)

        def control(name):
            sequence = len(raw['control_sequence']) + 1
            row = dict(sequence=sequence, owner=session.owner, action=name)
            raw['control_sequence'].append(row); scope['controls'].append(sequence)
            try:
                row['result'] = getattr(session, name)()
            except Exception as error:
                row['error'] = dict(type=type(error).__name__, reason=str(error))
                raise
            finally:
                checkpoint(raw)
            return row['result']

        def inference(phase):
            attempt=dict(phase=phase,index=sum(x['phase']==phase for x in scope['attempts']))
            scope['attempts'].append(attempt)
            try:
                if callable(getattr(adapter,'infer_observed',None)):
                    observed=adapter.infer_observed(prompts,sampling)
                    attempt.update(output=observed['output'],trace=observed['trace'])
                else:
                    attempt['output']=adapter.infer(prompts,sampling)
                scope['outputs'][phase]=attempt['output']
            except Exception as error:
                attempt['error']=dict(type=type(error).__name__,reason=str(error))
                raise
            finally:
                checkpoint(raw)
            return attempt

        control('attach')
        try:
            baseline=inference('baseline')
            api_values(scope['outputs']['baseline'])
            control('enable')
            inference('fault')
            scope['observed'] = control('status')
        finally:
            # An uncertain enable is still the adapter's recovery responsibility.
            scope['disabled'] = control('disable')
        try:
            for _ in range(max_off_attempts if batch else 1):
                off=inference('off')
                if batch:
                    from tools.effect_rounds import same_composition
                    off['composition_matches_baseline']=same_composition(baseline['trace'],off['trace'],raw['identity']['tp'])
                    checkpoint(raw)
                    if not off['composition_matches_baseline']:
                        continue
                break
            if batch:
                require(off['composition_matches_baseline'],'BATCH_COMPOSITION_RETRY_EXHAUSTED')
        finally:
            scope['detached'] = control('detach')
        scope['evidence'] = session.evidence()  # facts + paths into original receipts
        try:
            scope['comparison'] = dict(fault=compare(scope['outputs']['baseline'], scope['outputs']['fault']),
                                       off=compare(scope['outputs']['baseline'], scope['outputs']['off']))
        except ValueError as error:
            scope['api_error']=str(error)
            checkpoint(raw)
            raise
        checkpoint(raw)
    return raw


def summarize(raw, raw_bytes):
    require(raw.get('schema') == 'dcat.effect.raw.v1', 'RAW_SCHEMA')
    scopes = raw.get('scopes', [])
    require(len(scopes) == 2 and len({s['owner'] for s in scopes}) == 2, 'TWO_DISTINCT_OWNERS_REQUIRED')
    require(scopes[0]['worker_identity'] == scopes[1]['worker_identity'], 'WORKERS_CHANGED')
    identity = raw['identity']; tp = identity.get('tp')
    require(type(tp) is int and tp > 0, 'TP_REQUIRED')
    rows = []; issues = []
    from tools.extract_registration_record import DEPLOYMENT
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_identity import registration_identity_issues
    issues.extend(registration_identity_issues(identity))
    issues.extend('IDENTITY_MISSING:' + k for k in DEPLOYMENT if k != 'image_digest' and identity.get(k) is None)
    for s in scopes:
        if identity.get('mechanism')=='quant.batch_shared_scale':
            from tools.effect_rounds import same_composition, signature
            attempts=s.get('attempts',[])
            base=[a for a in attempts if a['phase']=='baseline']
            faults=[a for a in attempts if a['phase']=='fault']
            offs=[a for a in attempts if a['phase']=='off']
            require(len(base)==len(faults)==1 and 1<=len(offs)<=8,'BATCH_ROUND_ATTEMPTS_REQUIRED')
            signature(faults[0]['trace'],tp)
            matches=[same_composition(base[0]['trace'],a['trace'],tp) for a in offs]
            require(matches[-1] and not any(matches[:-1]),'BATCH_RETRY_MUST_STOP_AT_FIRST_MATCH')
            require(all(canonical(s['outputs'][phase])==canonical(a['output']) for phase,a in
                        [('baseline',base[0]),('fault',faults[0]),('off',offs[-1])]),'BATCH_SELECTED_REPLY_MISMATCH')
        e = s['evidence']
        require(sorted(r['rank'] for r in e['ranks']) == list(range(tp)), 'EXACT_RANK_SET_REQUIRED')
        packed = []
        for r in sorted(e['ranks'], key=lambda x: x['rank']):
            c = r['counts']
            require(all(type(c.get(k)) is int and c[k] >= 0 for k in COUNTERS), 'EIGHT_COUNTERS_REQUIRED')
            require(c['failed'] == 0 and c['eligible'] > 0 and c['observed'] >= c['eligible']
                    and c['eligible'] == c['attempted'] == c['applied'] == c['consumed'] == c['restored']
                    and c['changed'] <= c['applied'], 'COUNTER_CONTRADICTION')
            require(r['owner'] == s['owner'] and r.get('request_ids') and r.get('evidence_paths'), 'RANK_OWNER_OR_PROVENANCE_MISSING')
            require(isinstance(r.get('provider_owner'), str) and r['provider_owner'], 'PROVIDER_OWNER_MISSING')
            require(type(r.get('changed_elements')) is int and r['changed_elements'] >= 0, 'CHANGED_ELEMENTS_MISSING')
            require(all(r.get(k) is True for k in ('consumer', 'bytes_restored', 'resources_zero', 'canary_exact')), 'RECOVERY_OR_CONSUMER_UNPROVEN')
            # Raw paths, request IDs and full proofs remain in the hashed original.
            packed.append([r['rank'], *[c[k] for k in COUNTERS], r['changed_elements']])
        f = compare(s['outputs']['baseline'], s['outputs']['fault'])
        off = compare(s['outputs']['baseline'], s['outputs']['off'])
        require(off['exact'], 'OFF_API_NOT_EXACT')
        rows.append(dict(owner=s['owner'], provider_owners=[r.get('provider_owner') for r in sorted(e['ranks'],key=lambda x:x['rank'])], seq=s['controls'], ranks=packed,
                         effect=[f['tokens_changed'], f['top5_changed'], f['text_changed'], e.get('logits_changed')],
                         recovery=[True, True, True], off_exact=off['exact']))
    out = dict(schema='dcat.effect.page.v1', identity=identity, scopes=rows,
               rank_columns=['rank', *COUNTERS, 'changed_elements'],
               effect_columns=['tokens_changed', 'top5_changed', 'text_changed', 'full_logits_changed'],
               recovery_columns=['bytes_restored', 'resources_zero', 'canary_exact'],
               seq_source='HOST_CONTROL_SEQUENCE_NOT_WIRE_RPC_PROOF',
               tool_sha256=raw.get('tool_sha256'),
               original=dict(bytes=len(raw_bytes), sha256=sha(raw_bytes)),
               identity_issues=issues, registration_ready=not issues, qualification_granted=False)
    out['extract_sha256'] = sha(canonical(out))
    require(len(canonical(out)) + 1 <= 4096, 'PAGE_EXCEEDS_4096_NO_TRUNCATION')
    return out


def verify(page):
    value = dict(page); digest = value.pop('extract_sha256', None)
    require(sha(canonical(value)) == digest, 'EXTRACT_SHA_MISMATCH')
    require(len(canonical(page)) + 1 <= 4096, 'PAGE_EXCEEDS_4096_NO_TRUNCATION')
    require(page['schema'] == 'dcat.effect.page.v1', 'PAGE_SCHEMA')
    require(len(page['scopes']) == 2 and len({s['owner'] for s in page['scopes']}) == 2, 'TWO_OWNERS_REQUIRED')
    require(not set(page['scopes'][0]['provider_owners']).intersection(page['scopes'][1]['provider_owners']), 'PROVIDER_OWNER_REUSED')
    from tools.extract_registration_record import DEPLOYMENT
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_identity import registration_identity_issues
    issues = registration_identity_issues(page['identity'])
    issues.extend('IDENTITY_MISSING:' + k for k in DEPLOYMENT if k != 'image_digest' and page['identity'].get(k) is None)
    require(issues == page['identity_issues'] and page['registration_ready'] == (not issues), 'IDENTITY_CONCLUSION_CONTRADICTORY')
    for s in page['scopes']:
        require(len(s['provider_owners'])==page['identity']['tp'] and all(isinstance(x,str) and x for x in s['provider_owners']), 'PROVIDER_OWNER_MISSING')
        require([r[0] for r in s['ranks']] == list(range(page['identity']['tp'])), 'EXACT_RANK_SET_REQUIRED')
        for r in s['ranks']:
            require(len(r) == 10 and all(type(v) is int and v >= 0 for v in r), 'INVALID_COUNTER_ROW')
            _, observed, eligible, attempted, applied, changed, consumed, restored, failed, elements = r
            require(observed >= eligible > 0 and eligible == attempted == applied == consumed == restored
                    and changed <= applied and failed == 0, 'COUNTER_CONTRADICTION')
        require(s['recovery'] == [True, True, True] and s['off_exact'] is True, 'RECOVERY_UNPROVEN')
    return dict(extract_sha256=digest, integrity_valid=True, registration_ready=page['registration_ready'],
                qualification_granted=False)


def run_negatives(adapter):
    """One deployment-wide run; an authenticated deployment supplies operations.

    Existing enabled session must stay intact on duplicate enable. A final
    cleanup audit is separate from that intermediate assertion.
    """
    rows=[]
    for name in ('missing_permit','unknown_name','already_enabled','repeat_disable'):
        r=adapter.check_negative(name)
        require(r.get('name')==name and r.get('engine_alive') is True, 'NEGATIVE_ENGINE_OR_ASSOCIATION')
        allowed=(0,4) if name=='repeat_disable' else (4,)
        require(r.get('exit_code') in allowed and (r['exit_code']!=4 or r.get('code')), 'NEGATIVE_EXIT_OR_CODE')
        require(r.get('resources_zero_after_cleanup') is True and r.get('positive_control') is True, 'NEGATIVE_CLEANUP_OR_POSITIVE')
        if name=='already_enabled':require(r.get('original_session_unchanged') is True, 'DUPLICATE_ENABLE_CHANGED_SESSION')
        require(r.get('original_receipt_sha256'), 'NEGATIVE_ORIGINAL_SHA_REQUIRED')
        rows.append(r)
    return dict(schema='dcat.effect.negatives.v1',cases=rows,qualification_granted=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter'); p.add_argument('--adapter-sha256'); p.add_argument('--config')
    p.add_argument('--output'); p.add_argument('--verify'); p.add_argument('--summarize')
    p.add_argument('--negatives',action='store_true')
    a = p.parse_args()
    if a.verify:
        print(canonical(verify(json.loads(Path(a.verify).read_bytes()))).decode()); return
    if a.summarize:
        data = Path(a.summarize).read_bytes()
        print(canonical(summarize(json.loads(data), data)).decode()); return
    require(a.adapter and a.adapter_sha256 and a.config and a.output, 'TRUSTED_ADAPTER_CONFIG_OUTPUT_REQUIRED')
    path = Path(a.adapter); require(sha(path.read_bytes()) == a.adapter_sha256, 'ADAPTER_SHA_MISMATCH')
    spec = importlib.util.spec_from_file_location('trusted_effect_adapter', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    config = json.loads(Path(a.config).read_bytes()); target = Path(a.output)
    require(not target.exists(), 'OUTPUT_EXISTS')
    adapter=module.create_adapter(config)
    if a.negatives:
        raw=run_negatives(adapter);data=canonical(raw)+b'\n';target.write_bytes(data)
        print(canonical(dict(schema=raw['schema'],passed=True,original_sha256=sha(data),bytes=len(data))).decode());return
    raw = run(adapter, config['prompts'], temperature=config.get('temperature', 0),
              max_tokens=config.get('max_tokens', 4), checkpoint=lambda value: target.write_bytes(canonical(value) + b'\n'))
    data = canonical(raw) + b'\n'; target.write_bytes(data)
    print(canonical(summarize(raw, data)).decode())


if __name__ == '__main__':
    main()
