"""Review explicit all-rank recover, separately from fault/propagation evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from experiments.worker_core_v11_20261118.review_m1_rms18 import review as tensor_review
from experiments.worker_core_v11_20261118.review_m1_output39 import review as output_review


def need(value, reason):
    if not value:
        raise ValueError(reason)


def state_of(event):
    return event['response']['provider_state']


def lifecycle(rank_state):
    evidence = rank_state.get('evidence', {})
    return evidence if evidence.get('mechanism') == '39.output_proxy' else rank_state['serving']


def recovery_protocol(host):
    history = host['history']
    positions = [i for i, event in enumerate(history) if event['action'] == 'recover']
    need(len(positions) == 1, 'ONE_EXPLICIT_RECOVER_REQUIRED')
    index = positions[0]
    event = history[index]
    need(event['status'] == 'ACTION_COMPLETED', 'RECOVER_RPC_FAILED')
    state = state_of(event)
    need(state['next_phase'] == 'off' and not state['complete'], 'RECOVER_MUST_REQUIRE_OFF')
    need(set(state['rank_states']) == {'0', '1'}, 'RECOVER_MISSING_RANK')
    for rank_state in state['rank_states'].values():
        local = lifecycle(rank_state)
        rows = local['actual_requests']
        need([row['phase'] for row in rows] == ['normal', 'fault'], 'RECOVER_MUST_FOLLOW_ACTUAL_FAULT')
        need(local.get('next_phase', rank_state.get('next_phase')) == 'off'
             and not local['complete'] and local.get('recovered', local.get('matched_off_and_health_confirmed')) is False,
             'RANK_RECOVERY_NOT_RESET')
    advances = [state_of(e)['next_phase'] for e in history[index + 1:] if e['action'] == 'advance']
    need(advances == ['off', 'health'], 'RECOVERY_REQUEST_SEQUENCE')
    finish = [i for i, event in enumerate(history) if event['action'] == 'finish']
    need(len(finish) == 1 and finish[0] > index, 'RECOVERY_FINISH_REQUIRED')
    queries = [e for e in history[index + 1:finish[0]] if e['action'] == 'query']
    need(queries and queries[-1]['status'] == 'QUERY_OK', 'COMPLETED_QUERY_REQUIRED')
    terminal = state_of(queries[-1])
    need(terminal['complete'] and terminal['next_phase'] == 'complete', 'GROUP_PLAN_NOT_COMPLETE')
    need(set(terminal['rank_states']) == {'0', '1'}, 'TERMINAL_MISSING_RANK')
    for rank_state in terminal['rank_states'].values():
        local = lifecycle(rank_state)
        need(local.get('recovered', local.get('matched_off_and_health_confirmed')) is True,
             'RANK_RECOVERY_UNCONFIRMED')
        need(local['complete'] and local.get('next_phase', rank_state.get('next_phase')) == 'complete',
             'RANK_PLAN_NOT_COMPLETE')
    return dict(explicit_recover_after_fault=True, off_health_requests_after_recover=2,
                all_rank_terminal_phase='complete')


def review(data, hosts):
    need(data.get('explicit_group_recovery') is True, 'EXPLICIT_RECOVERY_CASE_REQUIRED')
    result = (output_review if data['mechanism'] == '39.output_proxy' else tensor_review)(data, hosts)
    result['recovery_protocol'] = [recovery_protocol(host) for host in hosts]
    result['mechanism'] = data['mechanism']
    result['recovery_test_scope'] = 'Explicit recover after successful F; no real worker crash injected.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    hosts = [json.loads((args.input.parent / ('scope' + str(i) + '_host.json')).read_bytes()) for i in range(2)]
    result = review(json.loads(args.input.read_bytes()), hosts)
    result.update(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  host_sha256=[hashlib.sha256((args.input.parent / ('scope' + str(i) + '_host.json')).read_bytes()).hexdigest() for i in range(2)],
                  reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['status'])
