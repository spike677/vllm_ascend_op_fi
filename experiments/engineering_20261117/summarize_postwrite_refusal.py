"""Summarize an actual refused pretrained run without publishing tensors or traces."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--remote-result', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    raw = a.result.read_bytes()
    result = json.loads(raw)
    state = result['provider_diagnostics']
    assert result['status'] == 'FAILED' and result['pretrained_model'] is True
    assert not state['qualified'] and not state['attempted']
    assert state['leases'] == 0 and not state['recovery_pending']
    observations = []
    for row in state['qualification_diagnostics']['observations']:
        diag = row['discovery_diagnostics']
        counts = diag['operator_counts']
        assert row['observed_candidates'] == 0 and row['eligible_with_consumers'] == 0
        assert counts.get('aten.mm.default', 0) == 0
        observations.append(dict(phase=row['phase'], original_operator_calls=row['original_operator_calls'],
            eligible_candidates=0, eligible_with_consumers=0, operator_counts=counts,
            intervention=row['intervention']))
    summary = dict(schema='demoncat.postwrite.pretrained_refusal.v1',
        mechanism='39.output_proxy', status='PRETRAINED_PREFILL_REFUSED_WITHOUT_INJECTION',
        model='Qwen3-0.6B', scope=result['scope'], device=result['actual_device'],
        torch=result['torch'], torch_npu=result['torch_npu'], error=result['error'],
        result=dict(path=a.remote_result, sha256=hashlib.sha256(raw).hexdigest()),
        source_sha256=result['source_sha256'], library_sha256=result['library_sha256'],
        observations=observations, leases=0, recovery_pending=False,
        valid_bf_pair=False, independent_first_difference=None,
        public_worker_qualified=False, all_eligible=False, every_match=False,
        conclusion='The admitted aten.mm producer was not observed in this actual HF inference path. '
            'aten.linear/matmul require separate producer and consumer qualification; no support is inferred.',
        next_step='Implement and qualify a distinct actual aten.linear/matmul adapter before any injection.')
    a.out.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=summary['status'], phases=len(observations))))


if __name__ == '__main__':
    main()
