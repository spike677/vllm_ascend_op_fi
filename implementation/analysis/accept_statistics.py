"""Apply the pre-registered acceptance gate AFTER independent localization."""
import argparse
import json
import re

RTOL = 1e-5
ATOL = 1e-8
MIN_RELATIVE = 1e-3
FIA = re.compile(r'^NPU\.npu_fused_infer_attention_score\.\d+\.forward$')


def accept(report):
    first = report.get('first_observed_api_difference')
    contract = report.get('comparison', {})
    reasons = []
    if contract.get('rtol') != RTOL or contract.get('atol') != ATOL or contract.get('reference') != 'baseline':
        reasons.append('FIXED_TOLERANCE_CONTRACT_MISMATCH')
    if report.get('status') != 'ALIGNED_ABOVE_TOLERANCE_API_DIFFERENCE' or not first:
        reasons.append('NO_ALIGNED_ABOVE_TOLERANCE_FIRST_DIFFERENCE')
    is_fia = bool(first and FIA.fullmatch(first['B_node']) and
                  FIA.fullmatch(first['F_node']) and first['boundary'] == 'output')
    if not is_fia:
        reasons.append('FIRST_DIFFERENCE_NOT_FIA_OUTPUT')
    fields = [d for d in ((first or {}).get('field_differences', []) if is_fia else [])
              if d['path'] in ('$[0].Max', '$[0].Min', '$[0].Norm') and
              d.get('exceeds_tolerance') and d.get('relative_delta_B') is not None]
    qualified = [d for d in fields if d['relative_delta_B'] >= MIN_RELATIVE]
    if not qualified:
        reasons.append('FIRST_FIA_RELIABLE_STATISTIC_BELOW_0_1_PERCENT')
    return {'schema':'fault28.visible.statistics_acceptance.v1',
            'status':'PASS_STATISTICS_GATE' if not reasons else 'NOT_ACCEPTED',
            'accepted':not reasons,'reasons':reasons,
            'minimum_relative_change':MIN_RELATIVE,'first_difference':first,
            'qualifying_fields':qualified,
            'best_reliable_relative_change':max((d['relative_delta_B'] for d in fields),default=0.0),
            'unobservable_fields':report.get('unobservable_fields'),
            'tensor_equality_proven':False,'physical_root_cause_proven':False}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    r=accept(json.load(open(a.report,encoding='utf-8')))
    with open(a.output,'x',encoding='utf-8') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(r));raise SystemExit(0 if r['accepted'] else 2)
