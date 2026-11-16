"""Statistics-only first difference, followed by separate native acceptance."""
import json
from pathlib import Path
import sys

ROOT=Path('/home/provider-adaptation-20261116')
sys.path.insert(0,str(ROOT/'source-a5/fault39/tools'))
from compare_model_capture import compare_groups

rows=[]
for tag in ('851','90','91','92'):
    root=ROOT/f'npu-{tag}-a3'
    # Select the first difference without loading any native/control result.
    independent=compare_groups(root/'operator_baseline/baseline',root/'operator_fault/fault')
    with (root/'independent_first_difference.json').open('x') as f:json.dump(independent,f,indent=2)
    b=json.loads((root/'operator_baseline/result.json').read_text())
    f=json.loads((root/'operator_fault/result.json').read_text())
    rows.append(dict(version=f['initial_native']['cann_build_version'],
        status_b=b['status'],status_f=f['status'],profile=f['initial_native']['build_profile_sha256'],
        off_exact=all(x['off_recovery']['baseline_bitwise_equal'] for x in (b,f)),
        input_unchanged=all(x['inputs']['full_storage_bitwise_unchanged'] for x in (b,f)),
        verdict=f['verdict'],statistics_status=independent['status'],
        first_difference=independent['first_difference']))
with (ROOT/'fault39-npu-summary-a3.json').open('x') as f:json.dump(rows,f,indent=2)
print(json.dumps(rows,indent=2))
