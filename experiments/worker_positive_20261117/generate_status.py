"""Apply only this round's evidenced deltas to the retained 17-mechanism audit."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = 'dcbef7a027bbc39e7796f12880305c3fa516c1f8'


def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


def main():
    previous_path = ROOT/'experiments/overnight_20261116/fault_matrix.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    rows = []
    for old in previous['mechanisms']:
        rows.append(dict(mechanism=old['mechanism'],
            previous_current_level=old['current_level'],
            previous_result=old['result']['conclusion'],
            change='UNCHANGED_FROM_DCBef7a0', common_worker_all_eligible=False,
            new_common_worker_fault_verified=False,
            remaining=old['gate'], next_action=old['next_action']))
    by_id = {row['mechanism']: row for row in rows}
    assert len(rows) == len(by_id) == 17
    for name in ('fault20-fault', 'fault20-smollm', 'fault39-qwen', 'fault39-final-build'):
        result = read('summaries/' + name + '.json')
        assert result['status'] == 'PASS_COMMON_WORKER_NATIVE_BFOFF'
        assert result['qualification']['recovery_verified'] is True
        assert all(row['actual_model_request_matches_enqueue'] for row in result['phases'])
    for name in ('fault20', 'fault39'):
        summary = read('summaries/' + name + '-independent-statistics.json')
        pair = summary['comparison']
        assert pair['native_logs_read'] is pair['private_tensors_read'] is False
        comparison = pair['comparison']
        assert comparison['status'] == 'FIRST_OBSERVABLE_STATISTICS_DIFFERENCE'
        assert comparison['identity'] == 'Functional.linear.112.forward/output/0'
        assert comparison['comparison_policy'] == dict(absolute_tolerance=1e-8,
            relative_tolerance=1e-5, rule='abs(F-B) > atol + rtol*abs(B)')
    by_id['20.buffer_substitute'].update(change='SCOPED_COMMON_WORKER_WEIGHTNZ_BFOFF_VERIFIED',
        new_common_worker_fault_verified=True,
        scope=['Qwen3-0.6B', 'SmolLM2-135M / actual Llama', 'TP1 BF16 eager fresh prefill',
               'one natural output projection', 'explicit NZ / controlled PTA deployment'],
        evidence=['summaries/fault20-fault.json', 'summaries/fault20-smollm.json',
                  'summaries/fault20-independent-statistics.json', 'summaries/qmm-readonly.json'],
        remaining=['DEFAULT_PTA_PATH_UNQUALIFIED', 'QMM_FAULT_QUALIFICATION_PENDING',
                   'MULTI_TARGET_UNQUALIFIED', 'OTHER_DEPLOYMENTS_UNVERIFIED'],
        next_action='Give an actual quantized intermediate module its own input/output/consumer contract; '
                    'qualify QMM separately without relabeling the FLOAT output head.')
    by_id['39.matmul_range'].update(change='SCOPED_COMMON_WORKER_MATMUL_BFOFF_VERIFIED',
        new_common_worker_fault_verified=True,
        scope=['Qwen3-0.6B', 'TP1 BF16 ND eager fresh prefill',
               'one natural output projection', 'CANN 8.5.1 / observed 910B4'],
        evidence=['summaries/fault39-qwen.json', 'summaries/fault39-final-build.json',
                  'summaries/fault39-independent-statistics.json'],
        remaining=['MULTI_TARGET_UNQUALIFIED', 'OTHER_PRETRAINED_MODEL_FAULTS_UNVERIFIED',
                   'OTHER_COMMON_WORKER_VERSION_COMBINATIONS_UNVERIFIED'],
        next_action='Qualify other actual models and runtime/build combinations; keep old native-version '
                    'evidence separate and define multi-target recovery before all_eligible.')
    models = {}
    for model in ('glm', 'deepseek'):
        result = read('summaries/' + model + '-admission.json')
        assert result['status'] == 'PASS_PRETRAINED_NPU_INFERENCE_NOT_FAULT_ACCEPTANCE'
        assert result['actual_fault_injection'] is False
        models[model] = dict(status=result['status'], actual_model_class=result['actual_model_class'],
            actual_model_type=result['actual_model_type'], native_fault_qualified=False,
            evidence='summaries/' + model + '-admission.json')
    out = dict(schema='demoncat.worker_positive.mechanism_delta.v1', baseline=BASELINE,
        previous_matrix_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),
        new_common_worker_positive_mechanisms=2, all_eligible_qualified_mechanisms=0,
        new_raw_statistics_published=False, models=models, mechanisms=rows,
        retained_failures=['15/17/18 strict ALIGNMENT_MISMATCH',
            '26 residual rank1 pre-write NaN and allocation initialization unproven',
            'Earlier same-Worker different-phase KV-slot statistics first difference'],
        source_of_truth='Actual scoped receipts; family names and historical qualification are not inherited')
    (HERE/'mechanism_delta.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
