"""Fresh final-package raw normal observations, never serialized F authority."""
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

from .paged33_s2_evidence import (_module,normal_observer_root,normalized_input,digest,VerifiedNormalEvidence)
from .serving_context import require

_MINT=object()
VARIANT='tnd_page16_arch22_group_s2_256'
CONTROL_PHASES=('reference','lossless','qualification_off')
PHASES=CONTROL_PHASES+tuple(name+f'_{index:02d}' for index in range(28) for name in ('fault','off'))+('cleanup_health',)


def payload_signatures(capture):
    values={key:capture[key] for key in ('hidden','logits_hidden','logits')}
    values.update({path+'/'+key:value for path,items in capture['attention'].items() for key,value in items.items()})
    require(len(values)==87,'GROUP_COMPLETE_87_PAYLOADS_REQUIRED')
    return {name:{key:value[key] for key in ('shape','dtype','bytes','sha256')} for name,value in values.items()}


def cost_gate(normal_seconds,case_seconds,*,initialization_seconds=0,ordinary_health_seconds=0,
              history_validation_seconds=None,constant_seconds=None):
    require(type(normal_seconds) in (float,int) and math.isfinite(normal_seconds) and normal_seconds>0
        and type(case_seconds) is int and 1<=case_seconds<=1800,'GROUP_MEASURED_NORMAL_AND_FIXED_CASE_BUDGET_REQUIRED')
    require(all(type(n) in (int,float) and math.isfinite(n) and n>=0 for n in
        (initialization_seconds,ordinary_health_seconds)),'GROUP_SEPARATE_MEASURED_FIXED_COST_REQUIRED')
    # Per-phase time includes reset/setup/real request/capture/cleanup. LLM init
    # happens once. This is admission, never a mid-run budget extension.
    horizon=[max(1,(index+1)/5) for index in range(len(PHASES))]
    history=normal_seconds if history_validation_seconds is None else history_validation_seconds
    constant=max(0,normal_seconds-history) if constant_seconds is None else constant_seconds
    require(all(type(n) in (float,int) and math.isfinite(n) and 0<=n<=normal_seconds for n in (history,constant)),
            'GROUP_MEASURED_VALIDATION_COST_COMPONENTS_REQUIRED')
    request_seconds=constant*len(PHASES)+history*sum(horizon);margin=max(120,math.ceil(.15*request_seconds))
    estimate=math.ceil(initialization_seconds+ordinary_health_seconds+request_seconds+margin)
    require(estimate<=case_seconds,'GROUP_60_REQUEST_COST_EXCEEDS_FIXED_CASE_BUDGET')
    return dict(protocol_requests=len(PHASES),normal_max_phase_seconds=normal_seconds,
        request_cost_multiplier=1,initialization_seconds=initialization_seconds,
        runtime_scan_policy='FULL_HISTORY_RESCAN_CONSERVATIVE_HORIZON',normal_history_requests=5,
        cumulative_scan_request_equivalents=sum(horizon),
        normal_constant_seconds=constant,history_validation_seconds=history,
        ordinary_health_seconds=ordinary_health_seconds,margin_seconds=margin,estimated_seconds=estimate,
        case_seconds=case_seconds,uncertain_fault_retry=False,budget_extension=False)


class GroupNormalEvidence(VerifiedNormalEvidence):
    def __init__(self,seal,data,*,simulation=False):
        require(seal is _MINT and data['variant']==VARIANT and len(data['targets'])==28,
                'GROUP_FINAL_PACKAGE_ALL_TARGET_RAW_REVIEW_REQUIRED')
        # The base object provides immutable canonical bytes and serialization refusal.
        from .paged33_s2_evidence import _MINT as base_mint
        super().__init__(base_mint,data,simulation=simulation)

    def match(self,binding):
        from .paged33_logical_binding import LogicalPagedInput
        require(type(binding) is LogicalPagedInput and not binding.closed and binding.cpu is self.simulation
            and binding.target_scope is not None,'GROUP_ACTUAL_TARGET_INPUT_REQUIRED')
        binding.target_scope.check(binding.worker,binding.module_path,binding.module,binding.owner,binding.generation)
        data=self.data();require(binding.module_path in data['targets'],'GROUP_TARGET_ABSENT_FROM_FINAL_PACKAGE_NORMAL')
        reference=data['targets'][binding.module_path]['reference_input'];current=binding.before['signature']
        require(normalized_input(reference)==normalized_input(current),'GROUP_CURRENT_INPUT_DIFFERS_FROM_TARGET_NORMAL')
        return dict(normal_evidence_sha256=self.fingerprint,module_path=binding.module_path,
            current_signature_sha256=digest(current),raw_input_exact=reference==current,logical_input_equivalent=True,
            allowed_difference_fields=['used_page_ids','full_page_table_sha256','derived_writer_consumer_slot'])

    def cost_gate(self,case_seconds):
        data=self.data()
        from fault_injection.native_adapters.attention28.group_resource_budget import plan
        return plan(data['resource_observations'],variant=VARIANT,source_sha256=data['source_manifest_sha256'],
                    build_sha256=data['completion_sha256'],case_seconds=case_seconds)


def load_group_normal_evidence(*,source,source_manifest_sha256,build_root,completion_sha256,observer,
        observer_receipt_sha256,stock_run,stock_sha256,baseline_run,baseline_sha256,report,report_sha256,current_execution):
    folder=Path(source).resolve(strict=True)/'experiments/c2c3_closure_20261117'
    common=_module('arch22_normal_bridge_evidence',folder/'arch22_normal_bridge_evidence.py')
    frozen=common.frozen_source(source,source_manifest_sha256,build_variant=VARIANT)
    reader=_module('review_arch22_all_normal',folder/'review_arch22_all_normal.py')
    require(common.sha(reader.__file__)==frozen['pins'][common.EXPERIMENT+'review_arch22_all_normal.py'],
            'GROUP_FROZEN_RAW_READER_CHANGED')
    args=SimpleNamespace(source=Path(source),source_manifest_sha256=source_manifest_sha256,build_root=Path(build_root),
        completion_sha256=completion_sha256,observer=normal_observer_root(observer,observer_receipt_sha256),
        observer_receipt_sha256=observer_receipt_sha256,stock_run=Path(stock_run),stock_sha256=stock_sha256,
        baseline_run=Path(baseline_run),baseline_sha256=baseline_sha256)
    fresh=reader.review(args,build_variant=VARIANT)
    require(not fresh['issues'] and fresh.get('normal_payloads_exact') is True and fresh.get('complete_payload_pairs')==435
        and fresh.get('per_target_input_output_pairs')==140 and fresh['build']['variant']==VARIANT,
        'GROUP_FRESH_FINAL_PACKAGE_28_TARGET_RAW_NORMAL_REPLAY_FAILED')
    require(common.sha(report)==report_sha256,'GROUP_PUBLISHED_NORMAL_REPORT_CHANGED')
    published=common.read(report)
    require(published['source']['manifest_sha256']==source_manifest_sha256
        and published['build']['completion_sha256']==completion_sha256
        and published['arms']['stock']['result_sha256']==stock_sha256
        and published['arms']['baseline']['result_sha256']==baseline_sha256,'GROUP_PUBLISHED_NORMAL_INPUTS_CHANGED')
    original=common.read(Path(baseline_run)/'result.json');execution=original['execution_contract']
    require({k:v for k,v in execution.items() if k!='driver_sha256'}==
        {k:v for k,v in current_execution.items() if k!='driver_sha256'},'GROUP_CURRENT_MODEL_PACKAGE_OR_RUNNER_CHANGED')
    capture=reader.phase_capture(Path(baseline_run),original['phases'][0],group=True);targets={}
    for path,boundary in capture['all_decode_boundaries'].items():
        live=boundary['logical_input_binding']['evidence'];normalized_input(live['before']['signature'])
        targets[path]=dict(reference_input=live['before']['signature'],outputs=live['outputs'])
    times=[phase['elapsed_seconds'] for phase in original['phases']]
    require(all(type(n) in (float,int) and math.isfinite(n) and n>0 for n in times),
            'GROUP_ACTUAL_NORMAL_PHASE_TIMINGS_REQUIRED')
    fixed=original['monotonic_costs']
    require(all(type(fixed[k]) in (int,float) and math.isfinite(fixed[k]) and fixed[k]>0 for k in
        ('initialization_seconds','ordinary_health_seconds')),'GROUP_ACTUAL_INIT_HEALTH_TIMINGS_REQUIRED')
    validation=[p['runtime_cost']['indexed_history_validation_seconds'] for p in fresh['arms']['baseline']['phases']]
    require(all(type(v) in (float,int) and math.isfinite(v) and 0<v<=elapsed for v,elapsed in zip(validation,times)),
            'GROUP_ACTUAL_INDEXED_VALIDATION_COST_REQUIRED')
    data=dict(schema='demoncat.paged33.group_normal_observations.v1',variant=VARIANT,targets=targets,
        payloads=payload_signatures(capture),source_manifest_sha256=source_manifest_sha256,
        completion_sha256=completion_sha256,observer_receipt_sha256=observer_receipt_sha256,
        replay_sha256=digest(fresh),normal_max_phase_seconds=max(times),published_report_sha256=report_sha256,
        initialization_seconds=fixed['initialization_seconds'],ordinary_health_seconds=fixed['ordinary_health_seconds'],
        history_validation_seconds=max(validation),normal_constant_seconds=max(t-v for t,v in zip(times,validation)),
        stock_result_sha256=stock_sha256,private_result_sha256=baseline_sha256,execution_contract=execution,
        resource_observations=fresh['arms']['baseline']['resource_observations'],
        source_pins=frozen['pins'],serializable_authority=False,qualification_granted=False)
    require(common.sha(Path(baseline_run)/'result.json')==baseline_sha256 and common.sha(report)==report_sha256,
            'GROUP_NORMAL_ARTIFACT_CHANGED_DURING_REVIEW')
    return GroupNormalEvidence(_MINT,data)
