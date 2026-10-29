"""Strict read-only causal replay analysis. CPU only; never creates native evidence.

The primary endpoint is the complete fixed target sequence. Requested labels
select the factorial contrast; actual tiling comes from public/raw launch bytes.
No numerical difference is required for a valid experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

from causal_core import PROFILES, compare_orders, digest, factorial, full_sequence_factorial, require
import profile_evidence as pe
import query_work_order as oracle

SOURCE_CALL_SHA = '458a683a616da397bd1eeba47b7348d0e2bf78c253ea43c158f7f393bb67c0c8'
TARGET_SHAS = {'query':'45755102b0f5ac710aa9ff30ab3bfe2ad69b66041281a1671eaccb144557dc00',
               'key':'73d1dec27ffe3873678a9daff7d422d51346997427fbc64279dbc0bfa8156fde',
               'value':'339fdb463b0ef9ba0931c739b2ee3db1fc705e1b6a5c28f0fd5daf6fea915947'}
UTIL_SHA = '442648c354814d149ba7f8185e4b16cc7d528d1333f8437f07c0f3c5c93187d9'
REFERENCE_QUERIES = [0,63,64,127,128,511,512,1023,1024,2047]


def read(path):
    return json.loads(Path(path).read_bytes())


def child(root, relative):
    p = (root/relative).resolve()
    require(p.is_relative_to(root.resolve()) and p.is_file(), 'Missing or escaping evidence path: '+str(relative))
    return p


def verify_input_source(path):
    m = read(path)
    require(m['schema'] == 'fault28.causal.inputs.v1' and m['source_call_sha256'] == SOURCE_CALL_SHA, 'Unapproved real model input origin')
    call_path = child(path.parent, m['source_call_file'])
    require(pe.sha(call_path) == SOURCE_CALL_SHA, 'Original call metadata changed')
    original = read(call_path)
    require(original['scalar_arguments'] == {'dropout_p':0.0,'is_causal':True,'scale':0.08838834764831845,'enable_gqa':True} and original['attention_mask'] is None,
            'Original native lowering contract changed')
    expected_kwargs = {'head_num':16,'input_layout':'BNSD','scale':original['scalar_arguments']['scale'],
        'keep_prob':1.0,'pre_tockens':2147483647,'next_tockens':0,'inner_precise':0,'sparse_mode':2,
        'gen_mask_parallel':True,'sync':False,'softmax_layout':''}
    require(m['native_kwargs'] == expected_kwargs, 'Frozen native arguments differ from original lowering')
    inputs = {}
    for name in ('query','key','value'):
        row = m['inputs'][name]
        raw = child(path.parent,row['file']).read_bytes()
        heads = 16 if name == 'query' else 8
        require(row['shape'] == [1,heads,2048,128] and row['dtype'] == 'torch.float16' and
                row['byteorder'] == 'little' and row['raw_order'] == 'C-contiguous logical elements', 'Fixed source geometry differs')
        require(len(raw) == row['bytes'] == heads*2048*128*2 and digest(raw) == row['sha256'] == TARGET_SHAS[name], 'Fixed target input bytes differ')
        require(original['target_inputs_before'][name]['logical_sha256'] == TARGET_SHAS[name] ==
                original['target_inputs_after'][name]['logical_sha256'], 'Original before/after provenance differs')
        source = original['selected_layer_private_inputs'][name]
        require(row['source_stride'] == source['stride'] and row['source_shape'] == source['shape'] and
                row['source_sha256'] == source['logical_sha256'], 'Frozen source geometry/SHA provenance differs')
        expected = {'1': digest(raw), '8': digest(raw*8)}
        require(row['expected_full_sha256'] == expected, 'Repeated-input SHA not independently reproducible')
        inputs[name] = {'record': row, 'raw': raw, 'expected': expected}
    return m, inputs


def check_snapshot(before, after, row, batch):
    a, z = dict(before), dict(after)
    formats = [a.pop('actual_format'), z.pop('actual_format')]
    require(a == z, 'Native API changed input values, shape, identity, or version')
    require(formats[0] == formats[1] or formats == [0,2], 'Unsupported native format change')
    r = row['record']
    require(a['shape'] == [batch,*r['shape'][1:]] and a['stride'] == r['source_stride'] and a['dtype'] == 'torch.float16', 'Replay input geometry differs from original')
    require(a['full_sha256'] == row['expected'][str(batch)] and a['target_sha256'] == r['sha256'] and
            a['per_batch_sha256'] == [r['sha256']]*batch, 'B1/B8 is not the same fixed input repeated')
    require(a['bytes'] == batch*r['bytes'] and a['byteorder'] == 'little' and a['raw_order'] == 'C-contiguous logical elements', 'Invalid input byte coverage')
    for field in ('object_id','data_ptr','version','storage_offset'):
        require(type(a[field]) is int, 'Missing within-call identity: '+field)
    return formats


def private_output(folder, result):
    full, tail = result['private_target_output'], result['target_last_query']
    require(full['shape'] == [16,2048,128] and full['dtype'] == 'torch.float16' and full['batch_slot'] == 0,
            'Wrong complete target output')
    path = child(folder, full['file'])
    require(full['byteorder'] == 'little' and full['raw_order'] == 'C-contiguous logical elements' and
            path.stat().st_size == full['bytes'] == 16*2048*128*2 and pe.sha(path) == full['sha256'] == result['output']['target_sha256'], 'Complete target bytes/SHA differ')
    require(tail['shape'] == [16,128] and tail['dtype'] == 'torch.float16' and tail['batch_slot'] == 0 and tail['query_index'] == 2047 and
            tail['byteorder'] == 'little' and tail['raw_order'] == 'C-contiguous logical elements', 'Wrong prespecified final query')
    raw_tail = child(folder,tail['file']).read_bytes()
    require(len(raw_tail) == tail['bytes'] == 4096 and digest(raw_tail) == tail['sha256'], 'Final query bytes differ')
    with path.open('rb') as stream:
        derived = b''
        for head in range(16):
            stream.seek((head*2048+2047)*128*2)
            derived += stream.read(128*2)
    require(derived == raw_tail, 'Final query is not a slice of the full target output')
    output = result['output']
    require(output['shape'] == [result['batch_size'],16,2048,128] and output['dtype'] == 'torch.float16' and
            len(output['per_batch_sha256']) == result['batch_size'] and output['per_batch_sha256'][0] == full['sha256'], 'Native full output/target mapping differs')
    return {'target_private_path': str(path), 'target_output_shape': full['shape'],
            'target_output_sha256': full['sha256'], 'endpoint_sha256': tail['sha256'], 'endpoint_raw': raw_tail}


def fixed_input_params(profile):
    return {k:v for k,v in profile['pod'].items() if k.startswith('inputParams.') and k != 'inputParams.bSize'}


def validate_stock_scope(case, result, folder):
    """Experiment stage and native API phase are distinct frozen fields."""
    require(case['phase']=='stock_bridge' and case['stock_bridge'] is True and
            case['operator_variant']==result['operator_variant']=='stock','Wrong stock experiment stage')
    call=result['native_call']
    require(call['phase']=='operator_replay' and call['scope_token'] is None and call['status']=='returned' and
            call['api']=='torch_npu.npu_fusion_attention_v3','Stock bridge acquired custom scope or wrong API phase')
    require(not (folder/'host_events.jsonl').exists() and result['vendor']['custom_opp_path'] is None,
            'Stock bridge has custom host evidence')


def stock_bridge_gate(args, inputs, manifest, rows, profiles, common, parser):
    matrix = read(args.stock_cases_dir/'manifest.json')
    require(len(matrix['cases']) == 4, 'Need all four stock bridge calls')
    original = read(child(args.input_manifest.parent,manifest['source_call_file']))
    old_output = original['target_output']['logical_sha256']
    primary = {(r['batch_size'],r['repeat_index']):r for r in rows if r['profile']=='base'}
    seen, report, stable = set(), [], {}
    for item in matrix['cases']:
        case_path=child(args.stock_cases_dir,item['file'])
        require(pe.sha(case_path)==item['sha256'],'Stock bridge frozen case SHA differs')
        case=read(case_path);folder=args.stock_results_dir/case['case_id'];result=read(folder/'result.json')
        b,repeat=case['batch_size'],case['repeat_index']
        require((b,repeat) not in seen and b in (1,8) and repeat in (0,1),'Duplicate or wrong stock cell')
        seen.add((b,repeat))
        require(case['operator_variant']==result['operator_variant']=='stock' and case['stock_bridge'] is True and
                result['status']=='CAPTURE_COMPLETED' and result['original_api_calls']==1 and result['model_forward'] is False,'Wrong stock bridge execution')
        require(result['source_case_sha256']==item['sha256'] and result['input_manifest_sha256']==pe.sha(args.input_manifest),'Stock bridge source/input differs')
        require(result['replay_source_sha256']==common[0] and result['versions']==common[1],'Stock/private replay code or version differs')
        validate_stock_scope(case,result,folder)
        for n in inputs:
            check_snapshot(result['inputs_before'][n],result['inputs_after'][n],inputs[n],b)
        expected_kwargs={**manifest['native_kwargs'],'pse':None,'padding_mask':None,'prefix':None,'actual_seq_qlen':None,'actual_seq_kvlen':None,'sink':None}
        require(result['native_kwargs']==expected_kwargs,'Stock bridge native scalar arguments differ')
        mask=result['mask_before']
        mask_expected=digest(b''.join(b'\0'*(q+1)+b'\1'*(2047-q) for q in range(2048)))
        require(mask==result['mask_after'] and mask['shape']==[2048,2048] and mask['dtype']=='torch.bool' and mask['full_sha256']==mask_expected,'Stock bridge causal mask differs')
        output=private_output(folder,result)
        native=pe.single_native_log(args.stock_results_dir/(case['case_id']+'.log'),result,parser)
        require(not any('libcust_opmaster' in x['path'] or 'libcust_opsproto' in x['path'] for x in native['loads']),
                'Stock bridge log contains custom operator library load')
        private=primary[b,repeat];private_profile=profiles[private['case_id']]
        require(output['target_output_sha256']==private['target_output_sha256'] and
                Path(output['target_private_path']).read_bytes()==Path(private['target_private_path']).read_bytes(),
                'Private BASE differs numerically from same-batch stock bridge')
        require(native['raw']==bytes.fromhex(private_profile['event']['raw_hex']) and
                native['tiling_key']==private_profile['event']['tiling_key'] and native['block_dim']==private_profile['event']['block_dim'],
                'Private BASE actual launched profile differs from same-batch stock')
        require(b!=8 or output['target_output_sha256']==old_output,'Stock B8 replay differs from original captured model b0 output')
        profile_key=(output['target_output_sha256'],digest(native['raw']))
        require(b not in stable or stable[b]==profile_key,'Stock bridge repeats are unstable')
        stable[b]=profile_key
        report.append({'case_id':case['case_id'],'batch_size':b,'repeat_index':repeat,
            'result_sha256':pe.sha(folder/'result.json'),'native_log_sha256':native['source_sha256'],
            'target_sha256':output['target_output_sha256'],'raw_sha256':digest(native['raw']),
            'same_batch_private_BASE_byte_equal':True,'old_B8_capture_target_equal':output['target_output_sha256']==old_output,
            'old_capture_equality_required':b==8,'stock_schedule_mode':None})
    require(seen=={(b,r) for b in (1,8) for r in (0,1)},'Stock bridge matrix incomplete')
    return {'status':'PASS_STOCK_REPLAY_AND_PRIVATE_BASE_COMPATIBILITY','calls':report,
            'B1_old_B8_difference_is_descriptive_only':True,'old_target_output_sha256':old_output}


def reference_metrics(inputs, rows):
    """Independent CPU FP64 reference at predeclared boundaries; no error pass threshold."""
    import numpy as np
    q,k,v = [np.frombuffer(inputs[n]['raw'],dtype='<f2').reshape(inputs[n]['record']['shape'])[0].astype(np.float64) for n in ('query','key','value')]
    scale = struct.unpack('<f',struct.pack('<f',1.0/(128**0.5)))[0]
    ref = np.empty((16,len(REFERENCE_QUERIES),128),dtype=np.float64)
    for head in range(16):
        for i,query in enumerate(REFERENCE_QUERIES):
            scores = k[head//2,:query+1] @ q[head,query] * scale
            weights = np.exp(scores-scores.max())
            weights /= weights.sum(dtype=np.float64)
            ref[head,i] = weights @ v[head//2,:query+1]
    reports = []
    for row in rows:
        values = np.memmap(row['target_private_path'],dtype='<f2',mode='r',shape=(16,2048,128))
        observed = np.asarray(values[:,REFERENCE_QUERIES,:],dtype=np.float64)
        require(np.isfinite(observed).all(), 'Nonfinite reference endpoint')
        delta = observed-ref
        reports.append({'case_id':row['case_id'], 'by_query':[{'query':query,
            'max_abs_error':float(np.max(np.abs(delta[:,i]))), 'rms_error':float(np.sqrt(np.mean(delta[:,i]**2))),
            'changed_from_fp64_elements':int(np.count_nonzero(delta[:,i]))} for i,query in enumerate(REFERENCE_QUERIES)]})
        del values
    return {'queries':REFERENCE_QUERIES,'effective_scale_float32':scale,'kv_head_rule':'query_head // 2',
            'reference_fp64_sha256':digest(ref.astype('<f8').tobytes()), 'cases':reports,
            'error_direction_is_not_an_acceptance_gate':True,'does_not_measure_model_quality':True}


def analyze(args):
    here = Path(__file__).parent
    dependencies = {}
    require(pe.sha(args.vendor_manifest)=='3d3ca8e0bc12003e19c6f3f4cfba21775e516385fe5807b10d59c6bb53cb1785',
            'Vendor differs from the CPU-reviewed causal a1 build')
    utilpath = args.previous_analysis/'verify_stock_off.py'
    util = pe.module(utilpath,'old_fa_public_util','71c63c54f3f32f90d82e3177385197b5830ce07117b766818945b4bfe5e25fe1')
    parserpath = args.previous_analysis/'extract_fa_log_profile.py'
    parser = pe.module(parserpath,'old_native_log_parser','fae9ce5be05d002376680a808b006711931334b83d8efb289de44931118a3f44')
    dependencies.update({str(utilpath):pe.sha(utilpath),str(parserpath):pe.sha(parserpath)})
    for relative, expected in {**oracle.SOURCE_FILES,'common/include/op_kernel/util.h':UTIL_SHA}.items():
        p = child(args.source_root,relative)
        require(pe.sha(p) == expected, 'Matched kernel/source helper changed: '+relative)
        dependencies[str(p)] = expected
    schema = read(args.pod_schema)
    require(pe.sha(args.pod_schema)=='82457b4440b60061e3a32345fb3b33adacf1a0f1b527f078f7d2557cb1af743a','SDK public POD schema changed')
    require(len(schema['fields']) == 205, 'Need 105 original POD fields and 100 SDK TCube fields')
    dependencies[str(args.pod_schema)] = pe.sha(args.pod_schema)
    inputs_manifest, inputs = verify_input_source(args.input_manifest)
    input_sha = pe.sha(args.input_manifest)
    matrix = read(args.cases_dir/'manifest.json')
    require(matrix['schema'] == 'fault28.causal.matrix.v1' and len(matrix['cases']) == 16, 'Primary frozen matrix must contain 16 cases')
    rows, profiles, observations, evidence, identities, auxiliaries = [], {}, [], {}, set(), {}
    common = None
    mask_raw = b''.join(b'\0'*(q+1)+b'\1'*(2047-q) for q in range(2048))
    mask_sha = digest(mask_raw)
    for item in matrix['cases']:
        case_path = child(args.cases_dir,item['file'])
        require(pe.sha(case_path) == item['sha256'], 'Frozen case SHA differs')
        case = read(case_path)
        folder = args.results_dir/case['case_id']
        resultpath = folder/'result.json'
        result = read(resultpath)
        require(result['schema'] == 'fault28.causal.replay.v1' and result['status'] == 'CAPTURE_COMPLETED', 'Replay did not complete')
        require(result['original_api_calls'] == 1 and result['model_forward'] is False and result['execution_kind'] == 'isolated_original_native_operator_replay', 'Replay call scope differs')
        require(result['source_case_sha256'] == item['sha256'] and result['input_manifest_sha256'] == case['input_manifest_sha256'] == input_sha, 'Case/input source SHA differs')
        for field in ('case_id','profile','batch_size','repeat_index'):
            require(case[field] == item[field] == result[field], 'Frozen cell metadata differs: '+field)
        call = result['native_call']
        require(call['api'] == 'torch_npu.npu_fusion_attention_v3' and call['status'] == 'returned' and call['phase'] == 'operator_replay', 'Wrong actual native API boundary')
        identity = (call['pid'],call['scope_token'])
        require(identity not in identities and call['enter_monotonic_ns'] < call['exit_monotonic_ns'], 'Repeated or invalid native call identity')
        identities.add(identity)
        collector=result['statistics_collector']
        collector_build={k:collector.get(k) for k in ('variant','precision','manifest_sha256','patched_processor_sha256','original_processor_sha256')}
        current = (result['replay_source_sha256'],result['versions'],result['vendor']['manifest_sha256'],collector_build)
        # Collector provenance may contain paths but is the same selected build;
        # do not compare per-case receipt paths as numerical execution factors.
        require(current == common if common is not None else True, 'Replay/vendor/version/collector differs across matrix')
        common = current
        require(case['approved_vendor_manifest_sha256'] == result['vendor']['manifest_sha256'] == pe.sha(args.vendor_manifest), 'Unapproved vendor source')
        formats = {n:check_snapshot(result['inputs_before'][n],result['inputs_after'][n],inputs[n],case['batch_size']) for n in inputs}
        mb,ma = result['mask_before'],result['mask_after']
        require(mb == ma and mb['shape'] == [2048,2048] and mb['dtype'] == 'torch.bool' and mb['full_sha256'] == mask_sha and mb['bytes'] == len(mask_raw), 'Causal mask changed')
        kwargs = {**inputs_manifest['native_kwargs'], 'pse':None,'padding_mask':None,'prefix':None,'actual_seq_qlen':None,'actual_seq_kvlen':None,'sink':None}
        require(result['native_kwargs'] == kwargs, 'Native precision/scale/mask/dropout arguments changed')
        outputs = private_output(folder,result)
        aux = result['auxiliary_outputs']
        require([x['name'] for x in aux] == ['softmax_max','softmax_sum','softmax_out','seed','offset'], 'Missing or reordered native returns')
        for x in aux:
            require(x['dtype'].startswith('torch.') and type(x['bytes']) is int and x['bytes'] >= 0 and
                    len(x['full_sha256']) == 64 and x['byteorder'] == 'little', 'Malformed auxiliary observation')
        auxiliaries[case['case_id']] = {x['name']:{k:x[k] for k in ('shape','dtype','bytes','full_sha256','target_sha256','per_batch_sha256','statistics') if k in x} for x in aux}
        log = args.results_dir/(case['case_id']+'.log')
        profile = pe.verified_profile(folder,log,result,schema,util,parser,oracle)
        typed = profile['event']['typed']
        actual = (typed['m'],typed['n_ratio'])
        require(actual == PROFILES[case['profile']] and typed['b'] == case['batch_size'], 'Actual launched profile does not implement frozen cell')
        require(typed['n1'] == 16 and typed['n2'] == 8 and typed['g'] == 2, 'Actual query/KV head geometry differs')
        event=profile['event']
        require(event['requested_profile']==event['resolved_profile']==case['profile'] and event['eligible'] is True,
                'Control intent or native eligibility differs')
        require(event['m_override_applied'] is (actual[0]==64) and event['r_override_applied'] is (actual[1]==4),
                'Control override witness differs from independently decoded profile')
        require(event['eligibility_checks']==len(event['guard_reports']) and event['eligibility_checks']>0 and
                all(x['passed'] is True and x['failed_guards']==[] for x in event['guard_reports']),'Failed host context guards')
        if profiles:
            previous = next(iter(profiles.values()))
            require(fixed_input_params(previous) == fixed_input_params(profile), 'Actual kernel input/precision/mask semantics changed')
            for k in ('template','layout','dtype_enum','calc_bytes','high_precision','enable_l1_reuse','sparse_mode','has_mask','has_dropout','has_pse'):
                require(previous['event']['typed'][k] == typed[k], 'Actual native route changed: '+k)
            require(previous['event']['tiling_key'] == profile['event']['tiling_key'], 'Kernel key changed')
        profiles[case['case_id']] = profile
        rows.append({'case_id':case['case_id'],'profile':case['profile'],'batch_size':case['batch_size'],
                     'repeat_index':case['repeat_index'],'call_identity':str(identity),
                     'profile_fingerprint':profile['profile_fingerprint'],**outputs})
        observations.append({'case_id':case['case_id'],'actual_m':actual[0],'actual_n_ratio':actual[1],
            'batch_size':case['batch_size'],'profile':case['profile'],'repeat_index':case['repeat_index'],
            'target_output_sha256':outputs['target_output_sha256'],'last_query_sha256':outputs['endpoint_sha256'],
            'actual_s2_group':typed['s2_group'],'raw_sha256':profile['raw_sha256'],'input_formats':formats,
            'output_per_batch_sha256':result['output']['per_batch_sha256'],
            'b8_slots_are_not_assumed_equal':True,'host_control_is_not_numerical_cause':True,
            'native_launch':profile['native'],'bmm1':profile['bmm']['bmm1'],'bmm2':profile['bmm']['bmm2'],
            'source_derived_coverage':{k:profile['trace'][k] for k in ('queries_checked','total_real_query_tasks',
                'every_real_query_owned_once','each_query_legal_keys_covered_once','query_split_between_cores','full_task_trace_sha256')}})
        evidence[str(resultpath)] = pe.sha(resultpath)
        evidence[str(folder/'host_events.jsonl')] = profile['host_events_sha256']
        evidence[str(log)] = profile['native']['source_sha256']
    bridge=stock_bridge_gate(args,inputs,inputs_manifest,rows,profiles,common,parser)
    tail = factorial(rows)
    full = full_sequence_factorial(rows)
    pairs = [('base','m_only'),('r_only','joint'),('base','r_only'),('m_only','joint'),('base','joint')]
    traces = {}
    chosen = {(x['batch_size'],x['profile']):profiles[x['case_id']] for x in rows if x['repeat_index'] == 0}
    for batch in (1,8):
        for a,z in pairs:
            pa,pz = chosen[batch,a],chosen[batch,z]
            traces[f'B{batch}.{a}_to_{z}'] = {**compare_orders(pa['trace'],pz['trace']),
                'bmm_changed_fields':{section:[k for k in pa['bmm'][section] if pa['bmm'][section][k] != pz['bmm'][section][k]] for section in ('bmm1','bmm2')}}
    for profile in PROFILES:
        traces['Batch.'+profile] = compare_orders(chosen[1,profile]['trace'],chosen[8,profile]['trace'])
    aux_pairs = {}
    cells = {(x['batch_size'],x['profile'],x['repeat_index']):x for x in rows}
    for batch in (1,8):
        for a,z in pairs:
            ar,zr = auxiliaries[cells[batch,a,0]['case_id']],auxiliaries[cells[batch,z,0]['case_id']]
            aux_pairs[f'B{batch}.{a}_to_{z}'] = {n:{'shape_equal':ar[n]['shape'] == zr[n]['shape'],
                'dtype_equal':ar[n]['dtype'] == zr[n]['dtype'],
                'full_return_bit_difference':ar[n]['full_sha256'] != zr[n]['full_sha256'],
                'target_bit_difference':ar[n].get('target_sha256') != zr[n].get('target_sha256') if 'target_sha256' in ar[n] and 'target_sha256' in zr[n] else None}
                for n in ar}
    aux_repeats = {f'B{b}.{p}':{n:auxiliaries[cells[b,p,0]['case_id']][n]['full_sha256'] ==
                 auxiliaries[cells[b,p,1]['case_id']][n]['full_sha256'] for n in ('softmax_max','softmax_sum','softmax_out','seed','offset')}
                 for b in (1,8) for p in PROFILES}
    refs = reference_metrics(inputs,rows)
    attribution={}
    if full['contrasts'] is not None:
        for name,value in full['contrasts'].items():
            attribution[name]={'primary_output_effect':'OBSERVED_REPEATABLE_INTERVENTION_EFFECT' if value['nonzero_elements'] else 'NO_OBSERVED_NUMERICAL_EFFECT',
                'micro_arithmetic_cause':'NOT_IDENTIFIED','applies_only_to_fixed_captured_input':True}
    return {'schema':'fault28.causal.independent_analysis.v1','status':full['status'],
            'preregistration_sha256':pe.sha(here/'preregistered_contract.json'),
            'input_manifest_sha256':input_sha,'matrix_manifest_sha256':pe.sha(args.cases_dir/'manifest.json'),
            'frozen_execution':{'replay_source_sha256':common[0],'versions':common[1],
                'vendor_manifest_sha256':common[2],'collector_build':common[3]},
            'dependencies':dependencies,'evidence_sha256':evidence,'calls':observations,
            'stock_bridge_compatibility':bridge,
            'full_sequence_numerical_contrasts':full,'last_query_numerical_contrasts':tail,
            'all_query_source_plan_comparisons':traces,'fixed_query_fp64_reference':refs,
            'attribution':attribution,
            'auxiliary_return_observations':auxiliaries,'auxiliary_pair_comparisons':aux_pairs,'auxiliary_repeat_bit_stability':aux_repeats,
            'auxiliary_change_does_not_imply_attention_output_change':True,
            'source_plan_is_not_hardware_trace':True,'floating_point_instruction_root_cause':'NOT_IDENTIFIED',
            'historical_fia_defect_reproduced':False,'joint_change_does_not_identify_M_cause':True,
            'no_numerical_difference_is_a_valid_outcome':True,'secondary_controlled_batch_policy':'NOT_YET_ANALYZED',
            'private_tensor_values_in_report':False,'msprobe_used_to_assign_cause':False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cases-dir','results-dir','stock-cases-dir','stock-results-dir','input-manifest','source-root','pod-schema','vendor-manifest','previous-analysis','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    require(not a.output.exists(), 'Analysis output must be new')
    try:
        report = analyze(a)
    except Exception as exc:
        report = {'schema':'fault28.causal.independent_analysis.v1','status':'FAIL_EVIDENCE_CONTRACT',
                  'error_type':type(exc).__name__,'error':str(exc),'causal_attribution':'NOT_IDENTIFIED'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'output':str(a.output)}))
    return 1 if report['status'] in ('FAIL_EVIDENCE_CONTRACT','NOT_IDENTIFIED_UNSTABLE_REPEATS') else 0


if __name__ == '__main__':
    raise SystemExit(main())
