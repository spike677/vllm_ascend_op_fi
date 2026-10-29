"""CPU synthetic regression checks. These are not NPU evidence or fault reproductions."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import io

import causal_core as core
import profile_evidence as pe
import query_work_order as oracle


def raw(*values):
    return struct.pack('<'+'e'*len(values),*values)


def rows(values=None):
    values = values or {p:[0.0,1.0] for p in core.PROFILES}
    result = []
    for b in (1,8):
        for p in core.PROFILES:
            for repeat in (0,1):
                data = raw(*values[p])
                result.append({'batch_size':b,'profile':p,'repeat_index':repeat,
                    'call_identity':f'{b}-{p}-{repeat}', 'endpoint_raw':data,
                    'endpoint_sha256':core.digest(data),'target_output_sha256':core.digest(data),
                    'profile_fingerprint':f'{b}-{p}'})
    return result


def fake_record(m,ratio,b=1):
    # Source-plan-only fixture; its fake one-byte raw is never passed to the
    # independent launched-POD verifier and cannot be accepted as native data.
    outer=2048//m
    p={'template':'FlashAttentionScoreS1s2Bn2gs1','layout':'BNSD','sparse_type':3,'sparse_mode':2,
       'has_mask':1,'has_dropout':0,'has_pse':0,'high_precision':True,'calc_bytes':4,'keep_prob':1,
       'enable_l1_reuse':False,'b':b,'n2':1,'g':1,'n1':1,'s1':2048,'s2':2048,'m':m,
       's2_base':128,'n_ratio':ratio,'s1_outer':outer,'s2_outer':16//ratio,'s2_group':128*ratio,
       'total_query_tasks':b*outer,'core_num':1,'split_factor':b*outer,
       'sparse_start_idx':[0]*48,'d':128,'dv':128}
    return {'schema':'fault28.fa.typed_host.v1','callback_status':0,'workspace_count':1,
            'workspace_sizes':[1],'raw_size':1,'raw_hex':'00','typed':p,
            'tiling_key':str(22619644976-(1<<34))}


def main():
    checks=[]
    def check(name, fn):
        fn()
        checks.append({'name':name,'status':'PASS_CPU_SYNTHETIC'})
    def rejects(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):
            checks.append({'name':name,'status':'PASS_REJECTED'})
            return
        raise AssertionError('Did not reject: '+name)

    check('fp16_exact_units_including_subnormal_signedzero_max', lambda: core.require(
        core.fp16_units(struct.pack('<6H',0,32768,1,0x3c00,0xbc00,0x7bff)) == [0,0,1,1<<24,-(1<<24),65504*(1<<24)], 'bad conversion'))
    rejects('nan',lambda:core.fp16_units(struct.pack('<H',0x7e00)))
    rejects('inf',lambda:core.fp16_units(struct.pack('<H',0x7c00)))
    check('all_equal_is_valid_not_forced_failure',lambda:core.require(core.factorial(rows())['all_numeric_contrasts_zero'],'all equal failed'))
    values={'base':[0,1],'m_only':[1,1],'r_only':[2,1],'joint':[4,1]}
    result=core.factorial(rows(values))
    check('exact_interaction_nonzero',lambda:core.require(result['contrasts']['1']['M_ratio_interaction']['max_abs_exact_units']==1<<24,'wrong interaction'))
    check('batch_effect_zero_when_same_factorial',lambda:core.require(all(x['nonzero_elements']==0 for x in result['batch_effects'].values()),'wrong batch effect'))
    z=rows();z[0]['endpoint_raw']=struct.pack('<2H',32768,0x3c00);z[0]['endpoint_sha256']=core.digest(z[0]['endpoint_raw'])
    check('unstable_repeats_not_identified',lambda:core.require(core.factorial(z)['status']=='NOT_IDENTIFIED_UNSTABLE_REPEATS','unstable accepted'))
    rejects('missing_matrix_cell',lambda:core.factorial(rows()[:-2]))
    z=rows();z[1]['repeat_index']=0
    rejects('duplicate_repeat_index',lambda:core.factorial(z))
    z=rows();z[1]['call_identity']=z[0]['call_identity']
    rejects('reused_native_identity',lambda:core.factorial(z))
    a,b=oracle.derive(fake_record(128,8)),oracle.derive(fake_record(64,8))
    pair=core.compare_orders(a,b)
    check('M_only_preserves_legal_groups_but_changes_computed_masked_extent',lambda:core.require(
        pair['changed_query_counts']['legal_key_groups']==0 and pair['changed_query_counts']['computed_key_groups']>0 and
        pair['changed_query_counts']['softmax_row_geometry']>0 and pair['last_query_changed_head_counts']['softmax_row_geometry']==0,'wrong M distinction'))
    c=oracle.derive(fake_record(128,4))
    check('ratio_changes_legal_groups',lambda:core.require(core.compare_orders(a,c)['changed_query_counts']['legal_key_groups']>0,'ratio invisible'))
    bad=copy.deepcopy(a);bad['target_batch_tasks'].append(copy.deepcopy(bad['target_batch_tasks'][0]))
    rejects('duplicate_query_owner',lambda:core.query_orders(bad))
    bad=copy.deepcopy(a);bad['target_batch_tasks'][0]['computed_key_groups']=[[1,128]]
    rejects('missing_key_zero',lambda:core.query_orders(bad))
    bad=copy.deepcopy(a);bad['query_split_between_cores']=True
    rejects('splitkv_rejected',lambda:core.query_orders(bad))
    bad=fake_record(128,8);bad['typed']['sparse_start_idx'][0]=1
    rejects('source_missing_task',lambda:oracle.derive(bad))
    bad=fake_record(128,8);bad['typed']['high_precision']=False
    rejects('precision_change',lambda:oracle.derive(bad))
    bad=fake_record(128,8);bad['tiling_key']='1'
    rejects('kernel_key_change',lambda:oracle.derive(bad))
    parser=pe.module(Path(__file__).resolve().parents[3]/'analysis/fault28_tiling_check_20261029/extract_fa_log_profile.py','cpu_native_parser')
    fake_call={'case_id':'fixture','api':'torch_npu.npu_fusion_attention_v3','pid':123,'thread_id':456,
               'scope_token':'synthetic_only','phase':'operator_replay','batch_size':1,'status':'returned'}
    enter={'event':'native_api_enter',**fake_call};exit={'event':'native_api_exit',**fake_call}
    prefix='OP(123, fixture) '
    launch=prefix+'[NNOP][Launch][789] OpName:[aclnnFlashAttentionScore_0_FlashAttentionScore] Tiling Key: 22619644976, len: 1464, numBlocks: 20, Workspace Num: 1.'
    chunk=prefix+'[SplitDataAndPrint][789] OpName:[aclnnFlashAttentionScore_0_FlashAttentionScore] PrintTilingData[0]: Aligned TilingData Size: 1472, data(uint32_t): '+' '.join(['0']*368)
    workspace=prefix+'[CommonOpExecutorRun][789] OpName:[aclnnFlashAttentionScore_0] Workspace addr: 0x1, size: 128,'
    log_lines=[json.dumps(enter),launch,chunk,workspace,json.dumps(exit)]
    # Exercise the complete streamed endpoint with a difference away from the
    # final query. Last-query-only analysis would miss this valid effect.
    temp_root=Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(prefix='_f28cpu_',dir=temp_root) as temp:
        path=Path(temp).resolve()
        core.require(path.is_relative_to(temp_root) and path.parent == temp_root, 'Unsafe test cleanup target')
        log=path/'synthetic.log'
        log.write_text('\n'.join(log_lines)+'\n',encoding='utf-8')
        check('single_native_marker_raw_parser_cpu_fixture',lambda:core.require(len(pe.single_native_log(log,{'native_call':fake_call},parser)['raw'])==1464,'wrong raw parse'))
        log.write_text('\n'.join([log_lines[0],launch.replace('OP(123','OP(124'),*log_lines[2:]])+'\n',encoding='utf-8')
        rejects('foreign_native_pid',lambda:pe.single_native_log(log,{'native_call':fake_call},parser))
        log.write_text('\n'.join([log_lines[0],launch,launch,*log_lines[2:]])+'\n',encoding='utf-8')
        rejects('multiple_native_launches',lambda:pe.single_native_log(log,{'native_call':fake_call},parser))
        log.write_text('\n'.join([x for x in log_lines if x!=chunk])+'\n',encoding='utf-8')
        rejects('missing_native_raw',lambda:pe.single_native_log(log,{'native_call':fake_call},parser))
        base=bytes(16*2048*128*2)
        dataset=[]
        for p in core.PROFILES:
            data=bytearray(base)
            if p in ('m_only','joint'):
                data[0:2]=raw(1.0)
            f=path/(p+'.bin');f.write_bytes(data)
            for batch in (1,8):
                for repeat in (0,1):
                    dataset.append({'batch_size':batch,'profile':p,'repeat_index':repeat,
                        'target_output_sha256':core.digest(data),'profile_fingerprint':str((batch,p)),
                        'target_private_path':str(f),'target_output_shape':[16,2048,128]})
        full=core.full_sequence_factorial(dataset)
        check('full_sequence_detects_early_query_only_effect',lambda:core.require(
            full['contrasts']['B1.M_at_ratio8']['nonzero_elements']==1 and
            full['contrasts']['B1.M_at_ratio8']['nonzero_by_query'][0]==1 and
            full['contrasts']['B1.M_at_ratio8']['nonzero_by_query'][2047]==0 and
            full['contrasts']['B1.M_ratio_interaction']['nonzero_elements']==0,'full endpoint wrong'))
        dataset[0]['target_output_sha256']='0'*64
        dataset[1]['target_output_sha256']='0'*64
        rejects('raw_output_sha_mismatch',lambda:core.full_sequence_factorial(dataset))
    report={'status':'PASS_CPU_SYNTHETIC_ONLY','native_runs':0,'checks':checks,
            'not_fault_evidence':True,'not_full_native_evidence_adapter_acceptance':True}
    (Path(__file__).parent/'cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'checks':len(checks)}))


if __name__=='__main__':main()
