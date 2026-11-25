"""Build a separate immutable reference supplement from reviewed A2 variants."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FAMILIES={
 'quant':(['row9_scale_half','row9_scale_quarter','row10_scale_double','row10_scale_quadruple'],
          ['reference_adapters/quant_scale.py','reference_adapters/review_quant.py',
           'tests/worker_extension/test_reference_quant_scale.py']),
 'hidden':(['row7_mlp_last_row_zero','row8_decoder_last_row_zero'],
          ['reference_adapters/hidden_rows.py','reference_adapters/review_reference.py',
           'tests/worker_extension/test_reference_hidden_rows.py']),
 'swiglu':(['row3_gate_fp16_round'],['reference_adapters/swiglu.py',
           'reference_adapters/review_reference.py','tests/worker_extension/test_reference_precision.py']),
 'logits':(['row12_logits_columns_rotate'],['reference_adapters/logits_mapping.py',
           'reference_adapters/review_reference.py','tests/worker_extension/test_reference_logits_mapping.py']),
 'vision-compute':(['row16_conv3d_compute_fp16'],['reference_adapters/vision_precision.py',
           'reference_adapters/review_reference.py','tests/worker_extension/test_reference_precision.py']),
 'vision-input':(['row32_conv3d_input_fp16'],['reference_adapters/vision_precision.py',
           'reference_adapters/review_reference.py','tests/worker_extension/test_reference_precision.py']),
 'matmul-order':(['row19_bf16_reverse_splitk'],['reference_adapters/matmul_order.py',
           'reference_adapters/review_reference.py','tests/worker_extension/test_reference_matmul.py']),
 'matmul-bias':(['row34_bf16_before_bias'],['reference_adapters/matmul_order.py',
           'reference_adapters/review_reference.py','tests/worker_extension/test_reference_matmul.py']),
 'vision-fp8':(['row32_conv3d_input_e4m3','row32_conv3d_input_e5m2'],
          ['reference_adapters/vision_precision.py','reference_adapters/precision_rules.py',
           'reference_adapters/review_reference.py','tests/worker_extension/test_reference_precision_rules.py',
           'reference_adapters/projection_round.py','reference_adapters/hidden_rows.py']),
 'projection-round':(['row34_projection_mantissa5','row34_projection_mantissa3'],
          ['reference_adapters/projection_round.py','reference_adapters/precision_rules.py',
           'reference_adapters/hidden_rows.py','reference_adapters/review_reference.py',
           'tests/worker_extension/test_reference_precision_rules.py','reference_adapters/vision_precision.py']),
}
DEPENDENCIES=['experiments/worker_core_v11_20261118/external_input_examples.py',
    'fault_injection/sites/norm18_variance.py','fault_injection/sites/norm18.py',
    'fault_injection/sites/projection39.py','fault_injection/sites/mla28.py',
    'fault_injection/engines/region_lease.py','fault_injection/product_contract.py',
    'tools/review_product_mode.py',
    'fault_injection/conv3d_natural.py','fault_injection/vision_input.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/external_provider.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/product_host.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/product_request.py',
    'experiments/worker_core_v11_20261118/product_mode_probe.py',
    'experiments/worker_core_v11_20261118/continuous_public_probe.py']

def sha(raw):return hashlib.sha256(raw).hexdigest()

def build(family,destination):
    destination=Path(destination)
    if destination.exists():raise ValueError('IMMUTABLE_SUPPLEMENT_ALREADY_EXISTS')
    variants,files=FAMILIES[family]
    evidence=json.loads((ROOT/'docs/worker_core_v1/reference_adapter_a2_20261124.json').read_bytes())
    for variant in variants:
        record=evidence['variants'].get(variant,{})
        if record.get('status')!='REFERENCE_A2_TP2_TWO_OWNERS_REVIEWED':
            raise ValueError('A2_REVIEW_NOT_PASSED:'+variant)
        if not record.get('scope_identities'):raise ValueError('OWNER_WORKER_IDENTITY_MISSING:'+variant)
    base=subprocess.check_output(['git','rev-parse','59f6aea5'],cwd=ROOT,text=True).strip()
    dependencies={}; checkout_forms={}
    for file in DEPENDENCIES:
        frozen=subprocess.check_output(['git','show',base+':'+file],cwd=ROOT)
        current=(ROOT/file).read_bytes()
        # Windows checkout uses CRLF; the immutable release archive uses LF.
        # Record both byte identities instead of pretending they match. Any
        # difference beyond the checkout line-ending transform is refused.
        committed=subprocess.check_output(['git','show','HEAD:'+file],cwd=ROOT)
        if frozen!=committed or current.replace(b'\r\n',b'\n')!=frozen.replace(b'\r\n',b'\n'):
            raise ValueError('FROZEN_DEPENDENCY_CHANGED:'+file)
        dependencies[file]=sha(frozen)
        checkout_forms[file]=dict(release_sha256=sha(frozen),a2_checkout_sha256=sha(current),
            identical_bytes=frozen==current,only_checkout_line_endings=frozen!=current)
    blobs={f:(ROOT/f).read_bytes() for f in ['reference_adapters/__init__.py','reference_adapters/README.md',
        'reference_adapters/bootstrap.py','reference_adapters/product_probe.py',
        'reference_adapters/control_guard.py','reference_adapters/refusal_probe.py',*files]}
    if family in ('vision-fp8','projection-round'):
        # Factory SHA alone is insufficient: helpers implement the numeric rule
        # and the public control preflight. Pin every shipped executable module.
        for variant in variants:
            observed=evidence['variants'][variant].get('execution_source_sha256',{})
            for file,raw in blobs.items():
                if file.startswith('reference_adapters/') and file.endswith('.py'):
                    if observed.get(file)!=sha(raw):
                        raise ValueError('EXECUTION_SOURCE_DIFFERS_FROM_A2_REVIEW:'+variant+':'+file)
    # Embedded test files may import sibling factories; do not advertise CPU
    # commands requiring absent modules. A2 product commands are per variant.
    recipes=[]
    for variant in variants:
        module=next(f for f in files if f.startswith('reference_adapters/') and '/review_' not in f)
        actual_sha=sha((ROOT/module).read_bytes())
        if evidence['variants'][variant].get('factory_sha256',{}).get(Path(module).name)!=actual_sha:
            raise ValueError('FACTORY_DIFFERS_FROM_A2_REVIEW:'+variant)
        stage='sampler' if family=='logits' else 'vision' if family.startswith('vision') or family=='matmul-bias' else 'model'
        key='17' if stage=='vision' else '18'
        recipes.append(dict(variant=variant,factory_module=module[:-3].replace('/','.'),
            factory_lookup='FACTORIES['+repr(variant)+']',factory_sha256=actual_sha,
            stage=stage,region_kind='transient',phenomenon_proxy=True,
            a2_command='python3 -B reference_adapters/product_probe.py --mechanism '+key+' --reference '+variant+' --negative-regression',
            review_command='python3 -B -m reference_adapters.'+('review_quant' if family=='quant' else 'review_reference')+
                ' /evidence --receipt product_http_'+key+'.json --negative product_control_negative.json --output /evidence/reference-review.json'))
    blobs['recipes.json']=(json.dumps(recipes,ensure_ascii=False,indent=2)+'\n').encode()
    blobs['dependency-byte-forms.json']=(json.dumps(checkout_forms,indent=2)+'\n').encode()
    result=dict(evidence,variants={v:evidence['variants'][v] for v in variants})
    blobs['a2-review.json']=(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode()
    blobs['HANDOFF.md']=(
        '# 黄区参考适配器独立补充件\n\n'
        '仅供黄区审查接入，现象复现，非原机理。A2/0.26/CANN9.1/社区Qwen3.8 W8A8/TP2结果不授予A3资格。'
        '本包不修改冻结r7.4.1、不新增公共编号、不自动登记。仅manifest列出的命名变体完成A2两owner复核。\n\n'
        '在独立目录解压；需要r7.4.1的三个wheel及其完整源码目录（含experiments）。Python 3.12、torch2.10/torch_npu2.10，'
        '与目标部署身份匹配。把本包和r7.4.1源码同时加入PYTHONPATH，不覆盖发行目录。逐项检查manifest中文件及依赖SHA。\n\n'
        '可信宿主对每个rank导入factory，以manifest中的工厂文件SHA构造ExternalSite，region_kind=transient，'
        'expected_deployment取该rank实际observe_deployment；通过register_external_site注册，再配置product profile和验收许可。'
        '只允许固定命名变体，不接受用户数值、层名或rank。CLI映射由黄区接入方补入，能力以ProductClient.capabilities为准。\n\n'
        '依赖文件的冻结Git内容未改。Windows验收快照部分文件为CRLF，r7.4.1源码包为LF；'
        'dependency-byte-forms.json逐项列出两种SHA，不声称它们原始字节相等。工厂SHA按本包实际字节审核。\n\n'
        '产品顺序为attach→enable→真实用户HTTP请求→status→disable（drain和金丝雀）→detach；'
        '每变体两个owner，缺许可/超范围/重复enable负例，逐rank八计数及changed_elements，真实消费者、区域字节恢复、资源清零、逐位金丝雀。'
        'changed小于applied不等于失败，但零变化不得宣称复现了数值现象。原始回执留在验收方，回传一页总结由负责人转交。\n\n'
        '完整A2命令和语义边界见开发线reference_adapters/README.md；本包只含本批工厂和复核器。'
        '受审源提交、source_manifest SHA、原回执SHA及同组Worker/两个owner均在a2-review.json。\n'
    ).encode()
    manifest=dict(schema='dcat.yellow.reference.supplement.v1',family=family,frozen_base=base,
        variants=variants,qualification_granted=False,dependency_sha256=dependencies,
        files={f:dict(bytes=len(b),sha256=sha(b)) for f,b in sorted(blobs.items())})
    blobs['manifest.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    destination.mkdir(parents=True)
    archive=destination/'reference-adapters.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file,raw in sorted(blobs.items()):
            info=zipfile.ZipInfo(file,date_time=(2026,11,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,raw)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(blobs)
        for file,raw in blobs.items():assert z.read(file)==raw
    (destination/'manifest.json').write_bytes(blobs['manifest.json'])
    (destination/'SHA256SUMS').write_text(sha(archive.read_bytes())+'  reference-adapters.zip\n'+sha(blobs['manifest.json'])+'  manifest.json\n',encoding='utf-8')
    return dict(archive=str(archive),sha256=sha(archive.read_bytes()),variants=variants)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=FAMILIES,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(build(a.family,a.output)))
