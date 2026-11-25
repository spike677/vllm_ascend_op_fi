"""Package row12 only from two independently reviewed real-model receipts."""
import argparse,hashlib,json,subprocess,tarfile,zipfile
from pathlib import Path
from reference_adapters.review_lm_head import review,VARIANTS
ROOT=Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()

def build(evidence,source_archive,destination):
    evidence=Path(evidence);destination=Path(destination)
    if destination.exists():raise ValueError('IMMUTABLE_DELIVERY_EXISTS')
    controller=json.loads((evidence/'controller.json').read_bytes())
    if (controller.get('status') not in ('FINISHED','CASE_FAILED') or controller.get('isolation_removed') is not True
            or len(controller.get('cases',[]))!=2):
        raise ValueError('A2_WINDOW_NOT_FINISHED')
    with tarfile.open(source_archive) as t:manifest_bytes=t.extractfile('source_manifest.json').read()
    source=json.loads(manifest_bytes);files={f['path']:f['sha256'] for f in source['files']}
    records={}
    for variant in VARIANTS:
        cases=[c for c in controller['cases'] if c['id'].endswith(variant.replace('_','-'))]
        if len(cases)!=1 or cases[0].get('exit_code') not in (0,1):raise ValueError('A2_CASE_NOT_PASSED:'+variant)
        case=cases[0]
        if sha(manifest_bytes)!=case['review']['source_manifest_sha256']:raise ValueError('A2_SOURCE_MANIFEST_MISMATCH')
        raw=(evidence/case['id']/'persistent_product.json').read_bytes();data=json.loads(raw);r=review(data)
        if not r['passed'] or r['variant']!=variant:raise ValueError('A2_INDEPENDENT_REVIEW_FAILED:'+variant)
        if case['exit_code']==1:
            original_raw=(evidence/case['id']/'independent-review.json').read_bytes()
            original=json.loads(original_raw)
            if (original.get('passed') is not False or original.get('issues')!=['OWNER_MISMATCH']
                    or original.get('receipt_sha256')!=sha(raw) or original.get('variant')!=variant):
                raise ValueError('A2_EXECUTION_FAILURE_NOT_REVIEWER_OWNER_BUG:'+variant)
            r['review_correction']=dict(original_review_sha256=sha(original_raw),original_process_exit=1,
                reason='Old reviewer equated group owner with random provider owner; new review verifies complete ticket/worker/generation chain and HTTP request mapping.',
                original_receipt_unchanged=True)
        # Preserve exact combination fields from the real all-rank configuration,
        # not from the harness name or a remembered image tag.
        r.update(receipt_sha256=sha(raw),source_commit=source['base_commit'],
                 source_manifest_sha256=sha(manifest_bytes),configuration=data.get('configure'),
                 window_seconds=controller['elapsed_seconds'],isolation_removed=True,
                 raw_receipt_retained_on_124=True)
        r['reviewer_sha256']=sha((ROOT/'reference_adapters/review_lm_head.py').read_bytes())
        records[variant]=r
    names=('__init__','lm_head_rows','lm_head_provider','lm_head_product','persistent_probe','review_lm_head')
    blobs={f'reference_adapters/{n}.py':(ROOT/f'reference_adapters/{n}.py').read_bytes() for n in names}
    for p,b in blobs.items():
        if p.endswith('review_lm_head.py'):continue # separately CPU-tested independent reviewer
        if sha(b)!=files.get(p):raise ValueError('A2_EXECUTION_SOURCE_CHANGED:'+p)
    for n in ('rows','provider','review'):
        p='tests/worker_extension/test_reference_lm_head_'+n+'.py';blobs[p]=(ROOT/p).read_bytes()
    doc='docs/worker_core_v1/reference_12_32_34_handoff_20261125.md';blobs[doc]=(ROOT/doc).read_bytes()
    dependencies={}
    for p in ('external_persistent_runtime','external_persistent_regions','persistent_product_host',
              'persistent_ingress','product_host','runtime','providers/_scope'):
        file='src/demoncat/runtime/vllm_ascend/worker_extension/'+p+'.py'
        frozen=subprocess.check_output(['git','show','59f6aea5:'+file],cwd=ROOT)
        current=subprocess.check_output(['git','show','HEAD:'+file],cwd=ROOT)
        if current!=frozen:raise ValueError('FROZEN_RUNTIME_CHANGED:'+file)
        dependencies[file]=sha(frozen)
    blobs['a2-review.json']=(json.dumps(records,indent=2)+'\n').encode()
    blobs['HANDOFF.md']=('''# 第12行 v2 持久权重行参考补充

仅列manifest的两个固定档，A2/0.26/CANN9.1/社区Qwen3.8 W8A8/TP2实机结果，不授予A3资格。
独立解压，需要Python3.12、torch/torch_npu2.10、冻结r7.4.1三个wheel及完整源码（含experiments）。
不覆盖冻结目录；将本目录、r7.4.1根目录、src、fault39/src和experiments/worker_core_v11_20261118加入PYTHONPATH。
实际lm_head布局、原权重读取和sampler路径须先审查；A2源码PIN不能盲目替换成A3新SHA。
静态复制/预打包/tied embedding/量化head未知时拒绝；只改拥有完整token10/11/12行组的分片，其余rank守卫。
开启期间全部真实请求读到改写权重；关闭时drain后同对象/分配代字节恢复，不覆盖后来合法更新。
写入计数每owner一次，读取计数为真实prefill/decode次数；非owner不制造写入。
两个owner、实际目标logits/概率和sampler消费、负例、逐位金丝雀见a2-review.json及原回执SHA。
可信PersistentSite注册采用LmHeadCollapseProvider/LmHeadSwapProvider、region_kind=persistent_weight，
必须保留部署鉴权、身份、许可、独占ingress和真实drain；本包验收RPC不能暴露公网。
完整命令与适用边界见docs中的交接单；CLI映射由黄区补入，资格以实际注册为准。

```bash
python -m unittest discover -s tests/worker_extension -p 'test_reference_lm_head*.py'
DCAT_REFERENCE_ROW12=row12_lm_head_collapse python -m reference_adapters.lm_head_product
python -m reference_adapters.review_lm_head /evidence/persistent_product.json --output /evidence/independent-review.json
```

交换档使用新的evidence目录和row12_lm_head_swap。原始张量留验收方，只交审查摘要和SHA。
''').encode('utf-8')
    manifest=dict(schema='dcat.reference.lm_head.supplement.v1',variants=list(VARIANTS),frozen_base='59f6aea5',
        qualification_granted=False,dependency_sha256=dependencies,
        files={p:dict(bytes=len(b),sha256=sha(b)) for p,b in sorted(blobs.items())})
    blobs['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    destination.mkdir(parents=True);out=destination/'reference-lm-head.zip'
    with zipfile.ZipFile(out,'w') as z:
        for p,b in sorted(blobs.items()):
            info=zipfile.ZipInfo(p,date_time=(2026,11,25,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
    with zipfile.ZipFile(out) as z:
        for p,b in blobs.items():assert z.read(p)==b
    (destination/'manifest.json').write_bytes(blobs['manifest.json'])
    (destination/'a2-review.json').write_bytes(blobs['a2-review.json'])
    (destination/'SHA256SUMS').write_bytes((sha(out.read_bytes())+'  reference-lm-head.zip\n'+sha(blobs['manifest.json'])+'  manifest.json\n').encode())
    return dict(path=str(out),bytes=out.stat().st_size,sha256=sha(out.read_bytes()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--source-archive',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(build(a.evidence,a.source_archive,a.output)))
