"""Immutable control-refusal supplement, separate from all frozen releases."""
import argparse,hashlib,json,zipfile,tarfile,subprocess
from pathlib import Path
from reference_adapters.review_refusal import review

ROOT=Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()

def build(evidence,destination,source_archive):
    evidence=Path(evidence);destination=Path(destination)
    if destination.exists():raise ValueError('IMMUTABLE_DELIVERY_EXISTS')
    raw=(evidence/'reference_refusal.json').read_bytes();result=review(json.loads(raw))
    if not result['passed']:raise ValueError('A2_REFUSAL_REVIEW_FAILED')
    controller=json.loads((evidence/'controller.json').read_bytes())
    if controller['status']!='FINISHED' or controller['isolation_removed'] is not True:
        raise ValueError('A2_WINDOW_NOT_FINISHED')
    source=controller['cases'][0]['review']
    with tarfile.open(source_archive) as tar:
        manifest_bytes=tar.extractfile('source_manifest.json').read()
    if sha(manifest_bytes)!=source['source_manifest_sha256']:raise ValueError('A2_SOURCE_MANIFEST_MISMATCH')
    source_files={r['path']:r['sha256'] for r in json.loads(manifest_bytes)['files']}
    result.update(receipt_sha256=sha(raw),source_commit=source['source_commit'],
        source_manifest_sha256=source['source_manifest_sha256'],
        combination=dict(hardware='Ascend910B4/A2',stack='vLLM0.26/CANN9.1',model='community Qwen3.8 W8A8',tp=2),
        window_seconds=controller['elapsed_seconds'],raw_receipt_retained_on_124=True)
    paths=['reference_adapters/'+n+'.py' for n in ('__init__','control_guard','refusal_probe',
        'product_probe','bootstrap','matmul_order','review_refusal')]
    paths+=['tests/worker_extension/test_reference_refusal_review.py']
    blobs={p:(ROOT/p).read_bytes() for p in paths}
    for p in paths:
        if p.startswith('reference_adapters/') and not p.endswith('review_refusal.py'):
            if sha(blobs[p])!=source_files.get(p):raise ValueError('A2_TESTED_SOURCE_CHANGED:'+p)
    # An independent module is importable alongside an old reference package;
    # it does not shadow that package's namespace or overwrite a frozen file.
    blobs['dcat_reference_control_guard.py']=blobs['reference_adapters/control_guard.py']
    blobs['a2-review.json']=(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode()
    blobs['HANDOFF.md']=('''# 参考适配器控制拒绝修复（T2）

独立补充件，不修改 r7.4.1、旧五个参考包或既有标签。不授予任何 A3 或新变体资格。
A2 / 0.26 / CANN9.1 / 社区Qwen3.8 W8A8 / TP2：缺点、身份漂移、超范围、重复缺点均结构化拒绝 exit4；
同一 client 随后真实图片请求四步的完整 hidden/logits SHA 一致；每rank租约、owner、binding、wrapper为零。
窗口331.238秒，结果和原回执SHA见a2-review.json。引擎在负例期间存活，验收完成后由工具主动关闭。

## 黄区可信 Worker 接线

需要 Python3.12、r7.4.1 三个wheel和其完整源码。独立解压，不覆盖发行目录。
将本目录加入可信 Worker 的PYTHONPATH，在黄区已有可信RPC包装类之前加入mixin：

```python
from dcat_reference_control_guard import ReferenceControlMixin
class ReferenceSafeExtension(ReferenceControlMixin, ExistingTrustedWorkerExtension):
    pass
```

`ExistingTrustedWorkerExtension` 是黄区现有、具有鉴权回调的Worker包装类，不能直接把示例测试RPC暴露到服务外网。
只处理 `reference_adapters.*` 工厂；公开机制及黄区其他provider不受影响。
在 attach/arm_canary 控制边界检查身份及静态选点，预检安装的临时hook随即移除，不运行模型、不写张量。
拒绝发生于未绑定状态时，rollback仅确认本机制没有binding/lease；已有所有权仍交原恢复路径处理。
实际kernel/forward异常不会被吞掉，也不宣称任意运行期故障都能安全恢复。

## A3最小复测

用实际不存在的候选、故意不匹配的部署身份、mode=single各一次，再重复缺候选一次。
每次应为exit4及原拒绝码，同client随后能完成真实请求；两个rank完整hidden/logits四步SHA与基线一致，资源全零。
不要把旧34无BF16权重候选的结论套用到A3；A3若有自然候选，需选择另一个确实不存在的静态候选。

## A2复现与CPU审查

源码同时需要r7.4.1的experiments目录，先核对manifest全部SHA。

```bash
export PYTHONPATH="$PWD:$R741_SOURCE:$R741_SOURCE/src:$R741_SOURCE/fault39/src:$R741_SOURCE/experiments/worker_core_v11_20261118"
python3 -m unittest discover -s tests/worker_extension -p test_reference_refusal_review.py -v
python3 reference_adapters/product_probe.py --mechanism 17 --reference row34_bf16_before_bias --refusal-only
python3 -m reference_adapters.review_refusal /evidence/reference_refusal.json --output /evidence/independent-review.json
```

真实运行仅限隔离的专用实例，模型/source/evidence挂载与原A2执行器一致；CPU命令不需要NPU。
本补充件不包含12/32/34新变体资格，其实机验收单独交付。
''').encode()
    dependencies={}
    for file in ['src/demoncat/runtime/vllm_ascend/worker_extension/product_host.py',
                 'src/demoncat/runtime/vllm_ascend/worker_extension/product_binding.py',
                 'src/demoncat/runtime/vllm_ascend/worker_extension/external_provider.py',
                 'fault_injection/engines/region_lease.py']:
        frozen=subprocess.check_output(['git','show','59f6aea5:'+file],cwd=ROOT)
        current=subprocess.check_output(['git','show','HEAD:'+file],cwd=ROOT)
        if frozen!=current:raise ValueError('FROZEN_RUNTIME_CHANGED:'+file)
        dependencies[file]=sha(frozen)
    manifest=dict(schema='dcat.reference.control.supplement.v1',frozen_base='59f6aea5',dependency_sha256=dependencies,
        qualification_granted=False,files={p:dict(bytes=len(b),sha256=sha(b)) for p,b in sorted(blobs.items())})
    blobs['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    destination.mkdir(parents=True)
    archive=destination/'reference-control.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for p,b in sorted(blobs.items()):
            info=zipfile.ZipInfo(p,date_time=(2026,11,25,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
    with zipfile.ZipFile(archive) as z:
        for p,b in blobs.items():assert z.read(p)==b
    (destination/'manifest.json').write_bytes(blobs['manifest.json'])
    (destination/'a2-review.json').write_bytes(blobs['a2-review.json'])
    (destination/'SHA256SUMS').write_bytes((sha(archive.read_bytes())+'  reference-control.zip\n'+sha(blobs['manifest.json'])+'  manifest.json\n').encode())
    return dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True);p.add_argument('--source-archive',required=True)
    a=p.parse_args();print(json.dumps(build(a.evidence,a.output,a.source_archive)))
