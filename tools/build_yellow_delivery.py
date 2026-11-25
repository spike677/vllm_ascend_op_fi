"""Immutable aggregate handoff; never rewrites an earlier index or release."""
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LIMIT=100_000_000

def sha(b):return hashlib.sha256(b).hexdigest()
def build(name):
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name):raise ValueError('DELIVERY_NAME')
    folder=ROOT/'releases'/name;index=ROOT/'releases'/('YELLOW-DELIVERY-INDEX-'+name.removeprefix('yellow-delivery-')+'.md')
    if folder.exists() or index.exists():raise ValueError('IMMUTABLE_DELIVERY_EXISTS')
    selections=[
      ('T1','releases/demoncatpfi-rc3-preview-20261124-r7.4.1','冻结Qwen完整发行及原校验清单'),
      ('T1','releases/qwen-r741-phased-handoff-20261124','分阶段正式测试/33补提交接'),
      ('T7','releases/fault33-evidence-candidate-20261124-r2','33候选采集工具r2，非放行'),
      ('T7','releases/fault33-evidence-candidate-20261124-r2-k3','33 K3补充'),
      ('T1','releases/yellow-tools-20261123-r2','原只读采集/提取工具'),
    ]
    selections += [('T1',p.as_posix(),'已通过A2的参考补充；待黄区审查') for p in sorted(Path('releases').glob('yellow-reference-*-r1'))]
    selections += [('T1',p,'交接/执行/台账') for p in [
      'docs/worker_core_v1/reference_adapters_execution_20261124.md','docs/worker_core_v1/reference_adapter_a2_20261124.json',
      'docs/worker_core_v1/fault_library_coverage_20261124.md','docs/worker_core_v1/fault_library_coverage_20261124.json',
      'docs/worker_core_v1/external_v2_handoff_20261124.md','docs/worker_core_v1/r74_v2_acceptance_20261124.md',
      'docs/worker_core_v1/product_effect_test_20261124.md','docs/worker_core_v1/yellow_intake_20261125_delivery.md',
      'docs/worker_core_v1/fault33_n2_existing_capture_20261124.md']]
    selections += [('T6','docs/worker_core_v1/v4_d0_d1_v030_handoff_20261125.md','V4 0.30新目标，只读；旧0.32噪声审查不能借用')]
    blobs={};records={}
    for task,name0,purpose in selections:
      p=ROOT/name0
      if not p.exists():raise ValueError('MISSING:'+name0)
      for f in sorted(p.rglob('*')) if p.is_dir() else [p]:
        if f.is_dir():continue
        if f.is_symlink():raise ValueError('SYMLINK_NOT_ALLOWED')
        key=f.relative_to(ROOT).as_posix();raw=f.read_bytes()
        if len(raw)>=LIMIT:raise ValueError('GITHUB_FILE_LIMIT:'+key)
        blobs[key]=raw;records[key]=dict(bytes=len(raw),sha256=sha(raw),task=task,purpose=purpose)
    manifest=dict(schema='dcat.yellow.aggregate.v1',delivery=name,
      source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      qualification_granted=False,files=records)
    manifest_raw=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    folder.mkdir();archive=folder/'yellow-delivery.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
      for key,raw in sorted(dict(blobs,**{'manifest.json':manifest_raw}).items()):
        info=zipfile.ZipInfo(key,date_time=(2026,11,25,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,raw)
    if archive.stat().st_size>=LIMIT:raise ValueError('AGGREGATE_GITHUB_LIMIT')
    with zipfile.ZipFile(archive) as z:
      assert set(z.namelist())==set(blobs)|{'manifest.json'}
      for key,raw in blobs.items():assert z.read(key)==raw
    (folder/'manifest.json').write_bytes(manifest_raw)
    sums=sha(archive.read_bytes())+'  yellow-delivery.zip\n'+sha(manifest_raw)+'  manifest.json\n'
    (folder/'SHA256SUMS').write_bytes(sums.encode())
    base='https://github.com/spike677/vllm_ascend_op_fi/blob/'+name+'/'
    text='# 黄区统一交付入口 '+name+'\n\n'
    text+='PC拉取本分支/同名冻结标签后，将汇总ZIP和SHA256SUMS转入黄区。Codex不直接向黄区发送。旧索引和发行不改；后续交付使用新索引文件和新标签，禁止覆盖本标签。\n\n'
    text+='本包不自动登记、不授予A3资格。r7.4.1仍是Qwen正式包；补充件按manifest分别审查。第12/32/34行新变体及T2安全拒绝修复尚未在此快照交付，不可把旧失败工厂当作通过项。V4只读步骤已改0.30，D包尚待证据。\n\n'
    text+='|汇总文件|字节数|SHA256|\n|---|---:|---|\n'
    for p in [archive,folder/'manifest.json',folder/'SHA256SUMS']:
      raw=p.read_bytes();key=p.relative_to(ROOT).as_posix();text+=f'|[{key}]({base}{key})|{len(raw)}|`{sha(raw)}`|\n'
    text+='\n校验：Linux使用 `sha256sum -c SHA256SUMS`；Windows使用 `Get-FileHash -Algorithm SHA256` 对照。解压后manifest逐文件字节数/SHA必须一致；原r7.4.1目录继续按其DELIVERY-SHA256SUMS检查并在独立环境安装。原包不能直接用开发目录替代。\n\n'
    text+='一页工具：先解压r7.4.1源码tar；按docs/worker_core_v1/product_effect_test_20261124.md运行product_effect_probe及--verify。v2接线按external_v2_handoff；不能把transient示例当作持久权重恢复协议。\n\n|GitHub路径|字节数|SHA256|任务|用途|\n|---|---:|---|---|---|\n'
    for key,r in sorted(records.items()):text+=f"|[{key}]({base}{key})|{r['bytes']}|`{r['sha256']}`|{r['task']}|{r['purpose']}|\n"
    index.write_bytes(text.encode());return dict(index=str(index.relative_to(ROOT)),zip=str(archive.relative_to(ROOT)),sha256=sha(archive.read_bytes()),bytes=archive.stat().st_size,files=len(records))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');print(json.dumps(build(p.parse_args().name)))
