"""Read-only V4 0.30 supplement with the frozen source package dependency closure."""
import argparse,hashlib,io,json,subprocess,tarfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()

def build(destination):
    destination=Path(destination)
    if destination.exists():raise ValueError('IMMUTABLE_DELIVERY_EXISTS')
    base='59f6aea5'
    frozen=['fault_injection','tools/extract_registration_record.py',
        'experiments/worker_core_v11_20261118/dsv4_noise_capture.py',
        'experiments/worker_core_v11_20261118/dsv4_observe_probe.py',
        'tests/worker_extension/test_dsv4_noise.py']
    raw=subprocess.check_output(['git','archive','--format=tar',base,'--',*frozen],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        blobs={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    dependencies={p:sha(b) for p,b in blobs.items()}
    added=['fault_injection/dsv4_noise_v030.py','tools/review_dsv4_noise_v030.py',
        'tools/extract_dsv4_v030_record.py','tests/worker_extension/test_dsv4_noise_v030.py',
        'docs/worker_core_v1/v4_d0_d1_v030_handoff_20261125.md']
    for p in added:
        if p in blobs:raise ValueError('FROZEN_FILE_WOULD_BE_REPLACED:'+p)
        blobs[p]=(ROOT/p).read_bytes()
    blobs['HANDOFF.md']=('''# V4 0.30 只读噪声审查补充

不修改r7.4.1或旧0.32工具，不包含D2注入资格；黄区完整数组尚未回传。
目标为A3、V4-Flash-0731-W8A8、vLLM0.30.0（保留实际+empty后缀）、
vllm-ascend0.30.0、CANN9.2.1、TP8/DP1。EP必须是实际bool并给ep_source。
不能将版本改写为旧0.32或将未知EP填True。

Python3.12；离线审查/提取只需标准库。可信Worker采集另需部署本身的torch/torch_npu，
通过部署已有的鉴权RPC调用，不创建Worker，不暴露任意代码RPC。
本包携带完整冻结fault_injection源码以避免Python常规包遮蔽新增模块，
所有旧文件逐字节来自r7.4.1 Git blob；不复制到现有安装目录，独立解压运行。

```bash
export PYTHONPATH="$PWD:$PWD/tests/worker_extension"
python -m unittest discover -s tests/worker_extension -p 'test_dsv4_noise*.py' -v
python tools/review_dsv4_noise_v030.py /private/calibration.json --output /private/summary.json
python tools/extract_dsv4_v030_record.py noise /private/summary.json
python tools/extract_dsv4_v030_record.py observe /private/observe.json
```

calibration.json是{combination,calibration}，每条prompt两个独立Worker组各至少5次、
每rank四步完整hidden/logits和top5，数组只留黄区。可选phases为normal/fault/off/health，
原文件格式与旧噪声工具一致，仅组合身份和summary schema区分0.30。
全有限、token稳定，两两最大距离×1.5；字节恢复和资源清零不放宽。
提取每段≤4096字节，超长拒绝、不截断；逐段原样粘贴，原件留黄区。
不包含缺失的dispatch/combine动态行映射，不能把只观测摘要当8或13已适配。
完整只读步骤及router两分支陷阱见docs中的交接单。
''').encode('utf-8')
    manifest=dict(schema='dcat.v030.noise.tools.v1',frozen_base=base,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        qualification_granted=False,dependency_sha256=dependencies,
        files={p:dict(bytes=len(b),sha256=sha(b)) for p,b in sorted(blobs.items())})
    blobs['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    destination.mkdir(parents=True);out=destination/'v030-noise-tools.zip'
    with zipfile.ZipFile(out,'w') as z:
        for p,b in sorted(blobs.items()):
            info=zipfile.ZipInfo(p,date_time=(2026,11,25,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
    with zipfile.ZipFile(out) as z:
        for p,b in blobs.items():assert z.read(p)==b
    (destination/'manifest.json').write_bytes(blobs['manifest.json'])
    (destination/'SHA256SUMS').write_bytes((sha(out.read_bytes())+'  v030-noise-tools.zip\n'+sha(blobs['manifest.json'])+'  manifest.json\n').encode())
    return dict(path=str(out),bytes=out.stat().st_size,sha256=sha(out.read_bytes()),files=len(blobs))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    print(json.dumps(build(p.parse_args().output)))
