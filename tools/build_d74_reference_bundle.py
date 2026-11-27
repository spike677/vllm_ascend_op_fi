"""Build a deterministic, non-installable D74 source-reference archive from Git blobs."""
import argparse
import ast
import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

BASE = "e38c4113b421ac979699071a60e80d3babd05e07"
ROUTE = "27cadf797014a6a34bbee23e5eaeae1977f5488f"
PROBE = "743611c4a8de9a7c0f762e6a3a38f7a55c350304"
R3 = "065931d986e44a970af6c8ed3af1423b24f1258a"
W = "src/demoncat/runtime/vllm_ascend/worker_extension/"
DOCS = "docs/worker_core_v1/"


def git(*args):
    return subprocess.check_output(["git", *args])


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


VERIFY = '''"""Reference-only byte verification and six original CPU contract tests."""
import argparse, hashlib, importlib.util, json, sys, types, unittest
from pathlib import Path
root = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--tests', action='store_true')
args = p.parse_args()
manifest = json.loads((root/'manifest.json').read_bytes())
for row in manifest['files']:
    f = (root/row['path']).resolve()
    if not f.is_relative_to(root) or not f.is_file():
        raise SystemExit('REFERENCE_PATH_INVALID:'+row['path'])
    raw = f.read_bytes()
    if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise SystemExit('REFERENCE_SHA_MISMATCH:'+row['path'])
print(json.dumps({'verified':True,'files':len(manifest['files']), 'qualification_granted':False}))
if args.tests:
    # Isolated namespace loads ONLY the two archived stdlib modules. It does not
    # import or simulate the production fault_injection package initializer.
    folder=root/'reference/27cadf79'
    pkg=types.ModuleType('fault_injection')
    pkg.__path__=[str(folder/'fault_injection')]
    sys.modules['fault_injection']=pkg
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        mod=importlib.util.module_from_spec(spec)
        sys.modules[name]=mod
        spec.loader.exec_module(mod)
        return mod
    load('fault_injection.product_contract',folder/'fault_injection/product_contract.py')
    load('fault_injection.dsv4_product_v030',folder/'fault_injection/dsv4_product_v030.py')
    module=load('reference_contract_tests',folder/'tests/worker_extension/test_dsv4_product_v030.py')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    print(json.dumps({'tests_run':result.testsRun,'failures':len(result.failures),
        'errors':len(result.errors),'passed':result.wasSuccessful(),
        'real_v4_environment_validated':False}))
    raise SystemExit(not result.wasSuccessful())
'''


def build(source_commit, out):
    source_commit = git("rev-parse", source_commit).decode().strip()
    payload, provenance = {}, {}

    def add(path, data, origin):
        if path in payload:
            raise ValueError("DUPLICATE_ARCHIVE_PATH:" + path)
        payload[path], provenance[path] = data, origin

    def blob(commit, path, prefix):
        add(prefix + path, git("show", commit + ":" + path), dict(commit=commit, path=path))

    for path in ("fault_injection/dsv4_product_v030.py", "fault_injection/product_contract.py",
                 "tests/worker_extension/test_dsv4_product_v030.py",
                 DOCS + "dsv4_v030_product_route_preparation_20261125.md"):
        blob(ROUTE, path, "reference/27cadf79/")
    add("reference/27cadf79.patch", git("show", "--format=fuller", "--binary", ROUTE),
        dict(commit=ROUTE, kind="original_commit_patch_not_full_branch"))
    for name in ("v4_d13_independent_release_20261125.md", "v4_public8_d1_review_plan_20261125.md",
                 "v4_d1_targeted_capture_20261125.md", "v4_v030_probe_r3_handoff_20261125.md"):
        blob(R3, DOCS + name, "reference/r3/")
    for path in (DOCS + "v4_upgrade_001_review_20261127.md",
                 DOCS + "v4_router_boundary_capture_design_20261127.md",
                 DOCS + "v4_upgrade_001_cpu_20261127.json",
                 DOCS + "v4_upgrade_001_sources_20261127.json",
                 "tools/review_v4_installed_sources.py",
                 "experiments/worker_core_v11_20261118/dsv4_boundary_observer.py",
                 "tests/worker_extension/test_v4_source_intake.py",
                 "tests/worker_extension/test_dsv4_boundary_observer.py"):
        blob(PROBE, path, "reference/743611c4/")
    for path in ("docs/decision-log/20261127.md", DOCS + "runtime_026_version_bound_checklist_20261127.md"):
        blob(source_commit, path, "")

    runtime = [W + x for x in (
        "v026_text_observer.py", "v026_source_review.json", "v026_installed_source_review.json",
        "preferred_a2_source_review.json", "v026_generation_observer.py", "v026_inspection.py",
        "serving_context.py", "serving.py", "product_observer.py", "product_request.py",
        "continuous_binding.py", "product_batch.py", "product_config.py", "product_binding.py",
        "product_host.py", "product_errors.py", "product_registration.py", "deployment_identity.py",
        "product_native.py", "external_provider.py", "external_control.py", "external_sampling.py",
        "external_persistent_runtime.py", "external_persistent_regions.py", "persistent_product_host.py",
        "persistent_kv_release.py")]
    runtime += ["fault_injection/product_contract.py", "fault_injection/environment.py"]
    locations = []
    for path in runtime:
        blob(BASE, path, "reference/r8/")
        raw = payload["reference/r8/" + path]
        definitions = []
        if path.endswith(".py"):
            definitions = [dict(name=n.name, line=n.lineno, end_line=n.end_lineno)
                           for n in ast.walk(ast.parse(raw))
                           if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
        locations.append(dict(path=path, commit=BASE, sha256=hashlib.sha256(raw).hexdigest(),
                              definitions=definitions))
    add("runtime-source-locations.json", encoded(locations), dict(kind="generated_frozen_source_index"))
    readme = f'''# D74 参考资料，不是安装包

**没有在真实 V4 0.30/0.32 环境验证过。** 本包不授予执行或验收资格。

D74 取代旧文档中的蓝区出包前置：0.30/0.32 的运行时、机制、工具、包与验收由黄区负责。
S0-UPGRADE-001 现在是咨询，不阻塞 S1/R0/R1/R2。蓝区不再等待源码来发布 D13/公共 8 包。

- reference/27cadf79：原提交 {ROUTE} 的长请求关闭准入、依赖、六项 CPU 测试和设计说明。
  helper 只生成 executable=false；enable_prepared_route 始终拒绝。它硬编码 global ranks 0–7、TP8/EP/DP1、
  软件版本字符串、2500–4092 token/4 token 等，仅是参考假设，须以实际环境重审，不可直接接入产品。
  patch 仅该提交的三个文件差异；不要把其整个开发分支当成冻结 r8 安装。
- reference/r3：四份观测/审查门槛文档。历史蓝区责任与等待要求已由 D74 覆盖。
  r3 历史观测结果不等于 D13 或公共 8 注入资格。
- reference/743611c4：源码审查和未接线的 Python 观测核心；旧 CPU 18/18 是 stdlib 核心测试，
  不是 V4 tensor fixture、设备消费事件、Worker 接线或实机证明。此次不继续开发它。
- reference/r8：冻结 {BASE} 的选定源码原始字节，只供对照，不是完整运行时。
- docs/worker_core_v1/runtime_026_version_bound_checklist_20261127.md：位置、目的、迁移重审面。
  runtime-source-locations.json 给出冻结文件 SHA 与函数行号。

不把任何 reference/ 目录加到生产 PYTHONPATH；不覆盖现役 SITE；没有 wheel、Native 库、许可或登记。
所有源文件从指定 Git blob 读取，绕过 Windows 换行转换。manifest.json 记录逐文件来源、字节数、SHA。

取件后先按交付索引核对 ZIP SHA，解压独立目录，然后：

```sh
python -I verify_reference.py
python -I verify_reference.py --tests
```

Python >=3.10，只需标准库。测试器建立隔离 namespace，加载原始 helper 与 product_contract，
不执行生产包 initializer；只验证原有六项 CPU 契约，不声称整包安装检查或 NPU 验收。

黄区另开 0.30 分支，独立会话审运行时；R1 两 owner/负例通过后逐项迁移重验并重新登记。
LEDGER-FINAL 分列 0.26/0.30；0.26 资格不得复制为 0.30。蓝区以后只按请求做源码咨询。
'''
    add("README.md", readme.encode("utf-8"), dict(kind="D74_reference_notice", source_commit=source_commit))
    add("verify_reference.py", VERIFY.encode("utf-8"), dict(kind="stdlib_reference_verifier", source_commit=source_commit))
    manifest = dict(schema="dcat.d74.reference.v1", source_commit=source_commit,
                    real_v4_environment_validated=False, installable=False,
                    files=[dict(path=p, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                                origin=provenance[p]) for p, raw in sorted(payload.items())])
    manifest_raw = encoded(manifest)
    sums = "".join(f"{row['sha256']}  {row['path']}\n" for row in manifest["files"])
    sums += hashlib.sha256(manifest_raw).hexdigest() + "  manifest.json\n"
    payload["manifest.json"] = manifest_raw
    payload["SHA256SUMS"] = sums.encode()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "d74-v030-reference-20261127.zip"
    if target.exists():
        raise ValueError("REFERENCE_ARCHIVE_ALREADY_EXISTS")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, raw in sorted(payload.items()):
            info = zipfile.ZipInfo(path, (2026, 11, 27, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    (out / "manifest.json").write_bytes(manifest_raw)
    (out / "SHA256SUMS").write_bytes(sums.encode())
    print(json.dumps(dict(zip=str(target), bytes=target.stat().st_size,
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest(), files=len(manifest["files"]))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_commit, args.output)
