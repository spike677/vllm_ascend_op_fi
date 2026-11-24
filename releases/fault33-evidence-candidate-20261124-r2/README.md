# 33只读取证补充件 r2

替代r1和黄区自改采集器。仅工具补充，不改变r7.4.1或授予33资格。源码提交 `1f54145fb46e00a48bd9c942b98b553e2b973876`。

修复aclrtBinaryGetFunction按名字加载句柄漏记；ByEntry后到时更新真实entry。每条bin增加binding sidecar，name-only key明确为null，不能用unknown=0证明可以解码。真实entry未知的目标FIA仍停止。

验证：Python36/36；124模拟ACL9/9（含9000条完整采集）；CANN9.1 SDK无设备编译。NPU运行0次，A3实际验证尚未进行。

- 源码SHA256：`2d472c2e3268080c52bfa3ba497e6239a22f77e062849bbed379e5177578a835`，8366字节。
- 参考库SHA256：`c87db1e157bca28e395e1c8280b0958fdea2d057b29affa21576c727996e023e`，171536字节。
- 参考库为A2镜像内aarch64构建，只供身份和编译参考；A3按实际SDK重建并回传新receipt/SHA，不能直接以此取得资格。
- 构建回执：docs/worker_core_v1/fault33_capture_r2_validation_20261124/build-receipt.json；包含源码提交、编译器、11个实际头文件SHA、组件版本和build-id。

## 部署顺序

1. 外层目录先执行 `sha256sum -c SHA256SUMS`。
2. 新建私有工作目录，解压冻结r7.4.1源码tar，再覆盖解压本r2 ZIP。不修改现役安装或冻结文件。
3. 在该目录校验manifest的所有文件（包括参考库），保留原始JSON：

```bash
python - <<'PY'
import json,hashlib,pathlib
m=json.load(open('manifest.json'))
for f in m['files']:
 p=pathlib.Path(f['path']);b=p.read_bytes()
 assert len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256'],f['path']
d=m['base_dependency']; assert hashlib.sha256(pathlib.Path(d['path']).read_bytes()).hexdigest()==d['sha256']
print('verified',len(m['files']))
PY
```

4. 以 `docs/worker_core_v1/fault33_candidate_windows_20261124.md` 的r2勘误为准：用tools/build_fault33_capture.py在A3实际CANN9.1 SDK构建；核对运行时加载的库SHA；窗口一ALL=1、串行128/512/2304三形状，各4token，两rank分别封存审查。
5. 旧执行单只复用可信RPC/加载/flush操作，不继续旧2304单形状流程。bin和`.bin.binding.json`同时保留，使用本包review_launch_capture_v2.py。
6. 回传字段按yellow_intake_20261124_fault33_full.md：唯一FIA decode映射、launch key/blockDim/tiling边界、实际mask/KV长度/block_table；GDN和unified_attention只计数。原件留黄区。
7. 窗口一审查通过后才配置窗口二，固定真实Q/K/V每形状每rank控制≥99%并列全量差异位置，任何门禁失败即停。不能把公开候选自洽当作真实kernel已经匹配。

已收到且未变的9库身份和金丝雀不单独重采。全部通过后才谈r7.5适配和两owner注入。本包由负责人转交，Codex不直接联系黄区。

## CPU复核

```bash
PYTHONPATH=src:.:fault39/src:tests/worker_extension python -m unittest test_launch_capture_v2 test_fault33_real_entry test_fault33_candidate_decode test_fault33_candidate_partitions test_paged33_difference_locations test_paged33_per_head_control test_paged33_private_capture test_fault33_capture_worker -v
# Linux/g++；假ACL，不调用NPU
python tests/worker_extension/native_capture_cpu_probe.py /tmp/new-mock-result.json
```

Qwen线产品统一正式测试请用r7.4.1；本包同时附最终测试补充单，不另做r7.3.1正式测试。冻结产品文件保持不变。
