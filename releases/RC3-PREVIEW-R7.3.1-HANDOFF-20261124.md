# RC3 preview r7.3.1（2026-11-24）

从r7.3发行提交1b816dd6开出，旧标签和包不变。标签 `demoncatpfi-rc3-preview-20261124-r7.3.1`。
源码提交 `524addfe410f1343fa555ea107215d9afe99cdbb`。维护分支 `codex/rc3-r7-3-1-20261124`。
许可source_manifest_sha256填写 **source-review.json** 的SHA：`846d9d628697dd382fad6de28d2272f16c562efa41e27cb46c060c4787a3daa3`，不是release-manifest.json。

## 内容与门禁

所有rank契约拒绝变为带类型的ProductContractRefused，CLI映射exit4并保留拒绝码；控制RPC结构化，Batch非串行/不完整金丝雀不再用预期拒绝杀死引擎。
黄区CLI需按docs/worker_core_v1/r731_control_handoff_20261124.md接product_control_reply；蓝区没有黄区CLI源码，不声称已替其修改。
直接RPC允许HTTP200+结构化body。模型/kernel异常不吞掉，不通过重试伪造成功。

新增统一效果工具product_effect_probe.py和ProductClient适配器；同服务baseline/fault/off、两个owner、一页≤4096字节、八项计数及元素数、原件SHA。
A2真实18和Batch均通过；18一页2172字节，Batch2234字节，重算逐字节一致，verify通过。
18仅top5变化，token/text不变；Batch两个owner均token/text/top5变化；OFF精确恢复。完整fault logits未采集为null，不能把top5称作完整logits。
HCCL仍需黄区真实控制/回执适配器，无HCCL或A3实测资格。工具运行/映射/一次性负例详见product_effect_test_20261124.md。

- CPU：1216项，1214通过，2项既有废弃研究入口带理由跳过。
- A2 / Ascend910B4 / 0.26 / CANN9.1 / Qwen3.8社区W8A8 / TP2：真实负例、引擎存活、清理及两个owner正例；详见r731_control_20261124原件SHA和审查摘要。
- 无设备安装：network=none、无设备映射、无Worker/NPU初始化；三个wheel安装源码逐字节匹配，见install-check.json。
- 316个机制文件与r7.3逐字节一致；Worker控制路径变化单独列source-delta.json。
- 三个wheel版本：demoncatpfi 0.4.0rc6.post4；worker-extension/fault39-core 0.3.0rc6.post4。

## 安装与交付

完整发行目录直接包含docs、源码tar.gz、三个wheel、清单、SHA和安装回执。负责人转交，Codex不直接发送黄区。

```sh
sha256sum -c DELIVERY-SHA256SUMS
python -m pip install --no-index --no-deps wheels/*.whl
```

|文件|SHA256|
|---|---|
| `wheels/demoncat_fault39_core-0.3.0rc6.post4-py3-none-any.whl` | `1c332a4e23e06a4a075846feb13cd265f7dd9c7db758de78aa4cd638556d742e` |
| `wheels/demoncat_worker_extension-0.3.0rc6.post4-py3-none-any.whl` | `2f2745a0ab7f8401f9f918780aa73b4c960ee389ca14bad89fd18d9a2c224aef` |
| `wheels/demoncatpfi-0.4.0rc6.post4-py3-none-any.whl` | `475661f1e976a5f9563ff3b20b971a02d4ce4601cb84c3254682742c97da218c` |

黄区master按最新任务直接从r7.3.1开始正式一次测试；结果用于验收和登记，不重复旧四页矩阵。未验收组合仍关闭。
v2持久区域、33 A3适配、8 V4不在此维护包中；分别进入r7.4/D包，真实回传审查后才进入r8登记。无新增YELLOW组合。
临时隔离只针对124卡2/5，窗口结束全部恢复；操作记录在审查摘要。vLLM退出时的semaphore/shared_memory警告单独披露，不与已证明为零的fault租约混淆。
