# RC3 preview r7.3：外部 provider v1（2026-11-24）

基于 r7.2 `7ffd9a602184ec445cd9ef9b4a5666c25b012e98` 的独立分支 `codex/rc3-r7-3-20261124`。
标签 `demoncatpfi-rc3-preview-20261124-r7.3`。不包含 r8 的 V2/V4 适配、免许可登记、正式 bootstrap RPC。
旧标签与包不变，未验收组合继续关闭。本次仅交负责人转交，不直接发送黄区。

## 身份与安装

- 源码提交：`d010fb78c241d833c75dd04feb061a47579d5141`。
- 许可 `source_manifest_sha256` 使用 **source-review.json** 的 SHA256：`c7c2bd39c581011b5661e46f48ae30683aeeb70f59cde514fb63c05574817664`，不是 release-manifest.json。
- 三个 wheel 版本：demoncatpfi `0.4.0rc6.post3`；worker-extension、fault39-core `0.3.0rc6.post3`。
- [发行目录](./)；[全部交付 SHA](DELIVERY-SHA256SUMS)。
- 目录中直接提供 `docs/worker_core_v1/`、示例 `experiments/worker_core_v11_20261118/external_gain_example.py`、源码 tar.gz、三个 wheel、安装检查与字节对照。

在独立环境校验 DELIVERY-SHA256SUMS 后安装（该命令从发行目录运行）：

```sh
sha256sum -c DELIVERY-SHA256SUMS
python -m pip install --no-index --no-deps wheels/*.whl
```

基础环境沿用已审查 vLLM/torch/CANN，不把三个 wheel 当作完整推理环境。

| 文件 | SHA256 |
|---|---|
| `wheels/demoncat_fault39_core-0.3.0rc6.post3-py3-none-any.whl` | `b515013c0b8971d6192e5f6ad2ef6750f73806c6fa6ff2aa297f5f0c480562ad` |
| `wheels/demoncat_worker_extension-0.3.0rc6.post3-py3-none-any.whl` | `60554e59d77e0fa95a7882b11cece0ae92883142ec2c00142e164de25f5fcf33` |
| `wheels/demoncatpfi-0.4.0rc6.post3-py3-none-any.whl` | `33474d32c7492a4da90640bd369879b6b10248079bb026982984d10e86a26361` |

## 门禁与实机范围

- CPU 1195 项：1193 通过，2 项既有废弃研究入口按理由跳过。
- 124 无设备安装检查见 install-check.json：三个 wheel 安装源码逐字节匹配；network=none，无设备映射，无 Worker/NPU 初始化。
- **316 个机制文件**与 r7.2 逐字节一致，逐文件 SHA 在 mechanism-byte-comparison.json；所有源码差异在 source-delta.json。Worker 接线及新增外部 site 是有意变更。
- A2 / Ascend910B4 / 0.26 / CANN9.1 / Qwen3.8 社区 W8A8 / TP2：模型示例和采样示例各在同组 Worker 完成两个 owner。
- 模型示例每 owner 每 rank applied/consumed/restored=1932，changed=1930，failed=0；采样示例对应计数均12，failed=0。
- 区域字节恢复、资源清零、关闭后金丝雀 hidden/logits 精确恢复通过；超范围负例通过，采样另有贪心不注入负例。
- 三个验证窗口及有界预清理全部结束、隔离已解除。性能仅记录，不作门禁。
- NPU 取证快照与最终发行源码分别记录；后补 external_control helper 经过 CPU 测试，等价结构化回包模式由实机 RPC 诊断验证。

## 黄区接入

详见 docs/worker_core_v1/external_provider_r73.md：可信部署先审核 factory 源码 SHA 与完整部署身份，再逐 rank register_external_site、configure_product_profiles、安装当前验收许可；宿主使用现有 ProductClient 生命周期。
CLI 仍由黄区使用 `python -m demoncat.cli fault ...` 并补可信命名映射。本包不接受普通用户传入代码、回调或裸地址，不授予外部能力产品资格。

采样 site 在真实 AscendSampler 的 `probs.div_(q)` 消费边界改变指数噪声 q，原采样器继续消费，设备读完后恢复 q。限定已审查源码、temperature>0、单请求、非推测路径；恢复 q 不撤销已经选择的 token。

v1 只支持瞬态区域。持久 weight_scale/int8 权重以及 KV cache 写入明确拒绝：仍缺全生命周期独占、别名/派生缓存一致性以及安全恢复点的证明，不能套用临时张量恢复。KV 错写会改变同请求后续 token，不能以恢复字节声称轨迹恢复。

## RPC、33 与登记

- r7.2 A2 实测：48 次正常/延迟/结构化拒绝控制调用均对齐；rank0 未捕获异常后，rank1 回包连续落后一拍。根因为底层 MultiprocExecutor 遇首个 FAILURE 提前退出，未排空其余 rank 队列。
- 自定义控制 RPC 应在 Worker 内返回结构化错误并校验每次 token 与完整 rank 集；诊断 helper 见 external_control。真正超时、传输异常或错位后停止并重建专用引擎，不靠补发查询冲刷队列。不吞掉模型/kernel 异常。
- 详见 rpc_rank_alignment_20261124.md。尚不能据此断言黄区第一起异常来自哪条探测调用，需要其原日志。
- 33 A3 继续关闭；fault33_a3_evidence_supplement_20261124.md 给出 sparseMode=3 的语义核对、分段完整捕获、精确版本 serializer、真实 A3 Q/K/V 控制检查≥99%的方法，不沿用 A2 分组猜测。
- 目前未收到 r7.2 各组合提取文本及有效负例原字段；yellow_registration_draft_20261124.md 仅为待审台账，不登记、不补默认身份。

此次 A2 结果只证明两个外部示例及声明范围；不代表黄区自有16项机制、A3、DeepSeek或EP已经验收。
