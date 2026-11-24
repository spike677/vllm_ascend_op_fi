# RC3 preview r7.2 维护交付（2026-11-24）

独立维护分支 `codex/rc3-r7-2-20261124`；标签 `demoncatpfi-rc3-preview-20261124-r7.2`。
从冻结 r7.1 发行提交 `ef53263bdf9b9139ce92c4c8eb27bbb5e98c124d` 开始，仅修控制拒绝与 enable 失败恢复。
r7.1 的标签、包、许可身份不变；不含 r8 外部 provider、新模型适配或 33 A3 放行。

## 发行身份

- 源码提交：`676b1113b040d9b3e443eba0844a268069303b88`。
- 许可 `source_manifest_sha256` 应填本包 **source-review.json** 的 SHA256：`8fb4a1e8dde8325432076ada708c3c3a53a9f37ef4322404a54a777665f12261`，不是 release-manifest.json 的 SHA。
- 三个 wheel：demoncatpfi `0.4.0rc6.post2`；worker-extension / fault39-core `0.3.0rc6.post2`。
- [发行目录](demoncatpfi-rc3-preview-20261124-r7.2/)、[完整交付校验表](demoncatpfi-rc3-preview-20261124-r7.2/DELIVERY-SHA256SUMS)。
- 源码 tar.gz SHA256：`76e91221db4732de25796ee38dd185fd95c929ba74af1e887a38576c27c69a7a`。

| wheel | SHA256 |
|---|---|
| `demoncat_fault39_core-0.3.0rc6.post2-py3-none-any.whl` | `a55b23bfcf18710826bc268aab5643f45d0556f8a5dec187cb22b42356bb3004` |
| `demoncat_worker_extension-0.3.0rc6.post2-py3-none-any.whl` | `af37d2700334e98dbf5721df9324b57fd91b346402e30706e76214eef19c80de` |
| `demoncatpfi-0.4.0rc6.post2-py3-none-any.whl` | `d85d7cbc7f888c81ebf3a591d4487c61dcb13ebbdde96ce178f07287231611db` |

## 修复与验证

未知 profile / 控制鉴权拒绝返回结构化 exit_code=4，不再使异常逃出 Worker RPC。
enable 任一 rank 部分失败时，复用 Runtime 清理本 owner；没有取得 lease 的 prepare attachment 也清理。
未确认清零时保留恢复责任，公共 disable 可重试；不触碰其他 owner。
同一 ProductClient 可在确认清理后安装正确许可并重新 enable。真实模型 forward/kernel 异常没有被吞掉。

- CPU：1178 项，1176 通过，2 项已废弃研究公共入口按既有理由跳过。
- A2 / Ascend910B4 / vLLM 0.26 / vllm-ascend 0.26.0rc1 / CANN 9.1 / Qwen3.8 社区 W8A8 / TP2：两轮独立窗口，290.649 秒和 282.165 秒，均通过。
- 缺许可后两个 rank 的 leases/owners/bindings/wrappers 均为零，健康 HTTP 继续成功；同客户端重试及同 Worker 第二 owner 通过。
- 每 owner 真实消费者、区域恢复、故障资源清零和部署金丝雀精确恢复通过。
- NPU 后仅补充 Host 内部异常分类和未完成清理重试的 CPU 检查，未改设备/数值路径。实测 manifest 与最终源码身份分开保存。
- 无设备安装：三个 wheel 的 388 个 Python/JSON 文件逐字节匹配源码清单，无 Worker、无 NPU 初始化、无网络。
- 相对 r7.1，316 个机制文件逐字节一致；范围为 fault_injection/、fault39/src/、native_execution_strategy/。Worker 控制代码是本次有意修改的内容。

证据：[控制修复与实机结果](../docs/worker_core_v1/r72_control_recovery_20261124/README.md)、
[安装检查](demoncatpfi-rc3-preview-20261124-r7.2/install-check.json)、
[机制字节对照](demoncatpfi-rc3-preview-20261124-r7.2/mechanism-byte-comparison.json)。

## 负责人转交的升级与复核事项

1. 在独立环境按 DELIVERY-SHA256SUMS 校验全部文件，再安装三个 wheel；复用既有可信 bootstrap 与身份观测，不改 runner 绕过门禁。
2. 使用新源码 SHA 生成当前验收许可；不要把 r7.1 许可重命名用于 r7.2。未验收组合继续默认关闭。
3. 黄区 CLI 使用 `python -m demoncat.cli fault ...`。库的 ValueError 契约拒绝应映射退出码 4；内部/恢复失败为 5。蓝区不持有黄区 CLI，不能宣称其接线已验证。
4. 在同一引擎复核：未知 profile → 健康请求；缺许可 enable → status/disable/detach → 零资源 → 正确许可阳性。
   若退出码非零，测试脚本仍须采集结构化拒绝和清理结果，不能用 KeyError 中断后冒充负例通过。
5. 原十一项黄区阳性回报不因本控制修复自动作废；其身份、owner、逐组合提取文本和有效负向记录仍需审查。
   本包不新增登记，A2 结果不代表 A3；33 的 A3 launch/serializer/控制重算证据不足时仍关闭。

本次仅交负责人转交，没有向黄区发送消息或执行操作。
