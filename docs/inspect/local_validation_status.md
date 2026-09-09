# Inspect V1 本轮交付状态（2026-09-09）

**状态：LOCAL_ADAPTERS_TESTED / LIVE_DEFERRED。不是全功能实机验收。**

| 项目 | 本轮结果 | 证据边界 |
| --- | --- | --- |
| 通用 Inspect 接口 | 已实现白名单 RPC、分页 CLI、三个 adapter | 本地 fixture 验证，不代表现场 vLLM 已加载 |
| 模型/参数观察 | 注册表 metadata、args/kwargs 嵌套路径、限额/TTL/清理测试通过 | 未在真实双卡安装本任务 observer |
| 量化解析 | 类型/指纹/源码合同接口、字段线索与 UNKNOWN 回退、reference/model scale 分离 | 未验证当前现场具体量化方法；没有捏造 activation scale |
| 候选与 DRAFT | 复用 Candidate；明确路径、锚定 selector、epoch、缺口、研究能力状态 | 不自动 arm；不能表达的 nested/alias/层精确选择拒绝 |
| QMM | 既有 typed schema、native formal/Run/weight/scale recorded adapter 完成 | 有真实已提交 candidate 回放；新增 typed/native 样例明确为 fixture，非本轮实机采集 |
| module→native | 未证明 | module_name=null，不按 shape、时序猜测 |
| backend 是否准确执行选择 | 导出等价检查、委托实际目标集核对、错误拒绝测试通过 | 当前仅 fixture；实机仍待 IDENTITY |
| lossless | 原 native fixture 回放通过；既有 lifecycle IDENTITY 委托测试通过 | 不能记为当前模型 lossless PASS |
| 原长任务 | 未修改源码/配置/extension；未发额外推理或 arm/disarm/remove | 仅曾只读核对已有记录；最后的远端复核被安全审核拒绝 |
| 最终服务/锁 | 本任务没有安装 observer、没有 arm、没有获取/释放 NPU_LOCK | 原任务最后已知仍运行、无 final_environment；当前状态未经最后复核，不声称新确认健康或已释放 |

## 本地验证

- `python -m unittest discover -s tests -q`：138/138 PASS（原有 91 项 + 本轮 47 项）。
- `tools/interactive_inspect.py --help`：PASS。
- `tools/replay_inspect_examples.py`：PASS，旧工件不覆盖。
- 本轮文档相对链接：PASS。
- `git diff --check`：PASS。
- 原 Research 分支 FakeRequant 数学测试：5/5 PASS（禁用 pyc 写入，只读运行，未改原分支）。
- 固定 SHA256 的原 Research extension 与 Inspect 组合导入：PASS；lazy state 默认 disabled。不是 vLLM/NPU 实机验证。
- 未安装 PyTorch/NPU 依赖来伪造实机验证；fixture 用受限 Tensor/Module 测试替身。

本地隔离回放目录：`artifacts/inspect/private/recorded_v1_final_20260909/`。

包含：`recorded_candidate.json`、`native_fixture.json`、`typed_probe_fixture.json`、`fixture_validation_result.json`、`inspect_session_manifest.json`。该目录已忽略，不上传现场路径与模型信息。源码 hash 记录在 manifest；早期中间目录保留未覆盖。

## 明确未完成项

1. 当前安装量化实现的源码语义核对与受信合同。
2. 新会话真实双卡 list/describe、args/kwargs 观察、四类投影枚举。
3. 从实际观察结果生成并独立核对冻结 R1 集合。
4. 真实 disabled/IDENTITY/recovery 及单请求 integration smoke。
5. 安全有界 QMM live 观察与模块层归属证明；当前不开发新的 native interceptor。

若继续，需要原评测任务完成安全交接。安全审核本轮分别拒绝了远端量化源码全文导出与最后的运行状态回读；两次均未通过其他路径重试。需要用户明确授权具体只读文件/状态核对及本地目的地，或用户自行提供这些记录；在此之前不启动本任务模型实验。
