# Inspect 实机交接清单

更新：2026-09-10 已完成按本清单的安全交接及 A–E 验收，详见[实机记录](live_validation_20260910_zh.md)。以下保留最初门禁，不表示原长任务仍在运行。

本任务本地基线为 `b4a1077`，独立分支 `feature/interactive-inspect-v1`。不推送。旧任务、源代码、参数、Fault 状态、NPU_LOCK 均不由本任务接管或修改。

## 必须满足的前置条件

1. 原监督任务退出，结果落盘；只看进程消失不够。
2. final_environment 证明原服务恢复健康，hook/fault/pending mutation 无残留。
3. 原 NPU_LOCK 已释放，按项目机制获取本任务独立 owner；禁止删除别人的锁。
4. 记录新 session、服务 epoch、源码及 R1 依赖 hash、模型、TP/PP、逻辑 rank 与物理卡映射。
5. 复用现有已验证 launcher，extension 变化在新服务启动时生效；不热装当前评测服务，不改宿主 vendor 库，不复制 native runtime。

当前原任务使用物理卡 2/5；授权仅适用于其清理后本任务的受控后续验证，不意味着可以抢占。源码检查遭安全审核拒绝的量化文件不绕过导出，需要用户明确授权只读审计或提供源文件后再核对语义。

## 最小验证（不是三轮评测）

- A：真实双卡 status/list/describe，记录实际枚举到的模块、rank/layer、quant_method 类型；不预填 64 层。
- B：独占单请求观察 args/kwargs，stop 后确认本任务 handles=0，baseline 不受修改。没有执行事件则记录未观察，不改执行模式求命中。
- C：按当前已核对的实现源码/哈希/运行时合同解析尺度。未知就 UNKNOWN，特别区分 model activation scale 与 reference absmax。
- D：从当前扫描结果按既有 capability 和 MLP_DOWN 角色选择候选，导出精确集合，独立对比原冻结 R1 manifest。不能仅以“都是 down_proj”判断等价。
- E：显式委托已有 owner 进行 disabled/IDENTITY/恢复，确认目标集合、rank 命中、changed=0、输出一致、服务与 hook 清理。子集不被固定 owner 支持时拒绝，不扩大。
- F：QMM recorded 已有；没有安全有界 native live adapter 时保留缺口。不为此开启无限 trace，不 dlopen preload 伪装已有绑定已替换。
- G：只有前述门禁通过且复用原冻结 R1 才可做单条 baseline/fault/recovery；不补齐 4096，不搜索强度。不以是否生成明显异常作为 Inspect 验收条件。

## 产物及结论边界

在全新隔离结果目录保存 inspect_session_manifest、model_modules、quantization_inventory、injection_candidates、DRAFT plan、ValidationResult、原始请求/响应及最终清理。现场源路径/模型信息不要提交到普通 Git。

必须分开报告：通用接口、本次实际参数观察、支持的现场量化实现、配置可表达性、当前 backend 绑定、recorded/live QMM、module→native 映射、lossless、原任务是否受影响、最终锁与服务状态。

本地 fixture 用来检验接口和防误绑定逻辑，不代表真实 NPU、vLLM 执行路径或 DeepSeek 实机验证。当前不用 `INSPECT_LIVE_VALIDATED`。
