# 项目概览

## 项目是什么

DemonCATPFI 是面向本地 vLLM-Ascend/CANN 开源模型的开发者模式故障能力研发、验证与效果评估仓库。其目标是为故障诊断工具提供受控、可审计、可恢复的模拟输入，而不是对公共服务、网络目标或未授权系统施加影响。

项目将“历史 Fault 的语义”与“当前运行时可以验证的技术路径”分开：一个代理能力可以复现故障表征、支持诊断工具测试，但不能被误写为健康 kernel 自然发生了同样的物理缺陷。

## 仓库边界

```text
DemonCAT 产品框架
  = 用户侧 PrimitiveSpec、RuntimeBinding、controller、授权、生产 CLI/API

本仓库
  = Ascend/NPU 执行平面实现、native interposer、策略、观察、发现、验证、评估和能力 metadata
```

本仓库不创建第二套产品 registry、controller 或生产注入 CLI。它产生的 capability manifest、验证结果和 effect profile 是对产品框架的证据型交接，而不是自动注册或自动 arm 指令。

## 当前目录职责

当前默认入口是[Qwen0.6B单卡手册](../inspect/qwen06b_single_npu_user_walkthrough_zh.md)，源码主体与固定参考依赖见[源码基线](../releases/source_baseline_v1/README.md)。Windows仅阅读维护，真实运行仍为Linux/Ascend；DeepSeek未实机认证。Inspect已是实现模块，不再仅是整合规划。

| 目录 | 职责 |
| --- | --- |
| `src/ascend_fi/` | 可复用的 backend、strategy、observer、discovery、validation 与 evaluation 集成边界 |
| `profiles/capabilities/` | capability metadata、已验证环境与限制 |
| `experiments/` | 历史案例、冻结策略与实验级配置 |
| `fault20_fast_validation/` | 已验证 native/runtime 来源、reports 与 evidence tree；本轮不移动 |
| `docs/` | 项目、评估、Fault、模型与历史归档的阅读入口 |

详细的迁移门禁见[仓库分层](repository_layers.md)，产品边界见 [DemonCAT 合并合同](demoncat_merge_contract.md)。

## 当前文档结论边界

本目录记录的是各自 source/evidence 所支持的结论。不同模型、CANN/vLLM/torch_npu 版本、硬件、并行拓扑或请求形态均需重新发现、lossless validation 与效果评估；不能跨环境直接继承“已验证”状态。
