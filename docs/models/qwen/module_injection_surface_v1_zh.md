# Qwen 双卡模块注入面发现 V1

## 方法和边界

本报告使用已有 FakeRequant extension 的 `IDENTITY` 模式进行 observe-only live discovery：它仅返回原 tensor，不做数值变换、fault arm 或模型质量评估。固定 completion 返回 HTTP 200，每个 rank 收集 512 个 identity hit。

```text
状态：QWEN_MODULE_INJECTION_SURFACE_DISCOVERY_V1_PASS
DSV4：DSV4_DISCOVERY_SCHEMA_READY；尚未进行 DSV4 live discovery
```

## 已确认的可观察模块族

| 逻辑角色 | 真实模块模式 | 覆盖 | 已观察能力 |
| --- | --- | --- | --- |
| 注意力输入 | `model.layers.N.self_attn.qkv_proj` | 64 层 × 2 rank = 128 | `torch_forward_pre_hook`，BF16 输入已观察 |
| 注意力输出 | `model.layers.N.self_attn.o_proj` | 64 层 × 2 rank = 128 | `torch_forward_pre_hook`，BF16 输入已观察 |
| MLP gate/up | `model.layers.N.mlp.gate_up_proj` | 64 层 × 2 rank = 128 | `torch_forward_pre_hook`，BF16 输入已观察 |
| MLP down | `model.layers.N.mlp.down_proj` | 64 层 × 2 rank = 128 | `torch_forward_pre_hook`，BF16 输入已观察 |

这些是后续 Candidate/Validation 流程可选择的候选面，不是自动注入许可。每个实际 Fault 仍需 target binding、lossless gate、rank policy 和恢复生命周期证明。

## 未发现的角色与限制

当前 Qwen dense 运行时未出现 `MOE_ROUTER`、`ROUTED_EXPERTS`、`COMPRESSOR`、`INDEXER`、`STATE_CACHE`。这不表示框架永不支持，只表示本模型的本次 runtime 不可见。

RMSNorm 可注册 pre-hook，但 identity scan 没观察到首个浮点实参，故不标为已确认 candidate。legacy selector 没导出实参序号时，`first_float_input_index` 保持 `null`，不靠猜测填写。

## DSV4 边界与 evidence

schema 已定义 DSV4 的融合/拆分 QKV、MoE router、FusedMoE experts、compressor、indexer 与 state cache 角色；实际 DSV4 使用前必须重新进行 observe-only discovery，不能复用 Qwen 的 module name、层数、shape 或地址。

```text
本地 profile: profiles/discovery/qwen_2card_injection_surface_v1.json
原始 live capture: artifacts/evaluation/qwen_2card_module_surface_live_capture_v1.json
远端 capture: /home/haoran/demoncat_overnight_20260907/evaluation/
module_injection_surface_discovery_v1_20260909T020000Z/module_surface_live_capture.json
```
