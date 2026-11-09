# M-only 接入位置与验收链

本轮复用已验证的 FA host、kernel、通用 runtime 及证书。新增的是严格的 M-only 完整模型入口、
Attention/末 query/KV 精确观察和独立验收；此前 joint 模型结果不作为 M 的归因证据。

```text
固定 input_ids / position_ids；每轮 cache=None
  → 原始 HF Qwen3-0.6B 的全部 28 层
  → 原始 Qwen3Attention → F.scaled_dot_product_attention
  → 共同 ModelObserver.observe → 本层 StrategySession.execute
  → 原生 SDPA backend → aclnnFlashAttentionScore
  → 匹配版本 host 的原生 factor 选择边界
      B/OFF：M128 / nRatio8
      F：    M64  / nRatio8
  → 原生 host 生成依赖 tiling → 原生 kernel 真正计算
  → 原始返回对象 → o_proj、残差、后续层、lm_head
  → 本轮真实 KV cache + 同一个固定后续 token → 原始 decode
```

`model_probe.py:ModelObserver.__init__` 对每个真实 `self_attn` 建立独立 adapter/Session 和无条件
module hook；`observe` 包装的是模型自然调用的 SDPA，共同的 `invoke_original` 恰好调用原 API 一次。
不是替换 Python `torch_npu.npu_fusion_attention_v3` 名字后假定截获 native 内部调用。

| 代码 | 本轮作用 |
| --- | --- |
| [build_cases.py](../build_cases.py) | 固定 B1/B8、五轮顺序、`query_tile` 选择、CPU FP64 统计版本及输入指纹。 |
| [m_only_contract.py](../m_only_contract.py) / `require_selection`、`require_plan` | 捕获和验收共同拒绝 joint、nRatio4；实际 F 必须 M64/R8，其他模式 M128/R8。 |
| [model_probe.py](../model_probe.py) / `ModelObserver.observe` | 保留 Q/K/V 与原始返回；绑定 PID/TID/token、真实 host/raw 与本层 Session。 |
| [model_context.py](../model_context.py) / `actual_native_call`、`prefill_context` | 从实际公共描述符重建上下文，核 Python/native 的 GQA、causal、scale、形状及请求映射。 |
| [session.py](../../../native_execution_strategy/session.py) / `StrategySession.execute` | 每次匹配重新选择，保留原调用一次和原返回身份。 |
| [fa_plan.h](../../../native_execution_strategy/native/fa_plan.h) / `DecideFactors`、[fa_bridge.h](../../../native_execution_strategy/native/fa_bridge.h) / `SelectFactors` | 对真实上下文选择证书内合法 M，保持 nRatio8。 |
| [apply_fa_bridge.py](../../../native_execution_strategy/native/apply_fa_bridge.py) | 将选择接入匹配 host 的 factor 边界；派生 tiling 仍由原生 host 生成。 |
| [model_helpers.py](../model_helpers.py) / `tensor_snapshot` | 同一只读诊断函数复制逻辑字节、记录几何和完整/逐请求 SHA，不替换模型数据。 |
| [numeric_observer.py](../numeric_observer.py) / `observe`、`compare_bytes` | CPU 逐元素比较完整 Attention、末 query、实际 cache keys/values 及 logits。 |
| [model_probe.py](../model_probe.py) / `main` | 每轮 fresh cache；真实 prefill 和固定 token decode；读取全部 28 层真实 cache。 |

私有 vendor 指纹为 `42173485d5d3c86b57cd8a847dbce2dfd1fc42004c1a451bb282b7aa211d9ef1`，
23 文件 runtime manifest 为 `ca0bf188bdc031ea3e4249c03ac13a9a8f751cd5ae8485655d9048c9f6cdba3b`，
validated 证书为 `be58b4e77098a8911ab97ad8711144a5a5999dc3742080726c594ad88e8fcc5b`。
本轮未重新构建或扩大底层证书；公开证据同时保存实际执行源码快照，优先按快照 SHA 验算。

M-only 控制不修改 Q/K/V、输出、权重或 stride，不直接覆盖序列化后的 raw tiling 字段。
M 的改变允许 host 重新计算与 M 依赖的计划；不能要求所有派生字段也不变。
nRatio 则同时在请求候选、实际 launch 解码和最终验收中核为 8。

## 三条独立证据

1. **实际调用和策略。** 框架分母由配置、无条件 module hook、msprobe 节点和 construct 共同建立。
   完整 DEBUG 日志独立枚举所有 attention launch；验收器重新读取每次 prefill 的原始窗口，
   核实际 raw、205 项 POD、169 项执行计划、34 项输入契约和证书，不以命中日志代替实际执行。
2. **数值传播。** 各层实际输入前后逻辑 SHA、完整输出和逐请求 SHA 与共同逐元素观察相互核对；
   cache 直接从本轮模型返回的真实对象逐层读取。后层输入允许继承前层差异；各层又同时接受 M-only，
   因而不能将后层每一项差异唯一归因于第 0 层。完整私有 logits 另行离线逐字节复核。
3. **msprobe 独立首差。** 仅按原始 `statistics/mix`、stack 和 construct 遍历真实模型 API 顺序，
   先冻结推断，再与 native 和精确观察连接。不会用注入层、期望 profile 或命中标记指定首差。
   统计相等不代表张量相等；null、源栈不齐和统计碰撞都保留。

## 范围和观察限制

当前适配器证书只覆盖 B1/B8、FP16 BNSD、Q16/KV8 heads、D128、S1=S2=2048 的固定 causal FA 分支。
本次所有受控 prefill 调用均应切换；Q1/KV2049 decode 显式范围外、原 API 透传，不能因此宣称
任意形状或 decode 策略已支持。该透传未走 Session 的 unsupported 分支。

统一 CPU FP64 collector 对每个实际浮点张量采用相同统计过程，用于减少小差异的摘要碰撞；
它不改变模型张量，且不是未修正 msprobe 默认统计的可观测性结论。真实 FP16 数值由 kernel 产生。
模型诊断读回在原 trace 中保留，只按 `tensor_snapshot` 的精确源码栈标记为诊断；真实 SDPA 不被过滤。

B Attention/KV 参考及各轮完整 logits 留在远端，F Attention/KV 没有保留完整张量文件；
后两者的逐元素差异来自两组共同的运行时 CPU 观察器。公开包可重算统计与哈希绑定，
不能离线复算未保留的完整 F Attention/KV，也不能替代完整日志及私有数据上的联合验收。

本轮回答“只变 M 能否在完整模型传播到 logits”。它没有实现盘古历史 FIA 图模式，
没有证明原生 Batch 必定触发该 FA 策略切换，也不唯一定位 kernel 的某条 Reduce 指令。
logits 数值变化与答案错误、准确率下降需要分别验收。
