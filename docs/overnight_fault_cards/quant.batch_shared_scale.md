# quant.batch_shared_scale

**本卡是源码及门禁审计；当前Worker实测结果另行绑定回执，不继承旧实验PASS。**

## 目的与语义

真实DynamicQuantV2从逐token行量化范围变为batch共享范围；这是后端归约范围错误，不是异常prompt自然诱发已证明。

## 当前实现

IMPLEMENTED_NATIVE_TLS

## 模型与逻辑角色

实际动态W8A8方法、独立健康decode输入、scheduler request-id→row→token-position/prefix映射，两rank分别证明。 Qwen/GLM/DeepSeek均须以实际结构及调用证明，不按品牌直接放行。

## 实际算子边界

aclnnDynamicQuantV2 native reduction descriptors and scale/Q outputs

## 环境/版本/硬件

CANN9.1、精确vLLM runner SHA、TP2/B8 pure decode、TLS owner；只选首个实际eligible模块。

## 具体实施

沿用native量化算术并改变内部[T,H]→[1,T*H]归约范围；不篡改Q/K/V/权重或事后输出。选点和arm按owner线程。

## 无损门禁

独立健康forward与B/F输入hash、prefix、actual scheduler rows一致；B逐行结果与原算子一致，未选模块透传。

## 故障效果

分别核对每rank每request row对应的scale与Q改变；原始statistics按整张量汇总，不用单纯shape[0]代表请求数。

## OFF与恢复

停止新arm、owner线程finish/context释放，重复相同decode OFF exact；异线程finish/disarm必须拒绝。

## 本轮目标

复用3步TP2模型通过，补每request量化结果证据的摘要/哈希，不从整张量Max推出全部请求都变化。

## 当前缺口

ALL_ELIGIBLE_MAPPING_UNVERIFIED；PER_REQUEST_SCALE_Q_IMPACT_UNPROVEN；COMMON_WORKER_ADAPTER_PENDING

## 下一步

核对已保存scheduler row映射、scale/Q逐行辅助证据；缺少则限定为全张量统计与请求输入映射已证。

## 历史证据及来源

quant-model-a2：每rank3命中；实际scheduler请求/token行映射及selected input一致；3step整批DynamicQuant output0 Max127→98不证明每请求scale/Q均变；6/8logprob变、token不变；OFF exact。

历史等级：NATIVE_VERIFIED, MODEL_VERIFIED, RECOVERY_VERIFIED。

历史恢复、当前Python/Native入口与共用Worker接线分别验收。本文不自动授权all_eligible/every_match。

完整G0–G9、覆盖资格、源文件SHA与统计合同见[机器可读能力卡](../../profiles/capabilities/quant.batch_shared_scale.json)。

- [fault_injection/quantization.py](../../fault_injection/quantization.py)
- [fault_injection/deployments/batch_scale.py](../../fault_injection/deployments/batch_scale.py)
- [fault_injection/native_adapters/batch_scale_native.cpp](../../fault_injection/native_adapters/batch_scale_native.cpp)
- [tests/test_quant_auto.py](../../tests/test_quant_auto.py)
- [experiments/provider_adaptation_20261116/test_quant_thread_native.py](../../experiments/provider_adaptation_20261116/test_quant_thread_native.py)
- [experiments/provider_adaptation_20261116/validation_summary.json](../../experiments/provider_adaptation_20261116/validation_summary.json)

## 本轮执行回执

本轮等级：**CPU_VERIFIED**。保留历史 TP2 decode；全张量统计不证明每请求 scale/Q 改变，逐行影响证据不足。

上文“当前缺口”是基线审计快照；执行后的缺口与历史状态分栏见[最终17项结果矩阵](../../experiments/overnight_20261116/fault_matrix.json)及[最终报告](../fault_closure_report.md)。本轮未授予共享 Worker 全量故障资格。
