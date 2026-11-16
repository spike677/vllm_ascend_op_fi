# 部署 provider：自动选点与支持范围

这里提供 20、26 两变体、共享量化尺度、33 的实际部署实现。
模型用户不填写层名、地址、rank PID 或 storage ID。模型结构发现负责候选，
实际 native 回调、allocator、scheduler 或 cache 关系决定该调用能否支持。
统一注册接口不是任意模型、形状和版本的通过证书。

实测结果与限制见[本轮验收](../../experiments/provider_adaptation_20261116/README.md)。
Fault39 使用独立 `demoncat_fault39` 生命周期与精确构建 profile；不能把该包的
通过结论授予统一目录中仍待部署的 `39.matmul_range` / `39.output_proxy` 桥。

| provider | 自动选择与真实注入边界 | 约束 |
|---|---|---|
| `buffer20.Buffer20Runtime` | 模型 Linear 对象 → 当前 forward → 实际 MatmulWeightNz Run slot；临时输入的原 v17 mantissa 变换 | CANN8.5.1、metadata ABI17、FP16/BF16 ND 完整输入 storage ≤64KiB；当前新实测是 BF16/NZ29 权重 |
| `residual26.Residual26Runtime` / `26.residual_stomp` | 真实量化模块 → QMM output-base；只读 allocator 证明两个 active block 相邻 → 原生写入 → 实际残差消费者后恢复 | CANN8.5.1、自定义 AddRmsNormBias；非邻接明确透传/拒绝，不伪造地址 |
| 同上 / `26.fia_destination` | 实际 Attention 对象 → FIA V3 构造输出 destination → 存活残差 → AddRmsNormBias 消费和恢复 | 真实32B W8A8 TP2推理已通过；仅接受当前forward的实际producer，不跨prefill/decode复用残差；不要求underflow邻接 |
| `batch_scale.QuantWorkerExtension` | 独立健康 decode 自动选择第一个 eligible 动态 W8A8 模块；实际 scheduler 行与前缀绑定 → DynamicQuantV2 归约范围 | CANN9.1、固定 runner 指纹、TP2/B8 pure decode；每次选定调用触发，其他候选明确透传 |
| `paged33.Paged33Runtime` | 实际共享 AttentionInterface 对象与 DynamicCache → 全量 page table 逆映射核验 → 原生 paged IFA host SInner 切换 | CANN9.1 审计 vendor，固定 B8/Q1/KV16384/H16/KV8/D128 FP16/page128；只验收 teacher decode 重复 |

部署人员先构建并预加载相匹配的原生库，提供已审查的 bundle、库 SHA 和新的证据目录。
不能在一个已启动的普通 worker 中晚加载 interposer，也不能将用户提供的 SHA 当作安全签名。
实际部署命令、只读挂载、环境及模型入口在
[`experiments/provider_adaptation_20261116`](../../experiments/provider_adaptation_20261116)。
`capture.py` 是该部署配套的原生 msprobe 配置入口；worker 示例将该目录加入 PYTHONPATH。

以下为已具备匹配预加载部署时的 20 使用方式，运行时从实际模型选择位置：

```python
from fault_injection import attach
from fault_injection.providers import register_runtime
from fault_injection.deployments.buffer20 import Buffer20Runtime

# 以下三个值由受信部署清单提供；用户不提供模型层路径。
register_runtime("20.buffer_substitute", lambda model: Buffer20Runtime(
    native_library, sha256=approved_sha256, evidence_dir=new_evidence_directory))
with attach(model, "20.buffer_substitute", arm="fault") as session:
    result = model(**inputs)
    session.assert_coverage()
```

`arm="baseline"` 走对应正常/无损路径；退出 session 移除 owned hook、恢复控制状态。
B/F/OFF 验收必须另行检查实际 kernel、完整输出和恢复，不以 `session.report()` 的
路由计数代替原生命中。该报告刻意保留 `native_fault_hit_proven=false`；实机结论来自
另存的 native 完成回执。清理不确定时保留 owner 并拒绝下一轮，不全局 reload 权重。

当前所有适配器要求专用同步 eager worker、`TASK_QUEUE_ENABLE=0`，不支持同进程
多故障并发。共享量化后端新增 TLS 选择和 context owner 校验以允许 worker 的后台
线程；它不开放异步模型执行。CPU 的 foreign-thread 测试不代替 TP2 实测。

20/26 的原始 C++ 算法保留，只在定义 `DEMONCAT_NATIVE_PROVIDER_BRIDGE` 的独立构建
中启用桥接 `.inc`。未定义该宏时保留历史入口行为与原证据。33 使用固定审计 bundle；
公开源码和清单不包含 vendor 二进制。其他具体 CANN 安装须单独适配和实测。

## 可复核的部署依赖

20/26 的原算法及桥接源码均在本分支，分别用本轮目录的 `build20.sh`、`build26.sh`
构建。Fault39用其独立 `native/build.sh`，不复用20/26的ABI验收。

共享量化保留[原始后端控制源码](https://github.com/spike677/vllm_ascend_op_fi/blob/297e03d1a03b08b31ec82b26b1940d89d9e49478/artifacts/quantization/batch_shared_scale_v2_bf_statistics_20261028/source/backend_control.py)
（SHA256 `5ce739c4272840f110044cfe898827fe403b5730273258e8ba0eda437685b38c`）。
本轮 `batch_scale_native.cpp` 在原算术上增加TLS选择；`build_quant.py` 从
`/work/quant_backend_control.py` 读取上述控制源码，生成独立构建清单。
调度校验复用旧 `model_hook.py` 的只读 `validate_runtime/capture_rows/digest`；
**不会调用其历史 `install/patch/load`、固定层选择或采集配置**。
实际依赖SHA为 `33b650ef368de1f5e8beb771e64aa6e06f1c64f99746dcc07a3b151253b8cdd0`；
它与公开旧稿仅有一段未调用的旧采集配置差别。
[`materialize_quant_dependencies.py`](../../experiments/provider_adaptation_20261116/materialize_quant_dependencies.py)
从精确Git对象重建并校验这些字节，不执行旧注入器；需先取得该commit。
输出的 `model_hook.py` 只读挂载到 `/quant_original/model_hook.py`，控制源码用于构建。
当前采集配置始终由本轮 `capture.py` 提供，为全API/module的statistics/mix。

33沿用提交 `c1dcb0f22438362bfdd7100f834b0be62cd425c9` 的
[vendor构建步骤](https://github.com/spike677/vllm_ascend_op_fi/blob/c1dcb0f22438362bfdd7100f834b0be62cd425c9/experiments/fault33_paged_tiling_20261105/source_policy/BUILD.md)、
[validator](https://github.com/spike677/vllm_ascend_op_fi/blob/c1dcb0f22438362bfdd7100f834b0be62cd425c9/experiments/fault33_paged_tiling_20261105/vendor_exec.py)
与[原生审计解析器](https://github.com/spike677/vllm_ascend_op_fi/blob/c1dcb0f22438362bfdd7100f834b0be62cd425c9/experiments/fault33_paged_tiling_20261105/analysis/native_evidence.py)。
validator SHA为 `75730d3cca29dd0ef2cb5d4e5112a99202fb149ba13e89d2ca5e9ab63d18cdb2`，
实际vendor清单SHA为 `f48a82e89fdf7bbfafba5ac3e05334dbde01b422c55dc726f786152db5874cfe`。
重新构建后仍需验证具体bundle指纹、合法策略和实际launch；不能把新二进制直接视为本轮已验收包。

上述依赖均为版本受限的部署适配，并不要求模型用户填写层名。迁移到其他runner或CANN安装
应先更新适配合同和构建证据，再完成实机B/F/OFF；不可只改SHA放行。
