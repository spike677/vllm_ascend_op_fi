# C 工作流：Tensor、索引、路由与通信实现

本轮从 `cae9cdf4685543df98598bbc2cf0f3b9892b7635` 增加可执行机制代码和公共 Worker 的 Python 执行边界。历史源码、能力卡及实测原件未修改。这里记录源码与 CPU 合同结果，实机结论由本轮总控的原始运行回执单独确认。

## 已实现的执行链

| 机制 | 实际干预 | 观察与恢复 | 当前范围 |
| --- | --- | --- | --- |
| 15.alias_detach | 复用实际 NPU format29→base format 转换及原 `as_strided`，只改变返回视图的存储关联 | 确认真实 `copy_` 更新视图、原 base 未更新、原 base 被消费，逐元素检查非目标区；同输入 OFF | 已有 NZ29/BF16 16×16→8×8 固定几何，当前公共重放唯一实际调用 |
| 17.spatial_padding | 原 Functional Conv3d 的 padding 参数由 0 改为 `(0,7,7)` | 原算子恰好一次、合法 patch 几何和输出形状、参数未改、OFF | 标准 `nn.Conv3d` 的实际实现，16×16 空间 patch；非固定模型层名 |
| 18.api_half | 标准 `nn.RMSNorm.forward` 原调用返回后乘 0.5 | 原返回、真实后续模型输出、完整 OFF | 返回增益代理，未把融合 RMSNorm 或 Reduce 内部错误宣称适配 |
| 8.restore_offset | 实际 `npu_moe_token_unpermute` 前将 `I.reshape(T,K)` 和 `P` 按相同 `min(t+1,T-1)` 成组映射 | H/I/P 原存储不变、配对内容设备读回、实际原生 consumer 一次；完整 F 原生输出须等于 B 原生输出的相应组；OFF | 非 padded、完整排列、同设备，T=1 透传。构造推理 workload 可执行，生产 scheduler/producer 归属仍需验收 |
| 22.wrong_row | 实际模块 `cos_sin_cache` 对象的原 `aten.index_select` 返回后，只将末行换为首个合法 source 行 | source/index/其他行不变、结果读回；同 storage 的真实后续 mul/add/rotary consumer；OFF | 合法错行输出代理，未复现 uint32 地址溢出；现有缓存对象协议之外拒绝 |
| 25.reduce_scatter | 原同步 `torch.distributed.reduce_scatter_tensor` 返回后，重放实际测得的同输入 alternate 输出 | 实际 group/members/global-local rank/Native sequence、原生库 SHA、输入与正常输出全部匹配；原通信不新增、不跳过；本地同步清理及 OFF | 没有原生差异样本时拒绝注入；本轮没有凭 CPU 值构造正向样本。跨 rank 及真实模型资格单独待验 |

8 的配对映射沿用历史故障语义，实际 unpermute 参数也与 CANN 的接口定义核对。不同版本仍需以实际调用和完整输出验收。[CANN 8.5 接口](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/aolapi/context/ops-transformer/aclnnMoeTokenUnpermute.md) · [CANN 9.1 接口](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnMoeTokenUnpermute.md)

## 公共 Provider 与生命周期

入口为 [TensorModuleProvider](../../src/demoncat/runtime/vllm_ascend/worker_extension/providers/tensor_modules.py)。可信部署先通过 `install_tensor_replay` 绑定已加载 model 与真实同步 replay；RPC 不接收 Python callable、层名、地址、`all_eligible` 或 `every_match` 参数。Extension 不启动、重启 Worker，也不替换 `execute_model` 或 `_prepare_inputs`。

一次资格尝试执行 `baseline → baseline_repeat → lossless → fault → off`。前两轮完整输入与输出须精确相同，并从真实执行事件中找到唯一稳定候选。15/17/18 使用公共 `CandidateSet` 与 typed Python 身份。实际干预恰好一次，完整 OFF 输出恢复后才记录本次恢复。无候选、歧义、输入改变、未知 forward、外部改写 hook、未完成原执行或恢复失败均保留失败状态。

状态来源为 `execution_boundary=python_hook`、`state_source=OWNED_PYTHON_HOOKS`，包含 owner/generation、真实 hook 字典归属、在途 forward 和未释放 lease。`native_loaded` 与 `native_registered` 都为 false。这些字段不伪装为 Native `.so` 的状态。构造模型身份与实际 vLLM Worker 身份分别记录。当前范围为 `unique + once`，没有授予 C2/C3。

25 的校准对象只能由本进程实际观察 session 生成。必须是同一仍存活的 group/model、同输入、两次已完成 collective，且完整输出不同。签发后校准内容再次按 SHA 核验，直接构造对象、传入任意 delta、零差异样本或改写校准 Tensor 都不能启用。此机制是实测输出重放代理，并不修改 HCCL 归约算法，也不自动证明 LCCL 的历史非确定性。

## 与 msprobe 的组合

124 本次 collector 的 `module_processor.py` SHA-256 为 `4876e89f7f726cd552fc1bf0e7b7745a898461deec2cb2abf33b02deba7329d7`。其 `wrap_forward_with_hook_safety` 给模块安装持久实例 wrapper。发现器只在文件字节、实际 nested code object、closure 中的 module 归属均匹配时，核对最终原 forward 是否为标准实现；统计 wrapper 继续保留。未知 collector 版本、自定义 forward、外来 module 的 wrapper 仍拒绝。

机制事件和完整数组比较不选择 msprobe 首差。独立统计仍从 B/F 原始 `statistics/mix`、stack/construct 计算，结构不齐和不可观测项必须保留。此处的五个阶段是五次实际 prefill/模块重放，不是 decode token 步数。

## 最小入口与已执行检查

[单卡构造 NPU 脚本](tensor_provider_probe.py) 支持 15、17、18、8、22，每项执行实际 B/B/L/F/OFF，标记 `CONSTRUCTED_NPU_PROVIDER_REPLAY`。它不冒充 vLLM Worker 或预训练模型。25 使用 [双 rank 校准脚本](collective25_probe.py)，可通过 torchrun 运行；`--cpu` 只用于 Gloo 合同测试。没有校准差异时输出 `NO_REPLAY_SAMPLE`，不能算故障注入成功。

本地执行了 49 项相关测试，其中 48 项通过、1 项因当前 Windows torch 构建没有可用 Gloo device 而跳过。跳过项尚未执行 collective。其余用例执行真实 CPU Tensor 与 dispatcher 操作，覆盖三条已有机制、8/22 输入与消费、单目标拒绝、输入漂移、OFF、外部 hook 冲突、collector 组合、25 未签发校准拒绝。源码编译与 `git diff --check` 通过。

```text
python -m unittest tests.test_module_msprobe_wrapper tests.test_tensor_worker_providers tests.test_route8_index22 tests.test_module_faults tests.test_alias_auto tests.test_collective25
```

以下事项仍需独立验收：预训练模型自然调用链与请求归属；真正 vLLM Worker 的部署回执；25 多 rank 恢复与可重放的原生差异；新的 CANN/SoC 组合；每机制 C2/C3。构造模型或 CPU 测试不关闭这些门禁。
