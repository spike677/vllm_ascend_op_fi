# Fault22 研究衍生代理：RoPE 表源行映射错误

## 1. 根因标注与证据等级

沿用仓库 Fault8、Fault17 的记录方式，将模拟的错误语义、实际注入动作和已证明结果分开。

| 字段 | 本次定义 |
|---|---|
| 研究来源 | Fault22 按行读取／地址运算研究 |
| 模拟根因 | 读取实现将目的行与应读取的源行错误关联，返回另一条合法源行的内容 |
| 模型中的具体语义 | 最后一个 prefill token 使用了 position 0 的 RoPE cos/sin 表行，而不是 position 13 的表行 |
| 注入 primitive | `INDEXSELECT_WRONG_ROW_OUTPUT_PROXY` |
| 实际注入位置 | 指定 RoPE 实例中的原生 IndexSelect 返回后、返回值被下游使用前 |
| 实际动作 | 框架设备 copy 将一条合法 source 行覆盖到选定 output 行 |
| 根因复现等级 | **错行读取结果的效果级代理，不是原生缺陷或乘法溢出复现** |

本次可以归因的实验原因是主动安装的 output proxy；不能将其归因于
vLLM、CANN、原生 IndexSelect kernel 或硬件自发出错。

## 2. 与原 Fault22 溢出机制的区别

原 Fault22 设备实验研究的是：

```text
相同合法 index 和 row_bytes
→ 错误地使用 uint32(index × row_bytes)
→ 乘法回绕
→ 设备实际读取另一地址
```

当前模型 source 为 BF16 `[40960,128]`，每行 256 B，总容量 10 MiB。
在这一连续行字节偏移模型下，首次 uint32 回绕要求 index=16,777,216，
而合法最大 index 为 40,959，因此当前合同内不可达。

本次将 row13 的结果换成 row0 **不是**计算 `uint32(13 × 256)` 的结果：
该计算在 32 位和 64 位下都等于 3328，绝不会自然变成 0。
所以本包不能作为独立大 Tensor 溢出测试的替代证据，也没有调用那个实验设备 kernel。

## 3. 正常路径和唯一故障规则

实际目标：`model.layers.0.self_attn.rotary_emb`，
`AscendRotaryEmbedding.forward_oot` 内的 `aten.index_select.default`。
本次动态验证 source 是该实例的 `cos_sin_cache`，index 是本次 positions。

| Tensor | 已验证合同 |
|---|---|
| source S | BF16、连续 `[40960,128]` |
| index I | int64 `[14]`，内容 `0..13` |
| dim | 0 |
| output Y | BF16 `[14,128]` |

正常 `Y[t,:] = S[I[t],:]`。代理执行：

```text
Y_native = native_index_select(S, dim=0, I)
Y_fault = Y_native                     # 本次真实返回 Tensor
Y_fault[13,:] ← S[0,:]                 # 只覆盖最后一行
```

这是说明逻辑的伪代码，不是 host 计算代替设备操作。真实实现使用设备 Tensor 的
`src[0].clone()` 和 `result[13].copy_(replacement)`，并同步、readback。
没有改 source、index、其他 output 行、shape、dtype、logits、采样器或 KV cache，
没有注入 NaN/Inf。后续模型状态可以受传播影响，不等于直接修改了这些对象。

## 4. 为什么能影响生成结果

```text
原生 IndexSelect 正常生成 cos/sin 表查找结果
→ 代理把最后一行的 position 13 表内容换成 position 0
→ 真实下游 npu_apply_rotary_pos_emb 使用改变后的表内容
→ Query/Key 的位置旋转结果改变
→ 后续 attention / 模型计算继续执行
→ 生成 token 与文本轨迹分叉
```

首个 prefill 中的行号 13 是输入位置，不是“第 13 个生成 token”。
这里是位置编码表内容错配，不是通用权重错读，也不是输入文本 token 被删除或交换。
下游 rotary 统计与模型响应可证明传播效果，但本包没有逐层完整数值 oracle，
不能声称已定位首个 logits 差异或证明每个中间算子的全部数值。

## 5. 范围、生命周期与恢复

- Qwen3-32B-w8a8-nopdmix，TP2、eager、单请求串行、关闭 prefix caching。
- 只选择 layer0 已绑定实例；两个 rank 使用相同逻辑规则，各命中一次。
- 仅首次完整 14-token prefill；decode 不注入，不是 all-hit 或公共实现全覆盖实验。
- 控制入口要求开发者模式，安装检查 owner，绑定 scheduler 请求身份和本次 Tensor 引用。
- 原生完成并同步后读取摘要；复制完成并同步后校验选定行、其他行及 source/index。
- 保留临时 Tensor 到设备 copy 完成；RoPE scope 返回后再次同步，记录完成状态。
- 卸载要求没有 in-flight、poison 或 owner 冲突；恢复原方法，再运行 fresh request。

卸载不会撤销故障请求已经产生的状态变化。恢复结论来自新请求文本与全部 token IDs
精确回归 baseline，不是来自“hook 已移除”这一单独事件。

## 6. 本次证据与可见效果

| 证据层 | 结果及文件 |
|---|---|
| 代理实际写入 | 两 rank 各一次，162 bytes / 89 BF16 elements 改变，目标行 13；`fault_hit_evidence.json` |
| 非目标保持 | 原 source/index、非目标 output 行不变，设备 copy 完成并 readback |
| dump 活跃窗口 | 命中事件记录 probe active、step0；交付 step0–2 双 rank B/F 原始日志 |
| 目标输出差异 | step0 IndexSelect 输入统计相同，输出统计不同；rank0 Mean 从 0.47265625 到 0.4765625 |
| 下游传播 | 同一步 rotary 输出统计不同；`selected_statistics_comparison.json` |
| 模型轨迹 | token0 相同；token1 为 B=19/F=17，token2 为 B=271/F=488；`token_comparison.json` |
| 恢复 | 卸载后新请求文本和全部整数 token IDs 与 B 精确一致 |

B 完整正文开头是 ` 4`，F 为 ` 2 + 2 2 + 2? - Quora`。
完整正文在 [README](README.md) 和 B/F response JSON 中，不以最坏片段代替完整结果。
两组均为 32 token、`finish_reason=length`；B 自身也有重复和截断。
因此支持 **MODEL_TRAJECTORY_DIVERGENCE** 和 fresh-request recovery，
不将一次短请求升级成正式准确率下降或普遍、可重复的严重输出病理。

这是已有 step0–7 采集的 step0–2 原字节子集，不是新一轮三步实机实验。
statistics 不是全量 Tensor；命中事件中的字节变化来自另外的 readback 验证。

## 7. 实现溯源与结论边界

研究目录：`experiments/indexselect_wrong_source_row_proxy/`。

- `source/quick_rule.py::selection_plan`：最后一个输出行映射到第一个有效 position 源行。
- `source/quick_extension.py::QuickWrongRowExtension`：实例／请求绑定、设备 copy、校验、卸载。
- `source/probe_quick_extension.py`：已有 debugger 的只读状态记录。

实测 `quick_extension.py` SHA256：
`a9bb9cc575e8df871d0a7056b3ad9ef41a5014052d329cf9e23d128ed5defffb`。
其余源码哈希见 B/F source manifest。源码不在本 dump 交付包内。

**最终根因标签应写作：模拟 RoPE 表源行映射错误；通过原生返回后的单行输出代理实现。**
不得标成原生 IndexSelect 算错、原生 uint32 乘法溢出、LD_PRELOAD 绑定通过，
或自定义设备错读 kernel 在本次模型采集中执行成功。
