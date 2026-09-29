# IndexSelect wrong-row output proxy：B/F step 0、1、2

这是 **INDEXSELECT_WRONG_ROW_OUTPUT_PROXY** 的观测证据，不是原生 Fault22
32 位地址溢出复现，也不证明 LD_PRELOAD Run 绑定成功。

本次交付从 2026-09-29 已完成的 B/F `step0..7` 原始采集中提取 `step0..2`，
没有重新启动模型或重跑 NPU。保留原始配置，未将其伪改为三步采集。
生成请求仍为 32 token 上限；**forward step 不等同于生成 token 序号**。
step0 是完整 prefill，后续为 decode。这里同时交付前三个生成 token 的比较。

## 文件与使用

- `B_dump_steps012.zip`：正常组。
- `F_dump_steps012.zip`：单次错行结果代理组。
- 每个 ZIP 含 18 个原始 JSON：3 step × 2 rank × `dump/stack/construct.json`。
- 解压后按 `stepN/rankN/dump.json` 查看；同一 step/rank 配对 B/F。
- `B_dump_config.json` / `F_dump_config.json`：原始 task=`statistics`、
  summary_mode=`statistics`、level=`mix`，原始 step 0–7。
- `selected_statistics_comparison.json`：选定 IndexSelect 和首个 rotary API 的统计对比。
- `B_response.json` / `F_response.json`：完整 32-token 正文及整数 token IDs。
- `token_comparison.json`：前三个生成 token、完整 B/F/R 正文和恢复比较。
- `fault_hit_evidence.json`：两个 rank 的实际命中事件及 dump-active 证据。
- `manifest.json`：每个 ZIP 及原始成员的 SHA256；`SHA256.json` 校验交付数据文件。
  SHA256.json 不包括自身与此说明文档。
- `B_source_manifest.json` / `F_source_manifest.json`：实测源码哈希；本目录不包含完整工具源码。

只包含 statistics 日志，没有全量 Tensor 文件、模型权重、编译库或服务日志。

## 请求、模型和故障合同

模型：Qwen3-32B-w8a8-nopdmix，TP2、eager，单请求串行，关闭 prefix caching。
请求正文：`Answer with just a number: What is 2 + 2?`

选定 `model.layers.0.self_attn.rotary_emb` 的真实 `forward_oot` IndexSelect，
本次 source 是实例的 `cos_sin_cache`。原生执行后，在首次完整 prefill 中，
两个 rank 各一次将 output 第 13 行改为本次 source 第 0 行内容。
source/index 保持不变，非目标 output 行保持不变。不是全局按名称拦截任意同形状算子。

每 rank：1 次命中，162 bytes / 89 BF16 elements 改变；
有效读取偏移从 3328 B 改为 0 B（均在同一合法 source 内）。
真实设备 copy 完成后返回下游，未调用自定义 Fault22 kernel。

## 已确认的输出差异

生成 token 为零基编号：

| token 序号 | B token ID | F token ID | 是否相同 |
|---|---:|---:|---|
| 0 | 220 | 220 | 是 |
| 1 | 19 | 17 | 否 |
| 2 | 271 | 488 | 否 |

**首次生成 token 差异是序号 1，不是序号 0。**

Baseline 完整正文（保留开头空格）：

```text
 4

Answer with just a number: What is 2 + 2?
Okay, let's see. The user is asking "What is 2
```

Fault 完整正文：

```text
 2 + 2 2 + 2? - Quora
Answer with just a number: What is 2 +  2 2 +
```

两组均生成 32 token，`finish_reason=length`，并非自然完整结束。
F 存在重复片段，但 B 也重复请求且截断；本实验确认轨迹分叉，不能据此宣称严谨的质量评测下降。
卸载后的 fresh-request recovery 文本和全部整数 token IDs 精确等于 B。
R 仅作为恢复摘要记录，不增加第三组 dump。

## dump 的差异及证据边界

两个 rank 的 step0 `Tensor.index_select.0.forward` 输入统计相同，输出统计不同。
例如 rank0 BF16 `[14,128]` output Mean：B=0.47265625，F=0.4765625。
同一步下游 `npu_apply_rotary_pos_emb` 统计也发生差异。

step1、2 没有再次修改 IndexSelect；其表读取输出统计相同。
首个 rotary API 在 step1 的输入/输出统计相同，step2 不同。
这不表示 step1 全模型没有差异，也不构成全局首次 logits 差异定位。

statistics 日志只能证明所记录统计量的差异；逐字节变化数来自独立命中 readback，
不能把 statistics 当成完整 Tensor 逐元素比较。

结论：代理写入命中、首个 prefill 的目标输出统计差异、生成 token 1/2 分叉及新请求恢复均有证据。
原生乘法溢出、独立 kernel 替换、自然硬件故障不在本包证明范围内。
