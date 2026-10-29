# FA 2048-token 策略设计依据

更新：该候选已完成私有 typed host 实现、CPU 构建和独立安装；stock 四组 112 份 FA raw 候选已解码为 M128/nRatio8，并用 SDK 宏确认 S1S2/half/float/ND/L1Reuse key。下面保留设计时的推导与验收门槛，最新状态、源码指纹和证据见 [FA policy 构建记录](fa_policy/README.md)。私有 OFF 对 stock 的 raw/输出一致性及 ON 数值现象仍待实机验收。

固定源码：ops-transformer `v9.1.0` / `c77c0829f1f0f0b8f882db1d9b051a758b605472`。安装包 FlashAttentionScore kernel 的 13 个文件均与该提交相同，见 [匹配清单](fa_kernel_source_match.json)。此清单没有证明 stock host 实现完全相同。

## 候选输入与实际路径门槛

HF SDPA 的固定版本源码预计将同一 Qwen checkpoint 的 FP16、BNSD、Q `[B,16,2048,128]`、K/V `[B,8,2048,128]`、无 padding/full prefill、无 cache、dropout=0 输入交给 `aclnnFlashAttentionScore`。预期 `inner_precise=0`、`sparse_mode=2`、causal bool mask、scale=`1/sqrt(128)`。这些是源码推断；实际模型捕获必须逐项确认，实际 route 或参数不同则不套用此候选。

当前 1024-token BNSD 输入不满足 S1S2 模板的 `s2Size > 1024`。2048-token 输入满足条件，不必改变 dtype、layout、mask 或精度策略。`FlashAttentionScoreTilingS1s2Bn2gs1SameAB` 的准入要求不被 D=128/S=2048 命中，VarLen 则要求 TND；因此预计进入 priority 96 的 `FlashAttentionScoreTilingS1s2Bn2gs1`。实际 selected template/key 仍须确认。

## 最小源码选择器

仅在 `attention/flash_attention_score/op_host/arch22/flash_attention_score_tiling_general.cpp` 的 `FlashAttentionScoreTilingS1s2Bn2gs1` 类中新增同一精确 scope 条件，不更改通用准入限制：

1. `CalcS1S2BasicBlock`（3255）：保留原计算，目标 scope 将 `s1BasicBlock` 从 128 选为 64；`s2BasicBlock` 仍为 128。
2. `CalcNRatio`（3268）：保留原计算，目标 scope 将 `nRatio` 从 8 选为 4。
3. 不改 `AnalyzeDtype`、`AnalyzeAttrs`、mask/scale、`GetTilingKey`、kernel 数值操作或最后序列化 raw。OFF 条件为 false 时逐字节走原选择路径。

scope 必须包括真实模型层/调用标记，而非仅凭 shape 猜测 layer；还必须核验 B=8、N1=16、N2=8、G=2、S1=S2=2048、D=Dv=128、FP16/high precision、BNSD、causal、无 dropout/pse/rope/sink。单次目标调用的命中与未匹配透传另验收。不能以 selector/hit 日志代替实际 profile。

M=64 和 nRatio=4 都已存在于同一类的原生候选集合，但其组合对当前输入仍是**待验证策略**。特别是源码的 `CalcUBSize` 并非完整合法性证明，仍需下面的编译/运行门槛。

## 完整依赖重算与已有 kernel

- `MatchTemplate:1845–1875` 先算 M/S2/nRatio，再算 D 与模板；保留此顺序。
- `SetCoreParams:1997–2014` 统一生成 tail、outer 和 `nRatio`，2048 的 S2 group 数由 2 变 4。
- `SetBmm1TilingInput:3351` / `SetBmm2TilingInput:3378` 使用新的 M 与 `S2base*nRatio` 设置 shape、buffer、FixSplit；保留原生返回值检查。
- `SetSoftMaxTiling:3495` 使用 `[M/nRatio, S2base*nRatio]`，OFF 为 `[16,1024]`，ON 候选为 `[16,512]`。
- 对应 `GetWorkspaceSize`、tensor buffer 和分核信息继续由原代码生成；不得复制 OFF raw 后改字段。
- `op_kernel/flash_attention_score.cpp:516–524` 已有该模板的 `half,float` 实例（ND/NZ 与 L1Reuse 标记），不像 FIA legacy PFA 的 TND 分支仅存在 BF16 实例。
- kernel `arch22/flash_attention_score_s1s2_bn2gs1.h:483,574` 从 typed tiling 读取 M 与 S2 chunk，不依赖 FIA SplitFuse 的固定 `Q_TILE_CEIL=128`。

## 为何能改变 online 归约，但不能预判故障已发生

kernel 的外层任务按 B/N2/G/S1 分块，内层对同一 query 任务的 key 范围连续循环：L1Reuse 路径 `:855–867`，普通路径 `:914–933`。`GetS2LoopRange:704–762` 确定 causal 范围；`:1057–1061` 修正最后 S2 tail。每个有效 key 属于一个连续 group，未采用跨核拆同一 query 的 SplitKV 合并。L1Reuse 可能将相邻 query 块成对执行，仍必须验证覆盖与无重复输出。

首组 softmax 在 `:1955` 初始化，后续组执行在线更新；`:2239–2244` 对已有结果重缩放并相加，最后一组才在 `:2247–2250` 归一化并写输出。尾 query 在 OFF 中经历 2 个 1024-key group，候选 ON 经历 4 个 512-key group。因此改变的是已有 kernel 的真实在线分组与浮点运算顺序，FP16 输入/输出及 FP32 中间计算保留。

这支持作为方案 A 的实施候选，**不证明数值必然不同，更不证明已经复现真实事故**。应允许实际输出完全相同，或差异未达到预设验收条件。

## 构建与验收门槛

原先已构建的 OFF 包只有 FIA/PFA/IFA，不能冒充 FA 包。FA 需在独立源码目录以同一固定提交构建 `--ops=flash_attention_score --pkg --jit --noaicpu --soc=ascend910b` 私有 vendor，保留原始 OFF 和目标 ON 于同一最终产物中。不可修改或替换系统 OPP。

正式注入前须完成：实际 HF route/参数核验；stock 与私有 OFF 的 QKV、完整 tiling profile 和输出等价；候选 profile 的 M/nRatio/S2 loop/workspace/尾块合法性；每 query 有效 key 覆盖与唯一输出；同一输入、同一最终二进制的 B/F 重复验证；msprobe 首个可观测数值差异在真实 FA 输出，且输入 QKV 同值。若 profile 采集仍缺 workspace 或 host ABI 未验证，明确保留缺口，不宣称通过。
