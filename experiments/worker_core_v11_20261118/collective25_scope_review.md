# Fault25 有界原生样本审查

审查日期：2026-11-18。结论为 **`NO_NATIVE_SAMPLE_IN_TESTED_SCOPE`**：下面两次已完成的双 rank HCCL 校准均没有合法的同输入原生差异，原 measured-output replay 在该已测范围不可执行。保留原件状态 `NO_REPLAY_SAMPLE`；结束本轮相同路径的样本调查，不新增健康重复、不制造 delta、不授予 C1/C2/C3。

本次只读本地源码、Git 冻结版本和主控从 124 提取的摘要，没有 SSH、Docker、NPU 调用或新的机制实测。只新增本审查文档，旧源码和原件不变。

## 输入与身份

审查输入是 `v11-collective-bounded-recheck.json`，123046 bytes，SHA256 `f3aa498f7a7edee76e2f335b885542328f0c3765cfbb986ae316938846ccba93`，本地重算一致。它是新生成的只读摘要，**不是完整原件**；包含原 controller、source manifest、supervisor、rank 文件的绝对路径、长度与 SHA，以及所选原始字段。原件继续位于 `R=/home/17-fault-c2c3-20261117`。

| 实际环境 | 原 attempt / 构造 workload | 冻结源码 | 结果 |
| --- | --- | --- | --- |
| CANN 8.5.1 路径；Torch `2.9.0+cpu` + TorchNPU `2.9.0.post1+gitee7ba04` | `collective25-bounded-a1/collective25-851-bounded-a1/collective` | `source-a8`，`3ff0db3eb1aeefd054e526ebe52eb87dca9aa166` | 两 rank 均无样本；controller/torchrun exit 0，无 timeout |
| CANN 9.1.0 路径；Torch `2.10.0+cpu` + TorchNPU `2.10.0.post2` | `collective25-910-bounded-a1/collective25-910-bounded-a1/collective` | `source-a18`，`0b3961a3710479735e526f7110b49d07fb97d116` | 两 rank 均无样本；controller/torchrun exit 0，无 timeout |

两次 controller 均记录物理卡 `[2,5]`，各启一个静态 `torchrun --nproc-per-node=2`，实际参数为 `--suite bounded --repeats 4 --timeout 120.0`，没有 `--cpu`。Torch 的版本字符串含 `+cpu` 不等于这次运行使用 CPU：实际 group backend 为 `hccl`，原生观察门禁要求 NPU 张量，且记录了加载中的 TorchNPU/HCCL 库。

对两次冻结提交分别执行 `git show <commit>:<path>` 并重算 SHA/长度，四项源码均与摘要中的原 source manifest 及原文件重算值相符：

| 文件 | SHA256 |
| --- | --- |
| 两次 `fault_injection/collective25.py` | `0a21fb9d34a1367dbaf0a937af6d688f739a8f4a45de4337dd3eada8a5b5ba51` |
| 两次 `collective25_closure_probe.py` | `990f49c9497df521261f1f1784bbf4146f001d46d12d04b81ee4e09d86cda6db` |
| 两次 `collective25_supervisor.py` | `aae75a7494b912455105b7c4bf4e42cc682a65e4e8e14087a458e3f83a82a689` |
| source-a8 的 `_scoped_dispatch.py` | `c85656176621ba6ff1263a56e42a870b7ba9025edef8def85bfcd8af7016d37a` |
| source-a18 的 `_scoped_dispatch.py` | `6a275bbeed7830b49b2d6f24f34576f1a2640dcb81968f7a29a1c8176916f887` |

两个历史 helper 均先 `torch.npu.synchronize()`，再 `detach().cpu().clone()` 保存观察值。source-a18 增加的可选 diagnostics context 在本探针中未配置。当前 helper 后续变化不用于解释这两次运行。

实际加载库在各次运行的两个 rank、全部六 case 中保持一致；两个版本分别绑定，不能互相继承：

| 路径所属版本 | `libhccl.so` SHA256 | `libtorch_npu.so` SHA256 |
| --- | --- | --- |
| `/usr/local/Ascend/cann-8.5.1/` | `1a72dbff32076e81fa474179dd60a191f57b6ad290cfbf279952631364e1fbf2` | `dd07e4afb4e31a5123d0b97a4a319acaa96158705a22d704bfc0a91f662a6041` |
| `/usr/local/Ascend/cann-9.1.0/` | `e7742392619bd0f7bfd1b55552070f9811b25ed3017eefd49ed2c1f9f45fc647` | `6f000ee3be32c15d062c9fc7a5c0889e9033e2b1959800608b7a3819249010f6` |

## 对账结果与源码依据

两版本采用同一六项配置，按表中顺序执行。count 为每 rank 接收元素数；发送元素数为其两倍。

| case | dtype | receive / send | 输入模式 |
| --- | --- | --- | --- |
| `fp32-ramp-128` | FP32 | 128 / 256 | ramp |
| `fp32-cancel-4096` | FP32 | 4096 / 8192 | cancellation |
| `fp32-mixed-65536` | FP32 | 65536 / 131072 | 固定种子的 mixed scale |
| `fp16-mixed-4096` | FP16 | 4096 / 8192 | 固定种子的 mixed scale |
| `bf16-mixed-4096` | BF16 | 4096 / 8192 | 固定种子的 mixed scale |
| `bf16-cancel-65536` | BF16 | 65536 / 131072 | cancellation |

- **后端与输入固定。** [Provider](../../fault_injection/collective25.py) 第 102–169 行调用原 `torch.distributed.reduce_scatter_tensor`，固定 `SUM`、同步执行；验证实际成员 `[0,1]`、global/local rank、dtype、收发形状、不同输入/输出存储、有限输入/输出、输入未变、native sequence 前进及 HCCL 加载库。每个 case 的四条 signature 相等；每个 rank 的 PID/group/model 对象身份跨六 case 保持固定。每个 rank 内的四次输入 SHA 相同；两个 rank 的输入可以彼此不同。
- **调用顺序一致。** 两版本各有 24 次被观察的组级 ReduceScatter，即各 48 条 rank 事件，合计 96 条。全部事件为 `observed`、`original_execution_verified=true`、`native_calls=1`、`applied=false`。case 编号为 `k=0..5` 时，两个 rank 均依次记录 `5k+j → 5k+j+1`，`j=0..3`；case 间空出一次序号对应源码中的原调用 OFF。六次 OFF 不混入四次健康重复的分母。
- **比较原生返回值。** [Probe](../c2c3_closure_20261117/collective25_closure_probe.py) 第 200–231 行直接比较 `session.observations[*]['output']`，不是 `output+1` 消费者；使用 `torch.equal` 和原生输出字节 SHA，没有容差、null 相等或 NaN/NaN 相等。所有 24 个 rank/case 组合均有四个相同输入 SHA、四个相同原生输出 SHA；`repeat_exact`、`repeat_bytes_exact`、`same_input_exact`、`cross_rank_inputs_fixed` 均为 true。哈希数组与对应四条 event 的字段一致。
- **没有可签发 pair。** [Provider](../../fault_injection/collective25.py) 第 201–216 行要求同一观察会话、group/model/signature、同一输入，并在原生输出相等时抛出 `NO_NATIVE_SAME_INPUT_DIFFERENCE_OBSERVED`。四次均等于首个输出，故这四次中的任意一对都不能成为原 replay 的合法样本。不同 case、rank 或 CANN 版本之间的差异不能拼成同一 live group 的样本。
- **健康生命周期另列。** 摘要中双方的健康 AllReduce、barrier、device synchronize、group destroy 回执一致，所有 rank `hits=0`、`injection_attempted=false`。两份源绑定 supervisor 均报告 `CALIBRATION_LIFECYCLE_PASSED`、正常退出且无超时；其代码要求原始 NPU shutdown callback 返回以及 shutdown synchronize 成功。本次没有重新读取完整 lifecycle JSONL，故不把摘要审查写成重新独立逐行审计了原 shutdown 日志。

以上摘要字段没有 null 或对账失败。身份、形状、重复哈希和 sequence 已逐项核对；8 个冻结源码文件的字节也分别复核。本次检查是对已有真实运行证据的只读审查，CPU 合同或检查数量不充当 NPU 验收。

## 范围与最终处置

结论限定于以上两套确切加载库、物理卡 2/5、双 rank、六项有限输入配置、同步 SUM、四次健康调用和默认 `HCCL_ALGO=DEFAULT_UNMODIFIED`。它确认本探针的 `torch.distributed → HCCL ProcessGroup` 路径，不是 LCCL/ATB 或自然 vLLM Worker 的调用调查；未独立追踪具体 HCCL 算法名、内部归约顺序或 kernel symbol。没有证明所有输入、更多 rank、其他 Worker 或其他版本永远不会产生差异。

`off_exact` 的源码实际比较 `output+1` 消费者，可能掩盖原生输出末位差异，因此只记为这个无注入消费者的精确健康对照。它不影响四次原生输出的无样本结论，也不能证明未曾执行的故障已恢复。`actual_fault_recovery_proven=false` 保留；没有 F、B/F statistics 首差或新增 C1/C2/C3。

本轮有界调查已足以关闭为 **“已测 HCCL 路径无合法原生样本，原 replay 当前不适用”**。不需要再提取张量、重跑相同配置或随机注入通信误差。若以后要更改原故障定义或调查一个确实不同的通信路径，应作为单独机制/范围重新论证，不能升级本负结果。
