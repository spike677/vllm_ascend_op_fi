# M2 B8 NaN 无注入定位（2026-11-21）

范围：BLUE_COMBINATION / A2 Ascend910B4 / TP2 / Qwen3.8-27B 社区 W8A8 / eager。镜像 c9c9432b69767cf1855efad44f20eaa8ab02520ae05b75c936e67ea4e7eb2702；vLLM 0.26.0+empty、vllm-ascend 0.26.0rc1、torch_npu 2.10.0.post4、CANN 9.1.0。不是黄区 A3 验收，不授予 Batch 故障准入。

更新结论：首次B8的GDN存在错误累加。有限哨兵逐位实验推翻了早先“整块没有写入”的解释：WY request1的BF16输出被按相邻两个BF16组成的FP32字参与旧存储加法。KKT还有同一物理core前序任务累加异常。**零初始化不是修复；原kernel内状态修正正在实测，Batch基线尚未通过。**

## 已定位的直接异常边界

第 0 层 `language_model.model.layers.0.linear_attn` 内的 `recompute_w_u_fwd` 返回 `u`（返回元组索引 1）首先出现非有限值。两个 rank 都是有限输入 → `u` 中 7 个 NaN；`w` 有限。原生状态清零后 initial_state 全零，不能把本次问题归为未清理 SSM 历史状态。

本机运行的 `vllm_ascend/ops/triton/fla/wy_fast.py` SHA256：`51e23f0876d91ae09b32ae70aa03318b2440b7bd3ba029c0d9838869359597ef`。第118–119行仍通过 `torch.empty_like(v)`、`k.new_empty(...)` 分配输出。第122行调用 `recompute_w_u_fwd_kernel`，输出写入语句位于第83行。

| rank0 首次 prefill 边界 | NaN 数 | 证据含义 |
|---|---:|---|
| Layer0 输入、投影、causal conv、状态清零、cumsum、KKT、solve_tril | 0 | 前置已观测张量均有限；清零后状态确为0 |
| recompute_w_u_fwd 的 u `[1,88,24,128]` | 7 | 首个非有限输出，事件 sequence=23 |
| fwd_h 的 v_new / final_state | 7 / 896 | 污染进入计算及缓存状态 |
| chunk_fwd_o | 77 | 继续传播 |
| linear_attn.norm | 8,448 | 归一化扩大污染 |
| linear_attn.out_proj `[88,5120]` | 56,320 | 扩散为整条请求的全部 hidden 通道 |
| 完整模型 prefill / 下一步 decode hidden | 56,320 / 5,120 | 持续影响同一请求 |

严格 B8 中每条 prompt 为11 token；NaN 覆盖 token 行11–21，即零起算 request1，decode 中对应第1行。请求原始token均相同，cu_seqlens=[0,11,22,33,44,55,66,77,88]，chunk_indices 为每序列一个尾块。

## 排除原故障工具与调度差异

没有 worker_extension_cls、DemonCAT binding、permit、Native provider 或 SourceBoundGenerationObserver。只用标准 vLLM RPC 装诊断 hooks，记录真实 model 输出；每轮严格移除。没有事后修补输出。

普通 generate 的两个新 Worker 对照没有 NaN，但调度是 B1 prefill → 1 decode+7 prefill，并非88-token整批，不能作为B8通过证据。

通过标准 sleep(level=0)/enqueue/wake_up(scheduling) 保证 B8 同时入队后，两个独立 Worker 进程组（root-only 与逐层 trace）均复现56,320个NaN。两组首次输出一致：request1为[0,0]，request5为[321,1179]，其余为[271,760]；重复B8和之后B1均为[271,760]。有限值不等于结果正确，request5也是需要保留的异常现象。

## 独立参考与上游依据

实际首次 WY 输入单独保存在 Git 外，仅用于回放。CPU-FP64按序列边界、chunk边界和 grouped heads 计算参考。两rank输入与参考均有限；rank0的 u[0,16,23,6] 实测NaN，而参考为 -0.006276627886109054。rank0 u 还存在最大绝对误差2.857840190937569e38；正常参考范围约[-9.86554,32.76764]。

上游 [issue17009](https://github.com/vllm-project/vllm-ascend/issues/17009) 报告同类首次GDN漏写并明确复核0.26.0rc1；[PR17437](https://github.com/vllm-project/vllm-ascend/pull/17437) 将两个输出改为零初始化。外部记录只作旁证，本机算子边界和独立参考才是本次定位依据。更底层为何漏写（编译生成、launch或同步）不能仅凭该PR推断。

## 初始化与固定输入回放控制

独立新 Worker 中仅将 WY 的 u/w 分配改为 zeros_like/new_zeros：首次B8与后续B8/B1 hidden均有限，request1由[0,0]恢复为[271,760]。但request5首次仍为[321,1179]、重复运行才为[271,760]。因此零初始化只缓解NaN，不是完整冷启动正确性修复。

固定rank0首次输入，在三个独立算子进程分别执行 empty/zero/NaN哨兵分配，每进程4次：12次输出均有限，NaN哨兵也全部被覆盖；最大绝对参考误差w=0.00343114991507254、u=0.060851410031318665。单算子环境未复现，不能把问题写成该输入必然触发。完整模型首次执行上下文、编译/launch/同步/分配因素仍需区分。

完整模型NaN哨兵控制完成：两个rank的w/u各残留33,792个NaN，严格对应request1的11行×24头×128维。w/u的其余请求行均有限；后续B8和B1不再残留。与原stock捕获的k/v/beta/A/g_cumsum/cu_seqlens/chunk_indices逐元素完全相同。

独立参考确认这些残留位置全部应为有限且非零值。rank0中w范围[-0.37699888178527885,0.8958594343028196]，u范围[-1.3085590153932571,32.767637491226196]。早期据此推断“未覆盖”不充分：NaN与任意值的原子加法也会保留NaN。有限哨兵实验已区分这两种解释；stock中w偶然有限仍不能视为正确。

不把诊断对照当作故障注入验收或已发布修复，不修改冻结包、公共准入或宿主软件。更底层原因仍需对实际Triton生成代码、launch与同步进一步隔离；本次没有证据把它唯一归因于编译器或驱动。

## 工具、证据和资源

诊断：`experiments/worker_core_v11_20261118/diagnose_b8_nan.py`；独立参考和回放：`replay_wy_reference.py`、`replay_wy_npu.py`。这不是 msprobe statistics/mix 交付，也不替代故障机制验收。

远端私有尝试：`/home/17-fault-c2c3-20261117/m2-nan-20261121-a1`（自然调度），`.../m2-nan-20261121-a3`（严格B8），`.../m2-nan-20261121-a4`（初始化与回放控制）、`.../m2-nan-20261121-a5`（完整模型哨兵）。a2只准备、未执行。私有完整张量不入Git，仓库仅提交工具和总结。

每窗口使用已有资源锁、逐case门禁及最多2100秒的设备2/5清理器；只允许停止显式独占2/5的外部容器，不改Docker入口/配置，不删外部数据。窗口结束停止清理器并记录恢复。4个窗口全部完成并解除隔离，没有停止任何外部容器。总设备窗口墙钟1516.381秒（约25.27分钟，包含加载、编译、清理，并非kernel执行时长）。

| 窗口 | UTC开始 | UTC结束 | 秒 |
|---|---|---|---:|
| a1 | 05:23:44.108466 | 05:31:44.916729 | 480.808 |
| a3 | 05:32:36.990533 | 05:40:31.139817 | 474.149 |
| a4 | 05:40:48.122478 | 05:46:11.063965 | 322.942 |
| a5 | 05:47:17.397671 | 05:51:15.879543 | 238.482 |

日期均为2026-11-21；北京时间为UTC+8。汇总回执见 [m2_b8_nan_summary_20261121.json](../../experiments/worker_core_v11_20261118/m2_b8_nan_summary_20261121.json)。Git外日志归档保存在D盘 intake/blue/20261121/b8-nan-localization；归档SHA写入汇总。3个诊断统计/准入源码检查与1个独立分段参考CPU测试通过。

## 逐位追因：错误FP32原子累加（历史诊断，已停止）

2026-11-21晚负责人决定：以下修正试验安排已终止，不继续追查第三方kernel。现行结论与部署建议见 [环境缺陷报告](env_defect_gdn_b8_20261121.md)；历史控制记录不代表修复验收。

有限哨兵BF16=1.25只用于诊断。独立分析器匹配同一次实际调用中k/v/beta/g/A全部相同的请求与head，以健康请求输出作为期望写入字。request1的23个可匹配head中，w/u各16,192个FP32字（每字两个BF16）全部逐位符合“旧字+正确字”，正常覆盖逐位匹配均为0。head17输入不匹配，保留不参与证明。

[派生证据](../../experiments/worker_core_v11_20261118/gdn_atomic_store_evidence_20261121.json)保留捕获SHA与逐head结果；完整捕获不提交Git。`analyze_gdn_atomic.py`不修改张量，CPU测试区分正常覆盖、原子加与未写入。

KKT更早出现有限值错误：grid20中异常task为21、41、61、…、181，均为owner1，首个task1正常。CPU-FP64参考显示，累加该core此前任务的Gram矩阵可以解释异常A（保留BF16舍入误差）。首个NaN边界WY不是首个错误计算边界。

原kernel记录的请求边界和chunk索引正确；冷/热binary、grid、形状一致。提前编译、克隆索引、关闭multibuffer、单独前置atomic-none复位均未修复。关闭auto-blockify仅移动异常请求。修复试验改为在原GDN kernel AIC/AIV入口复位，保留empty分配、公式和dtype。完整冷启动、重复B8、独立重启的hidden/logits一致性通过前，不进入Batch故障验收，不宣布最终编译器/运行时责任边界。

### 控制状态接口依据及验证边界

[AscendNPU IR 的 set_atomic 文档](https://ascendnpu-ir.gitcode.com/en/developer_guide/dialects/hivm_dialect.html)说明，原子模式会影响后续GM写入，NONE用于恢复普通写入。[Ascend C asc_init文档](https://asc.gitcode.com/api/SIMD-API/c_api/utils/sys_init/asc_init.html)明确区分控制状态初始化与张量清零，并说明前序算子遗留状态可导致精度错误。这支持当前试验方向，但不替代本机核验，也不证明是哪一个前序算子留下了状态。

实际修正试验仅在私有源码快照和隔离容器内重编译两个GDN kernel。新JIT名字防止复用镜像内旧缓存；原数学输入、公式、输出分配不变。分别记录原IR、修正IR、辅助bitcode与实际binary的SHA。只有重新编译记录和真实执行同时成立才判定干预有效；此前命中旧缓存的运行保留为无效修复试验。
