# 全 rank 持续触发：开发记录（2026-11-21）

本页以负责人最新 12 项任务为准。以下基础开发记录保留；28 已开始真实模型验收，最新进展见 [实机报告](continuous28_real_model_20261121.md)。公共 Runtime 的28/Batch已接线，最新复核见 [公共入口记录](continuous_runtime_20261121.md)；不继承旧单次验收。冻结发行物未修改。

## 已实现的第一批基础代码

- `delivery_scope.py`、`ordinary_admission.py`、`registry.py`：26 residual_stomp 查询显示本轮排除，普通创建在加载 provider 前拒绝。历史实现和证据保留。
- `continuous_contract.py`：12 项范围、四阶段四 token 请求计划、至少两请求 Batch、按 rank 的八项覆盖计数审查。EP 无调用 rank 允许零命中，但整个组零实际注入不能通过；失败、未覆盖 site、资源残留不能伪装跳过。
- 原 `MultiRankTaskClient` 增加可信部署构造参数 `continuous=True`，复用已有生命周期/RPC/owner/generation/恢复，不另建状态机。1/2/8 个模拟 rank 走同一代码；全 rank 为 injector，条件命中处不新增 collective。新 provider 必须回读相同策略，历史单 rank 证据不能直接接受。
- 历史默认协议保留，新的持续路径不会因为构造参数而获得部署准入；公共 FaultConfig 不增加用户指定 rank 选项。
- `engines/attention.py`：FP32 blockwise online softmax 的单 query 行核心；GQA 和吸收形式 MLA 支持不同 QK/V 维度，显式唯一稀疏 key 集合，指定块仅重复分母贡献。不是合法重排，也不声称复刻历史 kernel。
- `engines/dynamic_quant.py`：当前真实行的共享 absmax/127、INT8 Q、FP32 行 scale；全零输入的 Q/scale 均为零；非有限输入明确拒绝。真实请求数、padding 过滤、原生 scale 布局由 site 校验。

## CPU 检查

第一组 35 项通过：continuous_group、continuous_contract、attention_reference_engine、ordinary_admission、multi_rank。
包含同一组模拟 Worker 两作用域、world_size=1/2/8、EP 零本地调用、不完整覆盖拒绝、26 查询/创建一致拒绝、GQA/MLA 与 FP64 参考、稀疏索引和尾块、分母重放以及历史多 rank 协议回归。

第二组 dynamic_quant_engine 两项独立测试通过，合计本轮 37 项通过；这不是实际 DynamicQuant/QMM 接线或验收。`git diff --check` 通过。

CPU 参考例：256 个等权 key、V 全 1，重放其中 128 个 key 的分母贡献后输出从 1 变为 2/3。只证明错误公式实现，不代替模型 hidden/logits 效果。

## 第一批基础开发时的缺口（历史，最新状态见总表）

1. 28 的实际 site 已补齐逐命中生命周期、重叠拒绝和同 stream 完成证据；其他能力及公共 provider 最终收尾待接线。
2. 覆盖审查器是纯校验函数，须接入实际 provider 的查询、finish 和最终收尾；计数不能由模拟回执代替实测。
3. 28 实验入口已接真实四 token 请求观察器；公共普通任务 adapter 仍待接入，旧单次 prefill provider 不符合持续契约。
4. V2-Lite 稠密 MLA 已绑定真实吸收 Q/K、latent V、scale、请求边界、投影与消费者；V4.1 稀疏 MLA 仍待实际调用适配。
5. Batch site已在真实Qwen3.8 B8 TP2证明原QMM消费：304个实际投影×4个forward，每rank每作用域1216次。公共Runtime接线未完成。
6. M4增加动态placement/token/request纯校验与CPU测试；实际dispatch/combine接线及EP完整汇总仍未完成。
7. 28 短请求性能见下方；其他方法未测。未授予公共 Runtime/RC3 准入。

## 首次实现阶段的124只读准备（历史）

本轮读取设备状态时卡2/5无 NPU 计算进程。已有 V2-Lite W8A8 制品目录已找到。未加载模型、未运行 NPU case、未停止容器、未开启隔离，NPU 用时为零。启动前仍需复查容器实际设备占用、资源锁与制品身份。

## 验证进度总表（新持续契约）

**口径补充（11-22产品模式任务）**：下表的“公共入口”是固定输入的WorkerFaultRuntime机制验收，仍与真实用户产品入口分开。五项产品模式候选与独立实测见[产品使用及范围](rc3_product_usage.md)，其他项不得继承。r4保留十项机制验收工具；33新分组8/16两次均因rank0无模型传播止损，见[paged33重新设计](paged33_redesign_20261122.md)。

产品入口单独登记：124 Qwen3.8 TP2的18、39op、22、28、17已通过；Batch、39mr、41、26fia、20待产品迁移，33关闭，8的Qwen dense场景不适用。124 V2-Lite仅保留历史机制验收，产品入口未验收；87 Qwen TP1/TP2和V4稳定版产品入口均未验收，五处主项目接线未确认。

|交付能力|124 Qwen3.8 TP2|124 V2-Lite TP2|87 Qwen3.8 TP1|87 Qwen3.8 TP2|87 DeepSeek V4稳定版|
|---|---|---|---|---|---|
|18|公共入口两作用域通过；161实际输出点×4步/每rank，644次；恢复通过|公共两作用域通过；55模块×4、220次；27个融合KV内部norm未覆盖|待持续验收|待持续验收|待V4稳定版身份与inspect；未验收|
|22|公共入口两作用域通过；16融合调用×4步/每rank，64次；恢复通过|公共两作用域通过；27MLA×4、108次；351 rotary行变化|待持续验收|待持续验收|待V4稳定版身份与inspect；未验收|
|39 output_proxy|公共入口两作用域通过；304真实投影×4步，每rank1216次；实际消费者与恢复通过|公共两作用域通过；observed463/eligible332，每rank332命中；131输出不适用，52结构候选无forward|待验收|待验收|待V4稳定版身份与inspect；未验收|
|39 matmul_range|公共两作用域通过；704 eligible/rank/作用域全部消费恢复，512非列分片调用不适用；仅W8A8|未列为首批蓝区模型|待验收|待验收|待V4稳定版身份与inspect；未验收|
|Batch|公共入口CPU/NPU各两作用域通过；304投影×4步/每rank，1216次；最终交付负向检查通过|未列为首批蓝区模型|待验收|待验收|待V4稳定版身份与inspect；未验收|
|41|Native公共两作用域通过；每rank observed64/eligible48，实际FIA消费和字节恢复、OFF/health精确恢复；1.01725×/1.04375×|MLA布局待适配|待独立构建/验收|待独立构建/验收|待V4稳定版身份与inspect；未验收|
|17|公共两作用域通过；每rank每作用域1次真实patch调用，全消费/恢复，原参数恢复，旧Qwen3-VL仅历史|不适用|TP1/TP2视觉验收待完成|TP1/TP2视觉验收待完成|缺图像预处理配置，不执行|
|8|dense 不适用|公共两作用域通过；26MoE×4、104次；26次新变化、78次单组decode保持原值|不适用|不适用|待V4稳定版身份与inspect；未验收|
|28 denominator torch|公共入口通过：16层×4步/每rank，两作用域，完整恢复与实际创建负向通过|公共入口两作用域通过：27层×4步/每rank，108次；恢复通过|待验收|待验收|待V4稳定版身份与inspect；未验收|
|20|Native公共两作用域通过；每rank1216次输入内容覆盖、真实DynamicQuant/QMM消费及字节恢复，OFF/health精确；2.36869×/2.39569×|待对应 site|待独立构建/验收|待独立构建/验收|不在 RC3 第二批清单|
|26 fia_destination|Native公共两作用域通过；每rank64次FIA数据写入真实residual并消费/恢复，OFF/health精确；1.11659×/1.12412×|未列为本批蓝区模型|待构建/验收|待构建/验收|不在第二批|
|33|方向A新控制公共两作用域通过；48次/rank应用/消费/恢复、failed0；最低control99.9349%，两rank hidden/logits传播，OFF/health精确；仅限定A2登记，产品模式未验收|未列为首批蓝区模型|待验收|待验收|不在 RC3 第二批清单|

以下为早期**实验binding**性能，不能替代下文公共入口结果：V2-Lite 28每rank108次，1.7266×/1.6460×；Qwen28每rank64次，1.2898×/1.2172×。实验Batch见[报告](continuous_batch_real_model_20261121.md)：Qwen每rank1216次，4.3012×/4.2694×；Qwen3-0.6B每rank336次，2.8584×/2.7968×。最新公共结果见总表和独立报告；旧单次通过结果不提升为持续契约通过。

最终 a4 独立重启复核逐调用恢复字节 SHA、Qwen 实际输出缓冲区 SHA 全部一致；两模型都通过同组 Worker 两作用域。元数据补充：V4.1 两版 config 已获取且完全相同；真实配置的 CPU 结构夹具通过，稀疏 MLA/EP 运行接线与准入未完成，见 deepseek_v41_metadata_20261121.md。

## 公共入口最新复核

RC3 预览 r2 已冻结推送，仅28/Batch；安装源码校验348文件通过。初次发行Git换行失配保留为失败版本，r2原始字节及Git blob SHA全通过。交付与局限见 `releases/RC3-PREVIEW-REPLACEMENT-20261121.md`。上方“实际设备和真实模型接线缺口”属于首次实现时记录；28/Batch公共Runtime与对象租约缺口已由阶段1闭合，其他项不得继承。

28 public-a5：每rank每作用域64次applied/changed/consumed/restored，failed=0；1.4265×/1.3416×；真实消费者与对象区间恢复、完整OFF/health均通过。Batch public-a3 NPU：1216次applied/consumed/restored，changed分别1214/1213，failed=0；6.2112×/6.0967×。public-a4 CPU同路径为6.0880×/6.4494×，尚无明确性能优势。这里只登记实际A2/0.26/Qwen3.8社区W8A8/TP2，不外推A3、MLA或其他输入长度。

39 public-a12：每rank每作用域1216次 applied/changed/consumed/restored，failed=0；2.42345×/2.43357×。上游非有限1208次，原位置已为正Inf为0次，两者分开记录。详见 [39公共入口报告](continuous39_real_model_20261121.md)。仅增加开发线候选，未变更预览r2，不冻结r3。

18 public-a14：每rank每作用域644次applied/changed/consumed/restored，failed=0；1.88848×/1.84664×；覆盖129模块输出和32融合Q/K归一化输出。见 [18报告](continuous18_real_model_20261121.md)。22 public-a15独立通过：每rank每作用域64次，rotary行变化672，Q/K变化64调用；1.16110×/1.09448×。见[22报告](continuous22_real_model_20261121.md)。

V2-Lite public-a16：27层108次/rank/作用域，全部applied/changed/consumed/restored，failed=0；2.02685×/1.82329×。同组Worker两作用域公共入口、完整恢复及无许可拒绝通过；不外推V4.1稀疏MLA/EP。见[MLA公共报告](continuous_mla28_public_20261121.md)。

V2-Lite public-a20：8的当前AllGather路由/反排列链已通过公共两作用域，26个site、104应用/消费/恢复，changed26，单组decode78次仍执行但无新变化；failed0，1.58518×/1.49855×。当前请求producer→专家输出→A2 abs→原生unpermute关联完整；EP/多请求未验收。见[MoE报告](continuous_moe8_public_20261121.md)。

Batch public-a17：可选forward延迟核对模式通过公共两作用域，样本数和精度标准不变，耗时5.93327×/5.97488×，默认immediate及r2不变。见[性能记录](continuous_batch_performance_20261121.md)。本轮35项持续框架/consumer/恢复/模式准入CPU测试通过，不代替历史测试欠账修复。

## 2026-11-22 r3 交付更新

r3 已冻结六项公共持续候选，发行提交 0a10cb2e，源提交112376ee；新标签 demoncatpfi-rc3-preview-20261122-r3。971项范围测试、33项补充检查及124无设备358文件字节核验通过。A3 TP1/TP2分别待验收，未外推登记。上文r2与未冻结描述保留历史时间范围；最新交付见 releases/RC3-PREVIEW-R3-HANDOFF-20261122.md。33维持未通过，17视觉塔下一项，Native门禁尚未满足。

## 2026-11-22 视觉17后续更新

vision17-a3公共入口通过：两rank每作用域observed/eligible/applied/changed/consumed/restored均1、failed0；hidden/logits传播和完整OFF/health精确恢复，1.05504×/1.00378×。详见[视觉报告](continuous_vision17_20261122.md)。仅开发线BLUE_COMBINATION，冻结r3不变；17与33均已有结论，Native进入可实施阶段。DeepSeek稀疏MLA新增CPU参数解析与sink参考，仍不代表实际接线或A3验收。

## 2026-11-22 Native41更新

41在A2/0.26/Qwen3.8社区W8A8/TP2通过公共入口两作用域，见[Native41报告](continuous_native41_20261122.md)。同stream真实KV区域改写与消费者事件恢复已验证，16次直接KV prefill/rank/作用域明确不适用；不把它记为全64次注入。冻结r3不变；Native26 FIA→residual和20输入覆盖继续，DeepSeek稀疏/EP及A3仍待接线与验收。

26 fia_destination 已通过独立复核，见[Native26报告](continuous_native26_20261122.md)。M4新增原生DispatchV2/CombineV2对象关联与CPU签名测试；实际runner分片行映射和接收端expert归属尚缺，见[MC2准备](mc2_dispatch_join_20261122.md)，不登记A3。

20已完成公共两作用域独立复核，见[Native20报告](continuous_native20_20261122.md)。首轮证据容量失败后改为全量校验、逐站点计数与有序列表SHA，未放宽容量或恢复标准。Native三项均仅开发线BLUE组合；原始数据不上传，冻结r3不改。[扩展CPU回归](native_cpu_review_20261122.md)1023项、0失败/错误、2旧研究skip；不代表所有历史欠账清零。

## 2026-11-22夜策略调整

DeepSeek目标改为V4稳定版，具体模型/权重/软件指纹等待黄区新回传；旧V4.1仅保留缺陷与历史CPU准备，不继续清零工具和A/B/C。故障8新路线为combine输出后同请求行成组错位代理，不能修改MC2通信元数据。r4准备原r3六项及17/41/26/20；33重新设计，在新公共实测通过前仍关闭。

## 2026-11-22隔夜最终增量

最新冻结物为r5，不修改r4/旧标签；见[r5交付单](../../releases/RC3-PREVIEW-R5-HANDOFF-20261122.md)。33方向A在A2/0.26/CANN9.1/Qwen3.8社区W8A8/TP2的限定公共入口验收已通过并记录实际rank身份；不在r5，不自动扩展到产品模式或A3。见[33报告](paged33_direction_a_20261122.md)。旧33失败保留，不覆盖新控制的判断。

8新增AllGather输出行错位代理参照：V2-Lite相同新栈TP2两作用域，每rank104observed、26eligible、78单行decode不适用、26次消费恢复；不替代历史反排列路径，也不授予MC2/EP资格。见[参照报告](allgather8_reference_20261122.md)。DeepSeek目标已收敛到V4-Flash-0731 W8A8/CANN9.2.1/TP8+EP/DP1；噪声1.5规则与只观测入口CPU通过，等待黄区实际源码SHA/标定/只观测结果，所有V4能力未登记。黄区r4缺身份的文本只记结论；等r5完整回传。
