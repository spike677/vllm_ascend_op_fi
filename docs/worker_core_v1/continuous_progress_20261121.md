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

|交付能力|新栈 Qwen3.8 TP2|V2-Lite TP2|87 Qwen3.8|87 DeepSeek|
|---|---|---|---|---|
|18|待持续接线/重验|待适配|待持续验收|待适配/验收|
|22|待持续接线/重验|待适配|待持续验收|待适配/验收|
|39 output_proxy|公共入口两作用域通过；304真实投影×4步，每rank1216次；实际消费者与恢复通过|待适配|待验收|待验收|
|39 matmul_range|待实现|未列为首批蓝区模型|待验收|待适配/验收|
|Batch|公共入口CPU/NPU各两作用域通过；304投影×4步/每rank，1216次；最终交付负向检查通过|未列为首批蓝区模型|待验收|W8A8独立DynamicQuant部分待适配|
|41|Native 最后阶段|MLA cache 待适配|待验收|待适配/验收|
|17|不扩大模型范围，仅保留既有视觉模型|不适用|仅既有视觉范围，持续验收待完成|不在该模型范围|
|8|dense 不适用|待持续适配|不适用|M4 待完成|
|28 denominator torch|公共入口通过：16层×4步/每rank，两作用域，完整恢复与实际创建负向通过|实验binding通过：27层×4步/每rank，两作用域；公共入口待阶段3|待验收|稀疏 MLA/M4 待完成|
|20|Native 最后阶段|待对应 site|待验收|不在 RC3 第二批清单|
|26 fia_destination|Native 最后阶段|待对应 site|待验收|不在 RC3 第二批清单|
|33|paged attention 待实现|未列为首批蓝区模型|待验收|不在 RC3 第二批清单|

覆盖统计与性能（28 新实机）：V2-Lite 每作用域每 rank 108 次 applied/changed/consumed/restored，failed=0，耗时 1.7266×/1.6460×；Qwen 每作用域每 rank 64 次，failed=0，耗时 1.2898×/1.2172×。Batch新增实机见 [报告](continuous_batch_real_model_20261121.md)：Qwen3.8每rank每作用域1216次applied/consumed/restored，changed为rank0 1214、rank1 1213，failed=0，4.3012×/4.2694×；Qwen3-0.6B每rank336次，2.8584×/2.7968×。39的最新公共入口结果见下文；其余9项仍没有本轮持续实测通过结论。既有单次通过结果作为历史证据保留，不提升为新契约通过。

最终 a4 独立重启复核逐调用恢复字节 SHA、Qwen 实际输出缓冲区 SHA 全部一致；两模型都通过同组 Worker 两作用域。元数据补充：V4.1 两版 config 已获取且完全相同；真实配置的 CPU 结构夹具通过，稀疏 MLA/EP 运行接线与准入未完成，见 deepseek_v41_metadata_20261121.md。

## 公共入口最新复核

RC3 预览 r2 已冻结推送，仅28/Batch；安装源码校验348文件通过。初次发行Git换行失配保留为失败版本，r2原始字节及Git blob SHA全通过。交付与局限见 `releases/RC3-PREVIEW-REPLACEMENT-20261121.md`。上方“实际设备和真实模型接线缺口”属于首次实现时记录；28/Batch公共Runtime与对象租约缺口已由阶段1闭合，其他项不得继承。

28 public-a5：每rank每作用域64次applied/changed/consumed/restored，failed=0；1.4265×/1.3416×；真实消费者与对象区间恢复、完整OFF/health均通过。Batch public-a3 NPU：1216次applied/consumed/restored，changed分别1214/1213，failed=0；6.2112×/6.0967×。public-a4 CPU同路径为6.0880×/6.4494×，尚无明确性能优势。这里只登记实际A2/0.26/Qwen3.8社区W8A8/TP2，不外推A3、MLA或其他输入长度。

39 public-a12：每rank每作用域1216次 applied/changed/consumed/restored，failed=0；2.42345×/2.43357×。上游非有限1208次，原位置已为正Inf为0次，两者分开记录。详见 [39公共入口报告](continuous39_real_model_20261121.md)。仅增加开发线候选，未变更预览r2，不冻结r3。
