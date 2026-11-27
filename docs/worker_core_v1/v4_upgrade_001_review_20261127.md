# S0-UPGRADE-001：蓝区启动与现场补证边界

2026-11-27，按新 D73 第 2 类升级请求启动。S1/M1 由黄区独立继续，不等本项；蓝区不操作 A3，不以 A2 结果授予 A3 资格。r8、r3 已冻结发行物不变。本次是开发提交，不是新的可执行 D 包或探针发行包。

## 已核查的材料

本地接收目录 `D:/CodexArchives/20261127/yellow-s0-complete-d1-upgrade-ledger/` 两份文字记录，以及负责人 11-27 附件。尚未找到 S0-UPGRADE-001-A、实际安装源码包、G0 原件或 runtime_groups 原件；摘要中的 SHA 只按黄区声明保存，不能据此声称蓝区已审过原件。

- G1 声明 8 ranks、每 rank 4 个样本，共 32 个。不能把 rank 数乘进每组样本数以满足“两独立进程组、每组至少 5 个样本”。先按 formal/canary、进程组、请求样本 ID 核对 G0/G1；同一请求的多个 logits step 不能冒充独立样本。
- 36 operations/rank 和无截断，只证明所选观测边界完整，不证明 router 被覆盖。shared expert → combine 后的输出链不是分数 → 选择器链。
- `runtime_groups` 是否存在须查 G1 原件的 targeted_observation.runtime_groups；字段 null 时不从 TP8、EP=true 推出成员和顺序。
- “不用起服务补 D0”只适用于文件静态身份及已保存的进程证据。旧 Worker 已退出且未保存 `/proc/<pid>/maps` 时，镜像、库文件 SHA 不能证明旧进程加载了哪个库。明确标为旧运行缺证，下一窗口另采当前运行，不回填旧运行。

## v0.30.0 候选源码初审（不是已安装源码结论）

已取回 7 个官方文件；SHA、大小、URL 及 AST 候选位置在 `v4_upgrade_001_sources_20261127.json`。原 11-25 清单中重合的 5 个文件 SHA 一致。尚缺实际 vLLM FusedMoE/selector 文件及模型 config；不根据标签猜测它们。

1. `models/deepseek_v4/model.py:333` 的 hash 条件同时涉及层索引、num_hash_layers 和 draft 标志。只有 config 的实际值、named_modules 对应关系和实际 hash=false 三者一致，才能选非 hash 层；层 3 不是无条件默认。若 num_hash_layers=3 且非 draft，层 3 才是源码规则下的首个候选。
2. 同文件 `:399–413` 有两支：内部 router 入口的 router_logits 可能是 hidden 输入；外部分支才直接 F.linear 产生分数。不能把同名参数无条件当作分数。内部候选生产点为 `AscendMoERunner._apply_router_linear`，需实际对象和分支确认。
3. `routed_experts.py:638` 的 `_select_experts` 委托真实 router 后，还可能做 log2phy、shared expert 拼接、force-EPLB 或 profile-run 随机路由。必须分别记录 selector 原始结果与真正交给 dispatch 的 IDs/weights，不能只核对两处 shape 一致。
4. `forward_impl` 经 prepare 修改 hidden/router_logits，选择专家，再交 quant_method.apply；dispatch/combine 由实际通信实现决定，最后 finalize。prepare/SP/padding、MC2 辅助整数和真实组都是行归属证据的一部分。

来源：[model.py](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/models/deepseek_v4/model.py)、[fused_moe.py](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/ops/fused_moe/fused_moe.py)、[routed_experts.py](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/vllm_ascend/ops/fused_moe/routed_experts.py)。以上仍须与黄区实际文件逐字节比较。

## 本次已实现、尚未接线的部分

- `tools/review_v4_installed_sources.py`：只读文件、AST 解析，不 import/执行部署源码；输出逐文件 SHA、候选调用、缺件、与上游的差异。缺 config 不补默认；始终不授予资格。
- `experiments/worker_core_v11_20261118/dsv4_boundary_observer.py`：Python 方法边界观测核心，复用 r3 MethodLease。全目标先核对源码 SHA、参数和严格 bound-method 守卫；只调用原方法一次、保留 kwargs、返回对象和异常；单独调用序号与父调用序号；预算耗尽只标截断；恢复不移除外部包装。
- 这不是可直接部署的 V4 接线。对象解析、实际 selector 文件、张量快照与同 stream 事件、runner step/request 映射、跨线程/共享通信对象的作用域过滤仍须按现场源码完成。没有这些，不能将记录当作消费者完成或行归属证明。
- CPU 夹具验证源码解析与观测协议，不模拟已取得 V4 设备证据。未占 NPU，未更改黄区组件。

离线使用（运行方已有源码时；无需起服务）：

```sh
python tools/review_v4_installed_sources.py \
  --root installed-source-tree --upstream verified-v030-tree \
  --config model-config.json --layer-index 3 \
  --extra vllm/实际选择器文件.py > source-review.json
```

`--extra` 要用实际导出的路径，可重复；示例路径不是可猜填的实现。完整报告留本地，回传文件名/大小/SHA 和必要差异摘要。源码可按逐文件文本或归档路径转交负责人；不能仅回 SHA 后要求蓝区分析未知内容。

## 下一步与交付门槛

1. 黄区现在补静态源码、config 字段、既有 ranks/maps/组/噪声原件索引；同时继续 S1。
2. 蓝区收到后完成实际差异与分支审查，锁定生产者、选择器、prepare/dispatch/combine/finalize 对象，完成只读 probe 的 Worker 接线、CPU 张量夹具与独立导入测试，交付一个完整包和执行单。未确认真实边界不要求黄区试错式重启。
3. 黄区确认 v4b-07:19301 与 0–7 可用时使用原窗口；否则按负责人安排在 S1 后、S2 前使用 8–15 的 TP8 短窗口。噪声采集与定点观测分开；工具代码更新不热替换已加载 Worker。
4. D13 独立审查 router/selector/dispatch 与噪声后再出候选；公共 8 额外要求 combine 输出行归属和真实消费者。任一缺失维持 `COMBINE_OUTPUT_ROW_JOIN_UNPROVEN`，不以 D13 放行替代。

LEDGER 的 20、13、16/17/32 更正按负责人本附件由黄区维护：20 主体 PASS(M1)、13 终态 PASS 或 CLOSED_YD、16/17/32“验收通过，等合入”（17 等 r8.1）。蓝区只引用文字决定，不擅自重分库行、不将这些文字状态写成公共资格登记。
