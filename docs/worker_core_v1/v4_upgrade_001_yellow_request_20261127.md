# 给黄区：S0-UPGRADE-001-A 最小补件

按 D73，由黄区自行处理 S1/M1，不等待蓝区常规审查。本次补件用于已发起的新包升级请求，不撤销黄区独立决策权。

请先做不需重启的部分，补充单正文≤4KiB；原件留黄区，附文件名、字节数、SHA256：

1. 实际镜像/容器安装的源码内容和逐文件 SHA：deepseek_v4/model.py；fused_moe 下 fused_moe.py、routed_experts.py、token_dispatcher.py、prepare_finalize.py、moe_comm_method.py；common/ops/sequence_parallel.py；实际 vLLM FusedMoE/selector 的实现文件与导入关系。源码内容通过负责人批准的通道交蓝区；只有 SHA 无法做差异审查。不要执行模型初始化来找文件。
2. 实际 config 的 num_hash_layers、MoE/专家/group/top-k 相关字段及 config SHA；已保存 named_modules 中目标路径与类。没有运行时值的 internal-router/hash/draft 字段保持 null，不推断层 3 一定是首个非 hash 层。
3. 原 D0/D1 运行 ID 对应的实际 rank 集合、各 rank 身份原件 SHA、PID、r3/扩展 SHA。已保存 maps/加载库 SHA 一并索引；旧进程未保存 maps 则标缺失，不能从镜像倒推。后续采集不能冒充旧运行身份。
4. 查 G1 的 targeted_observation.runtime_groups：实际 EP/TP/MC2 size、成员顺序、local rank、来源；查 G0/G1 每个独立进程组 formal/canary 的请求样本数。8 ranks×4 samples 不等于单 rank/进程组已满足 5 samples。
5. 把 shared expert/combine 后处理链标为已观察的输出链；router scores→真实 selector→最终 dispatch IDs/weights 仍缺证。不要将 SOURCE_REVIEW_REQUIRED 改为已证明。

蓝区已开始候选源码审查、只读观测核心与 CPU 测试。实际源码到达后锁定对象/参数/张量边界，才发布完整探针和执行单；此开发提交不可直接替换已运行的 r3。D13 与公共 8 分开判定，不因公共 8 缺证阻塞已满足条件的 D13，也不因 D13 通过解除公共 8 门禁。

运行窗口由黄区确认：v4b-07（0–7、19301）仍健康且获准用卡则原窗口采；否则按负责人安排 S1 后、S2 前的 8–15 短窗口，采完切回 Qwen。噪声与观测分开，不热替换已加载模块。

LEDGER 按负责人最新更正执行：20 主体 PASS(M1)，Y-D12 仅黄区降精度部分；13 目标 PASS 或 CLOSED_YD；16/17/32 验收通过等合入（17 等 M1/r8.1）。蓝区不重新审查黄区自有能力。
