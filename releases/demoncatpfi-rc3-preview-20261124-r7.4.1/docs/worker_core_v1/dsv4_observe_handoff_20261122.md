# V4-0731只观测试跑（不注入）

目标A3 / V4-Flash-0731-W8A8 / v0.32.0rc1-a3 / CANN9.2.1 / TP8+EP / DP1，4096以内长prompt单chunk。蓝区不运行V4。不得与V4.1 W8A8的16卡OOM/1M上下文实验混合。

11-23更正：源码候选清单改用[dsv4_0731_source_review_20261123.json](dsv4_0731_source_review_20261123.json)，它按同一上游提交的Git blob原始字节生成。旧11-22清单的路径有效，但SHA来自Windows CRLF工作树，不再用于Linux字节匹配。旧发行包保持原样，不能通过修改黄区文件换行去“匹配”旧SHA。请在镜像已加载的vllm_ascend包上逐个返回新清单paths的SHA，不用Git候选的SHA覆盖实际文件。`dsv4_observe_probe.source_fingerprints(paths)`以实际import包根目录读取，只返回相对路径和SHA，缺失为null。工具r1的source动作无需改动，调用方传入新清单路径即可。相同也只是源码身份，不是注入通过。

可信验收驱动使用`experiments/worker_core_v11_20261118/dsv4_observe_probe.py`的observe callable，经现有受信collective_rpc：`source`（options.paths为清单路径）、`start`、真实服务推理、`stop`。不创建Worker，不注入，不写入张量，不改变调度，不创建fault许可。必须先完成源码/硬件/软件/模型身份收集；只观测可接收不匹配源码以发现实际差异，绝不据此放行注入。

start仅在现有runner.execute_model边界装观察上下文，每次实际execute内部进入、退出TorchDispatchMode；stop在空闲控制边界移除自身包装。已有边界owner时拒绝，不能覆盖其他任务的wrapper。没有调用的rank记录0，不制造调用。观察器不会让原算子返回新张量。

每rank返回18、22、39op、39mr、Batch、41、8、28的observed/eligible、prefill/decode/unknown计数、实际operator、Python调用路径、输入输出shape/dtype、不适用原因和实际request ID。这里eligible只表示类型和请求映射具备候选条件；`injection_adapter_validated=false`，不等于可注入。28/41的压缩/稀疏布局没有运行join证据时eligible=0并明确原因。C++融合内部未暴露到TorchDispatch的调用报告不可观测，不能将0解释为模型没有算子。

每次推理≥2500 token，固定4生成token、≤4096单chunk。输出按rank分文件，文本回传时按rank/机制拆分，保持所有不适用和不可观测项。stop结果、源码SHA、运行时实际CANN身份和prefill/decode独立计数交负责人转回。这里只观测入口CPU测试通过；等待真实结果后调整site，再制作第二批验收包。

## 11-23 decode计数复核（不注入）

本轮报告39992/40081之差89，只能称“两次总采样差”。未给rank、窗口、工具过滤和图配置，不能称完整decode算子覆盖，也不能据此判定根因。以下作为负责人转交的下一轮只观测口径；不要求改runner或重启生产服务。

1. 在既有专用Worker读实际`vllm_config.compilation_config`、`cudagraph_mode`、`model_config.enforce_eager`，保留值和读取来源；缺失为null，不从启动命令补填。同时记录runner类、实际模型类及包module、hf_config.architectures/model_type、实际并行参数、已加载CANN。`deepseek_v41/model.py`存在不能证明被0731执行；还需实际forward/观测调用栈。
2. 确认本次decode实际运行模式为NONE/eager。配置之外，保留实际执行的graph/replay日志或受信只读调度观测。候选MRV1在`worker/v1/model_runner.py:2507`把enforce_eager传给force_eager，并在2513附近打印本批实际cudagraph_mode；其他runner需从其实际源码找等价证据，不套字段默认值。若图关闭无法证明，结论为“decode覆盖未证明”，不放行Python持续注入。
3. **每rank单独计数**：rank0…7各有独立会话/输出，记录工具版本、活动类型、过滤规则、profile路径的私有身份/摘要。不能用一个rank×8，也不混合TorchDispatch API次数和msprof底层kernel次数。工具r1.observe只统计候选机制，不能将其observed总和当全部kernel数。
4. 同一Worker组先按实际方案预热，排除编译、图捕获、权重加载、canary和其他请求。固定实际prompt token SHA、参数和请求顺序，关闭prefix cache/MTP，保证单chunk和请求不提前EOS；分别运行max_tokens=1与2，返回实际completion_tokens及每步调度。按rank逐算子计算`C2(op)-C1(op)`；单chunk且无投机时两请求应各有一次prefill，后者额外一次decode。
5. 每次窗口都在请求完成且设备活动收尾后结束/flush profiler，使用新输出目录，不能重复读取上一会话。对照前两次的prefill是否相同；任何负差、缺rank、缺步、前缀命中或额外prefill说明差分不可直接归因decode，保留失败原因。可重复一对确认窗口完整；直接按实际scheduler步分割的decode区间应与差分相互印证。不能只报告减法后正值而丢弃负项。
6. 回传每rank：实际图配置/本批模式、C1/C2总数及按算子差分、prefill/decode/unknown步数、是否覆盖所有rank、观察器移除和injected_calls=0。已有extract_dsv4_record observe用于site结果；完整内核计数可分页为≤4KiB文本并带原文件字节/SHA，不冒充该提取器的schema。

### 图回放的影响

候选源码`compilation/acl_graph.py:161–167`在NONE模式调用Python runnable；回放分支约300行执行`entry.aclgraph.replay()`并返回缓存输出。故图内Python算子/模块hook通常不会逐decode重执行，TorchDispatch可能仅看到图外活动；capture阶段曾命中不能证明replay阶段持续命中。图外hook仍可能执行，设备profiler也可能看到图内kernel，因此**不能笼统说所有hook失效，或说89必然由图造成**。当前能力要求按实际decode符合条件调用取证，不能用静态prefill清单推断图内覆盖。

## 噪声条件对齐

正式标定仍用4生成token和两个独立Worker组各≥5样本，计数用max_tokens1/2是另一项只观测控制，不作为噪声样本混入。额外记录prompt token SHA、请求顺序、进程身份、seed、torch deterministic开关、NPU deterministic等级（可观测时）、HCCL相关设置及指标定义；读不到写null。11-22的首选logprob本来只有微小变化，其他候选可达约1；不能将新“1e-5”概要直接用作上界。规则始终是原始hidden/logits和top5指标两两最大值×1.5。
