# V4 G1目标层与四token logits次数：书面确认

2026-11-26。适用于黄区A3、V4 W8A8、实际vLLM/vllm-ascend0.30、CANN9.2.1、TP8+运行时EP的专用实例。是源码审查和采集规则，**不是V4实机通过声明**。无需等待目标层才能开始合格G0；G0仍用r2，G1的noise完成后才用r3定点观测。不改r2/r3冻结代码或噪声门禁。

## 已核对的源码

官方v0.30.0文件下载原字节SHA与黄区本轮回传前缀相符，完整身份仍以黄区已加载文件核对为准：

- [DeepSeek V4 model.py](https://raw.githubusercontent.com/vllm-project/vllm-ascend/v0.30.0/vllm_ascend/models/deepseek_v4/model.py)：`e834918a8c9f9458452a437e8c4a66d25e76c816758358edf3eef06dc0588c71`。本地原文件行333：非draft层的hash条件为 `layer_idx < config.num_hash_layers`；行399–415分内部router与外部F.linear；行1106为compute_logits。
- [NPU model_runner_v1.py](https://raw.githubusercontent.com/vllm-project/vllm-ascend/v0.30.0/vllm_ascend/worker/model_runner_v1.py)：`7fdd01b9e77c69a93282a7ca8e63bd80f30b3781e85a62369264efd5c312f75e`。行2634/2645是互斥分支的logits调用，2657保存执行状态，2763起sample_tokens消费状态；4242/4325还存在dummy路径调用。因此只能对隔离的请求窗口计数，不能把进程全部logits调用都当该请求。

网页显示行号可能折叠空行，复核以完整SHA和函数上下文为准。这里只引用公开源码作候选语义，未以tag替代安装身份。

## G1目标层规则（现在确认）

1. 从**当前已加载模型**的有效text config读整数`num_hash_layers=h`，缺失就报告，不能默认0。只用当前V4-0731实例的config，不借本地V4.1-1017社区配置。
2. 层编号是零基。首个候选是`layer_idx == h`，不是h+1；如该处没有实际MoE父模块，则按实际层索引递增找首个`layer_idx>=h`的MoE，并记录跳过原因。
3. 候选必须是正常主模型的MoE父模块，具有真实gate/experts，`hash is False`、`gate.tid2eid is None`；排除draft/MTP/视觉和PP占位。记录父模块、gate、experts的**实际named_modules完整名**、类名、layer_idx及`experts.is_internal_router`。TP8各rank对照，不能按字符串模板猜`model.layers.0`，不能把配置TP当EP组成员。
4. `target_layers` 填**MoE父模块的实际名称**，不要只填gate。外部router分支直接F.linear读取gate.weight，不保证调用gate.forward；仅hook gate会漏掉分数产生。内部router分支中名为router_logits的参数在进入experts时可能仍是hidden，不能仅凭参数名把它当分数。
5. 用r3一次定点请求确认该父模块实际执行，记录scores生产→选择器→dispatch的同一步同层关联。只有结构候选不等于已证明score路由；融合内部不可见时记缺口，不替换算子制造证据。公共8还需要combine输出行归属、实际TP/EP/MC2组和真实消费者，库行13成功不替它放行。

当前回传没有h的原值和MoE named_modules清单，因此不能负责任地写死某个数值层号；上述规则是可直接执行的选择算法，不等待新的负责人决定。可在黄区已有可信只读Worker方法内使用下段（不是新增的probe action，也不执行模型）：

```python
runner = worker.model_runner  # 如有代理，只用inspect_binding审定的真实计算层
model = runner.model
cfg = runner.vllm_config.model_config.hf_text_config
h = getattr(cfg, "num_hash_layers", None)
if type(h) is not int or h < 0:
    raise ValueError("G1_NUM_HASH_LAYERS_UNOBSERVED")
modules = dict(model.named_modules())
names = {id(module): name for name, module in modules.items()}
candidates = []
for name, module in modules.items():
    idx = getattr(module, "layer_idx", None)
    gate, experts = getattr(module, "gate", None), getattr(module, "experts", None)
    if type(idx) is not int or idx < h or gate is None or experts is None:
        continue
    if getattr(module, "hash", None) is not False or not hasattr(gate, "tid2eid") or gate.tid2eid is not None:
        continue
    if id(gate) not in names or id(experts) not in names:
        continue
    candidates.append({"layer_idx": idx, "parent": name,
        "class": type(module).__module__ + "." + type(module).__qualname__,
        "gate": names[id(gate)], "experts": names[id(experts)],
        "is_internal_router": getattr(experts, "is_internal_router", None)})
candidates.sort(key=lambda row: (row["layer_idx"], row["parent"]))
result = {"num_hash_layers": h, "candidates": candidates,
          "status": "STRUCTURAL_CANDIDATES_NOT_RUNTIME_PROOF"}
```

在实际主模型路径和源码排除draft/MTP后，选排序首个候选；如果同索引多套主模型/代理无法区分，标目标不明，不盲选。该只读方法若当前Worker没有部署入口，不为G0临时重启；将核对并入已计划的G1初始化或已有named_modules导出流程。缺层名只阻塞targeted，不阻塞noise。

r3 options使用已交付字段，不新增参数：

```python
options = {
    "target_layers": [actual_moe_parent_name],
    "max_calls": 16384, "integer_budget": 4194304,
    "targeted_observation": {
        "max_operations": 2048, "float_budget": 1048576,
        "integer_budget": 1048576, "sample_rows": 1, "sample_columns": 2048,
    },
}
```

先全rankcapture stop且installed=false，再observe start→真实请求→等待完成→observe stop。按r3单先1token、再2token定位prefill/decode，需要时再4token；这不替代4token噪声组。耗尽预算、缺scores或缺消费者按D64记缺证，不扩大硬上限、不让observed结果假装qualified。

## 四token请求：预期4次，不是默认5次

前提是单真实请求、实际生成满4token、单chunk、无MTP/投机解码、无异步超前执行、无额外请求/健康生成流量，capture在warmup完成后开始且只覆盖该请求。**prefill产生第1个输出token；之后3次decode产生第2–4个token。** 因而每个承担logits计算的rank预期4次compute_logits及4次一一对应采样，不是prefill再加4次decode。HTTP完成后的排空/缓存清理不应被假定需要再采一个token。

这是受上述前提限定的请求计数预期，不是所有运行配置的普遍次数定理。EOS提前停止、chunked prefill、异步调度、spec/MTP、dummy/profile路径及其它请求会影响调用观察；先读真实启动与调度记录，不能用max_tokens=4直接证明恰有4次。

r2/r3的`dsv4_noise_capture.py`逐字节相同：只保留最多4步，第五次及以后记录`NOISE_MORE_THAN_FOUR_FORWARDS`；stop在缺4步时记录缺失，sample_join另核对4次一一关联。**capture result的steps长度4不能反证没有第5次。** 必须同时审errors、capture_valid、sample_join_proven、samples关联和请求ID；出现上限错误后不能宣称保存了第5步完整tensor，也不能由“4条steps”推断总调用数。

### 校验样本的自动分支

1. G0批量采集前只跑一个校验样本：全rankinspect_binding干净，capture start成功，发送已固定prompt生成4token，等待真实完成/排空，逐rankstop并保存完整结果。不要同时开启observe。
2. 若各参与rank恰4步、4个sample关联、无error、capture_valid和join通过、请求ID一致且数值满足noise标准，继续G0。无须再等codex确认。
3. 若看到5次（或上限错误证明至少5次）：当前样本无效，完整保留原始回包/错误/采样序号/token数/请求ID/时间范围/配置/源码指纹。检查是否混入warmup、额外生成健康检查、重发请求、MTP/spec、异步超前或非单chunk；不能直接宣称“0.30正常就需要5次”。
4. 若已有日志明确证明可纠正的宿主窗口/配置错误，按D64在同一项**最多再校验一次**；只调整黄区宿主采集窗口或已批准测试参数，保存新旧差异。不得手删一步、把max_tokens改3凑4步、把上限改5、热改探针或扩宽审查器。
5. 第二次仍多于4/少于4、采样关联不明，或无安全修正依据：G0记`NOISE_CAPTURE_CALL_COUNT_UNPROVEN`（本补充的记录分类，不是假称已有工具错误码），停止该噪声组，不提交为合格标定；保留原件，继续r8切换、离线准备等独立工作。G1同工具逻辑也不能假定自动解决，noise资格保持关闭，targeted可按已批准的独立只读窗口取证。
6. 若需要改变公共capture行为或审查口径，属于公共工具维护，交codex/负责人处理；不阻塞其它任务，也不在黄区改冻结工具。只读现有日志不足以还原第5次归属时明确缺证，不要求为解释它在G0叠加journal污染噪声样本。

库行13须router生产/选择/dispatch证据与合格G0/G1噪声均通过才出产品包；公共8独立审combine。噪声失败时可以继续准备适配，不放行任何D产品资格。

## 书面依据及回传

本结论结合完整SHA对应的官方源码、r4.2冻结r3捕获逻辑，以及黄区P0/P2/P3文本；未在A3实际运行。回传：h、实际候选/选择层、每rank路径和源码SHA、internal-router分支、四token的实际返回token数、steps/errors/samples及join、请求窗口和所选D64规则。EP组仍缺时保留null，不用TP8代填。无需回传机器地址或容器名到Git。
