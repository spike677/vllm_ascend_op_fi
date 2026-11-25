# V4 0.30 只读观测r3：G1同进程补采执行单

基线：r4.1内r2。r2的max_calls/integer_budget/target_layers不能产生router浮点分数、选择器张量消费链或combine消费者事件。r3只新增观测，不能注入、不能自动放行D13或公共8。

## G1是否需要第三次重启

r3的capture、噪声模块、噪声提取/审查器、probe_support、probe_extension及全部fault_injection与r2逐字节一致，见包内shared-byte-comparison.json。唯一改变的既有Python文件为dsv4_observe_probe.py；新增targeted模块、独立提取器、测试与冒烟程序。不替换r2冻结文件。

若G1尚未启动：将r3独立目录置于该次启动的PYTHONPATH最前，扩展类仍为experiments.worker_core_v11_20261118.dsv4_probe_extension.Dsv4ProbeExtension。inspect_binding规则与r2不变。先完成未启用observe的G1全部噪声采集，再做下面定点观测，可共用G1这次重启，不因本次字节差异要求第三次。

若G1已用r2启动：不要在活跃Worker热替换模块或修改sys.path假装升级。保留合格G1；r3需单独计划一次重启。文件相同只保证采集逻辑身份，不保证新进程状态相同；G0/G1仍各自按原判据审查。

禁止observe/targeted和noise capture同时开启。工具会拒绝已有capture时开启targeted；反向顺序仍须宿主遵守：先所有rank observe stop且installed=false，再capture start。不得将带观测同步开销的运行混入G0/G1样本。

## 执行顺序

1. 校验r4.2索引、ZIP、manifest与shared-byte-comparison；旧r4.1仍有效。
2. P3如仍用r2，按原计划采D1，并据真实call.layer/layer_path选出一个实际MoE层名。选择非hash、真实score-based router层可同时服务D13；不猜层0。源码显示内部router时，必须覆盖真正产生scores和选择器的那层。
3. G1启动r3，逐rank D0/inspect_binding，完成既有正式和金丝雀prompt-manifest及G1噪声组。未知插件绑定仍拒绝；不扩大准入。
4. 全部capture stop、请求完成后，先保存一份无观测的确定性推理结果。执行下述targeted观测，stop后再发相同请求，保留输出和方法恢复证据。
5. 在同组Worker分别采正式长prompt生成1、2 token；若两次无截断，可按需再采4 token定位完整持续调用。这个1/2 token取证不替代4 token正式验收，也不代替G1样本。
6. 用独立提取器回传定点摘要。蓝区分开审D13真实router消费与公共8 combine行映射。

## 实际支持的options与调用

下面沿用已鉴权的宿主executor，不暴露开发RPC给公网；actual_moe_layer为D1已观察到的named_modules名称。代理target_paths只沿用此前源码审定值。

```python
options = {
    "target_layers": [actual_moe_layer],
    "max_calls": 16384,
    "integer_budget": 4194304,
    "targeted_observation": {
        "max_operations": 2048,
        "float_budget": 1048576,
        "integer_budget": 1048576,
        "sample_rows": 1,
        "sample_columns": 2048,
    },
}
# options["target_paths"] = reviewed_target_paths  # 仅在已有审定路径时
rows = executor.collective_rpc("demoncat_dsv4_probe", kwargs={
    "tool": "observe", "action": "start", "options": options}, timeout=120)
# 逐rank成功后发送上述真实请求，等待请求实际完成。
rows = executor.collective_rpc("demoncat_dsv4_probe", kwargs={
    "tool": "observe", "action": "stop", "options": {}}, timeout=120)
# 使用现有黄区宿主回执保存方式，将rows原样写入targeted-stop.json。
```

任一rank安装失败时，等待在途完成后向全rankstop，核对仅本工具包装已恢复。缺失层名或options超限在安装前拒绝。合格材料要有全部rank，不把无调用rank伪造成命中。

## 新证据字段与限制

result.targeted_observation.operations记录真实自然算子（包括F.linear分解后的mm/addmm，以及可见的topk或融合选择器），原算子只执行一次。每条包括step、rank、layer_path、schema、标量、源码调用上下文、输入输出区域与版本、浮点/整数样本、上游输出→本次输入重叠区间及完成事件。scheduler_maps按step保留调度行；runtime_groups分别记录实际TP/EP/MC2组成员和来源，缺失为unavailable，不推默认值。

浮点仅采以0为起点的明确坐标片段，sample_shape/sample_origin/sample_full给出覆盖；sample_sha256是原dtype字节SHA。默认不采权重存储，权重转置视图同样排除。整数按预算采完整张量，否则明确缺失。非有限值用字符串保留，不抹成0。选中层外只延伸到可见消费者/别名链，不追全模型。

edges只在仍活跃对象、同设备、同存储、连续精确区间且版本未漂移时建立；非连续包络不当作准确重叠。inference_mode张量没有版本计数器，明确标记observed_schema_writes_only：根据所有观察到的schema写入使旧关联失效；不可见的Native写入仍需源码审查，不能宣称已排除。view/transpose等别名操作不算读取消费者。consumer_candidate仍需源码审查其参数语义：例如F.linear→topk输入或combine输出→实际算子输入。设备事件同步后记录complete及stream，只证明当前stream；自定义算子内部多stream契约、TP/EP行语义仍须源码证据。不能以事件或指针相等单独宣布COMBINE_OUTPUT_ROW_JOIN_PROVEN。

prepare的部分上下文从真实调用栈局部值/对象字段读取；第一次选中层调用记录ascend_forward_context._EXTRA_CTX的padded_num_tokens和mc2_mask。实际分支、缺失上下文、辅助编码均按[公共8方案](v4_public8_d1_review_plan_20261125.md)审查。不可见融合内部scores不能伪造；若只有hidden而无真实score，就明确缺项。

硬上限：4096条targeted算子、1048576个浮点/整数样本元素各一份预算、最多4行×2048列片段；每次start整段累计。耗尽标truncated，观测错误标errors，原推理继续。不得把截断当作闭合；缩到单层/单请求仍缺证就报告，不扩大硬上限。事件同步会增加观测耗时，不作产品性能测量。

## 提取与回传

```bash
python tools/extract_dsv4_targeted_record.py targeted-stop.json > targeted-extract.jsonl
python tools/extract_dsv4_targeted_record.py --verify targeted-extract.jsonl
```

每段≤4096 UTF-8字节，包含原件字节/SHA、页号/总页数、规范化extract_sha256。张量只回传样本前16项及完整样本SHA/覆盖，完整样本留黄区。按蓝区需要回传summary、router生产/选择/dispatch关联、combine→consumer关联的相关段；保留全量提取文本供本地核验。某条超过上限则拒绝输出，不静默截断。旧noise/identity/binding仍用r2相同工具。

共同参考：[定点补采单](v4_d1_targeted_capture_20261125.md)、[公共8审查方案](v4_public8_d1_review_plan_20261125.md)、[库行13独立放行](v4_d13_independent_release_20261125.md)。其中B档的观测实现由本r3承接；正式D产品适配与资格仍单独门禁。D13优先，不等待公共8。

