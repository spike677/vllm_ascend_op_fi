# V4 D1定点补采预案：公共8与库行13

由负责人转交。先审现有D1，能用已有证据闭合的不重采。r4.1有效；不改冻结包，不要求黄区修改公共探针。目标A3/V4 W8A8/实际0.30/CANN9.2.1/TP8+运行时EP。只读补采与G0/G1噪声标定分开，不把带journal开销的运行混作噪声基线。

## A档：现有r2可执行，不需重启

前提：同一健康Worker，上一轮observe/capture已全部stop，绑定检查通过，prompt-manifest已确定。保留原件与SHA，不覆盖原D1。

层选择：从现有D1的call.layer/layer_path选择一个真实发生MC2 dispatch/combine的named_modules层；需要同时服务库行13时优先选已证明采用分数选专家的非hash层。不得填写猜测的model.layers.0。若没有真实层名或并非MC2，先报告，不强行过滤。

使用既有宿主executor（不是HTTP端点；不是新CLI），同一进程分别做两次start→真实请求完成→stop：一份已固定长prompt生成1 token，另一轮同prompt生成2 token。1/2 token用于定位prefill与后续decode，正式持续注入验收仍为4 token。记录实际调度轨迹；若请求被切块则不按单chunk解释。

```python
# 由D1真实层名填写；非可直接复制的占位默认值。
options = {
    "max_calls": 16384,
    "integer_budget": 4194304,
    "target_layers": [actual_observed_module_name],
}
# 如果之前已审定显式target_paths，沿用同一值；不能自动追随代理。
# options["target_paths"] = reviewed_target_paths
rows = executor.collective_rpc("demoncat_dsv4_probe", kwargs={
    "tool": "identity", "action": "inspect_binding",
    "options": {k: v for k, v in options.items() if k == "target_paths"},
}, timeout=120)
# 逐rank检查后才start。
rows = executor.collective_rpc("demoncat_dsv4_probe", kwargs={
    "tool": "observe", "action": "start", "options": options,
}, timeout=120)
# 宿主发送上述真实请求，等待实际完成，保存请求及结果。
rows = executor.collective_rpc("demoncat_dsv4_probe", kwargs={
    "tool": "observe", "action": "stop", "options": {},
}, timeout=120)
```

每次start后整数预算按整个采集累计，不是每次算子预算；过滤只限已选层。任何rank失败时停止新请求、等待在途完成，所有rank清理本探针；不得重复start掩盖部分安装。stop仍按保存目标恢复。超过硬上限不再扩大预算；一次缩小到单层/单请求仍截断，就转缺证清单，不反复采全模型。

按安装schema提取：dispatch的expert_ids/mask及输出assist_info、expert_token_nums、ep_recv_counts、tp_recv_counts；combine的对应assist/counts/expert_ids；各自schema、标量参数、输入输出元数据、step/layer/rank、InputBatch映射、dispatch_candidates。完整整数原件留黄区。r2不采expert_scales浮点值；某些分支不存在mask时保留缺失并查global_bs语义，不补全true。

## B档：现有r2不足时，纳入D候选采集能力

以下不是r2 options，也不是已交付功能。不要传capture_context/capture_float/consumer_trace等虚构参数。只有安装源码、已有只读接口无法补齐时，Codex才按实际D1缺项实现定点采集，随D候选交付，不新增独立适配器补充包。

|位置|最少补采内容|用途|
|---|---|---|
|选中层prepare前后|本步调用序号、replace_allreduce、num_tokens、padded_num_tokens、SP切片和有效mask、输入输出对象区间|连接调度行与本地dispatch行|
|MC2 dispatcher|实际MC2/TP/EP成员全局rank列表及顺序、本地rank、通信名称到组的对应|不把配置EP或部署TP冒充算子组|
|dispatch/MLP/combine/finalize|完整schema和路由整数；对象生产消费链；有效行/去padding；下游输入区间；同stream设备事件|闭合公共8行归属和消费者完成证据|
|非hash router分数→选择器|真实scores形状/有限性/受控行值，top-k ids/weights，实际评分函数、bias、group选择、normalization；精确层/请求行|为库行13独立证明分数被真实选专家路径消费|
|router内部实现|is_internal_router分支及真正分数产生位置/选择函数源码SHA|排除把hidden当成scores、hook未命中的情况|

新采集需先定义总字节/调用/整数/浮点样本预算，张量在已证明的同stream消费边界复制、完成后读取；超预算明确缺证且不改变推理。只保留选中层必要样本，不dump权重或全模型hidden；不能在条件性调用中增加全组collective。只读阶段必须验证卸载后方法身份与推理恢复。

重启判断：A档不需重启；B档若必须更新已导入Worker代码或增加新边界，则将全部缺项合为一次有计划的专用实例重启，不在活跃Worker热替换模块。若现有可信只读接口已能给字段，可无重启补证。G0/G1及profiler回退的既有计划不因此擅自改变；新软件/插件状态需重新绑定身份与相应标定。

## 回传与当日处理

继续r2已有提取方式，附一段定点说明：run/组、源码清单SHA、安装源码SHA、层名、生成长度、各rank调用索引和原件字节/SHA、errors/truncated/预算、映射或router消费结论、缺失字段。提取文本不含完整整数值不等于未采到；先在黄区原件核对，再按需粘贴少量指定调用的整数片段与字段路径，每段≤4KiB。不得手工编造完整映射或默认组。

Codex收到D1当天按[公共8审查方案](v4_public8_d1_review_plan_20261125.md)列明已满足/缺项；只发必要的A档步骤或合并B档字段规格。库行13优先独立推进，公共8不满足时继续关闭；见[独立放行方案](v4_d13_independent_release_20261125.md)。



## r3观测补充

本文保留r2证据边界说明。新增浮点样本、自然算子关联和设备事件的执行方式见[v4_v030_probe_r3_handoff_20261125.md](v4_v030_probe_r3_handoff_20261125.md)；B档观测由r3承接，仍不自动证明router或combine语义。G1先采噪声、关闭capture后再定点观测，不能同时启用。
