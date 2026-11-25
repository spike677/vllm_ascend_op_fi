# V4 0.30 探针 r2：严格绑定与连续采集独立补件

本补件修复Codex交付的只读工具，不修改r7.4.1/r8运行时、噪声判据或任何故障机制。A3目标为实际vLLM/vllm-ascend 0.30、CANN9.2.1、V4 W8A8、TP8；A2 0.26烟测只验证框架对象包装/恢复可移植性，不能代替A3或公共8资格。

## 重启前现在可做的事

保留当前健康实例，不重复start/capture，不自行删除守卫或替换方法。保存已有成功/失败请求体、完整stdout/stderr、探针源码SHA，以及八个Worker分别的maps与日志。API进程89个库的记录不能代替Worker映射。按rank核对installed/清理状态，顶层500不说明八rank全未安装。

负责人最新安排：先检查本包manifest、diff和离线测试，再由黄区重启专用实例完成D0、绑定检查、D1、G0；随后立即第二次重启做G1，不等待G0回传。若绑定检查实际指向profiler插件，允许先按下方限定插件后额外重启一次；失败或缺证不得跳过守卫。

## 安装变化及安全边界

Python3.12，采集需要部署自带torch/torch_npu；审查/文本提取只需标准库。独立解压到新工具目录，把此目录放在Worker启动PYTHONPATH最前，不覆盖公共wheel或旧工具目录。

D0/D1专用实例的Worker扩展只加载本包探针扩展：

```text
--worker-extension-cls experiments.worker_core_v11_20261118.dsv4_probe_extension.Dsv4ProbeExtension
```

黄区宿主/RPC入口继续实施部署方鉴权；本mixin不提供鉴权或公开服务。禁止把开发用collective_rpc接口暴露给不可信网络。**不要继承或组合黄区旧fault_injection部署扩展**，包括会安装batch_scale、residual26_worker、vision17_worker/MethodPin的组件。不要使用A2专用SmokeExtension。启动差异记录须列出旧/新worker-extension-cls、PYTHONPATH、扩展源码SHA、启动命令和Worker PID/rank，不得仅说“已重启”。

如果另有必要扩展，先在隔离初始化阶段运行inspect_binding证明它未给runner/model挂实例属性，再决定组合；不得以来源指纹、模块名或人工批准绕过守卫。鉴权应保留在可信宿主控制边界，不借此保留会改写模型方法的旧扩展。

冻结RPC只用kwargs，八rank逐一检查：

```python
rows = executor.collective_rpc(
    'demoncat_dsv4_probe',
    kwargs={'tool': 'identity', 'action': 'read', 'options': {}}, timeout=120)
# 每rank结果：{schema:'dcat.probe.rpc.v2', rank, status, exit_code, result}
# 失败保留code/reason/error_type/traceback；不以HTTP200替代逐rank检查。
```

D0列出parallel_config候选来源及真实字段，TP/DP/EP组大小、成员与rank；runner/model/sampler类、对象和方法源码身份。不存在的字段为null并给原因，不能用默认EP。多个配置来源冲突需要人工审查。

先执行只读检查，再安装。此调用不添加或删除任何包装：

```python
rows = executor.collective_rpc('demoncat_dsv4_probe', kwargs={
    'tool':'identity','action':'inspect_binding','options':{}}, timeout=120)
```

同时返回root_runner/root_model及显式选择的selected_runner/selected_model的完整类名、MRO、对象身份，以及runnable/model/_inner/__wrapped__的直接内层引用类型。存在model引用不等于它是代理；不能仅凭字段名判定包装来源。每个callable记录包名、模块、源码文件/SHA/行号。

每rank返回execute_model、sample_tokens、compute_logits的实例属性状态、type/module/qualname、类函数、__func__/__self__关系、最长16层__wrapped__链、源码文件/SHA/行号，以及两种Worker状态残留标志。能从闭包或对象字典读到MethodPin类包装固定的原函数时，列入pinned_original_candidates；这只是来源线索，不会执行或拆除包装，也不自动证明谁安装了它。

**准入仅接受普通bound method，__self__是当前对象，__func__与类上同名函数是同一函数。** 原来没有实例属性的类方法符合这一条件；实例上保存同一个类bound method也可接受。闭包、partial、不同函数、绑定其他对象、未知包装一律拒绝。approved_baselines/approved_baselines_by_rank已撤销，RPC携带它们返回PROBE_BASELINE_OVERRIDE_FORBIDDEN。指纹仅作诊断，不是放行凭证。

- OBSERVE_WORKER_STATE_PRESENT / NOISE_CAPTURE_WORKER_STATE_PRESENT：本探针Worker状态已存在，禁止重复安装。
- PROBE_INSTANCE_NOT_CLASS_BOUND_METHOD：实例覆盖不是该对象的同名类函数。
- PROBE_CLASS_METHOD_BASELINE_REQUIRED：类方法不满足受支持的普通bound method契约。
- PROBE_BOUNDARY_ALREADY_OWNED / PROBE_OWNER_CHANGED：边界已被占用或卸载前被替换。

RPC拒绝返回binding全详情、code/reason/error_type/traceback；不能只回传顶层HTTP状态。确认所有rank检查正常后：

```python
rows = executor.collective_rpc('demoncat_dsv4_probe', kwargs={
    'tool':'observe','action':'start','options':{}}, timeout=120)
# 宿主发送真实推理请求，等待真实完成。
rows = executor.collective_rpc('demoncat_dsv4_probe', kwargs={
    'tool':'observe','action':'stop','options':{}}, timeout=120)
```

任何start的双/多rank结果中只要有一项失败，宿主保留所有回包、停止新请求并等待在途完成，再向所有rank发该工具stop/abort；未安装rank的结构化NOT_INSTALLED可接受，已安装rank必须证明恢复。不要把部分成功记成全组安装失败后直接重装。

安装保存目标、原callable、是否实例属性和来源；调用始终走原callable。卸载仅在当前属性仍属于本工具时恢复；原实例值恢复原对象，原类方法删掉本工具覆盖。重复start拒绝；部分安装失败只回滚本工具写入。未安装status返回结构化NOT_INSTALLED。外来替换不强拆，保留清理责任，按专用实例恢复流程处理。

## 功能表与不能宣称的结论

|能力|支持范围|仍需证据/限制|
|---|---|---|
|D0身份|配置原值/来源、组成员、对象/类/源码链|不存在接口记null；来源指纹不能放行包装|
|inspect_binding|只读绑定/包装链/源码位置/可读的MethodPin原函数；拒绝时附完整诊断|启动路径未被证明前，不将八rank触发归因为框架或黄区组件|
|D1逐调用|序号、forward步、rank/组、阶段、实际调用栈中的模块层、具名输入/输出张量身份与形状|torch dispatch未暴露的融合内部调用不可观测；layer可能null|
|D1行证据|实际InputBatch顺序+query_start_loc CPU偏移+computed位置形成row→request/token；保存路由/索引整数张量；dispatch输出与combine输入的同指针同SHA候选关系|无真实offset不推断；padding/有效输出行仍null；跨EP与算子语义未证明时始终COMBINE_OUTPUT_ROW_JOIN_UNPROVEN；不授公共8资格|
|预算|默认4096条匹配调用、262144个MoE路由整数元素；普通量化权重只记元数据；可信options可设max_calls≤16384、integer_budget≤4194304|超限标truncated/缺证，不声称完整；不自动增加采样或制造调用|
|capture|四步每步完整hidden/logits、shape、top5；sample_tokens调用序号、请求ID、可见token与logits关联|异步/multi-logits关联不唯一记unproven；非driver无完整logits则不合格，不借其它rank补齐|
|返回/异常|原callable参数、返回和原异常不改；观察失败记错误|capture_valid=false或observations有错误时不能进入审查PASS|
|恢复|逐rank原方法精确身份恢复、重复owner可重新安装|不证明第三方状态修复；不等于39op归属异常已修复|

observe stop原结果兼容旧observe.v1摘要，新增calls/errors/truncated及identity。完整私有calls/数组落盘；文本回传使用工具，不手工拼N2或EP映射。

```python
rows = executor.collective_rpc('demoncat_dsv4_probe', kwargs={
    'tool':'capture','action':'start',
    'options':{'phase':'calibration'}}, timeout=120)
# 一条已批准长prompt请求，生成4token；真实完成后：
rows = executor.collective_rpc('demoncat_dsv4_probe', kwargs={
    'tool':'capture','action':'stop','options':{}}, timeout=120)
```

每rank确认exit0、capture_removed=true、capture_valid=true、4步完整数组与sample_join_proven=true。用result.steps进入原calibration结构；result.errors等旁证必须保留，不能只截steps绕过工具失败。capture中不通过抛异常改变模型推理；失败结论在status/stop中返回。abort只清理，不算采集通过。不要在同一轮混装外部性能/故障wrapper。

```bash
export PYTHONPATH="$PWD:$PWD/tests/worker_extension:$PYTHONPATH"
python -m unittest test_dsv4_probe_v2 test_dsv4_noise_capture test_dsv4_observer test_dsv4_noise test_dsv4_noise_v030 -v
python tools/extract_dsv4_v030_record.py identity /private/d0-rpc.json
python tools/extract_dsv4_v030_record.py binding /private/binding-rpc.json
python tools/extract_dsv4_v030_record.py observe /private/observe-rpc.json
python tools/review_dsv4_noise_v030.py /private/calibration.json --output /private/summary.json
python tools/extract_dsv4_v030_record.py noise /private/summary.json
```

提取每段≤4096字节，无法满足时拒绝，不截断。原数组/完整调用表只留黄区；noise输入结构、噪声上界、有限性和恢复标准保持r1不变。工具成功安装不等于D1行映射完成。

## 与Y1包装器失败的关系

黄区更正原文记载：完整日志425620字节，SHA256为7d1f9a48c8ae2fa967e0493f82bf03ef8bc7b405c9eda9f25395bc50d1d199e1，覆盖11:12:34–12:16:57及11:59/12:05守卫错误；119403字节属于旧快照。源码检索只见vLLM/vllm-ascend的类方法定义，未证明框架会给execute_model或compute_logits挂实例包装。旧的“框架基线包装可经指纹放行”方案撤销。

后续《组件加载核查》回传已说明旧部署组件未加载，MethodPin只保存并校验原方法、不设置实例属性；撤销把它当作已证实赋值方的说法。《插件入口与外部包核查》又声明LMCache/Arctic未启用。静态检索和加载声明仍不能替代实际绑定身份，不据此断言所有外部包装均不存在。没有成功安装日志，不能推断“上次成功安装遗留”。计划重启前后分别保留绑定检查和扩展组合；若干净组合不再有实例覆盖，只能先确认该启动组合下无覆盖，再结合被移除扩展的加载/初始化证据确认来源。未知包装触发守卫属于正确拒绝，不能绕过它安装。

本包bound method身份断言比较__func__与__self__，不以两次属性读取的is比较。v1-to-v2.part-00至03的内容尚未收到，本包按需求独立实现并测试；不声称已逐段审查或采用黄区diff。收到原文后再登记审查结论。

r7.4.1 AutomaticRequests在close报OWNER_CHANGED的条件是当前函数对象与其安装值不同。39op黄区scope1 enable成功、请求有覆盖，不能用启动时已有实例方法直接解释后续close变化。41已更正为harness漏枚举。因此尚未证明相同根因；本补件不改产品包装器，r8不凭推断移除其保护。

## 包装对象与真正计算层

默认只检查worker.model_runner及其model；不自动追随内层引用，也不把代理对象的转发方法当作其自身类函数。确认类源码的委托路径和真实调用后，可在可信options显式指定：

```python
# 仅为经检查确认为runnable代理时的例子；不得原样套用未知对象。
options = {'target_paths': {'runner': ['runnable'], 'model': []}}
```

每段只允许runnable/model/_inner/__wrapped__，最多4段，仅读对象字典或nn.Module的_modules；不存在、环路或非法路径拒绝。对选中的真实计算层仍执行相同bound-method严格守卫。先用identity/inspect_binding携带同一options核对selected对象，再给observe/capture start携带它。stop按保存的租约对象恢复，不重新解析路径。必须在实际请求证明该层被调用；零调用不是成功。CPU代理夹具已证明外层委托自然进入内层、仅在内层安装及恢复；不代表任意插件代理已适配。

## 插件回退（仅绑定来源实际指向profiler时）

核对的是vLLM v0.30.0官方源码：load_plugins_by_group按entry point的**name精确匹配**，不是包名；VLLM_PLUGINS按逗号拆分，不清理空格。所有经此函数加载的组（包括general和platform，以及调用它的IO/stat/endpoint等组）受影响；endpoint另要求显式启用。它不屏蔽普通Python直接import或其它插件框架。[插件加载器](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/plugins/__init__.py)、[环境解析](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/envs.py)。

vllm-ascend v0.30.0的entry points与本轮黄区metadata回传一致。平台ascend必须保留；保留connector/model_loader/model注册，但不启动KV connector，不启用profiling。确切值为：

```bash
export VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model_loader,ascend_model
```

这会排除msserviceprofiler及ascend_service_profiling，也排除其它未列出的entry point。依据：[Ascend注册表](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/setup.py)。黄区实际镜像为0.30.0+empty，启动前只读比对安装元数据与上述源码，保留实际差异；不能把tag源码身份当作已安装源码身份。仅当inspect_binding显示profiler来源时使用此回退，记录旧/新环境、完整命令、源码和绑定变化，再重启；不能仅因怀疑就更改插件。重启后再次逐rankinspect_binding，不放宽守卫。

## journal与公共8的证据边界

CallJournal在原算子执行一次前后，保存逐调用序号、forward步、rank/组、实际模块层、schema、张量身份/shape/dtype/stride、真实InputBatch行→请求→token位置，以及MoE整数路由张量和dispatch/combine候选关联。默认max_calls=4096、integer_budget=262144；硬上限分别16384和4194304。用完预算标truncated/缺失原因，不截掉失败结论、不变更推理结果。普通int8量化权重不占路由整数预算。

可信options还可给target_layers（最多64个真实named_modules名称），例如先用1token的实际call.layer结果选择一个层，再在同进程用该层重采1token/2token；名称不存在在安装前拒绝。过滤数filtered_calls与target_layers随原件保留，过滤范围外的调用不能当作已覆盖。先在当前进程检查预算/截断，必要时调预算或限定已观察层并重采，不需为此重启。

```python
options = {'max_calls': 16384, 'integer_budget': 4194304,
           'target_layers': [actual_observed_module_name]}
# observe start(options) →真实1token请求→stop，另一次start→真实2token请求→stop。
# 不混用两个请求的调用序号。原始calls/整数数组留黄区。
```

提取命令为`extract_dsv4_v030_record.py observe`：输出摘要及MoE逐调用/索引SHA；完整索引值仍需在黄区保存供本地语义核对。包内工具不解除COMBINE_OUTPUT_ROW_JOIN_UNPROVEN。

公共8的“combine之后同一请求token行成组错位”需要的是**该combine输出处**的行→请求归属、有效行/padding边界、group/rank语义及后继消费者，而不是只有runner输入处的映射。若已经由真实算子语义和逐调用证据证明这条输出映射，可以不复原历史恢复索引故障；但当前仍为UNPROVEN，不能据此出可执行D包。需结合dispatch路由整数、combine输入索引、同step/layer关联、TP/EP运行时组及0.30源码审查输出是否恢复请求顺序。跨rank不能用指针地址相等作关联；若缺映射继续关闭公共8，第13行也不借此获得资格。数据回来后先审查四类文本，再决定D包或最小补采。

## G0、G1连续流程与噪声输入

1. G0启动后保存每个rank的PID、进程启动时间、组成员、模型/软件身份。D0与绑定检查全rank通过后，先做D1的1token、2token两轮。
2. 固定两份输入的原件：`formal-prompt.txt`（正式验收）与`canary-prompt.txt`（产品金丝雀），由黄区从其实际任务选定并保存在本次私有采集目录。**本轮未收到这两份原文或完整SHA，本文不杜撰其值，也不借旧prompt的标定。** 开始G0之前生成并保留下述prompt-manifest，把实际SHA填入；G1复核相同原件，不能换文案。
3. 记录文件原字节SHA、服务端实际token ID列表的规范JSON SHA及真实长度。两个SHA不是同一字段：噪声run.prompt_sha256使用实际token ID列表SHA。使用completions传固定原文，若走chat则还需冻结模板后的实际token序列。每条实际长度2500–4092；max_model_len=4096，max_num_batched_tokens至少实际长度；生成4token，temperature=0，seed=42，prefix cache/MTP/KV connector关闭，实际调度证明单chunk。
4. 顺序固定为formal-0、canary-0、formal-1、canary-1……formal-4、canary-4。每条前start capture、真实完成后stop；每rank必须capture_valid、sample_join_proven、四步完整且有限、方法身份恢复和状态清零。第一条合格样本可计入，不合格不能靠复制或改sample ID补足。
5. G0每份prompt各≥5样本后，立即关闭该Worker组并重启G1。保存独立的进程组身份（每rank PID加启动时间；不能只用可能复用的PID），再次identity/inspect_binding，然后严格相同顺序采formal/canary各≥5样本。G1不必重复D1，除非配置或来源改变。
6. 每份prompt分别将两组各5个run组成calibration；不得把formal和canary混成一份输入。额外保留全局交错顺序与group启动记录。审查器固定两组两两距离最大值×1.5、token一致、全部有限；不改变规则。输出identity/binding/observe/noise四类提取文本，完整数组留黄区。

prompt-manifest必须来自原件和实际tokenizer/调度结果，不能手填默认值：

```json
{"formal":{"file":"formal-prompt.txt","file_sha256":"<从原件计算>","token_ids_sha256":"<实际token列表SHA>","prompt_tokens":"<实际整数>"},
 "canary":{"file":"canary-prompt.txt","file_sha256":"<从原件计算>","token_ids_sha256":"<实际token列表SHA>","prompt_tokens":"<实际整数>"},
 "order":"formal-0,canary-0,...,formal-4,canary-4","seed":42,"temperature":0,"max_tokens":4}
```

可用`sha256sum formal-prompt.txt canary-prompt.txt`核对文件；token列表用UTF-8 `json.dumps(ids,sort_keys=True,separators=(',',':'),allow_nan=False)`后计算SHA256。manifest中的占位符不是可运行输入，未绑定真实值不得开始G0。

## 最终门禁与自包含安装

冻结后的工具源码以source_manifest.json为准，CPU结果与A2回执都绑定同一SHA；旧21:09烟测不作本包门禁。最终CPU45项通过；A2/0.26/CANN9.1/Qwen3.8社区W8A8/TP2烟测还覆盖rank1拒绝、rank0成功后仅回滚本工具，外来包装保留；不证明V4/EP机制资格。退出时resource_tracker有IPC清理警告，原件保留，不能宣称宿主IPC全量认证。

ZIP自带Python项目依赖（包括fault_injection.sites.dsv4_observer）；Python、torch/torch_npu、vLLM由目标环境提供，不复制第三方运行栈。从独立空目录解压，只设`PYTHONPATH=<解压目录>`即可导入扩展，不需要tools-r1根目录。包内manifest列每文件SHA，source_manifest列与A2实际运行一致的186个源码文件；CPU测试目录随包，用上面的unittest命令运行。不要把旧r1目录加入PYTHONPATH。
