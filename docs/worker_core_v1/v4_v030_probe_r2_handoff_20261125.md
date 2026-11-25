# V4 0.30 探针 r2：一次重启的独立工具补件

本补件修复Codex交付的只读工具，不修改r7.4.1/r8运行时、噪声判据或任何故障机制。A3目标为实际vLLM/vllm-ascend 0.30、CANN9.2.1、V4 W8A8、TP8；A2 0.26烟测只验证框架对象包装/恢复可移植性，不能代替A3或公共8资格。

## 重启前现在可做的事

保留当前健康实例，不重复start/capture，不自行删除守卫或替换方法。保存已有成功/失败请求体、完整stdout/stderr、探针源码SHA，以及八个Worker分别的maps与日志。API进程89个库的记录不能代替Worker映射。按rank核对installed/清理状态，顶层500不说明八rank全未安装。

只计划一次重启：先检查本包manifest、diff和离线测试；确认可信扩展组合、实际基线方法来源审查，再由黄区按服务管理流程重启。重启后的第一组有效标定为G0，G1仍必须是另一独立Worker组；一次重启并不免除噪声标定的两个进程组要求。

## 安装变化及安全边界

Python3.12，采集需要部署自带torch/torch_npu；审查/文本提取只需标准库。独立解压到新工具目录，把此目录放在Worker启动PYTHONPATH最前，保留黄区原宿主/鉴权路径，不复制覆盖公共wheel或旧工具目录。部署的可信扩展类组合新增：

```python
from experiments.worker_core_v11_20261118.dsv4_probe_extension import Dsv4ProbeExtension

class TrustedWorkerExtension(ExistingAuthenticatedExtension, Dsv4ProbeExtension):
    pass
```

上面ExistingAuthenticatedExtension由黄区实际类替换，必须继续对demoncat_dsv4_probe实施既有鉴权；mixin本身不鉴权、不创建服务，也不授权任意RPC。部署worker-extension-cls指向这一个组合类，不可丢弃已有扩展。不要部署仅用于隔离A2验收的dsv4_probe_smoke.SmokeExtension。

冻结RPC只用kwargs，八rank逐一检查：

```python
rows = executor.collective_rpc(
    'demoncat_dsv4_probe',
    kwargs={'tool': 'identity', 'action': 'read', 'options': {}}, timeout=120)
# 每rank结果：{schema:'dcat.probe.rpc.v2', rank, status, exit_code, result}
# 失败保留code/reason/error_type/traceback；不以HTTP200替代逐rank检查。
```

D0列出parallel_config候选来源及真实字段，TP/DP/EP组大小、成员与rank；runner/model/sampler类、对象和方法源码身份。不存在的字段为null并给原因，不能用默认EP。多个配置来源冲突需要人工审查。

实例上的方法不再一律拒绝，但也不会自动批准D0的输出。可信部署方审查callable_chain中每个__func__/module/qualname/文件SHA与__wrapped__来源，确认它是核验过的框架基线，再构造approved_baselines_by_rank。指纹包含进程内真实函数对象身份，重启后须重新审查，不能复用旧rank指纹。不能让工具把刚读到的任意指纹自动加入白名单。未知wrapper保留拒绝。

```python
# reviewed 为负责人/可信部署审查后的值，不是无条件复制D0的输出。
reviewed = {'0': {'execute_model': '<核验指纹>',
                  'compute_logits': '<核验指纹>', 'sample_tokens': '<核验指纹>'}}
# 必须填写所有实际rank；缺失rank上的实例覆盖将被拒绝。
rows = executor.collective_rpc('demoncat_dsv4_probe', kwargs={
    'tool':'observe', 'action':'start',
    'options':{'approved_baselines_by_rank':reviewed}}, timeout=120)
# 宿主按现有推理入口发送真实请求，不在观察RPC里启动推理。
rows = executor.collective_rpc('demoncat_dsv4_probe', kwargs={
    'tool':'observe','action':'stop','options':{}}, timeout=120)
```

任何start的双/多rank结果中只要有一项失败，宿主保留所有回包、停止新请求并等待在途完成，再向所有rank发该工具stop/abort；未安装rank的结构化NOT_INSTALLED可接受，已安装rank必须证明恢复。不要把部分成功记成全组安装失败后直接重装。

安装保存目标、原callable、是否实例属性和来源；调用始终走原callable。卸载仅在当前属性仍属于本工具时恢复；原实例值恢复原对象，原类方法删掉本工具覆盖。重复start拒绝；部分安装失败只回滚本工具写入。未安装status返回结构化NOT_INSTALLED。外来替换不强拆，保留清理责任，按专用实例恢复流程处理。

## 功能表与不能宣称的结论

|能力|支持范围|仍需证据/限制|
|---|---|---|
|D0身份|配置原值/来源、组成员、对象/类/源码链|不存在接口记null；来源指纹不是审批本身|
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
    'options':{'phase':'calibration','approved_baselines_by_rank':reviewed}}, timeout=120)
# 一条已批准长prompt请求，生成4token；真实完成后：
rows = executor.collective_rpc('demoncat_dsv4_probe', kwargs={
    'tool':'capture','action':'stop','options':{}}, timeout=120)
```

每rank确认exit0、capture_removed=true、capture_valid=true、4步完整数组与sample_join_proven=true。用result.steps进入原calibration结构；result.errors等旁证必须保留，不能只截steps绕过工具失败。capture中不通过抛异常改变模型推理；失败结论在status/stop中返回。abort只清理，不算采集通过。不要在同一轮混装外部性能/故障wrapper。

```bash
export PYTHONPATH="$PWD:$PWD/tests/worker_extension:$PYTHONPATH"
python -m unittest test_dsv4_probe_v2 test_dsv4_noise_capture test_dsv4_observer test_dsv4_noise test_dsv4_noise_v030 -v
python tools/extract_dsv4_v030_record.py identity /private/d0-rpc.json
python tools/extract_dsv4_v030_record.py observe /private/observe-rpc.json
python tools/review_dsv4_noise_v030.py /private/calibration.json --output /private/summary.json
python tools/extract_dsv4_v030_record.py noise /private/summary.json
```

提取每段≤4096字节，无法满足时拒绝，不截断。原数组/完整调用表只留黄区；noise输入结构、噪声上界、有限性和恢复标准保持r1不变。工具成功安装不等于D1行映射完成。

## 与Y1包装器失败的关系

原探针的误拒是安装前`name in vars(object)`判断；删除框架实例方法是绕过该守卫后可能出现的卸载缺陷。本次修复两者。r7.4.1 AutomaticRequests也有实例覆盖安装守卫，但它在close报OWNER_CHANGED的必要条件是当前函数对象与本工具安装值不同。39op黄区scope1 enable成功、请求有覆盖，不能用“启动时已有实例方法”直接解释之后close的identity变化。41已更正为harness漏枚举，没有第二owner失败证据。因此属于相似风险领域，尚未证明同一根因；本补件不改产品包装器，r8不凭推断移除其保护。具体现场仍需要失败stdout及边界身份。
