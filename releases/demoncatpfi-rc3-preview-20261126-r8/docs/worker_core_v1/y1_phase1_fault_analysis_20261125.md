# 阶段1故障结果：证据与推断分开

截至2026-11-25的后到现场文本。以下是对实际r7.4.1源码的判断，不把黄区初步解释直接当作根因。

|项|已经证明的事实|本轮定性|
|---|---|---|
|41|harness参数枚举不含41，未进入configure或注入|黄区接线漏项，未验收；不是公共缺陷|
|26fia|出现Native库准入拒绝；补配置后黄区声明两owner、两rank通过|部署/接线问题有支持；登记仍待效果与身份页|
|39op|第二owner的推理和status完成，两个rank在disable的包装器对象检查失败|包装器归属异常已确认；修改者及根因未证明|

## 为什么39op不能按“owner字符串不同”结案

`product_observer.py` 的条件是：

```python
all(vars(self.runner).get(name) is wrapper for name, wrapper in self.patches)
```

`is`比较函数对象本身，不比较owner文本，也不比较许可。报错意味着 `execute_model` 或 `sample_tokens` 当前对象不再是该观察器记录的对象（包括被删除）；尚不知道是哪一个、何时变更、由谁变更。

公共框架存在三种不同身份，不能把整个JSON中所有名为owner的字符串混成集合比较：

|身份|生成与含义|应该如何核对|
|group/ticket owner|`ProductClient`新建 `worker-group-…`，同一scope跨rank相同|本scope宿主client、binding.group、disable ticket之间一致|
|provider内部owner|`WorkerFaultRuntime.prepare` 为每rank lease生成独立 `secrets.token_hex(16)`，以 `lease.native_owner` 调用provider|本rank provider_state与该rank lease对应；不要求等于group owner或其他rank|
|permit_id|许可自己的随机标识，附机制、attempt、源码清单等|permit使用期间身份不变；许可结构根本没有owner字段|

因此补充里两个32位十六进制值可能是provider内部owner，另一个 `worker-group-…` 可能来自scope0；只有**明确字段路径＋scope＋rank**才能判断。即使全回执集合未出现失败ticket owner，也只说明回执缺失、错关联或某次调用链不一致的可能，不能证明这个函数对象拒绝码的原因。

真正的外部ticket owner或generation不匹配，`WorkerFaultRuntime._lease()`会在调用provider前返回 `OWNER_MISMATCH` 或 `STALE_GENERATION`；provider自身ticket不匹配则是 `PROVIDER_OWNER_MISMATCH`。此次已经进入provider并返回 `PRODUCT_AUTOMATIC_WRAPPER_OWNER_CHANGED`，是更后面的检查。generation=2是第二轮正常递增，不是故障证据。许可安装返回的 `acceptance_granted=false`是“不授予正式资格”，不是安装失败。

**不要让permit复用group owner，也不要为了通过检查复用旧ticket、忽略函数对象归属或强行删除wrapper。** 本轮CPU覆盖正常两owner、迟到的旧close、未清理旧包装器、新包装器被替换四种边界；原版A2实际流程的结果另在维护记录中列出。

本轮A2 TP2原版39op两个owner已经通过独立审查，83次控制调用中112次wrapper身份核对全部一致；41和26fia也各完成相同两个owner检查。因此尚不能提交一个有证据支撑的公共观察器修复。蓝区不会把黄区的暂定归因改写成已复现根因。

下一次只需聚焦：同一Worker的scope1 enable结束、最后一个真实request结束、status结束、disable入口这四个边界，记录两个runner函数的当前对象与观察器期望对象、闭包所指observer/owner，以及本scope的group owner、generation。结合实际黄区宿主/RPC片段，才能区分诊断/宿主替换、公共重复清理、代码加载漂移或其他路径。失败实例没有这些历史数据时，普通status不能倒推出已经消失的历史对象。

## 26fia的两个解释不能拼接

早期文本声称forkserver信号处理异常，后到文本却说同时间RPC全为200，Worker只有GDN warning，并给出Native库拒绝和补配置后的通过。未给共同run ID、进程身份与完整异常链，故不采用“FIA路径和预加载库信号注册组合错误”的解释。

configure尚未安装FIA site，也没有执行Native区域改写； `PRODUCT_NATIVE_REVIEWED_LIBRARY_REQUIRED`是在检查绝对库路径和审定SHA。20使用同一库通过，证明库可以用于那个已测组合，不证明26fia此前的部署JSON/挂载/实际RPC参数一致。此次最有力的证据是补配置后通过的对照，但原始失败字段和对应运行身份仍决定能否细分为“缺路径”“错误SHA”或接线取错配置。

## 处置和交付

41和26fia按最新事实撤销“已确认公共缺陷”的标签。39op仍需定位；没有证据就不改共享观察器、不宣称修复。recovery_required的旧实例按[安全恢复步骤](r8_y1_defect_review_20261125.md)处理；黄区其它任务继续。

r8交付仍优先于33/D增量，但不能为赶交付把错误根因写进代码。六项原PASS与26fia补测PASS均只是黄区文本声明，正式登记等逐组合提取页；不会因这份分析自动放行。
