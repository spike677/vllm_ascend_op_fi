# 其余15个机制：统一接入与验收边界

后续更新：5条部署 provider 已新增实际实现及124推理测试，最新范围与限制见
[provider验收总结](../../experiments/provider_adaptation_20261116/README.md)和
[`fault_injection/catalog.py`](../../fault_injection/catalog.py)。下文的CPU回执与旧状态表
保留首次整合时的范围，不作为新版provider的当前验收结论。

本轮以已发布 Fault39 提交 `c1f5726a` 为基线，只整理其余机制。
原工作区、Fault39实现及统计、历史实验与原生源码均不覆盖。
旧统一实现以 [导入清单](import_manifest.json) 保存文件身份；FA策略框架的31份源码/资源
继续按原 `IMPORT_MANIFEST.json` 逐字节保留。

**本轮完成本地接口、生命周期、宿主接线契约与CPU回归；没有新增NPU验收或msprobe采集。**
蓝区仍无法取得当前内源 `src/demoncat`，状态明确为
`INNER_FRAMEWORK_INTEGRATION_PENDING`。目录登记、结构候选、真实运行、NPU验收分别报告。

## 使用入口

模型用户只给模型对象和机制，不填层名、层号、head数、地址或base对象：

```python
from fault_injection import FaultConfig, ModelFaultBackend, inspect_support

print(inspect_support(model, "18.api_half"))  # 只读，不运行模型或provider
backend = ModelFaultBackend(model)
backend.prepare()
backend.register(FaultConfig("18.api_half"))  # 注册保持关闭
try:
    backend.enable()
    output = model(**inputs)
finally:
    backend.disable()  # 真正恢复本会话的hook/配置；失败会抛出并保留恢复状态
print(backend.query())
backend.clean()
backend.unregister()
```

这个例子要求真实标准 `torch.nn.RMSNorm`。HF/vLLM自定义类即使同名也不能冒充受支持实例。
使用 `arm="baseline"` 运行相同选择路径；`disable()` 后的普通模型调用是OFF恢复对照。
原有 `with attach(...) as session`、`session.assert_coverage()` 接口保留。
覆盖断言检查实际路由分母，不验证数值故障、模型质量或所有请求的原生行映射。

只读导出不导入torch、不初始化设备，也不提供一个看似能用的内源CLI替身：

```bash
python -m fault_injection catalog
python -m fault_injection host-manifest --output remaining-host-contract.json
```

安装使用已经匹配设备的环境：`pip install --no-deps -e .`，不会替换torch/torch-npu。
版本为 `demoncatpfi 0.2.0`；Fault39继续使用其独立包和已验收原生控制面。

## 逐项状态

| 机制 | 当前接入 | 自动选择及待适配项 | 验收边界 |
|---|---|---|---|
| `8.restore_offset` | 目录/契约，执行拒绝 | 缺完整旧worker；需真实MoE恢复indices与top-k配对、消费者 | 旧固定模型证据，不是新通用实现 |
| `15.alias_detach` | 自动API适配 | 模型forward内公开as_strided，实际私有格式/base/几何；native内部调用未覆盖 | 旧NZ29机制级；新路由仅CPU模拟，不代表自然模型写入链成立 |
| `17.spatial_padding` | 自动模块适配 | 标准Conv3d及真实16×16 patch几何，修改padding并调用真实conv3d | 重建CPU实现；旧视觉NPU证据不转授新入口 |
| `18.api_half` | 自动模块适配 | 标准RMSNorm返回增益0.5，全部符合条件的实际调用 | CPU返回代理；不是Reduce内部错误或原native wrapper |
| `20.buffer_substitute` | 原生provider桥 | 实际Linear/量化对象；部署provider须证明ABI、slot、storage及原变换 | 新桥仅CPU模拟；旧固定版本NPU另列 |
| `22.wrong_row` | 目录/契约，执行拒绝 | 缺完整worker；实际RoPE cache/index/consumer关系待接入 | 保留旧合法行输出代理，不称地址溢出 |
| `25.reduce_scatter` | 未实现，执行拒绝 | 通信组、rank、collective及专属机制待实现 | 无可迁移的通过结论 |
| `26.residual_stomp` | 原生provider桥 | 实际量化模块、allocator邻接与live residual消费链 | 旧231次限定命中；新桥仅CPU模拟 |
| `26.fia_destination` | 原生provider桥 | 实际Attention对象、writer destination与residual生命周期 | 保留旧441/447完整消费者证明缺口 |
| `quant.batch_shared_scale` | 自动method选择＋部署桥 | 实际动态W8A8包装链；还需健康输入身份、TP2 pure-decode请求行映射 | 不把shape[0]当请求数；旧md5/张量采集不改称新statistics通过 |
| `28.fa.legal_tiling` | 原FA策略桥 | 复用证书及真实调用上下文；通用模型到native FA provider待接入 | 原56次调用＋4次validated及限定模型；attach明确拒绝 |
| `28.fia.legal_tiling` | 自动Attention适配 | 真实共享注册路由；固定vendor的健康M/S2 512→128 | 新路由CPU；旧合法切分未达0.1%统计目标 |
| `28.fia.denominator_replay` | 自动Attention适配 | 同一共享路由，真实kernel分母重复累计 | 与合法切分分开；旧Qwen0.7968%限于旧冻结实验 |
| `33.paged_decode` | 原生provider桥 | 实际分页Attention对象、page table、KV存活及请求映射待部署 | 旧限定decode；不能用28的dense prefill替代 |
| `41.tail_column` | 未实现/阻塞，执行拒绝 | 真实dispatcher、cache writer、storage和consumer待证明 | 设计可行不算运行通过 |

机器可读清单由 `capabilities.py` 生成，保留独立的 `dispatch_adapter_present`、
`runtime_factory_required`、`native_launch_verified=false` 和
`current_model_npu_accepted=false`。`inspect_support` 只返回结构候选和缺口；
存在provider factory不代表已经加载、调用或验证该provider。

15条中5条有直接自动适配器，5条需要部署provider（含自动量化选择），1条是策略桥，
4条明确不可执行。历史目录把15写成27的别名仍保留，未凭空新增“故障16”。

## 生命周期与并发边界

- `prepare/register` 不安装故障，register默认关闭；每次enable创建独立会话/generation。
- `disable/clean` 真正撤销当前owner的hook、包装器或运行适配，不修改权重，不调用全局reload。
- 清理失败保留所有者与会话隔离，不能释放锁后让下一条故障继续启用。
  可以证明重试安全的资源保留引用供原线程重试；无法证明原生恢复时保持失败隔离。
- 在forward/原生回调尚未结束时关闭会话必须拒绝，不能在执行中拆hook。
  此接口是同步、单owner线程接口；不是异步服务的drain实现。
- 同一统一接口内的多会话并发组合明确拒绝；独立Fault39或外部工具不共享这把Python锁，
  部署需隔离，不能声称已验收跨工具的多fault组合。公共rank/cards/request筛选未实现时拒绝未知配置，
  不把这些条件只写进query。native调度与实际rank作用域必须由匹配provider验证。
- `unregister_runtime` 仅撤下部署factory，不能停止已建立的会话；实际撤销必须调用
  `ModelFaultBackend.disable/clean/unregister`。两种注册关系分别管理。
- 清理完成与运行成功分开。发生过模型或注入错误，即使后续恢复成功，也不能将本轮覆盖改成通过。

原生量化provider的finish失败且无法确认disarm、alias观察的TorchDispatchMode退出失败，
以及native provider安装状态不明且没有拿到owned removal，
当前都需要停止使用该进程并重新启动；没有虚构“强制释放后成功”的恢复入口。
其他可重试的hook/provider移除失败，由原owner在线程上下文中修复原因后重试，保留首次错误与后续回执。

本轮修复的是可影响真实恢复的生命周期问题，保留各故障数值机制及拒绝门槛。
参数支持范围、FP16/BF16、布局、版本、私有格式和分页要求不会因统一接口而放宽。

## 内源项目接线契约

`demoncat.model_faults.host_contract.v1` 的方法名是版本化文档协议，
并非对不可访问内源API的猜测。宿主需要实现catalog/binding方法，保留owner校验，
在实际controller中使用 `ModelFaultBackend`。每个factory绑定指定机制，拒绝跨机制配置。

安装会记录每项注册意图，部分失败按owner回滚；回滚不完整时保留待恢复项目。
卸载遇到仍注册或清理未确认的backend时拒绝，要求controller先做scoped clean/unregister；
不会以删除目录条目代替撤销故障。旧factory或旧backend不能在宿主条目移除后再次启用。

这些是15个独立机制描述，不声称内源增加了15个primitive，也不把变体拆成虚构的故障编号。
宿主应按自己的primitive参数规范映射；现有Fault39契约不改动。

## 统一取证与独立首差

每条机制声明预期可观测边界，但不把它预设为比较器答案：

- 保存B/F原始 `statistics/mix` 的dump、stack、construct、配置与采集器身份，保留真实step数。
- 固定 `abs(F-B) > 1e-8 + 1e-5*abs(B)`；null保持不可观测，不补零，不用完整张量或命中日志选首差。
- 先从原始统计独立定位，再查原生hit/launch、存储、rank、请求和消费者关系。
  统计差异不能单独唯一证明物理根因。
- 15的view初值/copy_返回可以一致，原base写后读才不同；17的padding输入本来就改变；
  18是返回代理；26可能恢复原输出但污染residual。不能统一要求所有机制都“输入相同、算子输出首差”。
- 合法FA/FIA切分可能没有超阈值差异；归约错误、输出代理、地址代理分别报告，不合并机制标签。
- 现有Fault39只读比较器可处理符合相同原始结构的采集；旧不同schema/配置必须明确拒绝或单独适配。
  本轮没有把历史不同标准的采集转换成新的statistics验收。

会话JSON只用于生命周期/路由观察，绝不是msprobe三件套。CPU回执中的
`actual_npu_calls=0` 必须保留。新的模型/版本需重新完成B/F/OFF、覆盖和独立统计验收。

## 本地测试与交付

测试覆盖真实小型Qwen3、GLM、DeepSeek-V3、Llama实现的CPU结构路由，padding/cache语义，
真实CPU Conv3d/RMSNorm，显式alias/quant/native模拟provider，注册不启用、重复启停、
owner隔离、活动调用拒绝清理、恢复失败隔离及宿主部分注册回滚。
原FA框架69项回归独立运行，Fault39源码和已有统计不变。

执行 `python tests/run_unified_validation.py --output <新的目录>` 生成独立CPU回执，
不覆盖先前的198项通过记录。最终结果及源码SHA见本目录的新验证回执和交付清单。

本次最终 **174项接口/机制测试＋69项原FA回归，共243项通过，无跳过**。
[CPU回执](validation/receipt.json) 固定torch 2.10.0+cpu、Transformers 5.5.4及32份测试相关源码SHA；
[统一测试日志](validation/unified.log)、[FA回归日志](validation/native_strategy_regression.log) 保留原始结果。
[交付核验](delivery.json) 检查31份冻结FA文件、wheel内44份源码/资源，以及脱离源码目录后的
真实CPU B/F/OFF运行。wheel是本地验证产物，不包含vendor或模型权重；本分支交付可离线构建的源码。
