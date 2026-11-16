# 其他故障的统一接口、自动定位与验收范围

本文保留先前本地整理及198项CPU验证范围。后续统一配置、生命周期、清理失败保留与
内源宿主契约见[其余15个机制合入说明](remaining_faults_integration_20261116/README.md)；
其中的新回执独立保存，不回填本页旧验收。

本轮只完成本地实现和CPU验证，未连接124、未新增NPU调用或msprobe采集。旧机制源码、统计原件、构建清单和限定验收保留；统一入口不会把旧Qwen验收推广到其他模型。

## 模型用户入口

```python
from fault_injection import attach, inspect_model

print(inspect_model(model))
with attach(model, "18.api_half", arm="fault", evidence_dir="run/fault") as session:
    output = model(**inputs)
    session.assert_coverage()
report = session.report()
```

使用者只指定模型对象、机制和B/F，不填写层名、层号、头数、base对象或地址。示例18要求模型实际包含标准`torch.nn.RMSNorm`；自定义RMSNorm即使名称相同也不会自动套用。所有机制都保留`baseline/fault`、严格拒绝/显式透传、`report()`、`assert_coverage()`和退出恢复接口。不同机制不能在同一进程同时进入统一session。

`inspect_model()`只读返回Attention、标准Conv3d/RMSNorm、动态量化和原生候选；候选不代表已执行。动态地址、视图写入、调度映射、分页cache和消费者必须在实际调用中证明。`assert_coverage()`验证已声明范围的路由覆盖，不是数值故障、完整模型质量或NPU验收。

统一入口在已进入的session关闭时保存唯一命名的`session-<id>.json`，包括显式`close()`；安装失败以异常返回，不伪造运行回执。报告先序列化再落盘，不覆盖旧文件。报告保存失败保留模型/清理错误。它不是msprobe的dump/stack/construct，也不参与独立统计首差定位。

## 逐项接入情况

| 机制 | 本轮接口与自动选择 | 仍待适配 | 已验收范围与本轮验证 |
| --- | --- | --- | --- |
| `8.restore_offset` | 已登记，执行明确拒绝 | 缺完整旧worker；需真实MoE routing、成对恢复indices/top-k和消费者 | 旧固定Qwen3-VL MoE证据保留；本轮未重建成任意索引扰动 |
| `15.alias_detach` | 已接入；在当前模型forward内识别公开`torch.as_strided`，由实际base/几何选择 | Tensor方法、ATen/native内部调用；自然模型的完整写入链需实测 | 旧NZ29/BF16机制级；新自动路由仅CPU显式模拟provider，不是NPU通过 |
| `17.spatial_padding` | 已接入；识别实际标准Conv3d与原生参数，全部匹配调用修改padding | 自定义Conv3d实现、其他patch几何、真实视觉分支验证 | 新重建实现的真实CPU Conv3d测试；旧视觉NPU证据单独保留 |
| `18.api_half` | 已接入；实际标准`nn.RMSNorm`的模块返回增益0.5 | HF/vLLM自定义RMSNorm及旧native wrapper采集边界 | 新模块返回代理CPU测试；不继承旧native API首差位置 |
| `20.buffer_substitute` | 统一桥接已接入；Linear/量化method实际实例自动发现 | 版本专用provider的ABI、semantic slot、live storage与变换 | 旧固定CANN/模型NPU证据；新bridge仅CPU模拟 |
| `22.wrong_row` | 已登记，执行明确拒绝 | 缺完整旧worker；需真实RoPE cache/index/consumer关系 | 旧合法行替换输出代理，不是uint32地址溢出 |
| `25.reduce_scatter` | 已登记为未实现 | 通信组/rank/真实collective调用及专属实现 | 没有可迁移的实机通过结论 |
| `26.residual_stomp` | 统一桥接已接入；量化模块对象自动选择 | allocator邻接、live residual、真实writer和consumer证明 | 旧限定allocation proxy；新bridge仅CPU模拟 |
| `26.fia_destination` | 统一桥接已接入；实际Attention对象自动选择 | writer destination及residual生命周期provider | 旧441/447完整consumer证明缺口保留；新bridge仅CPU模拟 |
| `quant.batch_shared_scale` | 已接入自动method链选择，并有校验旧BackendControl的部署桥 | 实际runner的调度行映射、健康输入身份及匹配构建部署 | 旧TP2动态W8A8；新自动选择仅CPU模拟，不声称全模型NPU通过 |
| `28.fa.legal_tiling` | 保留现有`StrategySession`桥接、证书和实际调用上下文 | 模型到真实FA入口的部署provider；`attach`仍明确拒绝 | 旧56次调用+4次validated及限定模型实验；原框架69项CPU回归 |
| `28.fia.legal_tiling` | 已接入共享Attention自动路由；健康coupled M/S2模板512→128 | 新路由NPU验收及更多几何；当前vendor固定 | 本轮CPU合同检查；旧合法切分实验未达0.1%统计目标，不能借用replay通过结论 |
| `28.fia.denominator_replay` | 保留共享Attention自动路由；B/F同M128/S2=128，F使用replay key | 更广kernel几何、其他模型NPU验收 | 旧Qwen3完整模型统计通过；新注册路由仅CPU，不等于历史28根因 |
| `33.paged_decode` | 统一桥接已接入；共享Attention或具有真实impl的分页模块候选 | page table、KV cache、请求映射及tiling版本provider | 旧Qwen3固定分页增量路径；不能用28 dense prefill合同替代 |
| `39.matmul_range` | 统一桥接已接入；真实Linear对象候选 | ABI/slot/live input范围provider | 旧Matmul有限输入数值实验；未定位累加器具体类型 |
| `39.output_proxy` | 与range分开接入桥接，选择真实Linear对象 | native返回写入边界、bounds与恢复provider | 旧非有限输出代理；不能归为真实计算根因 |
| `41.tail_column` | 保留未实现/阻塞状态 | 真正dispatcher/cache writer/storage/consumer链 | 旧设计不能变成已验收能力 |

历史源码、证据位置及版本见[完整清单](unified_fault_inventory.md)。桥接协议及六种native机制所需证明见[原生适配要求](native_adapter_requirements.md)。

## 新实现的机制边界

**15：**沿用健康NZ29/BF16的`base[16,16]`、`view[8,8]`、stride`[16,1]`、offset17合同。生产provider只调用真实NPU格式转换到format2；确认逻辑初值一致且存储独立，然后调用原`as_strided`。CPU测试必须显式使用测试provider，公共`attach`不能以CPU clone冒充私有格式。运行时分别观察真实`copy_`、原base写后值和明确的数据读取消费者；`empty_like`等元数据/覆盖操作不算consumer。缺写入/消费、未知读取和任意消费者结果参考保持未证明。`assert_coverage`仅指公开view构造覆盖。

**17：**历史完整worker未找到，因此这是依据已保存机制说明重建的本地实现，未覆盖原件。实际标准Conv3d需要kernel/stride一致、零padding、dilation1、groups1、输入空间16×16且输入extent等于kernel，F仅将padding改为`(0,7,7)`。由原`Functional.conv3d`计算，输出仍为`[B,Cout,1,1,1]`，不裁剪、不改权重和返回值。Batch、通道数和满足上述公式的时间kernel从实际模块读取；这个几何扩展只有CPU验证。

**18：**本轮为明确标注的`module_return_gain_proxy`，标准RMSNorm返回乘0.5。旧实现位于native API返回与消费之间，两者的hook顺序和msprobe可见位置可能不同，不能继承旧首差报告。它一直是数值代理，不宣称复现Reduce内部错误。

**量化：**识别实际动态W8A8 method包装链、INT8权重和TP2元数据，处理多个模块共享同一method的情况。部署桥核验旧控制源码、native源码、批准的构建清单、库字节及实际resolver，复用旧原生描述符归约，不修改Q/scale返回值。每次调用需要真实pure-decode一请求一行映射和独立健康输入SHA；不能把tensor首维自行解释为request。前层故障传播导致后层输入与健康记录不同会明确拒绝，不能擅自放宽为“全层均通过”。

**FIA合法切分：**公共入口增加`28.fia.legal_tiling`。同一已披露vendor中的B=`coupled_base512`，F=`coupled_m128`，两者都使用健康模板，无replay key。该受控M/S2联合切分不同于原生默认策略，也不是新的NPU验收。replay变体继续单独注册，B/F均M128，F才重复分母贡献。头数、维度、dtype、mask、版本合同没有因为接口通用而放宽。

## 部署适配不交给模型用户填写层名

`register_runtime(mechanism, factory)`供平台维护者注册已经完成版本适配的运行时；factory收到实际model，注入时再收到自动识别的module/call。`attach(model, mechanism)`自动取用它。生产native provider当前没有通用实现，缺失时在安装hook前明确拒绝；这是表中的待适配项，不要求模型用户手写内部层名或填地址来绕过。

provider只能给quant/native桥接注册，其他机制拒绝无效注册。注册采用所有权检查，不能静默覆盖他人的factory。哈希和调用绑定证明保证记录对应关系，不保证provider内容真实；需要独立NPU执行证据。原生桥接不会把provider自报命中升级为NPU验收。

## 本地验证与证据保留

新测试覆盖真实CPU Conv3d/RMSNorm数值与OFF恢复、跨模型共享子模块隔离、逐forward漏层、递归/并发拒绝、别名独立存储及写入观察、量化共享method和调度合同、native证明绑定及失败清理、无回调不通过、公开接口报告。Alias/quant/native测试provider均明确为CPU fixture，不算真实NPU故障。

共享Attention继续使用实际Qwen3、GLM、DeepSeek-V3、Llama随机小模型代码验证prefill、padding、cache decode和恢复。Q/K维度与V维度分别读取；不按品牌或28层模板选择，不用padding/crop伪装kernel兼容。标准模块测试覆盖任意属性路径和不依赖Transformer的组合。

本轮**129项接口/机制测试＋69项原策略回归，共198项通过，无跳过**。[CPU验证回执](unified_mechanisms_validation_20261116/receipt.json)记录版本、所有接口/测试源码SHA及`actual_npu_calls=0`；同目录保存完整日志。[交付校验](unified_mechanisms_validation_20261116/delivery.json)核验24份测试相关源码、31份冻结策略文件，以及离线wheel中的41份源码/资源。早前35+69项回执保留在`unified_validation_20261116`，不改写成新结果。

本轮不改变B/F原始`statistics/mix`字节、固定容差比较器或独立首差报告。任何新适配器的msprobe验收都须在对应模型/版本重新实测；路由报告、provider命中和完整私有张量均不能替代独立统计定位。
