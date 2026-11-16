# 自动模型适配与统一故障入口

后续生命周期与宿主接线规范见[其余15个机制合入说明](remaining_faults_integration_20261116/README.md)。
清理失败会保留隔离与待恢复状态，不能把退出异常解释为已经恢复成功。

本轮将本地故障资产纳入一个能力目录，并增加按模型对象自动定位的接入层。无需用户填写 `model.layers.0.self_attn`、层号、头数或Qwen专用注册代码。接口统一不意味着全部旧故障已经迁移，具体状态可运行 `python -m fault_injection` 查看。

后续其他机制的实现、自动选择方式及逐项验收边界见[2026-11-16扩展说明](unified_mechanisms_20261116.md)。公开入口统一为`attach(model, fault, arm, unsupported, evidence_dir)`，部署侧另可提供`runtime`，不向模型用户索要层名。

在已经匹配 torch/torch-npu/Transformers 的推理环境中执行 `pip install --no-deps -e .` 安装本地接口；不会自行替换设备依赖。只读清单和策略框架本身不导入torch或初始化NPU。

```python
from fault_injection import inspect_model, attach

print(inspect_model(model))  # 只读；模型可为Qwen、GLM、DeepSeek或其他结构
with attach(model, "28.fia.denominator_replay", arm="fault") as session:
    result = model(**inputs)  # 使用原模型调用，不另造forward
    session.assert_coverage()
print(session.report())
```

`baseline` 与 `fault` 使用相同的模型接口。`assert_coverage()`确认已发现的候选模块实际经过路由；不证明kernel命中、历史根因或msprobe首差。默认遇到不支持的调用立即报错；显式 `unsupported="passthrough"` 仅允许能力预检拒绝的调用返回原后端，报告中会记录，不能算全覆盖。进入后端执行后发生的异常不会偷偷重跑原算子。

## 模型发现与掩码语义

1. 扫描 `named_modules()`，分析实际 `forward` 的共享注册表调用，核对真实注册表对象，而非模型名称白名单或固定属性路径。
2. 使用公开 `transformers.AttentionInterface.register`；同时为临时后端名称登记原后端的 `AttentionMaskInterface` mask factory。只注册Attention名称可能导致上层丢弃mask，不能接受。
3. 回调从实际Q/K/V读取请求数、Q头/KV头、序列长度、QK维度及V维度。DeepSeek的QK与V维度分开描述；不通过pad/crop强行适配。
4. 只控制发现的模块对象。共享config的未知模块、绕过注册表的remote-code实现、未知mask后端默认拒绝并给出原因。源码检查是候选发现，真实回调才是覆盖证据。
5. 退出、模型异常或安装失败均恢复原config与自身注册项。嵌套/并行session拒绝，避免进程共享注册表冲突。不要在同进程中同时运行共享这些config的其他模型。

目前路由支持从原 `eager`/`sdpa` 后端接入；其他后端需核验掩码语义。运行环境需要支持本文协议的 Transformers，测试版本固定为5.5.4。公共接口可能跨版本变化，不能从单版本测试推导所有版本兼容。

## 原生执行策略与FIA边界

`fault_injection/strategy.py`复用已有`native_execution_strategy.StrategySession`，原框架文件逐字节复制，来源与SHA在`native_execution_strategy/IMPORT_MANIFEST.json`。FA桥接读取真实调用参数和已验证请求映射，保留合法候选、证书、原生raw和launch门槛。没有另造一套可绕过旧检查的策略接口。它是平台提供者接口，尚未自动接入所有模型中的原生FA调用。

FIA分母重复累计由`fault_injection/fia.py`接入。Python层按真实模块位置自动路由，kernel仍只接受B1/B8、Q16/KV8、2048-token prefill、QK/V维度128、FP16、causal、固定scale及匹配版本/构建。故障位于真实NPU kernel，Python不修改返回数值。新模型即使能注册，几何不匹配也会明确拒绝。模型品牌不参与kernel选择。

需要通过既有不可变vendor启动器启动新进程（不能在已初始化进程换库）。部署配置由运行环境提供：

- `DEMONCAT_FIA_VENDOR_MANIFEST`：已经验收的分母重复累计vendor清单路径，固定SHA在代码中；依赖完整原vendor，不能只下载证据分支中的源码。
- `ASCEND_CUSTOM_OPP_PATH`和动态库路径：由已有验证启动器设置；`ACLNN_CACHE_LIMIT=0`。
- `DEMONCAT_EVIDENCE_DIR`：`/evidence`下专用目录，自动产生逐调用scope与host日志。

这些是构建与证据目录设置，不是让模型用户定位注入层。新会话仅报告Python路由及host eligibility，`native_launch_verified`保留false；实际launch和B/F/msprobe验收仍使用独立原生观察工具，不能拿host命中日志替代。磁盘vendor校验也不等于实际加载身份验证。

现有native策略是进程作用域，必须使用专用推理进程，禁止其他线程/服务在此进程并行调用FIA；Python门禁拒绝已知活动线程与嵌套会话，但不能替代C++调用方隔离。host回执严格核对唯一scope、owner、mode、M、key和serializer。部署到并发服务之前仍需独立的native调度作用域适配，不能直接套用。

旧Qwen3/124的NPU结论属于旧冻结入口。本轮新的通用路由没有获得GLM、DeepSeek、Llama的NPU验收。图模式、分页decode、滑窗、padding、训练dropout等不属于当前FIA故障kernel支持范围。固定策略下也不保证每份输入产生差异。

## 完整故障整理

统一能力目录将注册与执行分开：

| 状态 | 含义 |
| --- | --- |
| `automatic_attention` | 通用模型路由已实现；实际调用还要过kernel门禁 |
| `automatic_operator` | 在真实模型作用域观察API/存储事实；目前为15公开as_strided |
| `automatic_module` | 按标准模块及真实参数选择；目前为17/18新本地重建实现 |
| `automatic_quantization_requires_runtime` | 实际量化method自动选择；仍需部署后端及调度provider |
| `native_provider_bridge` | 自动候选与原生callback桥接完成；具体版本provider待适配 |
| `strategy_bridge` | 旧原生执行策略已桥接；需要平台实际调用/launch观察接入 |
| `legacy_source` | 有旧实现及限定验收；尚不能由新入口自动启动 |
| `evidence_only` | 当前仅有历史证据，不能虚构可执行插件 |
| `not_implemented` | 方案存在，但没有可运行且验收的适配器 |

15的旧27误称、28合法切分与新的分母重复累计、33分页decode分别登记。20/26/33/39已增加统一桥接，仍需实际ABI、slot、存储、分页与生命周期证明；发现一个Linear或Attention模块不能代替这些事实。完整条目和源位置见[清单](unified_fault_inventory.md)。

## 验证与统计

结构回归、mask保留、拒绝策略、恢复、覆盖计数及FA桥接测试：

```bash
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s native_execution_strategy/tests
```

真实模型测试使用本地随机初始化的小模型，只验证接口语义，不下载权重，不占NPU，不代表模型质量或原生故障泛化。首次整理结果：**35项新接口测试＋69项原策略回归通过，无跳过**。见[旧验证回执](unified_validation_20261116/receipt.json)。后续机制扩展结果单独保存在[新验证回执](unified_mechanisms_validation_20261116/receipt.json)，不覆盖旧日志。

| 真实CPU模型代码 | 测试层数 | 验证内容 |
| --- | ---: | --- |
| Qwen3 | 3 | eager与SDPA两条路径；GQA、padding、prefill、cache decode及恢复 |
| GLM | 2 | eager；不同head_dim的实际几何、padding/cache与恢复 |
| DeepSeek-V3 | 2 | eager；QK维12、V维8分别传递、cache与恢复 |
| Llama | 2 | eager；GQA、padding/cache与恢复 |

测试环境为torch2.10.0+cpu、Transformers5.5.4、NumPy2.2.6。Windows最初的NumPy2.5.5扩展无法加载，固定依赖后重新完整执行；没有改动系统安全策略。可用 `python -B tests/run_unified_validation.py --output NEW_DIRECTORY` 生成新的带源码SHA回执。

msprobe仍在外部按同配置采集B/F的原始 `statistics/mix`，保留dump/stack/construct、配置和collector身份。新接口动态解析公开NPU API，避免缓存msprobe安装前的函数。覆盖报告与注入日志不进入独立首差的输入。历史统计原件及固定容差比较器均未修改。

公共注册与mask接口依据：[Transformers Attention backends](https://huggingface.co/docs/transformers/attention_interface)。GLM和DeepSeek共享接口的存在不等于当前故障kernel已支持其全部调用。
