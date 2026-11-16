# Worker Extension 部署

在目标 serving 镜像自身的 Python 环境安装蓝区扩展。发行目录为仓库的 `src/`。Provider 依其既有发行方式进入同一个 Worker 解释器。

```bash
python -m pip install ./src
python -c 'from demoncat.runtime.vllm_ascend.controller import DemoncatWorkerExtension; print(DemoncatWorkerExtension.__module__)'
```

机制动作需要相应的 Python Provider。Fault20 随仓库根包部署，Fault39 还需其独立包。已有同名内源 `demoncat` 包的环境以其现有 Controller/扩展的组合方式为准。蓝区试验使用独立环境核对包的实际导入路径和 serving 镜像。

```bash
python -m pip install .          # 机制 catalog、运行时绑定与 Fault20
python -m pip install ./fault39  # 需要 Fault39 时
```

## 首轮 serving 探测

在已具备模型与目标设备的镜像中，通过正常 vLLM-Ascend 启动参数附加扩展。以下为 TP1、eager 的示例，真实用例由既有 serving 控制端执行。

```bash
vllm serve "$MODEL" \
  --tensor-parallel-size 1 --enforce-eager \
  --worker-extension-cls demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension
```

先验证基础模型健康请求，然后通过本镜像已有 serving RPC 查询扩展和 Worker 并做被动 `attach/detach`。vLLM v0.18.0 Python 验证程序可在它已有的 `LLM` 对象上使用如下调用。示例仅适用于经过身份校验的单 Worker 环境。

```python
identity, = llm.collective_rpc("demoncat_worker_identity")
assert identity["status"] == "WORKER_IDENTIFIED"
worker = identity["worker"]
capabilities, = llm.collective_rpc("demoncat_capabilities")
attached, = llm.collective_rpc(
    "demoncat_attach", args=({"tenant_id": "diagnostic", "worker": worker},))
assert attached["status"] == "ATTACHED"
ticket = attached["ticket"]
detached, = llm.collective_rpc("demoncat_detach", args=(ticket,))
assert detached["status"] == "DETACHED"
```

collective RPC 调用每个 Worker。TP2/DP 的变更动作须使用明确的目标身份和已验收的宿主路由语义。健康检查、Worker 重启和正常模型生成使用现有 serving 路径。[vLLM v0.18.0 RPC](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/entrypoints/llm.py#L594-L621)

## Native Profile

Native `LD_PRELOAD`、匹配库搜索路径和分裂代理在创建 Worker 前准备。附着确认 Worker 身份，机制准备阶段由 Provider 检查实际映射、所需动态符号、构建身份和能力。初始化成功、API Server 的环境变量或磁盘上存在 ELF 单独只提供部分证据。实际映射及版本应逐 Worker 核对。

在 CANN 8.5.1 的独立验证容器中，既有构建使用以下组合。bundle 和其构建环境是成套的，两个机制的 bundle 分别在新进程验证。

| Profile | 启动前准备 | 独立检查 |
| --- | --- | --- |
| 无注入扩展探测 | Python 扩展可导入，使用正常服务镜像和已初始化模型。 | MRO、现有 RPC、身份、附着和健康请求。 |
| Fault20 isolated-cache | 匹配 `build_qmm.sh` 生成的 build profile。启动前 preload `libdcat20_qmm.so:libopapi.so`，用同一 bundle 的 `libopapi_nn.so` 覆盖验证容器的版本匹配加载位置。 | 该构建包含 WeightNz 和 QMM 两个独立适配子路径。按真实 API、self/x1、实际 Worker profile 和证据目录分别注册。 |
| Fault39 native ABI v1 | 匹配 `build.sh` 及部署 profile。启动前 preload `libfault39_native_v1.so:libopapi.so`，其 split 代理和 real 副本与运行时匹配。 | 结合已加载映射、构建、rank、可用请求范围及 lossless 证明检查实际能力。 |

历史验证命令将匹配 bundle 挂在 `/native`，如：

```bash
source /usr/local/Ascend/cann-8.5.1/set_env.sh
export LD_LIBRARY_PATH="/native:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD=/native/libdcat20_qmm.so:/native/libopapi.so
export TASK_QUEUE_ENABLE=0
export VLLM_SERVER_DEV_MODE=1
# Pod / container 按部署 profile 把同一 bundle 的 split 代理挂到匹配位置。
```

上述 Fault20 文件名来自新隔离构建，和旧 `build20-a1` 的 `libdcat20.so` 对应不同实现。Native 单点验证包括 B/F、实际 Run 和完整 OFF。Serving 进程的多 preload 组合应检查 SONAME、符号以及初始化顺序，才能分别验收。

## 部署方提供的回放

扩展可在普通 serving Worker 上做身份、附着、Native 准备和只读状态查询。正式注册需要部署方在 Worker 内绑定可验证的排他回放。Fault20 通过 `register_worker_matmul` 绑定实际 Worker 身份、构建 profile 和回放，`prepare` 默认查询 `aclnnMatmulWeightNz`，QMM 使用 `settings.operator="aclnnQuantMatmulWeightNz"`。Fault39 通过现有 `register_runtime` 接收 `Fault39WorkerReplay`，由调度集成提供实际请求 ID、作用域和排他预约。当前探测验证普通 Worker 缺少该绑定时的拒绝路径，不包含生产调度适配。

124 每次设备实验使用既有锁和空闲门禁。原始采集、vendor bundle 和张量保留在验证机，公开结果按路径和 SHA-256 追溯。[Worker 实机探测](../../experiments/worker_extension_20261116/probe.py) · [Fault20 隔离构建与回执](../../experiments/production_wiring_20261116/README.md) · [Fault39 原生版本适配](../../fault39/docs/cann_version_adaptation.md) · [附着合同](../contracts/worker_extension_contract.md)
