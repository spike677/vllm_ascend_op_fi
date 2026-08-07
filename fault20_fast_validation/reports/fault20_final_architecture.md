# Fault 20 最终架构（工程固化版）

日期：2026-08-07

## 1. 结论与边界

Fault 20 当前是一个**版本锁定的 CANN executor-cache precision fault injection 框架原型**：它在 `OpExecCache::Run` replay 边界替换本次调用的地址向量槽位，用同 dtype、同 shape、同 bytes 的 temporary buffer 承载退化后的输入，并继续调用真实 Run。

当前已经在两个 exact typed adapter 上完成模型级 E2E：

- `aclnnQuantMatmulWeightNz`：`addresses[0] = x1`，INT8；
- `aclnnMatmulWeightNz`：`addresses[0] = self`，BF16。

它当前不是：

- kernel 修改或芯片物理故障；
- custom op；
- `torch_npu` 修改；
- CANN kernel 重编译；
- packed INT2/INT4 kernel；
- 任意 ACLNN 算子的自动适配器。

全部 `PROVEN_*` 结论限定于 Qwen3-32B W8A8 dense、vLLM 0.12.0、torch_npu 2.8.0、CANN 8.3.RC2 构建边界、aarch64、`_GLIBCXX_USE_CXX11_ABI=0`、eager、repeat executor cache。

## 2. 当前物理架构

当前代码还没有按最终模块边界拆文件。真实运行由两个 native DSO、一个 SHM 控制工具和一个实验 harness 协作：

```text
Fault Controller / fault20_buffer_ctl
              |
              | POSIX SHM v5
              | /fault20_l4_buffer_substitute_ctl_v5
              v
+------------------------------------------------------------------+
| LD_PRELOAD injection runtime                                     |
| libfault20_buffer_substitute_abi0.so                              |
|                                                                  |
| precise interpose:                                                |
|   OpExecCache::Run                                                |
|   AddOpExecCache                                                  |
|   AddParamToBuf(const aclTensor*) / AddParamToBuf(aclTensor*)      |
|                                                                  |
| cache registry | selector | hit control | buffer lifecycle | log  |
+-----------------------------+------------------------------------+
                              ^ exported GWS begin/commit/abort
                              |
+-----------------------------+------------------------------------+
| typed GWS metadata front-end                                     |
| libopapi metadata proxy (remote final: trace_proxy_v4)            |
|                                                                  |
| aclnnQuantMatmulWeightNz GWS  -> capture x1 metadata              |
| aclnnAscendQuantV3            -> producer address evidence        |
| aclnnMatmulWeightNz GWS       -> capture self/mat2/out metadata   |
+------------------+-------------------------------+---------------+
                   |                               |
                   v                               v
       QuantMatmulWeightNz adapter       MatmulWeightNz adapter
       slot0=x1, INT8                    slot0=self, BF16
                   |                               |
                   +---------------+---------------+
                                   v
                         precision primitive
                   INT2-equivalent / INT4-equivalent
                         BF16 mantissa truncate
                                   |
                                   v
             pinned host buffer + temporary device buffer
                                   |
                  copied address vector[target_slot] replaced
                                   |
                                   v
                     real OpExecCache::Run
```

重要物理事实：

1. typed GWS metadata capture 目前在 `libopapi` metadata proxy 中；
2. cache registry、selector、两个 adapter 的 Run guard、primitive 调用和 buffer 生命周期目前集中在一个 LD_PRELOAD DSO；
3. 因此“runtime / adapter / primitive”是已经验证的**概念边界**，还不是完成拆分的源码目录；
4. `LD_LIBRARY_PATH` 选择 metadata proxy，`LD_PRELOAD` 只加载精确 executor-cache runtime；
5. runtime DSO 没有 `libopapi`、`libnnopbase` 或 `libascendcl` 的直接 `DT_NEEDED`，运行时只对已经加载的 DSO 做精确解析。

## 3. 请求期数据流

### 3.1 cache 建立阶段

```text
typed GetWorkspaceSize
  -> adapter front-end 读取正式 aclTensor metadata
  -> fault20_*_gws_begin()
  -> CANN 内部 AddParamToBuf / AddOpExecCache
  -> TLS 记录本次 raw cache pointer 和参数收集顺序
  -> fault20_*_gws_commit(metadata)
  -> registry[raw_cache] = immutable metadata
```

registry 只保存 immutable metadata 和 device 身份，不保存 activation device address。

### 3.2 repeat Run 阶段

```text
OpExecCache::Run(cache, addresses)
  -> GetLogApiInfo() 确认 exact ACLNN API
  -> registry[cache] 取得 adapter metadata
  -> 校验 enabled/API/device/dtype/format/shape/stride/offset/bytes
  -> QMM 额外校验 QuantV3 producer address == addresses[0]
  -> candidate/hit 原子计数
  -> 复制 addresses vector
  -> 准备 temporary buffer
  -> 仅替换 vector_copy[target_slot]
  -> real OpExecCache::Run(..., vector_copy)
  -> stream sync 后释放 temporary/host buffer
  -> 记录结构化 event
```

任何 guard、解析、分配、拷贝或同步前置条件不满足时，默认行为是原样调用真实 Run。数值变换已经开始后出现的资源错误同样记录状态并尽可能回退；不猜测 private layout，也不原地修改调用方 vector。

## 4. 当前目录树

```text
fault20_fast_validation/
├── source/
│   ├── l4_buffer_substitute_allhit_v1/       # 当前最终 v5/all-hit runtime
│   │   ├── buffer_substitute_preload.cpp
│   │   ├── buffer_control.cpp
│   │   ├── fault20_buffer_api.h
│   │   ├── int_equivalent_transform.h
│   │   ├── mantissa_truncate.h
│   │   ├── build.sh
│   │   └── create_allhit_container_v2.sh
│   └── l4_metadata_lifecycle_proxy/          # typed GWS metadata front-end
│       ├── libopapi_metadata_proxy.cpp
│       └── build.sh
└── reports/
    ├── fault20_final_architecture.md
    ├── fault20_capability_matrix_final.md
    ├── fault20_fault_primitive_design.md
    └── fault20_injection_capability_inventory_20260807.md
```

公开仓库不包含远端构建产物、请求/响应、trace、结构化 evidence、离线测试源码和历史探针；这些验证资产保留在原 Windows/服务器工作区。

## 5. 源码职责清单

| 路径 | 当前职责 | 工程分类 |
|---|---|---|
| `source/l4_buffer_substitute_allhit_v1/buffer_substitute_preload.cpp` | 精确 interpose、ABI 校验、两个 registry、selector、hit、同步 buffer 生命周期、两个 adapter 的 Run guard、日志 | runtime core + adapter selector + buffer manager |
| `.../fault20_buffer_api.h` | metadata/control ABI、v5 magic/version/SHM、mode/API/selection 枚举、GWS callback C ABI | shared ABI + control plane |
| `.../buffer_control.cpp` | reset/status/disarm、bounded/multi-rank/all-hit arm CLI | control plane |
| `.../int_equivalent_transform.h` | INT2/INT4-equivalent 单值纯函数 | fault primitive |
| `.../mantissa_truncate.h` | BF16/FP16 mantissa bit truncation | fault primitive |
| `.../build.sh` | ABI0 构建、DT_NEEDED/SONAME/symbol/LD_BIND_NOW 门禁 | build harness |
| `.../create_allhit_container_v2.sh` | 最终正确的 all-hit v2 容器配置 | container script |
| `source/l4_metadata_lifecycle_proxy/libopapi_metadata_proxy.cpp` | 正式 typed GWS metadata capture，QMM/QuantV3/MatmulWeightNz trace，调用 runtime GWS callbacks | operator adapter front-end |
| `.../build.sh` | 复制并改 SONAME 的真实 `libopapi`，构建受控 metadata proxy | adapter build harness |

## 6. 验证资产策略

历史 discovery、单阶段验证、token 对比和原始 evidence 用于形成本文结论，但不属于可部署运行包，因此不上传。公开文档只保留最终架构、能力矩阵、primitive 设计和汇总统计；模型级结论仍按原始实验的证据强度表述，不把 trace-only 结果写成 E2E。

## 7. 组件依赖与责任边界

| 组件 | 可以做 | 不可以假设 |
|---|---|---|
| runtime core | 精确 hook、cache lookup、控制读取、地址副本替换、资源生命周期、real Run | 不知道任意算子的槽位语义 |
| operator adapter | 用正式 ABI 捕获 metadata，定义 slot、dtype、layout/producer guard | 不实现故障数学，不缓存动态 activation 地址 |
| fault primitive | `same dtype/shape/bytes` 的确定性数值变换 | 不选择算子、rank 或 hit |
| control plane | 选择 API/device/shape/mode/hit 范围并 enable/disarm | 不直接触碰 tensor/device memory |
| experiment harness | 可复现请求、证据窗口、coverage/token 判定 | 不作为产品控制 API |
| evidence parser | 校验结构化事件与 response | 不从日志反推未记录的 kernel 内部行为 |

## 8. 已知架构缺口

- 源码仍是 monolith，需要按 runtime/adapter/primitive 拆分；本轮只给出计划，不改行为。
- typed GWS front-end 仍采用受控 `libopapi` proxy；正式产品需要明确部署、版本选择与回滚策略。
- 当前 temporary 生命周期使用 pre/post stream synchronization，没有 event-fenced async pool。
- 不支持 non-repeat-cache、prefill 大 shape、warmup/cold 注入、并发/soak 或跨版本自动选择。
- 第三个算子、FP16 模型 E2E、MoE/GroupedMatmul/MatmulAllReduce 均未开始。

## 9. 关联文档

- [最终能力矩阵](fault20_capability_matrix_final.md)
- [primitive 设计](fault20_fault_primitive_design.md)
- [当前注入能力统计](fault20_injection_capability_inventory_20260807.md)
