# DemonCAT 合并合同

## 目标

本合同确保 DemonCAT 产品框架和 Ascend backend 研究仓可组合，而不会出现平行 primitive registry、target schema、controller 或生产 CLI。

## 产品侧职责

DemonCAT 拥有：

- 面向用户的 `position + primitive + target + params` 合同；
- `PrimitiveSpec`、语义 primitive catalog 和用户 target 规范化；
- `RuntimeBinding` 选择、环境策略、FaultHandler dispatch、授权与恢复策略；
- 生产 CLI 与服务 API。

## backend 仓职责

本仓拥有：

- execution-plane backend 和 native interposer；
- 已绑定 target 上工作的 mutation strategy；
- 只读 observer 与 event format；
- 环境相关 physical binding 的 discovery/fingerprinting；
- baseline/fault/recovery evaluation helper；
- capability manifest、ABI/build 检查、测试和 scoped evidence。

开发 CLI 可以 trace、probe、validate 或 benchmark，但不得演变成第二个生产控制面。

## 合同边界

```text
PrimitiveSpec / normalized target       (DemonCAT)
                ↓
RuntimeBinding and request policy      (DemonCAT)
                ↓
logical capability request             (contract boundary)
                ↓
symbol / tensor / storage / stream     (Ascend backend runtime binding)
                ↓
mutation or observation                (strategy / observer)
                ↓
evidence, result and recovery status   (contract boundary)
```

raw address、cache pointer、call index 和 allocation generation 是 backend evidence/binding 输入，不能自动成为产品 target 字段。

## 接入门禁

任何 injectable backend 在被产品登记前都必须：默认 disabled；通过 ABI/library/version/architecture/symbol/tensor/target relation 检查；在不确定 binding、copy/sync 失败、重入或 stale allocation 时 fail closed；提供 bounded hit 与确定 disarm；报告 mutation 与 recovery；并且不超出证据支持的根因表述。
