# Native 可信配置边界负例（C3）

适用对象：冻结 r8/e38c4113 的 41、26fia、20 可信部署文件加载路径。A2 证据与 A3 组合资格分别记录。

## 对黄区构造方法的结论

**满足执行单第7节第5条的测试设计。** 但“计划如此构造”不是已经通过；需保存实际逐rank结果。两种负例分别构造，不能把用户 RPC 不接受 `native_sha256`（`PRODUCT_RPC_UNKNOWN_FIELD`）当成可信配置 SHA 校验通过。

|可信部署文件|预期拒绝|必须观察|
|---|---|---|
|绝对库路径、错误 `native_sha256`|`PRODUCT_NATIVE_REVIEWED_LIBRARY_REQUIRED`|configure exit4/结构化body；两个rank均拒绝；未安装profile、binding、租约；目标库拒绝前后均未映射|
|不存在的绝对库路径、已审查SHA d9aa0d94…ce274|`PRODUCT_NATIVE_LIBRARY_HASH_MISMATCH`|同上；不能因传了错误SHA而提前停在上一条门禁|

通过 `product_control_reply` 包装可信configure调用时，HTTP200加body.exit_code=4可以接受；不要让未捕获异常终止引擎。不要修改冻结运行时以使测试通过。

`load_product_deployment` 首先执行 `validate_blue_native_library`；`configure_product_deployments` 在所有文件校验完成后才安装profile。上述拒绝发生在加载动态库及实际机制执行之前。审查时必须核对具体拒绝码，不能只看“发生异常/exit4”。

建议按黄区已有计划在干净实例上测坏文件，然后另起正确配置实例做41正对照。每个rank保存：Worker身份与PID、实际导入源码路径/SHA、部署文件SHA（剔除宿主秘密）、configure返回、前后审计、前后 `/proc/<实际Worker PID>/maps`、健康检查。maps中一个库可以有多段；正对照按rank/PID和真实路径分组，不能断言映射行数必须等于2。

## 可以复用与不能复用的范围

三项Native共用同一部署文件验证函数，因此同版本、相同可信配置路径的错误SHA/不存在路径测试可证明共享边界；不需要假称三项都执行了Native注入。蓝区A2同字节验证结果可作为该函数的实现回归证据，**不能替代A3配置实测、A3运行库/CANN身份、三项机制实际命中与恢复，不能直接填入黄区组合的negative_controls。**

已有用户RPC未知字段负例、或此前26fia部署SHA被拒的文字记录，都不是完整的“两个rank实际configure拒绝且没有加载库”原件。本轮A2 C1/C2探针附带对此边界的独立检查；最终结果见 `r8_owner_rollback_a2_20261126.md`。该蓝区检查会在未加载目标Native库的同进程中，在18/22诊断完成后提交坏文件；不能把它描述为黄区另起干净实例的测试。

Native三项登记仍需：d9aa产物与sidecar匹配、两个rank实读身份、正确配置的真实请求计数、disable/detach/remove_permit后审计零与金丝雀恢复。库映射存在本身不等于完成这些条件。
