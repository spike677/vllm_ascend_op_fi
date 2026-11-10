# step0 / rank0 原始统计

这是本组一次decode的msprobe采集编号，配置为`statistics/mix`，只有step0。

- [dump.json](dump.json)：完整API/module统计。
- [stack.json](stack.json)：原始调用栈。
- [construct.json](construct.json)：原始构造关系。

三份文件与[本组ZIP](../../../../downloads/fault.zip)内的`data/cases/fault/observation/step0/rank0/`逐字节一致。完整dump约2.2 MB，可下载查看；快速查看可用[首个IFA记录摘录](../../../first_ifa_excerpt.json)和[独立首差摘要](../../../../first_difference.json)。

原始case ID为`model_fault_r0`，原始配置中的dump_path与运行时调用栈保持采集时内容。
