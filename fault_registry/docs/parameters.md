# 配置与触发语义

`schemas/common.schema.json` 定义统一外层；`mechanism.version` 当前必须为字符串 `1`。机制 ID 可用 canonical 或 legacy，内部统一返回 legacy 以兼容历史证据。CANN `9.1` 归一为 `9.1.0`。未知字段、未知后端、错 Primitive、缺少专属参数和版本不匹配均拒绝。参数的解析通过不代表已有可执行后端或实机资格。

每个机制的专属 Schema 位于 `schemas/mechanisms/`，不能用一个 value/factor 替代不同语义。当前 CPU 模块代理只接受实际实现的 `padding=[0,7,7] / kernel_spatial=[16,16]` 与 `gain=0.5`；不支持的数值拒绝，不静默忽略。其余参数定义是统一接口规范，执行仍由尚待接入的旧 Provider 决定，当前返回 NOT_IMPLEMENTED。

Trigger Engine 支持：

| mode | 特有字段 | 选择语义 |
| --- | --- | --- |
| first | 无 | 本作用域第 1 个合格调用 |
| nth | ordinal，正整数 | 第 N 个合格调用 |
| every_nth | ordinal，正整数 | N、2N、3N… |
| every_match | 无 | 每个合格调用 |
| probability | probability ∈ [0,1]、seed | 基于 seed、作用域键和序号的稳定 SHA256 抽样 |
| condition | field、op、value | 白名单元数据比较；无 eval、脚本或任意属性访问 |

`max_hits` 为每个计数桶的选择上限，null 表示不额外限制。选择会先占用预算，执行失败不回退预算。计数顺序是：实际资格判断 → layer/rank/device/request 筛选 → 合格序号递增 → trigger 与 operator_call 选择 → hit 上限。`operator_call` 表示筛选后的合格序号，不能当作原生 Kernel launch 编号。独立观察器仍保留被筛掉的调用；当前框架触发计数只接收已合格的回调。

| scope | 计数键 |
| --- | --- |
| process | process |
| worker | process、worker |
| rank | process、worker、rank |
| device | process、worker、rank、device |
| request | process、worker、rank、device、request_id |
| operator | process、worker、rank、device、operator |
| layer | process、worker、rank、device、layer |

每个 owner/generation 创建独立 TriggerEngine。process 作用域明确共享同一引擎内的请求与 Rank；没有跨进程共享计数或跨会话继承。并发由锁保护，顺序是获得锁的实际顺序，不承诺不同线程调度间相同概率序列。重复 event_id 拒绝；内存地址不作为身份或去重依据。默认最多记录 100000 个事件，超过上限拒绝；缺少选定作用域身份也拒绝。

target.request_scope=selected 必须给出非空 request_ids；all 必须为空。condition.field 仅允许 operator、layer、rank、device、request_id；op 为 eq/ne/in，in 使用列表。

旧 `FaultConfig`、attach、Provider 配置原样保留。`migrate_legacy()` 仅转换精确且无歧义的旧 fault/error/auto 配置为 every_match，不猜测 occurrence、phase、step、mode 或 evidence_dir 的新含义；其他情况返回明确错误并使用原接口。
