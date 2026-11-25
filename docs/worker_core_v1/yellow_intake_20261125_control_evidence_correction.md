# 黄区控制证据更正：39op、26fia与启动状态

接收日期：2026-11-25。来源：负责人提供的《39op控制证据缺口_26fia复测更正与启动状态_原文.txt》，3328字节，SHA256 `f1a7780a365ce8bfa2cca58f426ea821ca404817d551bde2bcdd24bad1d24061`。原文保留仓库外；本记录不复制机器、容器、进程或黄区内部路径。用途为失败定性和排期，不能替代产品登记效果页。

## 39op：旧证据已确认缺失，不再重复索要旧stdout

同次scope1的解析字段显示enable成功，双rank预热有效，前三个请求各1216次完整覆盖、failed0，第四个请求因生成长度超范围不注入。两个owner串行，非Batch金丝雀为单请求。这些记录支持enable和推理已经发生，不支持“没有许可所以未启动”的说法；`product_qualified=false`和`acceptance_granted=false`不是这一结论的反证。`attach=None`也不能直接判定从未attach，因为摘要不等于完整控制轨迹。

失败disable的原始stdout未归档，receipt中disable为None，只留下`MAIN_CLI_FAILED:5:`和空stderr。daemon单独日志为空，stderr曾进入harness输出。原文同时表述“stdout没有存档”和“stdout为空”，应区分：除非有当次字节数或原始管道记录，只能判为**内容未保留**，不能把未保留证明成服务端返回了空body。

此前`PRODUCT_AUTOMATIC_WRAPPER_OWNER_CHANGED`仍作为先前回传中的错误线索保留；本份材料本身没有该失败body，不能补写为已再次核验。A2原版标准流程未复现，公共缺陷仍未定性，不改变owner保护。

## 黄区宿主最小诊断调整（由负责人转交）

同意在黄区harness中补“失败也保存stdout”，这属于既定黄区宿主职责，不需要修改公共运行时或等待新包。应一次补齐以下信息，避免只补一个stdout文件仍丢失daemon根因：

1. 每次CLI调用在解析JSON或检查returncode**之前**，保存原始stdout/stderr字节、各自长度/SHA、returncode、动作、运行别名、owner、调用序号、开始/结束绝对时间。文件按运行/owner/序号/动作独立命名，不能覆盖前次结果；记录实际超时而非统一转exit5。
2. stdout是JSON时同时保存解析结果；解析失败仍保留原始字节。先保存证据再抛`MAIN_CLI_FAILED`，不修改原退出码，也不把exit5改成成功或exit4。日志不记录鉴权密钥。
3. session daemon的stdout与stderr分别归档，异常路径记录完整traceback和原始异常链；Worker双rank回包先完整保存，再做聚合。不能只保留某一个rank或最终外层字符串。
4. 39op只复测受影响项：同组Worker，scope0正常关闭/清理完成后换新owner；保存scope1 enable结束、末个请求结束、status结束、disable入口的控制关联。若已有只读wrapper观察接口，记录当前/期望`execute_model`、`sample_tokens`的对象身份；没有则明确缺失，不给公共方法打替换补丁。
5. 每个owner结束核对cleanup、三项恢复与全rank资源；失败时不自动重试enable、不强删wrapper。继续使用[r8恢复步骤](r8_y1_defect_review_20261125.md)。只处理受影响实例，其它黄区测试继续。

旧原件继续保留，新的受控复测不能反填旧失败运行。回传可用脱敏文本及原件SHA，不要求复制私有大型回执。

## Batch：存在具体接线疑点，但exit5尚未定因

这份文本描述Batch“2/4/8三个计数依次执行，每组内部Barrier并发”。如果这是第四节所回答的`run_canary`实现，则与r7.4.1的`SERIAL_B1_FIXED_PROMPT_ORDER_V1`不符：**金丝雀内每条prompt也必须逐条发送并等待实际完成**，不能仅让各组串行；正常故障批次的真实用户请求仍应并发。

公共`product_batch.py`会对非串行金丝雀记录`PRODUCT_BATCH_CANARY_REQUIRES_SERIAL_B1`，控制路径预期返回结构化拒绝/CLI exit4。已有CPU回归覆盖这一点。因此该接线可解释“enable被拒”的候选原因，却不能单独解释本次为什么exit5。也可能存在宿主丢失/重包装拒绝，或回滚、传输、内部异常；需要新保存的body才能判定。

黄区先核对Barrier到底位于金丝雀还是故障请求函数：若仅用于故障请求则保留；若用于`run_canary`则改为固定prompt列表逐条等待。每次开启前连续三轮稳定的串行金丝雀，关闭后两轮，公共Host负责轮次判断；不要为了凑轮数在一次arm内额外发送多套请求。补日志后再测Batch，保留单请求不注入负例，不放宽有限性、精确恢复或调度匹配标准。

## 26fia：撤销“补配置后出现新产品缺陷”的说法

负责人现已明确：signal handler TypeError发生在补配置前的configure阶段；补齐只读CANN9.1绑定和Native库注册后，黄区声明两个scope、双rank各64次、changed64、failed0，恢复字段通过。20也有补CANN绑定后通过的记录。

当前状态改为**旧环境/启动或部署接线失败，补配置后黄区声明通过，待正式文本审查**；没有新失败，不再把26fia列为已证实公共产品缺陷。`multiprocessing/forkserver.py → signal.signal`的TypeError与缺少CANN绑定在同次旧部署中出现，不足以证明后者是前者的直接根因；20的CANN准入拒绝也不是26fia signal异常的调用栈解释。不为这一关联编写公共修复。

继续需要两个owner/rank的八项计数、真实消费者、恢复/金丝雀、效果、安装/Native运行库身份和原件SHA用于登记，不重复索要不存在的新失败。负责人此前要求的r8增量双owner检查仍保留。

## V4：启动受阻，尚无可用D0/D1运行证据

黄区报告CANN9.2.1、设备可见和ACL导入已修，但server仍报`_pickle.UnpicklingError: pickle data was truncated`，处于重启循环。只能记录为启动受阻；截断pickle可能是子进程通信中断的后果，不能仅凭尾部异常判CANN、模型或公共故障机制有缺陷。

黄区按既定职责先保留一次完整启动的父/子进程最早异常、退出码、环境身份及资源终止记录，进行有界诊断，避免无限重启覆盖日志。Codex不操作黄区服务、不将失败启动当作D0/D1通过，也不据此提前开放D包。数据准备好后再推进，r8其它工作不等待V4。

## 更新结论

- 39op：证据缺口已定位为宿主归档不足，**故障根因仍未定**；补诊断后复测。
- Batch：先核对并修正金丝雀串行接线，保留exit5内层证据，再定性。
- 26fia：补配置后通过声明，无新失败；待登记证据，不记公共缺陷。
- V4：启动阻塞由黄区本地诊断，D0/D1未就绪。
- 本次仅更新证据与执行建议，无运行时/机制改动，无NPU操作，无新发行或A3登记。
