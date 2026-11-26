# r8 冻结交接

负责人已决定立即以候选 `81643582d2983b574b97208ddb3530312b167f07` 冻结，不等待 A3 登记。标签 `demoncatpfi-rc3-preview-20261126-r8`；发行提交取该标签指向的提交，源码提交以 source-review.json 为准，二者不混用。

发行目录：`releases/demoncatpfi-rc3-preview-20261126-r8/`。本次直接冻结已实测候选的三个wheel和源码压缩包内容；未修改运行时、机制或登记记录。外层补入正式执行单和后续验证证据；这些文档不改变 source-review.json 的SHA。

- demoncatpfi：0.4.0rc7；worker-extension与fault39-core：0.3.0rc7。
- source-review.json SHA256：`8568f8a4b59fbfc98060238bbd4acf526f0056d67ddd8eb2f91835dddcc311f5`。验收许可填写这个SHA，不能填release-manifest或A2实验清单SHA。
- A3登记为0；唯一内置产品登记是已验收的BLUE 18.api_half精确组合，不适用于A3。
- A2组合：Ascend910B4、0.26/CANN9.1、社区Qwen3.8 W8A8、TP2。16项同组Worker双owner回归通过；Batch初次信号退出保留，完整复跑通过，信号发送者未定性。规范安装后的免许可18双owner及未登记22负例通过。
- CPU：1360项、1358通过、2项既有研究入口有理由跳过；登记JSON更新相关18项另通过。所有执行Python与CPU规范快照相同，唯一运行时后续差异为已验证登记JSON。
- 无设备安装：三个wheel、406个Python/JSON文件核对；最终发行路径另留安装执行记录。
- r7.4.1基线481个既有运行时文件：474相同，7个为已批准的共享准入/控制改动；机制/site/observer/Native字节不变。详见shared-byte-comparison及影响分析，不能声称所有共享文件都没改。

## 黄区现在执行

按发行目录顶层 `r8_a3_formal_acceptance_20261126.md` 做16项正式测试。它是本次有效执行单；源码包内历史文档仅作参考。四个TP2槽使用0–7卡，不影响8–15卡及黄区当前安全收尾。全部走PRODUCT_ACCEPTANCE_ONLY及验收许可，不使用A3免许可模式。18三档分开；三个外部示例保留外部许可和逐rank注册。

故障运行、效果比较和一页提取一起做，同一Worker两个owner；不为登记另跑一套旧测试。保留黄区CLI接线与实际HTTP请求。原件留黄区，回传一页文本、关联身份、负例和SHA。公共问题交codex，黄区自有适配器/CLI/HCCL/台账仍由黄区本地维护。

## r8.1与关闭项

正式回传到达当天逐份审查。只有字段齐全、rank/owner一致、消费者/字节恢复/资源/金丝雀及负例通过的精确公共产品组合才进入r8.1。维护分支从本r8发行提交开出，只改登记表；附 `r8_incremental_acceptance_20261126.md` 中免许可冒烟、未登记拒绝和身份负例。该文档在r8阶段只作后续步骤，不授权现在跳过许可。

39op、Batch、26fia的黄区旧问题没有因A2通过而宣称已修复；新正式结果按同一次运行定性。缺证/失败组合不登记。33 A3、V4/D13/公共8继续关闭，待N2/ABI/布局及D0/D1/噪声审查；不阻塞本次r8冻结。外部三个示例不进入公共免许可表。

所有冻结标签和旧包保持不变。r8不是product_closed声明，也不代表黄区master已合入或A3已验收。
