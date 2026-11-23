# 黄区 r7.1 S0 文本接收与审查

收到 S0 安装/导入失败、修复后复测的合并原文，共18420字节，SHA256 d748401376cfc8f13309e7fc8c2a4fe5d2b708a8ee99272249341dfc401bdaff。原文留仓库外接收目录；不传播机器地址、容器名或私有部署路径。全部依据为黄区文本证明，不是蓝区结果。

组合：A3 Ascend910_9382，Qwen3.8 W8A8，vLLM0.26.0/vllm-ascend0.26.0rc1、CANN9.1、TP2；r7.1源码3f242e04，source-review SHA77e30525…f7e2。安装388文件检查通过；未改冻结物。

先前 ModuleNotFoundError 定性为部署接线/导入来源混用，不是已证明的 wheel 缺文件。原文关于两个regular package“自动覆盖”的解释不足：解析顺序、sys.modules缓存和真实Worker启动路径都可能影响，不能仅凭namespace存在定根因。黄区随后使Worker仅从已安装site及其host加载，CLI使用黄区自己的源码，复测S0已通过。不得删 __init__.py、复制 product_config 或修改冻结包来遮盖。

四页 dcat.product.extract.v1 各1764字节：scope0/1×rank0/1；canonical extract SHA全部正确，reviewer SHA536cf416…053b5e与冻结r7.1逐字节相符。每页 observed=eligible=attempted=applied=consumed=restored=1932，changed=1931，failed=0；消费者、金丝雀、资源清零均通过。image_digest=null按4.9为非必需，不拦截。机器可读审查见 yellow_r71_s0_review_20261123.json。

结论：功能/恢复提取一致性通过；正式登记需退补。原文 negative=null，只有permit ID，不能将permit ID当作产品owner；单个Worker身份文字也不能证明两rank身份完整。

由负责人转交的补充字段（从现有回执只读提取，无需先要求重跑）：

- 每scope的实际product owner、同scope各rank Worker/model身份，以及对应回执SHA。不要补默认值。
- 缺许可、未知profile、未知机制的实际拒绝结论/码与合法正对照，连同各rank资源清零；若当时未执行就明确未执行。
- 回传中样例将HTTP放在disable之后，不能作为注入操作顺序；有效顺序仍是list→enable→HTTP→status→disable，同一session不能在list之后再attach。

V4、Native三项与33均不能由这份18的S0结果代替。完整登记身份仍要按r8正式产品登记流程单独核对；本记录不授资格，product_closed=false。
