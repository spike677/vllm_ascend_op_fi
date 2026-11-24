# 33 A3库身份文本审查

来源：负责人提供的《故障33_A3库身份与待补项_原文.txt》，2026-11-24收到；A3 Qwen3.8 W8A8 TP2、r7.2 site。原始文本3233字节，SHA256 `b45f8a827c158322c4d1ddb1a88805f0c26763d4735cd4e8e164519c010b642c`，保留仓库外，不复制容器名、端口、设备编号和私有原件路径。本页不生成部署登记。

结论：**库身份文本部分通过；完整33取证仍需退补**。不重复索取已经提供的9个SHA。以下是黄区文本声明，尚非蓝区读取实际映射文件得到的结果。

|库|SHA256|ELF build-id|
|---|---|---|
|libascendcl.so|123d67c313f743e5d6f2e856f40c24d5edac376975cf0a7a1019e079b4171480|未观测到|
|libopapi.so|3e10def2f7158482f3967b9db6ad2707cca8691198ddae8cc52bbbaf61bb2675|未观测到|
|libopapi_transformer.so|9a79ed65cdebf70a4ee13c0e5fe0df331f8482a079907e4f184d9851e1183100|a7de1793a47988a5f6bca9a614a17946885f5466|
|libnnopbase.so|308dda461704a6e5f70de6e0eac4495e9829b9cec3328563b75a5b382c6e438b|未观测到|
|libopskernel.so|0a79294dd33e2b75065510bef9ba40b0717decd7a0ead43cb376b82fb24101d3|未观测到|
|libhccl.so|e7742392619bd0f7bfd1b55552070f9811b25ed3017eefd49ed2c1f9f45fc647|1a6ea4d467da5027deb04e2edecbfd43781698ac|
|libhcomm.so|648b248b39fb91164b08f20a12937ca31e780b28383b989590a7d6acc636fcb3|80c48a57edb267d2064b60264bbcf9b46afd9454|
|libacl_rt.so|5a6518505ef66688f02b4bde97517ba217212f29304d59b3a6ef0db14ded6c83|未观测到|
|libge_runner.so|d4854710dab10135bce0a5bb4ecc20b4e223cea96e24c531ab26a41c1a8a9301|未观测到|

## 已经补齐

- 安装元数据version.cfg为9.1.0，组件条目runtime/ge-executor/metadef为9.1；报告声明这些库已在Worker中映射。
- 9个候选operator/host/runtime/通信库的字节数与完整SHA，3个build-id。其余6个按“未观测到build-id”保留，不补造，也不单凭这一现象证明原因一定是strip。
- 安装包未附source的现场结论。现有两卡TP2足够本次Qwen取证，不需要九卡或TP8。

## 还不能成立的结论

1. “SERIALIZER_VERSION_9_1_0”目前只能表示该库被标注为9.1.0包的组件。列出的FfnWorkerBatching、SwinTransformer、MoeComputeExpertTokens、MoeFinalizeRouting和通用NnopbaseAddTilingId，未建立与本次FusedInferAttentionScore的关联，不能当作FIA serializer字段布局。
2. 目前没有库SHA/build-id→构建清单→源码commit→FIA serializer→sparseMode=3分支→实际key/模板的证据链。同版本号不足以替代。
3. version.cfg是安装元数据，不把它重标为运行时API观测。保留黄区“9.1.0实际加载”的声明，同时补充其读取来源或映射到该安装包的核对方式。
4. 9库表未逐rank展开；libopapi_nn、libopapi_math、libexe_graph被提到但未列SHA。仅当实际FIA路径使用它们才补相应身份，不扩大成无目的全库重复hash。
5. 远端A3-LIB-IDENTITY.md报告3308字节，但未给SHA；这与收到的3233字节转述文本是两份不同材料，不按字节数相等处理。

## 发给负责人的下一步

按[fault33_a3_evidence_execution_20261124.md](fault33_a3_evidence_execution_20261124.md)执行：复用本次9库表，只补两rank映射关联、原件SHA、实际FIA的sparse_mode/mask/布局/序列长度/block_table元数据、完整launch key/核数及错误计数。匹配源码向对应安装包提供方索取，重点为FIA serializer和host/key/template，而不是任意Transformer/MoE tiling符号。

可以补同进程、同输入的两次无注入token_ids与完整logits逐位金丝雀；记录每rank形状、dtype、全部元素比较和原件SHA。不能用HTTP top-k logprobs冒充完整logits，也不能用金丝雀相同代替固定Q/K/V的FIA控制重算≥99%。发生非有限值或不稳定时先记录基线问题，不注入。

33仍为“需退补”，不按时限结案，不放开产品门禁。r7.3/r7.3.1已发行；r7.4与r8尚未完成发行，不能把本执行单补充件称为r7.4正式包。
