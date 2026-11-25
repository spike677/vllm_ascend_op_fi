# D产品路线0.30预适配（2026-11-25）

范围：V4 W8A8 / A3 / CANN9.2.1 / vLLM0.30.0或0.30.0+empty / vllm-ascend0.30.0 / TP8+实际EP。库行13先行，公共8独立门禁。

已实现fault_injection/dsv4_product_v030.py的独立候选准入/请求范围逻辑：真实调度prompt 2500–4092 token、生成4 token、总长≤4096、单请求单chunk、fresh cache、无speculative/LoRA/prompt embeds/视觉；同时核对部署max_model_len/max_num_batched_tokens，不只相信用户参数。TP/EP/MC2分别核对8个rank的成员顺序、本地rank和实际来源；不把算子tp_world_size=1当部署TP。

6项CPU测试通过：上下边界、调度预算/标志、版本与身份缺失、组成员漂移、禁止调用者布尔值放行、Qwen原1024限制未放宽。该模块未接入公共enable，prepare返回executable=false；enable_prepared_route始终拒绝。这是提前完成的准备代码，不是已完成D产品适配。

剩余接线必须形成可审查的D专属路径：
1. D0实际runner/model/source身份及D1 router边界锁定后，给可信D binding加显式profile；不复用Qwen模型准入。
2. 同步连接capabilities范围、canary token限制、自动请求观察器范围、step/request映射；不能只把capabilities显示从1024改4096。
3. 采用独立demoncat.dsv4_0731.v030.noise.v1的固定标定证据；把normal/off/health和干净金丝雀接到现有生命周期。区域字节恢复/资源归零仍精确，噪声不放宽存储标准。
4. EP动态路由时仅记录真实调用，无调用rank守卫；只在共同控制边界汇总，不在条件性site通信。D13不要求combine输出归属，但router输入的请求归属仍须证明。
5. CPU结构/失败恢复回归及A2可移植性准备，不宣称A2具备V4 TP8+EP资格。A3同组Worker两个owner与负例通过后再登记。

r3探针A2烟测只验证观测框架，不算本D产品路由的A2验收。D0/D1未到之前不猜patch版本、模型内部模块位置或组；现有Qwen机制文件及公开产品路径未因此改动。

## r8主线

39op仍按同一次运行的wrapper/owner/control证据定性；26fia按补配置成功声明及对应身份/效果证据审查，不把其它运行的forkserver错误拼接成根因。当前接收目录最近增加的是V4准备材料，未发现新的同次39op/26fia正式效果页。保持既有未定性/待登记状态；全项r8候选A2回归、A3正式文本批量审查与最终冻结仍是独立待办，不能以本探针烟测替代。

本轮P2文字回传已说明canary为2680 token、formal为3201 token，均在D请求范围内；完整prompt-manifest SHA仅给前缀，不能据此声称已核对完整manifest。两个prompt的标定分别审查，不因长度合法自动接受噪声组。

