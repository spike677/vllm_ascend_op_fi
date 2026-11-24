# 22 融合 mRoPE 公共持续验收

组合：A2 Ascend910B4，CANN9.1.0，vLLM0.26.0/vllm-ascend0.26.0rc1，torch2.10+cpu/torch_npu2.10.post4，Qwen3.8社区W8A8，TP2，eager纯文本。源码清单SHA `02912cccfa9bbda43918ec88d52322450e0e0dee14a016e5077bcb50353b674d`。独立审查与原件SHA在 `experiments/worker_core_v11_20261118/continuous_public_22_review_20261121.json`。

同组Worker两个作用域、公共Runtime四阶段、每请求prefill+3decode通过。每rank每作用域16个实际融合层、64次observed/eligible/attempted/applied/changed/consumed/restored，failed=0。真实无许可创建拒绝，完成后许可、租约、hook和在途资源清零。

实际边界是原 `vllm::triton_split_qkv_rmsnorm_mrope` 的cos/sin输入。先证明它逐值等于真实 `cos_sin_cache[positions]`，再按 `(positions+1) % cache_length` 选合法错误行，写入当前调用的临时输入对象，由原融合算子恰好执行一次。同stream消费完成后恢复该临时输入，原positions和共享cache未改动。保留真实partial rotary、interleaved及三维positions，不重写融合kernel。

每作用域每rank：rotary_rows_changed=672，position_rows_changed=672，按实际调用对齐的Q/K输出指纹变化64次。报告显式提供这些效果字段，不只以最终hidden/logits变化替代22行级效果。这里后续decode由真实生成继续，因此后续层/步的QK差异可能包含传播；每次错误查表的行变化则由当前实际cos/sin与错误行逐值核对。

所有阶段输出有限；fault的hidden/logits有变化，OFF/health完整输出精确恢复。耗时1.1610967×/1.0944814×。不要求生成文本改变，不据此认定A3通过。

窗口a15：2026-11-21 12:56:27.897317至13:02:28.146344 UTC，360.249秒（包含加载与预热）。卡2/5有界隔离已结束，清理器停止，派发能力恢复；不修改Docker入口/驱动/CANN。当前仅加入开发线候选和BLUE_COMBINATION，r2包及标签不变，待黄区反馈后决定r3。
