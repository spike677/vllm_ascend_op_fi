# 交付索引 r2：仅文档勘误，不替换r8包

2026-11-26，负责人转交。原[r8发行索引r1](YELLOW-DELIVERY-INDEX-20261126-r1.md)、标签`demoncatpfi-rc3-preview-20261126-r8`及发行提交`e38c4113b421ac979699071a60e80d3babd05e07`保持有效；没有新wheel、源码包或探针包。

优先使用两份独立补充：

1. [卡位勘误及D64处理分支](../docs/worker_core_v1/r8_cards_d64_addendum_20261126.md)：0–7不可用；A0=8/9（39op及Native）、A1=10/11（18三档/22/15）、A2=12/13（Batch/39mr/28）、A3=14/15（视觉及外部示例）。项目分组仍按原执行单，V4与r8在8–15先后运行；按D60的剩余10分钟规则切换。工具缺口、单项最多重跑一次、依赖截止、必须请示条件及回传选择记录都已明确。
2. [V4 G1目标层与logits次数确认](../docs/worker_core_v1/v4_g1_layer_logits_confirmation_20261126.md)：从实际config/named_modules选零基`layer_idx>=num_hash_layers`的首个主模型非hash MoE父模块，核对gate/experts和internal-router分支；不只选gate。单请求、单chunk、无spec/异步超前且实际生成满4token时预期prefill一次+decode三次。第五次不能截断或改工具放行，按校验和D64分支处理。

两份文档覆盖原r8包内卡位文字及缺失的阻塞分支；不改`source-review.json`，验收许可SHA仍为`8568f8a4b59fbfc98060238bbd4acf526f0056d67ddd8eb2f91835dddcc311f5`。执行方无需修改冻结文件，运行时/机制未变化，本次不需新NPU回归。

r8正式文本到达当天审查，从e38c4113开r8.1维护分支仅修改通过组合的登记表，附免许可冒烟。D13先于公共8独立审查；33收到N2六条与ABI/布局再做候选解码、窗口二和r7.5。尚未收到的结果不登记、不冒充已通过。
