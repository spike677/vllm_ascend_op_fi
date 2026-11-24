# D 包产品效果噪声审查准备（2026-11-24）

状态：CPU 工具已实现，未获得 V4 实机资格，不是已冻结 D 包。实际 D0/D1 及 0.32/CANN 9.2.1 的部署适配仍是前提。Qwen 继续使用原有精确比对工具。

`tools/product_effect_noise.py` 复用 `demoncat.dsv4_0731.noise.v1`。标定必须来自两个进程、每进程至少五个同输入正常样本，使用完整 hidden/logits 和 token。不能用 top-5 logits、统计摘要或故障阶段数据扩大噪声上界。完整私有数组留在黄区；回传页最多 4096 UTF-8 字节，超限拒绝而不截断。

可信 adapter 提供 `identity()`、`new_session()`、`capture_noise(phase,prompt,parameters,session)`；session 复用既有 attach/enable/status/disable/detach 和 `evidence()`。工具不创建 Worker、不安装许可、不改变准入。`run()` 在同一组 Worker 上创建两个 owner，各跑 normal/fault/off/health，enable 或 fault 失败也尝试 disable；失败时不得把捕获到的部分数据登记为成功。

`capture_noise` 返回原有 noise.v1 全数组结构，并附真实 owner、request_id、worker_identity_sha256 和 evidence_paths。各 rank 的 provider_owner、请求关联、八项计数、changed_elements、实际消费者及字节/资源恢复由既有 provider 提供。EP 无真实调用的 rank 允许 eligible=0，但必须写明不适用原因，consumer 为 null/false，不能为了齐计数制造调用。两个作用域仍须有真实命中。

只对 OFF/health 的数值恢复使用已冻结噪声上界；区域字节和资源恢复必须严格通过。噪声内的 fault 记为模型效果不可见，不包装成传播成功。工具输出始终 `registration_ready=false` 和 `qualification_granted=false`；身份完整也不自动放行未审查的 V4 组合。

可信接线完成后的本地审查命令：

```bash
export PYTHONPATH=src:.
python tools/product_effect_noise.py --raw "$PRIVATE/product-noise-raw.json" \
  --calibration "$PRIVATE/noise-calibration.json" > "$PRIVATE/product-noise-page.json"
python tools/product_effect_noise.py --verify "$PRIVATE/product-noise-page.json"
```

calibration 文件使用既有 `combination` / `calibration` 两字段，组合身份须与产品原始回执一致。`--verify` 只核对文本完整性、身份与计数的自洽，不替代黄区本地从完整数组重跑审查。

CPU：`test_product_effect_noise` 9 项与 `test_dsv4_noise` 10 项通过。覆盖不精确但在噪声内的恢复、字节恢复不放宽、owner/进程/请求错配、未登记身份、伪造计数、token 变化、超界、Qwen 拒用、失败清理与 EP 无调用 rank。

剩余：收到 D0/D1 后锁定实际 site 和 EP 分组映射；公共 8 的分组错位审查必须依据实际路由/反排列，不把本工具的通用计数校验充作该机制的证明。第 13 行使用黄区已有 router 输出噪声变体及外部 provider v1，不新增机制；第 7 行仅 Qwen，第 37 行配置项加黄区舍入变体。
