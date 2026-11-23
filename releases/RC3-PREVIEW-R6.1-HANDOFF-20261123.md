# r6.1交付：Batch串行金丝雀修正

维护基线 `ea42a2461bc783daea172668c03a4549f992d758`；源码 `3b1e9f34b8f4e4130442c21f883772081031cb67`。标签 `demoncatpfi-rc3-preview-20261123-r6.1`。r6不修改；不新增产品登记。由负责人交A1/A2，蓝区不直接发给黄区。

- [发行目录](demoncatpfi-rc3-preview-20261123-r6.1/)：三个wheel、源码包、source-review、release-manifest、SHA256SUMS及安装检查。
- wheel版本：demoncatpfi `0.4.0rc5.post1`；worker-extension / fault39-core `0.3.0rc5.post1`。
- 许可 `source_manifest_sha256` 填 **source-review.json** 的SHA：`0a5ea7527d93bd0b2b86dab5a14aa7664804a392e2e42b3ddfa454c4938b806c`，不填release-manifest的SHA。
- 源码tar.gz SHA256：`0740b86bbe62cec027a7269e752fe91d6933b878a2aa171a466287a7cb95a3a3`。各wheel完整SHA见release-manifest。
- 124无设备安装通过：385文件逐字节相同，无Worker、无NPU初始化。CPU全套1104项，1102通过、2项带理由跳过；最终TP1/TP2离线审查调整后另跑35项全部通过。
- A2/0.26/CANN9.1/社区Qwen3.8 W8A8/TP2：同组Worker两作用域通过，窗口434.813秒，隔离已解除；不代表黄区TP1/TP2已通过。

## 安装和接线

独立安装三个wheel，不混用r6 Runtime、r6.1宿主或旧审查器。先核验`SHA256SUMS`；安装检查及外围交付文件见`DELIVERY-SHA256SUMS`。现有binding、许可和ProductClient动作不变。

Batch的 `run_canary(spec)` **逐条发送**固定prompt列表，每条必须等HTTP请求真实完成后再发下一条。不要一次发送列表，也不要并发线程。每条4个生成token。可执行参考是同包 `product_http_probe.py`；主项目只调整金丝雀回调，用户故障请求仍并发。

```python
def run_canary(spec):
    for prompt in spec['prompt']:
        result = send_completion_and_wait(prompt=prompt, max_tokens=4)
        assert result.completed
```

协议：最多6轮内连续3轮全有限、完整hidden/logits和真实调度一致才enable；disable释放owner后再跑2轮，均与正常参考精确一致。回执记录B1、prompt顺序、实际prefill/decode调度。未达标保留失败，不改旧回执、不重启制造恢复。

A3 TP1和TP2分别在同组Worker跑两个作用域；保留B1用户不注入负例及真实并发B2/B4/B8。使用本包审查器：

```sh
PYTHONPATH=src:. python tools/review_product_mode.py "$RECEIPT" --tp 1 --output "$REVIEW"
# TP2使用 --tp 2；完整原件留黄区，按文本协议提取回传。
```

工具补充包r2的产品提取器可用于分页字段，但其旧审查器不包含3+2串行门槛。将上述r6.1独立审查的结论、审查器SHA、回执SHA一并回传，不能用旧提取页的通过替代新门槛。

## D4/D5及缺许可负例

**D4/D5以r6.1受控复测取代。** r6旧回执没有足够调度轨迹，不能事后恢复并发批次或断言差异来自残留。保留原失败及“原因未定”；r6.1复测验证固定串行协议下能否恢复，不反推r6失败的唯一根因。若新协议仍失败，再依据新轨迹定位。

`ACTUAL_CALL_NOT_YET_OBSERVED` 是尚未执行真实请求时的预期actual_call层待观测项。`ordinary_route_report.can_prepare`明确不将这一层计入prepare拒绝原因，因此它与缺许可一起出现，**本身不算负例不纯**。纯缺许可负例仍必须用有效机制/profile及已匹配的部署/binding，只移除许可；实际拒绝应来自许可/部署授权，且没有注入或残留。若同时存在source、模型、TP、profile或binding不匹配，就不是纯缺许可，需单列，不用该待观测项为其开脱。
