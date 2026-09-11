# 测试与验证边界

仓库根目录、已有Python环境：

```powershell
# Windows PowerShell；解释器使用当前安装位置
$env:PYTHONPATH="src;tests"
$env:PYTHONIOENCODING="utf-8"
python -m unittest discover -s tests -v
python -m unittest discover -s experiments/fake_requant_reference/tests -v
```

Linux CPU使用 `PYTHONPATH=src:tests`。contracts/discovery/validation/profiles验证共享数据；runtime_inspect验证规则/绑定/计划/部署/有界统计；evaluation验证解析/冻结阈值/配对/报告。参考测试覆盖原CPU数学/config，不能证明tensor极小scale所有边界。

```text
python tools/audit_source_baseline.py verify --manifest docs/releases/source_baseline_v1/source_manifest.json --out <新的审计文件.json>
```

测试替身只证明离线控制流与合同，不冒充Ascend。真实输入、NPU数值、跨进程资源生命周期需实机；本轮只绑定已有证据，不启动NPU。实际结果见[发布记录](../docs/releases/source_baseline_v1/README.md)。
