# Distortion：规则、配对与展示

## 职责与环境

消费既有B/F/R原始结果和机制证据，使用冻结规则区分任务、终止与病态，输出现有EvaluationResult的extra字段和Dashboard。可离线运行；不负责安装hook，不修改fault数学，不使用LLM judge。

## 输入／输出与实现

[contracts.py](contracts.py)管理冻结suite和扩展记录；[mechanism.py](mechanism.py)先校验命中/清理；[detectors.py](detectors.py)解析答案、轨迹和规则；[evaluator.py](evaluator.py)配对周期；[replay.py](replay.py)、[report.py](report.py)输出结果和HTML。

输入需完整原文、HTTP/termination/token序列、明确的配置和机制query。只有pieces时不得冒充IDs，缺概率/正确答案不填0。输出由上层dataset/report消费，不可反向作arm许可。

## 状态与边界

无效fault不能纳入效果分母；length单独不是病态，开放题不能臆造Accuracy。全量数值与采样分开，阈值/分母见[唯一指标字典](../../../../docs/evaluation/fake_requant_tool_v1_metrics_zh.md)。历史与新配置不能互补缺指标。新输出另存，原始raw保留。

[tests/evaluation](../../../../tests/evaluation)覆盖冻结规则、合法结构、解析、重放、报告。测试不是新NPU/PPL实测；本次三题N=1不达ROBUST_VISIBLE_DISTORTION的N≥3固定矩阵门禁。
