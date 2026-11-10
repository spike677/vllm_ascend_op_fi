# baseline：完整原始statistics/mix

这里的[原始dump](dump.json)、[原始stack](stack.json)、[原始construct](construct.json)与[完整ZIP](../../../../downloads/baseline.zip)中的字节完全一致，不是单算子摘录或GE转换结果。

真实采集只有step0/rank0：完整Qwen3 eager第二次前向的采集编号；第一轮预热、第三轮未采集重复，不能解释为decode第0步或生成token数。完整dump含1616条API/module记录，原生FIA调用28次；null不可观测项保留。

可下载较大的原始JSON或使用ZIP。index.json仅为便利索引，first_fia_excerpt.json是明确标记的摘录，均不替代完整原始文件。
