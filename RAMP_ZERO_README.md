# RAMP-Zero 完整实现

文件：src/ramp_zero.py。独立完整实现，只复用框架数据、模型和 Agent 本地训练，不调用 ramp_b 或 research 入口。

运行：
```bash
cd /home/risegpu/sunjiahao/Alignins
python src/ramp_zero.py --device cuda:0
```

默认 CIFAR10/ResNet9/BadNets，20 客户端、4 恶意，Non-IID beta=0.5，poison_frac=0.3（非目标类候选集比例），100 轮、local_ep=2。参考中间 L2 25%-75%，median，可配置 mean；Top-K=0.001，epsilon=0.001。

差集为 relative Top-K 减 raw Top-K。将选中客户端这些坐标的本地更新 delta 置零，再全部客户端按原数据量加权 FedAvg。不是将全局模型权重清零，不删除整个客户端，不按坐标重新归一化权重。没有配对正常训练，没有正常更新恢复。

默认 --zero_scope oracle_malicious：与 B 保持同样的已知恶意干预对象，仅用于 Oracle 对照。--zero_scope all：所有参与客户端的差集都置零，真实身份只用于日志角色展示。

```bash
python src/ramp_zero.py --device cuda:0 --zero_scope all --name zero_all_v1
```

结果独立写入 research_results/RAMP_zero/cifar10_resnet9/noniid_beta0.5/clients20_malicious4_20pct/badnet/<zero_scope>/<name>/，默认 name=zero_v1。日志、CSV、图文报告、源码快照、最后模型分别保存。默认只在最后一轮保存完整更新快照。重复实验换 --name，旧目录不覆盖。

修改 RAMPZero.build_reference、select_coordinates、zero_update、aggregate 即可测试参考/坐标/置零/聚合规则。CLI 参数和训练循环均在同一文件。

验证：服务器 CPU 小规模 1 轮、2 客户端已完整跑通；两种 scope 的按数据量加权数值均通过，未改动原更新、空集合处理通过。确认每客户端只运行一次本地训练、无正常分支；图文报告生成通过。验证产物已清理。未启动新的正式 100 轮训练，未打断已有 B/D 实验。

## 参考构造更新（2026-11-12）

先取已知良性客户端，在良性池内部计算 L2 的 25%—75% 分位区间，再对候选逐坐标取 median（默认）。候选不足只回退全部良性；没有良性参与时明确报错。仍保留 --reference mean 供显式对照。日志、run.json 与快照均标记 reference_scope=oracle_benign_l2。这是使用真实身份的 Oracle 参考。

已经运行中的进程仍使用其启动时的旧代码；重新运行请使用新的 --name，旧结果保留。
