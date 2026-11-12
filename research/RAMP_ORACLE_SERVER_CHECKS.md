# RAMP Oracle 服务器验证记录

检查日期：2026-11-12；项目 /home/risegpu/sunjiahao/Alignins；环境 sunPytorch；cuda:0 与 cuda:1 两张 RTX 3090。

## 已通过

- 两个 shell 脚本语法及 Python 编译检查。
- 随机集合每层数量匹配差集、排除原始 Top-K、可复现且不消耗训练 RNG。
- 稀疏/全量/空集合替换、原更新不被原地修改。
- 原生按数据量加权 FedAvg 与 server_lr 更新数值检查。
- 原生 Agent 配对训练重放：更新、模型权重和 Python/NumPy/CPU/CUDA RNG 与攻击单分支一致。
- 两张卡并行，每卡顺序两组；4 客户端、1 恶意、CIFAR10 前 2000 张训练/前 200 张测试、每组 2 轮、local_ep=1、BadNets poison_frac=0.3，四组端到端通过。
- 完整 CIFAR10 50000/10000、ResNet9、20 客户端、4 恶意、local_ep=2、每组 1 轮、poison_frac=0：四组完全相同最终模型 SHA256。
- 初始模型、客户端划分、投毒计划、第一轮所有攻击更新及配对正常更新的组间指纹相同。
- 所有参与客户端保留，良性更新未修改；逐层替换预算与指标行数检查。
- 加入与 ASR 分批相同的干净目标类额外评估后，四组各重新跑 2 轮；各组最终模型指纹与加入前相同，额外评估没有改变训练。
- 报告、并发汇总与三类对比图自动生成；逐轮曲线与预算图已查看确认布局。
- 已存在的实验目录拒绝覆盖。

以上只验证实现和运行链路，不代表 100 轮正式研究结果。正式 oracle_v1 尚未创建或启动。

## 原框架核心文件未改动

- src/federated.py SHA256: 361ca0e18b16949d35c22c722cd5982e361447012302ee0c99a5793d2b884149
- src/aggregation.py SHA256: 633b0213f3b5875ebf7bf420747b3498d0c944dde68a763314d0061fee79c45a
- src/agent.py SHA256: 5ffbb42c5016a5d60bbae0132746957f545984e367a71301dd30922464046813
- src/utils.py SHA256: ef5adcc6cf8587ce8ee1a6f1b52f8d44a560ce544183c6f4d8b83bc79ca78274

完整临时验证数据存放在本任务创建的 /data/ramp_oracle_check_risegpu_* 中，校验完成后已清理。只保留本说明与可重跑检查脚本 research/check_oracle.py。
