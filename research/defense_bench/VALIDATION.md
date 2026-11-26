# 接入验证记录

日期：2026-11-17。以下验证只证明公式、接口与短流程能运行，**不证明防御性能达到论文结果**。

## 环境

- Python：`/data/miniconda3/envs/sunPytorch/bin/python`
- PyTorch `2.6.0+cu124`、torchvision `0.21.0+cu124`、NumPy `2.4.4`
- hdbscan `0.8.44`、scikit-learn `1.9.0`
- 实际计算设备：CPU；每个训练进程 1 个 CPU 线程。没有安装或升级共享依赖，没有启动 GPU 训练。

## 自动测试

```bash
/data/miniconda3/envs/sunPytorch/bin/python -m unittest research.defense_bench.test_defenses -v
```

结果：**20 个测试全部通过**。其中全流程测试包含 22 个方法—攻击子场景，每个运行两轮：

- FedAvg、RLR、FLAME、CRFL、Prodigal × BadNet/DBA/Neurotoxin/PGD = 20 个场景。
- Lockdown × BadNet/DBA = 2 个场景。
- 全流程使用合成 CIFAR 形状数据、小型线性分类器，但使用原生数据投毒、Agent、本地训练及评估组件。
- 另外覆盖 RLR 等号边界、样本权重、FLAME 的本地模型余弦距离与偶数中位数、真实 HDBSCAN、无簇失败策略、CRFL 全局参数裁剪/最终轮不加噪、平滑投票的固定模型抽样、Prodigal 严格共识/跨轮个性化、Lockdown 评估副本不污染训练、原输出目录保护。
- 单独强制触发 PGD 投影分支，验证投影半径及原模块／类均未被全局修改。

## 与固定上游实现的数值核对

从 `sources.json` 对应提交的源码中读取指定方法，在临时 CPU 小张量上比较：

| 核对项 | 结果 |
|---|---|
| RLR 上游 `compute_robustLR * agg_avg` 与独立实现 | 相同 |
| Prodigal 上游公共模型与每客户端个性化模型 | 相同 |
| Prodigal 上游最后梯度逐层 Top-K 与独立 mask | 相同 |
| Lockdown 上游与现有 `agent_sparse` 的剪枝、梯度恢复 | 相同 |

这些是定点算法核对，不是整个作者训练流水线等价性证明；不同训练模型和攻击路径的差异仍以 `SOURCES.md` 为准。

## 真实本地 CIFAR / 原生 ResNet9

没有下载新的训练数据；使用本机已有 CIFAR 数据。

公共设置：4 客户端、1 恶意客户端、BadNet、seed=0、Dirichlet beta=0.5、batch=2、本地 epoch=1、每客户端每轮只取 1 个 batch、测试集前 20 张。RLR 阈值 2；CRFL clip=100，训练／测试 sigma=0.001，平滑采样数 2。这些 CRFL 参数**只用于接入测试**。

| 方法 | CIFAR10 / ResNet9 / 2 轮 / LR=0.1 | CIFAR100 / ResNet9 / 1 轮 / LR=0.05 |
|---|---|---|
| FedAvg | completed | completed |
| RLR | completed | completed |
| FLAME | completed | completed |
| Lockdown | completed | completed |
| CRFL（经验平滑） | completed | completed |
| Prodigal | completed | completed |

全部运行的 `smoke_only=true`。同一数据集下六种方法的分区哈希和初始模型哈希各自完全一致。没有将这些 ACC/ASR 写成正式对比结果。

本次临时运行输出位于 `/tmp/alignins-defense-sources/smoke-cifar10-<method>/` 和 `/tmp/alignins-defense-sources/smoke-cifar100-<method>/`；临时目录可能被系统清理。持久化的复跑命令见 `README.md`。每次配置记录了该次启动时源码的哈希。

## 隔离检查

- 工作开始时为 `src/`、`research/` 中已有的 **52 个 Python 文件**记录 SHA256；工作结束时全部一致。
- 新增实现、文档和测试均在 `research/defense_bench/` 下。
- 没有修改现有 RAMP/AlphaGuard 代码、训练入口、原始配置或已有实验结果。
- 没有对原有训练进程发出终止/重启命令；没有运行全量训练；没有 Git commit 或 push。
- 新增 Python 模块通过语法编译。已有实验 CSV 的全仓库 whitespace 检查告警不在本次修改范围内，没有修改它们。

## 未覆盖或未完成

- APR-FL、FedTOF：尚无可忠实执行的实现，不能出现在实际结果表中。
- CRFL：没有认证概率区间、认证半径、certified accuracy，也没有作者提供的 CIFAR 超参数日程。
- 没有 200 轮收敛实验、跨种子统计显著性或防御性能结论。
- 没有验证部分客户端参与、带持久化 BN buffer 的模型、Dynamic Trigger/AlterMin/Model Replacement 的协议兼容性，因此入口未开放这些组合。
