# RAMP 第一批实验：Top-K 坐标结构与稳定性

服务器项目：`/home/risegpu/sunjiahao/Alignins`。本次只增加独立研究入口，不修改原生训练、攻击、AlphaGuard 的源码。所有运行与检查均在服务器完成。

## 运行

```bash
conda activate sunPytorch
cd ~/sunjiahao/Alignins

# 单次运行一个攻击，默认 100 个全局轮次，每客户端每轮训练 2 个本地 epoch。
bash run_ramp_topk.sh badnet
bash run_ramp_topk.sh pgd
bash run_ramp_topk.sh neurotoxin

# 或者依次运行全部三个攻击，自动生成整批对比。
bash run_ramp_topk.sh all

# 指定 GPU；不需要再激活 .venv。
DEVICE=cuda:1 bash run_ramp_topk.sh all

# 快速检查：至少 2 轮才能观察相邻轮次的集合稳定性。
ROUNDS=2 DEVICE=cuda:1 bash run_ramp_topk.sh all
```

脚本使用当前激活环境的 `python`；也可显式指定 `PYTHON=/data/miniconda3/envs/sunPytorch/bin/python`。各攻击串行运行，不会同时占用三份 GPU 显存。

`badnets` 可作为 shell 参数别名，传给框架时转换为原生名称 `badnet`。`-u` 仅用于让终端及时输出日志，不改变训练。

## 固定的首批条件

- CIFAR-10、ResNet-9、原生 AlphaGuard。
- 20 个客户端，全参与，4 个恶意客户端（20%）；本地数据投毒比例 30%。
- Non-IID Dirichlet beta=0.5，100 个全局轮次，local_ep=2。
- client_lr=0.1、server_lr=1、lr_decay=0.99、batch_size=64，其他参数沿用框架默认值。
- 原生随机种子为 0；实际目标类、触发器、优化器等全部配置会保存到各实验的 run.json。
- Top-K 比例为 0.001（0.1%）；诊断 epsilon=0.001；诊断参考默认为逐坐标中位数。
- 每客户端每轮增加 30 次同层、同数量随机坐标对照；只统计，不修改更新、不进入聚合。

可通过 `ROUNDS`、`LOCAL_EP`、`DEVICE`、`TOPK_RATIO`、`EPSILON`、`REFERENCE` 环境变量调整。`REFERENCE=mean` 只改变诊断参考，原生 AlphaGuard 参考仍为 median；这会明确保存在配置中。

随机对照可通过 `RANDOM_REPEATS` 和 `RANDOM_SEED` 调整；默认分别为 30、20261111。随机种子独立于训练的 Python、NumPy、torch 随机状态，不改变原有种子 0 的训练流程。

## 观察什么

在每次全局聚合前，对全部参与客户端的模型参数差值计算：

1. 原始集合：`I_raw = TopK(abs(delta))`。
2. 相对集合：`I_rel = TopK(abs(delta)/(abs(reference)+epsilon))`。
3. 排序提升集合：`P = I_rel - I_raw`。

参考候选由可训练参数更新 L2 的第 25—75 百分位区间选择；候选不足 8 个时仿照当前 AlphaGuard 回退到全部参与客户端，小规模参与时最低数不超过参与总数。参考候选不是已知良性客户端，真实角色只用于结果分组。

参考中位数使用 torch.median 的偶数样本下中位数约定，与当前 AlphaGuard 一致。新诊断分母按 RAMP 设计采用 `abs(reference)+epsilon`，不会改写原生 AlphaGuard 的分母或聚合结果。

统计包括：原始/相对 Top-K 重合率、差集比例、同一客户端跨轮集合 Jaccard、差集保留率、每层选中数量、按层参数量归一化的富集度，以及参考幅值/epsilon 的关系、边界并列值和额外计算耗时。

## 随机坐标对照

对每个客户端、每个全局轮次，严格匹配实际差集在各层的坐标数量，从对应层的原始 Top-K 以外无放回抽取随机集合。每轮重新抽样；每个重复分别与该客户端上次参与时的同一重复比较，再汇总随机时间 Jaccard 的均值和模拟分位数。

默认重复 30 次，不增加训练实验：仍只有 BadNets、PGD、Neurotoxin 三项，各 100 个全局轮次，原生 AlphaGuard 继续正常过滤。被排除的客户端会被观察，但不会参与最终聚合。

随机对照用于判断实际差集的坐标重合度是否高于条件随机水平。由于层数量已经人为匹配，不能用它证明层集中性；层集中性使用按参数量归一化的富集度。

随机曲线按角色展示：先对每个随机重复内的客户端求均值，再对重复均值取 2.5%—97.5% 分位数。阴影为条件随机模拟范围，不是置信区间，也不证明跨种子显著性。第一轮没有时间对照；两个空集合的 Jaccard 记为缺失值，避免把它们误记为稳定性 1。

client_round.csv 保存每客户端的随机均值、标准差、分位数以及“实际减随机”；random_control.csv 保存逐重复数值。稀疏随机集合不另存完整坐标，可由保存的实际索引、独立随机种子和源码重建对应快照轮次的抽样。

层指具名参数张量；卷积权重、BN 可训练权重等分别统计，BN running 状态不进入诊断。每层富集度为 1 表示与按参数量均匀分布相当，不能仅凭大层选中数量多推断其更重要。

## 独立保存目录

```text
/home/risegpu/sunjiahao/Alignins/research_results/RAMP_topk/
  cifar10_resnet9/noniid_beta0.5/clients20_malicious4_20pct/
    2026-11-11_16-30-00/             # 一次脚本调用，不重复使用旧目录
      README.md                     # 整批入口、状态、攻击间对比图片
      comparison/
        summary.csv
        01_training_comparison.png
        02_sets_comparison.png
        03_random_control_comparison.png
      badnet/                       # pgd/、neurotoxin/ 结构相同
        report.md                   # 本攻击的阅读入口，包含表格及图片
        run.json                    # 实际配置、版本、状态、SHA256、参数映射
        console.log                 # 完整终端日志，不再重复写 training.log
        tables/
          client_round.csv
          layer_round.csv
          round.csv
          metrics.csv
          random_control.csv
        figures/
          01_training.png
          02_overlap_and_stability.png
          03_layer_distribution.png
          04_layer_over_rounds.png
          05_reference_and_cost.png
          06_matched_random_control.png
        indices/                    # 仅索引；默认第1、25、50、75、100轮
        source_snapshot/            # 当时使用的研究入口和 src Python 源码
```

`RESULT_ROOT=/你的路径` 可指定整批结果的父目录。目录时间默认使用 Asia/Shanghai，可通过 TZ 覆盖。程序拒绝覆盖已有实验目录；训练报错或 Ctrl-C 中断时保留已写出的统计，报告中明确标记状态。普通 `python federated.py ...` 的日志目录保持原样。

不保存每轮完整客户端更新张量，不额外保存模型 checkpoint。索引为可训练参数拼接空间中的坐标，结合 run.json 的 parameter_layout 可映射到参数名和形状；P 可通过两个已保存集合重算。

## 自动图片

每个攻击输出训练指标曲线、良性/恶意集合差异与时间稳定性、归一化层分布、逐轮层热图、分母及开销诊断，以及实际差集对随机基线的时间曲线。整批输出攻击间训练曲线、集合统计及实际/随机稳定性对比。

图片由 matplotlib 无界面生成，不依赖服务器桌面。NaN/空白意味着统计未定义（例如首轮没有前一轮，或差集为空），不代表数值为零。

## 重新生成图片

```bash
cd ~/sunjiahao/Alignins
python research_topk.py --report-only /该次实验/badnet
python research_topk.py --compare /该次实验批次目录
```

这些命令读取已保存的 CSV，不会重新训练。报告失败不会把训练状态伪装成训练失败，run.json 分开记录训练和报告状态，入口会返回非零码以提示检查。

## 研究解释范围

这批实验判断集合是否不同、是否跨轮稳定、集中在哪些参数张量。它不证明差集是后门参数，也不直接比较不同防御的优劣。各攻击在原生 AlphaGuard 下的训练轨迹会不同；严格因果结论需要后续同客户端、同初始化的配对训练和 Oracle 坐标恢复实验。

先检查 abs(reference)<epsilon 的比例、边界并列计数和参考候选污染情况，再解释相对异常。只跑 1 轮可检查目录与层分布；时间稳定性至少需要 2 轮，正式趋势应使用完整轮次及后续多随机种子复验。
