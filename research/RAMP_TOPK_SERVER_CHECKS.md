# 服务器检查记录

项目：`/home/risegpu/sunjiahao/Alignins`。

## 已实际运行

2026-11-11，在 sunPytorch 环境、GPU cuda:1 上，分别运行 CIFAR-10 / ResNet-9 / 原生 AlphaGuard 下的 badnet、pgd、neurotoxin。每项 2 个全局轮次、local_ep=2、20 个客户端、4 个恶意客户端、poison_frac=0.3、Non-IID beta=0.5，使用完整数据集。

三项均正常结束，报告及图片生成成功。本次尚未运行正式 100 轮实验。

成功检查批次位于：

```text
/home/risegpu/sunjiahao/Alignins/research_results/RAMP_topk_validation/clients20_malicious4_20pct/2026-11-11_14-50-58/
```

该预检查批次的目录时间为服务器原 UTC 时间，实际时区保存在 run.json。正式脚本已将默认目录时间设为 Asia/Shanghai。

## 输出完整性

|攻击|客户端轮次记录|层记录|评估轮次|独立图片|
|---|---:|---:|---:|---:|
|badnet|40|3000|2|5|
|pgd|40|3000|2|5|
|neurotoxin|40|3000|2|5|

整批额外生成 summary.csv、2 张对比图及 README.md。

已核对：各层计数之和等于对应集合大小；原始/相对重合率与差集比例之和为 1；保存索引重算的差集大小与 CSV 一致；PNG 均能解码。另已查看代表性层分布图并调整排版，从同一份 CSV 重生成所有图片，未重新训练或改变统计。

实际可训练参数数为 6,573,120；Top-K 比例 0.001 对应 K=6,573。

## 算法接口检查

- 已知人工样例的原始、相对 Top-K 与差集符合预期。
- 非训练 buffer 不进入诊断坐标；索引映射正确。
- 相同更新重复输入时，非空集合的时间 Jaccard 为 1。
- 诊断不修改输入更新、模型状态或 torch 随机状态。
- 诊断前后的输入交给原生聚合器，得到完全一致的全局模型。
- 参考全零时相对特征仍有限，空差集可记录。
- 逐轮指标不会重复计入末尾 Best results 汇总。
- 最终版本的 shell 语法、Python 编译、人工样例及报告重新生成检查通过。

## 原生源码保持不变

|文件|SHA256|
|---|---|
|src/federated.py|361ca0e18b16949d35c22c722cd5982e361447012302ee0c99a5793d2b884149|
|src/aggregation.py|633b0213f3b5875ebf7bf420747b3498d0c944dde68a763314d0061fee79c45a|
|src/agent.py|5ffbb42c5016a5d60bbae0132746957f545984e367a71301dd30922464046813|
|src/utils.py|ef5adcc6cf8587ce8ee1a6f1b52f8d44a560ce544183c6f4d8b83bc79ca78274|

仅新增独立研究入口、诊断/出图模块、启动脚本和说明，并将 research_results/ 加入 .gitignore。

## 初期观察的解释范围

前两轮中，原始与相对 Top-K 的差异明显，差集坐标在相邻轮次的 Jaccard 较低。该结果只能说明初期集合确实不同，不能据此判断后门坐标、长期稳定性或防御有效性。请通过正式 100 轮观察趋势，再决定是否进入配对训练与 Oracle 恢复实验。

## 随机坐标对照更新（2026-11-11）

在服务器的独立诊断入口增加按层、按数量匹配的随机坐标对照。默认每客户端每轮重复 30 次，从对应层原始 Top-K 以外无放回抽样；独立种子 20261111。AlphaGuard 仍正常过滤，随机集合不参与训练或聚合，三种攻击各 100 轮、local_ep=2 的正式设置保持不变。

新增 research/matched_random.py、tables/random_control.csv、客户端随机汇总字段、06_matched_random_control.png 和整批 03_random_control_comparison.png。随机范围先在重复内对客户端求均值，再跨重复取分位数；它是条件模拟范围，不是置信区间。

服务器人工样例检查通过：层数量匹配、排除原始集合、无重复抽样、合法候选集合覆盖、可复现、空集合/被约束为唯一集合、训练 Python/NumPy/torch RNG 不变、输入和模型不变、原生聚合结果一致、统计分位数汇总顺序正确。

另在 sunPytorch、cuda:1 下用完整 CIFAR-10 运行 BadNets 2 个全局轮次，local_ep=2，20 个客户端、4 恶意、poison_frac=0.3、beta=0.5。训练及报告正常结束，退出码 0。保存 40 条客户端轮次记录、1,200 条随机重复记录、6 张单项图及 3 张批次图。已从稀疏索引重建全部随机抽样，并核对每层数量、原始集合排除约束、逐重复 Jaccard 和汇总字段，均完全一致。

该次训练指标、原有客户端统计及两轮原始/相对 Top-K 索引，与同配置未加随机对照的此前 BadNets 检查完全一致。原生 src/federated.py、src/aggregation.py、src/agent.py、src/utils.py 的 SHA256 保持不变。已查看随机对照图片，排版正常。正式 100 轮实验尚未启动。

本次检查结果：
/home/risegpu/sunjiahao/Alignins/research_results/RAMP_topk_random_validation/clients20_malicious4_20pct/2026-11-11_23-15-01/

正式运行：
conda activate sunPytorch
cd /home/risegpu/sunjiahao/Alignins
bash run_ramp_topk.sh all

也可以分别运行 badnet、pgd、neurotoxin；默认 cuda:0，可使用 DEVICE=cuda:1 指定 GPU。
