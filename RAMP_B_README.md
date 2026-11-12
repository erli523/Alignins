# 可直接修改的 B 组完整实现

服务器文件：`/home/risegpu/sunjiahao/Alignins/src/ramp_b.py`。

这个文件包含自己的参数解析、完整训练循环、配对正常分支、参考构造、原始/相对 Top-K、差集修复、加权 FedAvg、指标评估、逐轮输出、快照及图文报告。没有转调 `research_oracle.py`，也不导入 `research/oracle_core.py` 或报告模块。

只复用 Alignins 的 `utils.py`（数据/触发器/向量转换）、`models.py`（现有 ResNet9）和 `Agent.local_train`（原生本地 SGD）。没有复制或修改原框架核心文件。

## 运行

在项目根目录：

```bash
conda activate sunPytorch
cd /home/risegpu/sunjiahao/Alignins
python src/ramp_b.py --device cuda:0
```

在 src 中：

```bash
python ramp_b.py --device cuda:0
```

默认 CIFAR10、ResNet9、BadNets、20 客户端/4 恶意、每轮全参与、Non-IID beta=0.5、poison_frac=0.3、local_ep=2、100 轮。每轮修复后全部参与客户端按数据量加权 FedAvg，server_lr=1，不淘汰。

参数可以直接修改文件中的 `parse_args` 默认值，也可以命令行覆盖，例如：

```bash
python src/ramp_b.py --device cuda:1 --rounds 100 --local_ep 2 \
  --num_agents 20 --num_corrupt 4 --poison_frac 0.3 --beta 0.5 \
  --reference median --topk_ratio 0.001 --epsilon 0.001 --name b_only_v2
```

## 修改方案

`RAMPB` 类里：

|方法|修改内容|
|---|---|
|paired_train|攻击与正常分支如何生成；同模型、同客户端、同随机状态|
|build_reference|L2 候选区间和参考 median/mean|
|select_coordinates|原始/相对排序、Top-K、差集坐标定义|
|repair_update|选中坐标用什么值修复；当前使用配对正常更新值|
|aggregate|哪些客户端修复、聚合权重和服务器更新|

其他函数：`parse_args` 为默认设置；`train` 为完整训练循环；`evaluate` 为指标；`generate_report` 为图文报告。

参考候选默认取参数更新 L2 的 25%—75%，不足 min(8,参与数) 则使用全部。相对幅值为 `abs(delta)/(abs(reference)+epsilon)`；差集为相对 Top-K 减原始 Top-K。Z 仅用于观察，不决定淘汰或修复对象。

当前修复对象使用真实恶意身份，替换值来自同客户端的配对正常更新，所以是 Oracle 验证，尚不代表服务器能独立获得这些正常更新。poison_frac 保留原生非目标类候选集比例语义。

## 独立结果

默认：

```text
/home/risegpu/sunjiahao/Alignins/research_results/RAMP_oracle_B/
  cifar10_resnet9/noniid_beta0.5/clients20_malicious4_20pct/badnet/b_only_v1/
```

每次保存 `run.json`、`console.log`、`report.md`、`tables/`、`figures/`、`snapshots/`、`final_model.pt` 和源码快照。已有目录拒绝覆盖，重新实验使用新的 `--name` 或 `--output`。

`tables/metrics.csv` 包含逐轮 Clean ACC、ASR、与 ASR 同分批的非目标类干净图片目标预测率、触发后原标签准确率。
`tables/clients.csv` 包含客户端角色、参数 L2、相对分数/Z、差集/替换数量、替换幅度和聚合权重。
`tables/layers.csv` 为恶意客户端每层修复数量。

默认只在最后一轮保存完整配对更新快照，减少磁盘占用；中间轮次全部保留统计。结束后自动生成训练曲线和独立 B 报告，不依赖 A/C/D。
