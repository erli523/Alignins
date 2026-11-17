# RAMP 跨攻击、跨数据集实验

服务器项目：`/home/risegpu/sunjiahao/Alignins`。

## 固定设置

| 配置 | 置零 Top-K | 惩罚强度 |
|---|---:|---:|
| A | 5% | 5 |
| B | 4% | 7 |

原 AlphaGuard 检测 Top-K 固定 0.1%，坐标比值严格大于 1 才置零。参考、检测、异常数量规则及惩罚聚合直接调用 `src/ramp_ag_alpha_zero.py` 的 `RAMPAGAlphaZero`，没有另写检测方法。异常名单用于坐标置零，全部客户端仍按惩罚权重归一化聚合。

20 客户端、4 恶意、Non-IID beta=0.5、本地 2 epoch、batch=64、seed=0。

| 数据集 | 客户端初始学习率 | 轮数 | 阶段结果 |
|---|---:|---:|---|
| CIFAR10 | 0.1 | 100 | 100 |
| CIFAR100 | 0.05 | 200 | 100、150、200 |

沿用原本地学习率公式 `client_lr * 0.99**round`；服务器学习率 1。阶段表同时记录单轮 ACC/ASR 和前 10 轮均值，指标单位为百分比；逐轮 `metrics.csv` 单位为 0—1。

## 攻击及差异

- `badnet`：固定 plus 触发器，投毒比例 0.3。CIFAR10 A/B 复用已完成网格结果；CIFAR100 新跑。
- `DBA`：四个恶意客户端各持一段 plus 触发器，评估使用完整触发器，投毒比例 0.3。
- `neurotoxin`：使用上一轮防御后聚合更新，在每层选绝对值最小的 25% 构造下一轮梯度掩码；第一轮无掩码。投毒比例 0.3。
- `pgd`：保留原实现，每个本地 epoch 末投影完整更新；CIFAR10 半径为 `0.1*||w||`，CIFAR100 为 `1*||w||`。是否触发投影可在日志中查看。投毒比例 0.3。
- `dynamic_trigger`：保留原攻击器的归一化输入空间触发器逻辑；size=5、trigger_lr=0.1、outer_epochs=50、优化间隔 5 轮、对抗模型数 3、对抗训练 2 epoch。批内投毒比例 0.3。ASR 使用客户端 0 当前优化后的触发器；测试数据不参与触发器优化。
- `altermin`：源类 3 → 目标类 7。从各恶意客户端自己的干净训练数据中选最多 100 个源类样本；保留正常训练 2 epoch、恶意阶段 1 epoch、boost=1.5、rho=1e-4。**不沿用原项目从测试集取攻击训练样本的做法。** 这里不用 poison_frac 决定样本数；ASR 在未加触发器的源类测试图像上计算，含义是定向误分类。
- `model_replacement`：保留 `federated.py` CLI 默认参数 gamma=50、constrain_weight=0.02、alpha=0.5；有效更新缩放为 **1 倍**，这不是强放大配置。投毒比例 0.3。

目标类别均为 7。不同攻击的 ASR 分母和任务可能不同，特别是 AlterMin，不应把它解释成固定触发器后门成功率。

## 数量、调度与目录

CIFAR10 新跑 6 攻击 × 2 配置 = 12 组；CIFAR100 新跑 7 × 2 = 14 组，共 26 组新实验，汇总另含 2 组复用的 BadNet 结果。

批次：

```text
research_results/RAMP_attack_eval/cifar10_cifar100_AB_20261116/
```

真实数据目录为 `/data/risegpu_ramp_attack_eval/cifar10_cifar100_AB_20261116/`，项目内入口是软链接，避免系统盘空间不足。

- `README.md`：各组状态、参数说明、结果链接、对比图。
- `summary.csv`：各组当前轮数和最近 10 轮均值，未跑完的不是最终结果。
- `milestones.csv`：100/150/200 轮的单轮结果与 10 轮平均。
- `figures/`：按数据集和阶段生成 A/B 对比图。
- `launcher.log`：队列启动、完成和失败信息。
- `<数据集>/<攻击>/<配置>/`：每组只有 `config.json`、`training.log`、`metrics.csv`、`milestones.csv`、`curves.png`。

不保存模型、代码副本、source snapshot 或坐标明细。源码哈希记录在 JSON 中，便于确认运行期间代码是否被改动。

队列 1/3 在 GPU0，2/4 在 GPU1。普通攻击每卡最多两个任务；Dynamic Trigger 开始前申请该卡独占锁，避免和其他本批次任务争抢显存、算力。其他攻击先跑，动态触发器最后跑。

## 查看进度

在项目根目录执行：

```bash
python research/ramp_attack_batch.py status
tail -n 30 research_results/RAMP_attack_eval/cifar10_cifar100_AB_20261116/launcher.log
```

`status` 会刷新汇总和图片，也可以单独 `python research/ramp_attack_batch.py report`。

## 启动命令（已有任务运行时无需重复执行）

```bash
nohup /data/miniconda3/envs/sunPytorch/bin/python -u research/ramp_attack_batch.py launch --queues 4 >> research_results/RAMP_attack_eval/cifar10_cifar100_AB_20261116/launcher.log 2>&1 < /dev/null &
```

也提供四个队列脚本：

```bash
bash run_ramp_eval_cifar10_A.sh
bash run_ramp_eval_cifar10_B.sh
bash run_ramp_eval_cifar100_A.sh
bash run_ramp_eval_cifar100_B.sh
```

统一调度器和四个独立脚本选择一种使用；队列锁会拒绝同一队列重复运行。完成的组自动跳过；中断或失败的目录保留并报错，不覆盖，也不声称能从模型断点恢复，因为本批次没有保存模型。

单独新实验示例：

```bash
python src/ramp_attack_eval.py --data cifar100 --attack neurotoxin --config A --device cuda:0 --output research_results/RAMP_attack_eval/manual_cifar100_neurotoxin_A
```

输出目录必须尚不存在。参数设置集中在 `src/ramp_attack_eval.py` 的 `parse_args()`，命令行支持 `--rounds`、`--client_lr` 等；修改源码后，已建批次会检测哈希差异并停止启动后续组，以免混入不同实现。
