# RAMP Alpha-Zero：相对比值 Top-K 置零

在 `/home/risegpu/sunjiahao/Alignins` 下运行：

```bash
python src/ramp_alpha_zero.py --device cuda:0 --topk_ratio 0.005 --alpha_threshold 0.8 --name alpha_topk0p5pct_gt0p8
```

默认实验是 CIFAR10 / ResNet9 / BadNet，20 个客户端、4 个恶意客户端，投毒率 0.3，Non-IID beta=0.5，100 轮，每轮本地训练 2 epoch。与已有 B / Zero 文件独立，不覆盖它们。

## 每轮规则

1. 所有参与客户端完成正常的本地训练，上传更新 `delta = local - global`。
2. 取已知良性参与客户端，在其内部按更新参数 L2 筛选 25%—75% 区间；逐坐标 median 得到 reference。参考样本不足时只回退全部良性，无良性时停止。可用 `--reference mean` 修改。此处依赖实验已知身份，是机制验证配置。
3. 计算每个坐标的相对比值 `r[j] = abs(delta[j]) / (abs(reference[j]) + epsilon)`，默认 `epsilon=0.001`。
4. **仅对已知恶意客户端**，按 `r` 从大到小取前 `K=max(1,floor(D*topk_ratio))` 个坐标；其中满足 `r[j] > alpha_threshold` 的坐标，将 delta 置为 0。两个条件取交集；达不到阈值不补足数量。Top-K 按整个模型的可训练参数取，不逐层单独取，不按更新原始幅值排名。
5. 良性更新完全保留。所有客户端仍按原始本地数据量加权，`global += server_lr * sum(n_i * masked_delta_i)/sum(n_i)`。保留恶意客户端未置零坐标；不整客户端剔除、不按坐标重新分配权重。

不是对全局模型参数置零；负向更新使用绝对值计算比值，同样处理。没有配对干净训练、差集或 AlphaGuard 的 Z 阈值客户端过滤。日志的 score 是全部坐标比值的均值，Z 仅供观察。

## 可修改参数

- `--topk_ratio 0.001`：前 0.1%（默认）。`0.005` 为 0.5%，`0.01` 为 1%。
- `--alpha_threshold 0.8`：严格大于 0.8；别名 `--threshold`。
- `--name ...`：为每次实验选择新名字。目录已存在时拒绝覆盖。
- `--rounds 100 --local_ep 2 --beta 0.5 --poison_frac 0.3`：训练与攻击参数。

例如换为前 1%、阈值 1.2：

```bash
python src/ramp_alpha_zero.py --device cuda:1 --topk_ratio 0.01 --alpha_threshold 1.2 --name alpha_topk1pct_gt1p2
```

## 结果与代码

默认目录：

```text
research_results/RAMP_alpha_zero/cifar10_resnet9/noniid_beta0.5/clients20_malicious4_20pct/badnet/oracle_malicious/<name>/
```

每次独立保存 `run.json`（全部参数、状态、源码哈希）、`console.log`、`tables/metrics.csv`（四项评估指标）、`tables/clients.csv`（客户端决策）、`tables/layers.csv`（分层置零数量）、`report.md`、`figures/training.png`、`source_snapshot/`、`snapshots/` 和最终模型。训练结束自动出图。默认保存最后一轮坐标快照，可用 `--snapshot_rounds 1 50 100` 增加轮次。

日志 `topk` 是候选 K，`above_threshold_in_topk` 是候选中满足阈值的数量，`zeroed` 是实际执行置零的坐标数量。良性客户端即使诊断候选非零，`zeroed` 仍为 0。CSV `threshold_count` 统计全维度超过阈值的数量，`above_threshold_count` 统计与 Top-K 取交集的数量，`selected_count` 为实际处理数量。`removed_L2` 是本次被移除的更新分量范数。

完整方案在 `src/ramp_alpha_zero.py`，可直接修改 `RAMPAlphaZero.build_reference`、`select_coordinates`、`zero_update`、`aggregate`。复用原生模型、数据与客户端训练代码，不修改原框架聚合器。
