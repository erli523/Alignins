# RAMP AG Alpha-Zero：原 AlphaGuard 检测 + 异常客户端坐标置零

独立完整实现 `src/ramp_ag_alpha_zero.py`。保留原生 `src/aggregation.py` 与旧 `src/ramp_alpha_zero.py`。

## 运行

在服务器项目根目录运行：

```bash
cd ~/sunjiahao/Alignins
python src/ramp_ag_alpha_zero.py --ag_topk_ratio 0.001 --topk_ratio 0.005 --alpha_threshold 1 --name ag_penalty_detect0.1pct_zero0.5pct_threshold1_v3 --device cuda:1
```

默认 CIFAR10 / ResNet9 / BadNet，20 客户端、4 恶意、投毒比例 0.3、Non-IID beta=0.5、100 全局轮、每轮本地 2 epoch。无配对干净训练。已有目录拒绝覆盖。

## 每轮流程

1. 本地训练复用原生模型、数据、Agent；每个参与客户端从同一全局模型出发。
2. 原 AlphaGuard Stage 1：全部参与更新计算 L2，取 25—75 百分位区间；不足 8 人时回退全部参与者；默认逐坐标 median 构造 reference。不预先排除真实恶意客户端。
3. 原 AlphaGuard Stage 2：使用原来的有符号参考分母稳定规则计算 Alpha，保留原来的 NaN/Inf 处理。没有替换成绝对值加 epsilon 的干预比值。
4. 原 AlphaGuard Stage 3：对 Alpha 绝对值取前 `ag_topk_ratio` 均值作为 score；默认 0.001 即 0.1%。沿用原 torch median/MAD、MAD 过小时的 IQR/std 回退、modified Z 计算和排名规则，得到原本会被淘汰的客户端名单。
5. 将该名单作为坐标干预对象，替代整客户端淘汰。对这些客户端计算 `r[j]=abs(delta[j])/(abs(reference[j])+epsilon)`，按 r 取前 `topk_ratio`，其中严格大于 `alpha_threshold` 的 delta 坐标置零。无差集、无干净配对更新、无恢复。未被检测为异常的更新保持原样，误报的良性客户端也会被处理，漏报的恶意客户端也不会额外处理。
6. 所有参与客户端使用原 AlphaGuard Stage 3 算出的惩罚权重 p_i：`global += server_lr * sum(p_i*masked_delta_i)/sum(p_i)`。权重在坐标置零前计算，分母包含所有实际参与聚合者，归一化后权重和为 1。不是按数据量加权，不乘 n_i，不按坐标重新分配权重。检测异常对象仍参与聚合，其完整更新不被删除。

## 惩罚权重设计（v3）

沿用原实现：先对客户端 Top-K score 排序去两端得到稳健参考，正常范围为参考的 0.9—1.1 倍；范围内 deviation=0，范围外按距离边界的相对偏离计算。`p_i=max(0.01, exp(-2*deviation_i))`。注意这里不是 exp(-2*Z)。0.01 是原始 p_i 的下限，不是最终归一化权重的 1% 下限。

真实恶意身份不直接设置权重。检测异常的客户端先坐标置零，再按其原始 p_i 聚合；未检测异常的客户端使用完整更新与其 p_i。漏检恶意客户端若分数正常仍可能获得高权重，误报良性客户端也会受影响。暂不引入额外异常组系数或置零后重算惩罚。

原 AlphaGuard 整客户端剔除后仅对保留集合归一化；这里采用同样 p_i 公式，但参与集合为全部参与者，因此归一化分母不同，最终权重值不必与原版保留客户端完全一致。

## 两个 Top-K 独立

- `--ag_topk_ratio 0.001`：前面 AlphaGuard 评分和 Z 检测用 0.1%；别名 `--ag_topk`。
- `--topk_ratio 0.005`：后面坐标置零最多取 0.5%，可自行调到 0.01 等。
- `--alpha_threshold 1`：后面坐标比值必须严格大于 1；别名 `--threshold`。
- `--epsilon 0.001`：仅后面干预比值的分母稳定项；原 AlphaGuard Stage 2 的固定 0.001 保持原样。

同一批输入更新下，修改后两个参数不会改变前面 score、Z 和异常名单。不同干预导致全局模型轨迹不同，所以不同完整实验后续轮次分数仍可能不同。

K 均按整个模型维度计算 `max(1,floor(D*比例))`。阈值与干预 Top-K 取交集，未达阈值不补足。0.8 表示相对比值超过 0.8，并非比参考大 80%；绝对值比值不区分方向。

## 原检测器的人数规则

服务器当前原生 AlphaGuard 按 Z 排序，选取人数为 `min(本轮实际恶意参与数+1, 参与人数-1)`，不是固定 Z 阈值。20 人全部参与、其中 4 人恶意时会标记 5 人。本方法保持这一规则，真实身份仍用于原检测器确定数量与打印角色，但不直接决定哪些客户端接受坐标处理。因此不能称为完全不使用身份信息的检测器。单轮至少需要两名参与者。

日志中原生 Stage 3 的 Removed/Remaining 是原检测器本来会淘汰/保留的名单；本方法随后打印 `detected` 和 `client_exclusion=OFF`，实际只做坐标置零。

## 结果与日志

独立目录：

```text
research_results/RAMP_ag_alpha_zero/cifar10_resnet9/noniid_beta0.5/clients20_malicious4_20pct/badnet/alphaguard_detected/<name>/
```

保存 run.json、console.log、tables/metrics.csv、tables/clients.csv、tables/layers.csv、tables/data_partition.json、source_snapshot、snapshots、final_model.pt；结束自动生成 report.md 与 figures/training.png。四项评估为干净准确率、ASR、干净非目标类图片预测为目标类的比例、触发后原标签准确率。

逐客户端日志/CSV：

- `score` / `relative_score`、`Z` / `z_score`：来自原 AlphaGuard，而非全坐标均值。
- `coordinate_ratio_mean`：额外诊断，后面干预比值的全坐标均值，不决定检测。
- `detected` / `detected_anomaly`：是否被原 AlphaGuard 判为异常。
- `detection_topk` / `detection_topk_count`：评分 K。
- `zero_topk` / `topk_count`：坐标干预候选 K。
- `above_threshold_in_topk` / `above_threshold_count`：候选中超过阈值数量；`threshold_count` 是全维超过阈值数量。
- `zeroed` / `selected_count`：实际选择处理数；未标记客户端为 0。
- `removed_L2`：被移除更新分量范数；`penalty` / `penalty_weight`：原生原始惩罚 p_i；`weight` / `aggregation_weight`：p_i/sum(p) 的实际聚合权重。`data_weight` 只记录原数据量比例用于对照，不参与最终聚合。
- ZERO 表示检测异常并进入坐标处理，不表示整客户端清零；实际数量看 zeroed。

默认最后一轮保存快照，可增加 `--snapshot_rounds 1 50 100`。快照含 reference_vector、检测分数/Z、异常 ID、干预对象原始更新 intervention_updates、真实恶意更新 attack_updates、坐标索引、聚合前模型、层布局与权重。快照 client_weights 是实际惩罚归一化权重，penalty_weights 是原始 p_i，data_weights 仅供对照。未保存全部正常更新或优化器，不是断点续训文件。

## 修改位置及版本

`build_reference` 和复制的 `_alphaguard_stage1/_compute_g_ref/_alphaguard_stage2/_alphaguard_stage3` 负责原生检测；stage3 仅增加中间结果捕获。`select_coordinates`、`zero_update` 负责后续干预，`aggregate` 衔接两者及数据量加权。

此前版本只对齐参考、用全坐标比值均值计算 Z，并直接处理已知恶意客户端；v2 已替换该评分与对象选择；v3 进一步将数据量加权替换为原生惩罚权重对全部参与者归一化。结果保存在 alphaguard_detected 子目录，请使用带 v3 的新实验名区分。已启动的旧进程不会自动更新，需要重新启动并使用新实验名。

原生来源 src/aggregation.py，SHA256：633b0213f3b5875ebf7bf420747b3498d0c944dde68a763314d0061fee79c45a。

验证：原生 score/Z/异常名单及惩罚权重一致；更改干预比例和阈值不改变同一批输入上的检测与 p_i；大幅更改数据量不改变聚合结果；归一化权重和为 1；异常良性对象同样按规则处理；一轮 CPU 小数据训练及日志/快照/报告检查通过。上述为实现验证，不能保证后门指标恢复到原版水平。
