# RAMP Oracle 坐标作用验证

项目：`/home/risegpu/sunjiahao/Alignins`。所有正式实验和验证均在服务器运行。

## 正式运行：两个终端，两张 GPU

终端一：

```bash
conda activate sunPytorch
cd /home/risegpu/sunjiahao/Alignins
bash run_ramp_oracle_gpu0.sh
```

终端二：

```bash
conda activate sunPytorch
cd /home/risegpu/sunjiahao/Alignins
bash run_ramp_oracle_gpu1.sh
```

两者可以同时启动，不需要等待对方。每张卡顺序执行两个完整实验，不会同时在一张卡塞入两个训练进程。

|脚本|默认设备|第一组|第二组|
|---|---|---|---|
|run_ramp_oracle_gpu0.sh|cuda:0|A_no_repair 不修复|B_promoted_repair 差集替换|
|run_ramp_oracle_gpu1.sh|cuda:1|C_random_repair 随机替换|D_full_clean 完整正常更新|

每组各 100 个全局轮次，每一轮均执行该组处理方式。不是只在少数轮次修复，也不是四组共享一条训练轨迹。

## 固定设置

- CIFAR10，现有 ResNet9；20 客户端，恶意客户端 ID 0—3（20%），每轮全部参与。
- BadNets，plus 触发器，目标类 7。
- poison_frac=0.3：保持原生语义，对每个客户端非目标类候选集取 30%，不是全部本地数据的 30%。实际投毒索引保存到本组 tables/data_partition.json。
- Non-IID Dirichlet beta=0.5；local_ep=2 个完整本地 epoch；batch_size=64；drop_last=True。
- SGD client_lr=0.1，lr_decay=0.99（沿用原生 round 从 1 开始的公式），momentum=0，wd=0.0001。
- 所有客户端参与按数据量加权 FedAvg，server_lr=1；绝不调用 AlphaGuard 淘汰或重加权。
- 差集定位沿用上一批 RAMP_topk 的诊断定义：只用可训练参数更新，按 L2 中间 25%—75% 选择参考候选，少于 min(8,N) 时使用全部；逐坐标 torch.median（偶数取较小中间值），或配置 mean。
- 相对幅值 `abs(delta)/(abs(reference)+epsilon)`，epsilon=0.001；Top-K=0.001（0.1%）；差集=相对 Top-K 减去原始幅值 Top-K。这不是原生 AlphaGuard 的完整检测/淘汰流程。
- 种子沿用框架 0，不放入目录名；随机坐标使用独立 NumPy 生成器，不消耗训练随机状态。

## 四组含义与公平性

A 原样聚合；B 只把真实恶意客户端差集坐标换成配对正常更新；C 同样替换真实恶意客户端，但随机集合在每一层的数量与其自身差集严格一致；D 将真实恶意客户端整份更新替换为配对正常更新。良性更新在四组中都不被修改。

每组每轮为恶意客户端额外执行一次正常分支：使用同一客户端原始图片/标签、同一轮初始全局模型、同样 batch 顺序和随机状态、同样优化器/步数。攻击分支先正常运行；正常分支结束后恢复攻击分支之后的模型状态和 Python/NumPy/CPU/CUDA RNG，不影响主训练后续随机流。A 也生成配对分支用于诊断和公平核对，但不替换任何坐标。

每组的初始模型、客户端划分、投毒计划都保存 SHA256。第一轮所有客户端攻击更新及配对正常更新也保存指纹；组间起点或设置不一致会拒绝汇总。后续各组模型因处理不同自然分叉，不应要求不同组后期差集坐标或数量完全相同。

干净目标类预测率单独在与 ASR 相同顺序、相同分批的非目标类干净图片上计算。现有 ResNet9 在评估时仍使用 batch 统计，不能直接复用包含目标类图片的全量 Clean ACC 批次预测来做严格对应。额外评估保存并恢复 RNG，不改变主训练随机流。

随机候选池沿用上一批：每层排除原始 Top-K，再按本层差集数量无放回抽样；允许随机集合与差集自然重合，并记录重合数量。C 每客户端每轮仅抽一次随机集合，构成一条完整训练轨迹，不是十次完整随机对照。多种子重复留待探索结果后开展。本批严格四组，不含原始幅值修复第五组。

这是 Oracle 机制验证：使用真实恶意身份及真实正常更新，不能作为可直接部署的 RAMP 防御性能。

## 结果位置

默认批次：

```text
/home/risegpu/sunjiahao/Alignins/research_results/RAMP_oracle/
└── cifar10_resnet9/noniid_beta0.5/clients20_malicious4_20pct/badnet/oracle_v1/
    ├── batch.json
    ├── README.md                    # 批次状态、四组汇总、对比图入口
    ├── comparison/                 # 自动生成 curves.png、acc_asr.png、repair_budget.png、summary.csv
    ├── A_no_repair/
    ├── B_promoted_repair/
    ├── C_random_repair/
    └── D_full_clean/
```

每组独立保存：

```text
run.json
console.log
report.md
tables/
    metrics.csv                     # 每轮全局 ACC/ASR/干净目标类比例/触发后原标签 ACC
    client_round.csv                # 客户端角色、定位分数、替换数量/幅度、聚合权重
    layer_round.csv                 # 恶意客户端每层差集/随机集合的预算
    data_partition.json             # 真实划分与投毒索引
figures/training.png
snapshots/global_100.pt              # 聚合前全局模型、4 份攻击+4 份正常更新、候选和替换索引
final_model.pt                      # 最后全局模型
source_snapshot/                    # 本次源码快照，便于复核
```

服务器 home 分区空间有限，默认仅在最后一轮保存完整配对快照，四组合计约 1 GB 量级；其余轮次仍完整保存统计。日志不写到共享 docs 或公共日志文件。快照只存恶意客户端更新，不存所有良性更新；因此它支持局部对比，但不是完整训练断点，不提供自动续训。

每组结束自动生成本组报告并更新批次对比。两张卡通过文件锁避免同时写坏汇总；只有 completed 组进入对比图。最后一组结束后 README.md 自动包含四组结果。单组失败会记录 failed 状态，后续同卡组停止，不会把失败当完成。

## 常用覆盖

使用其他设备：

```bash
DEVICE=cuda:1 bash run_ramp_oracle_gpu0.sh
```

重新开始整批时，两个终端必须使用同一个新目录名；已存在的组绝不覆盖：

```bash
BATCH_NAME=oracle_v2 bash run_ramp_oracle_gpu0.sh
BATCH_NAME=oracle_v2 bash run_ramp_oracle_gpu1.sh
```

两边 ROUNDS、LOCAL_EP、TOPK_RATIO、EPSILON、REFERENCE 必须一致；不一致会拒绝加入同一批次。脚本默认不修改这些设置。也可使用 RESULT_ROOT 指定完整批次路径。

如磁盘充足并需要更多完整配对快照，可在两个脚本后添加同样的：

```bash
--snapshot-rounds 20 40 60 80 100
```

这会显著增加存储，默认不启用。

如果 A 已完成、B 因中断失败：不自动续训，也不覆盖 B。保留失败组记录，改用新的批次重新训练，或明确指定一个新的单组输出目录进行排查。需要补跑尚未创建的单组时可运行 `python research_oracle.py train --group B_promoted_repair --output /完整批次路径/B_promoted_repair --device cuda:0`，其余参数须与原批一致。

重建已有报告：

```bash
python research_oracle.py compare /完整批次路径
```

## 如何判断

先检查 A 确实产生攻击效果、D 能恢复正常训练表现，再比较 B 对 A 的 ASR 降幅与正常 ACC 代价，最后比较 B 是否稳定优于 C。若 B 与 C 接近、或 ASR 降低只伴随 ACC 崩溃，不支持差集具有额外修复价值。单种子四条曲线仅作探索，不作统计显著性结论。
