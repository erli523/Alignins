# 独立后门防御对比实验

本模块只新增文件，不修改 `src/federated.py`、`src/aggregation.py`、任何 RAMP/AlphaGuard 实现或既有实验目录。它是**统一框架下的适配实验**，不是原论文数值结果的完全复现。

## 当前实现范围

| 方法 | 状态 | 本模块的关键行为 |
|---|---|---|
| FedAvg | 对照组 | 按本地样本数加权 |
| RLR | 原算法适配 | 无权重符号投票；样本数加权更新；坐标级正/负服务器学习率 |
| FLAME | 论文重实现，非官方代码 | **本地模型**余弦距离聚类、所有客户端更新范数中位数裁剪、均匀聚合、加噪 |
| Lockdown | 现有稀疏协议独立接入 | 原生 `agent_sparse.Agent`、剪枝/恢复、加权聚合、旧掩码共识清理 |
| CRFL | **经验防御适配，不提供认证** | 裁剪聚合后的**整个模型**、训练加噪、测试时模型参数平滑投票 |
| Prodigal | 作者代码算法适配 | 最后一个本地 minibatch 的逐层梯度 Top-K、严格共识阈值、持续个性化模型 |
| APR-FL | **尚未实现** | 缺完整参数选择公式、多任务损失及干净数据协议；入口明确拒绝运行 |
| FedTOF | **尚未实现** | 缺 ATSD/FOF 完整公式、历史窗口和阈值规则；入口明确拒绝运行 |

来源、固定提交、区别和检索限制见 [SOURCES.md](SOURCES.md) 与机器可读的 [sources.json](sources.json)。没有用普通微调冒充 APR-FL，也没有用任意 SVD 投影冒充 FedTOF。

## 与现有项目的关系

- 复用现有 CIFAR-10/CIFAR-100 数据增强、Dirichlet 分区、ResNet9、`Agent.local_train` 和攻击评估。
- 复用 `ramp_attack_eval` 中的参数构造及评估工具，但**不实例化或运行任何 RAMP 防御**。
- 当前只支持全参与、无持久化 buffer 的原生 ResNet9。其他带 BN running statistics 的模型明确拒绝，不擅自改变 buffer 聚合规则。
- RLR/FLAME/CRFL/Prodigal/FedAvg 支持原生 BadNet、DBA、Neurotoxin、PGD；Lockdown 此版本只开放 BadNet/DBA，避免混用不同稀疏攻击语义。
- PGD 存在一个原生接口问题：`vector_to_model_wo_load` 没有返回值，但调用者将返回值传给 `load_state_dict`。独立入口仅为本次 Agent 实例提供正确的投影参数字典；不修改源文件、全局模块函数、攻击半径或本地 epoch 末投影时机。该适配记录在 `config.json`。
- 不使用恶意客户端真实身份进行防御。`num_corrupt` 只用于构造攻击数据/客户端，不传给防御函数。
- 每个方法独立进程、独立新目录。已有输出会报错，不覆盖、不续写其他实验。
- 训练默认 CPU，不会自动抢占 GPU；本次验证没有启动 200 轮训练。

## 使用

以下命令从项目根目录运行。已验证解释器为 `/data/miniconda3/envs/sunPytorch/bin/python`。

```bash
cd /home/risegpu/sunjiahao/Alignins
PYTHON=/data/miniconda3/envs/sunPytorch/bin/python
$PYTHON -m research.defense_bench --list-methods
$PYTHON -m research.defense_bench --method flame --dry-run
$PYTHON -m unittest research.defense_bench.test_defenses -v
```

FLAME 额外依赖 `hdbscan`，当前环境已安装。新环境使用本目录的 `requirements.txt`；不要为本实验升级正在运行其他训练任务的共享环境。其余依赖沿用项目；每次运行会记录实际版本。

### 两轮 CPU 冒烟验证

```bash
$PYTHON -m research.defense_bench \
  --method rlr --data cifar10 --attack badnet --device cpu \
  --rounds 2 --num-agents 4 --num-corrupt 1 --rlr-threshold 2 \
  --local-ep 1 --bs 2 --max-batches 1 --test-limit 20 \
  --output research_results/defense_comparison/smoke_rlr_seed0
```

改为 `--method fedavg/flame/lockdown/prodigal` 可验证其他入口。注意 shell 中应填写单个方法名，而不是带 `/` 的字符串。

CRFL 必须显式给出裁剪与噪声设置。例如下面仅用于**接入测试**，数值没有经过 CIFAR 防御效果调优：

```bash
$PYTHON -m research.defense_bench \
  --method crfl --data cifar10 --attack badnet --device cpu \
  --rounds 2 --num-agents 4 --num-corrupt 1 \
  --local-ep 1 --bs 2 --max-batches 1 --test-limit 20 \
  --crfl-clip 100 --crfl-noise 0.001 --crfl-test-noise 0.001 --crfl-samples 2 \
  --output research_results/defense_comparison/smoke_crfl_seed0
```

### 正式实验入口示例（不会自动执行）

```bash
$PYTHON -m research.defense_bench \
  --method prodigal --data cifar10 --attack badnet --device cuda:0 \
  --rounds 200 --seed 0 --num-agents 20 --num-corrupt 4 \
  --beta 0.5 --poison-frac 0.3 --local-ep 2 --bs 64 \
  --client-lr 0.1 --lr-decay 0.99 \
  --prodigal-topk 0.6 --prodigal-threshold 10 \
  --output research_results/defense_comparison/prodigal_cifar10_badnet_seed0
```

GPU 示例要在确认设备资源后手动运行；本入口不调度其他任务，也不接管既有 GPU 锁。CIFAR100 若对齐现有 `0.05` 学习率实验，明确加 `--data cifar100 --client-lr 0.05`。

正式对比应移除 `--max-batches`、`--test-limit`，使用多个相同种子，并保持训练/攻击设置一致。两种限制任何一个非零都会记录 `smoke_only=true`，不能作为最终性能。测试集只用于评估，不参与防御调参或训练。

## 防御参数与比较边界

- RLR：`--rlr-threshold 8`，是**人数阈值而不是比例**；改变客户端数后须显式检查。`abs(sum(sign(update))) >= threshold` 才使用正学习率。
- FLAME：`--flame-noise 0.001` 是论文图像分类实验的噪声乘子，实际标准差乘以当轮更新范数中位数。`min_samples=1`、`min_cluster_size=floor(n/2)+1`。没有多数簇就失败，绝不退化成 FedAvg。
- Lockdown：`--lockdown-density 0.25`、`--lockdown-anneal 0.0001`、`--lockdown-same-mask 1` 沿用该代码路径的默认配置，不宣称为最优设置；共识门限默认 `floor(n/2)`，保留 `votes >= threshold` 的坐标。共识仅作用于评估副本，不写回训练全局模型。
- Prodigal：`--prodigal-topk 0.6`；共识默认 `floor(n/2)`，但原实现用 **`votes > threshold`**。每层数量取整，允许小层选中零个参数，不强制至少一个。Top-K 来自最后一个 minibatch 的梯度，不是模型更新幅值。
- CRFL：明确设置 `--crfl-clip`、`--crfl-noise`、`--crfl-test-noise`；当前是用户选择的**常数裁剪日程**。原作 MNIST/EMNIST/LOAN 的动态日程不被冒称为 CIFAR 日程。最后一轮不加训练噪声；评估使用加噪前裁剪模型的参数平滑硬投票。`--crfl-samples` 默认 100，只有 Monte Carlo ACC/ASR，没有置信界、认证半径或 certified accuracy。
- FLAME、CRFL、Prodigal 固定服务器学习率为 1，与各自公式一致；若传其他值直接报错。
- Prodigal 的服务器评估的是公共模型，各客户端下一轮拿到各自的公共部分＋私人离群参数。不会把全局模型统一广播给所有客户端，也不会根据真实攻击者身份决定谁获得个性化参数。

不同方法本来就使用不同聚合权重或训练协议：不能为了“统一”把 FLAME/Prodigal 改成按样本数加权，或把 Lockdown 改成只做服务端 mask。报告时注明这些差异。

## 输出与可重复性

每次运行只生成：

- `config.json`：完整 CLI、来源与固定提交、当前本地源码 SHA256、环境版本、初始模型/分区哈希、开始/结束状态、实际轮数。
- `training.log`：本次训练日志，不与其他实验日志混用。
- `metrics.csv`：逐轮 ACC、ASR、干净非目标样本被判为目标的比例、触发样本原标签准确率、时间与模型变化范数。准确率均为 `[0,1]`，不是百分数。
- `rounds.jsonl`：裁剪界、噪声、保留客户端/坐标数等方法特定诊断，不包含完整模型坐标。
- `summary.json`：最后至多十轮的均值与实际窗口大小，附 `smoke_only`、评估方式和 `certified=false`。

不保存模型权重、数据集副本、完整参数轨迹或来源仓库副本。`source_sha256` 可用来检查既有代码在运行后是否发生变化。

分区/初始模型由相同种子产生；本地训练按 `(seed, round, client_id)` 确定独立随机流。防御噪声与初始化稀疏掩码使用独立随机流，避免改变其他方法的客户端采样/数据增强随机顺序。因此可以在**本独立套件内**配对比较；不能把旧日志与新日志视为随机轨迹完全相同的配对试验。

## 验证层次

`test_defenses.py` 包含公式边界、真实 HDBSCAN、小模型两轮全流程、个性化跨轮保留、共识不污染训练模型、平滑随机流不污染训练、缺失方法拒绝执行以及旧输出保护测试。合成数据测试不会下载数据。

另外对本机已存在的 CIFAR/ResNet9 执行 CPU 冒烟，详见 [VALIDATION.md](VALIDATION.md)。任何短跑的 ACC/ASR 都不是复现效果证据。
