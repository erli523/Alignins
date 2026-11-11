服务器目录整理和历史日志位置：[docs/SERVER_LAYOUT.md](docs/SERVER_LAYOUT.md)。脚本用途：[docs/SCRIPT_GUIDE.md](docs/SCRIPT_GUIDE.md)。

# 本机 AlignIns-master 的服务器副本

来源：`D:\pycharm\AlignIns-master`。服务器目录：`/home/risegpu/sunjiahao/Alignins`。

本次复制项目代码与实验脚本，未搬运旧 logs、数据副本、分析结果、缓存和编辑器记录。AlphaGuard、其他攻击防御、训练入口及模型均保留本机实现；不是之前 AlignIns-AlphaGuard 的防御版本。

## 根目录单次入口（2026-11-11 更新）

现在可在项目根目录运行，无需进入 src：

```bash
cd ~/sunjiahao/Alignins
conda activate sunPytorch
python federated.py --data cifar10 --model resnet9 --attack badnet --aggr alphaguard --num_agents 20 --num_corrupt 4 --poison_frac 0.3 --non_iid --beta 0.5 --rounds 100 --local_ep 2 --device cuda:0
```

根目录 federated.py 仅转发参数，并自动切到 src 工作目录后运行原入口，训练、攻击、防御代码没有改动。`-u` 可选，只影响日志是否缓冲。已在 sunPytorch 中补齐 geom-median 0.1.0，并验证根目录 --help 正常；没有改动该环境的 NumPy 或 PyTorch 版本。此前完整训练验证仍使用独立 .venv，不能等同于已在 sunPytorch 完成完整训练验证。

下面是原 src 入口和独立环境的历史运行说明，该方式仍可用。

## 原 src 入口运行一个实验

```bash
cd ~/sunjiahao/Alignins/src
source ../.venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u federated.py --data cifar10 --model resnet9 --aggr alphaguard --attack badnet --num_agents 20 --num_corrupt 4 --poison_frac 0.3 --non_iid --beta 0.5 --rounds 100 --local_ep 2 --device cuda:0
```

这一条只运行一个实验：CIFAR10 / ResNet9 / AlphaGuard / BadNets / 20 客户端且每轮全选 / 4 恶意客户端 / 本地投毒比例 30% / non-IID 0.5 / 100 全局轮次 / 每轮 2 个完整本地 epoch。`local_ep` 不是 batch 数。不要启动根目录的 run_sweep、run_ablation 等批量脚本。

参数都可以直接在命令里修改，如 `--attack pgd`、`--aggr alignins`、`--device cuda:1`、`--rounds 10`。IID 实验去掉 `--non_iid --beta 0.5`；无攻击使用 `--clean`；无防御平均基线使用 `--aggr avg`。

查看完整选项：`python federated.py --help`。必须从本项目 `src` 目录运行，因为原程序按相对路径读取数据并备份代码。

## 参数位置

- `src/federated.py`：命令行选项与缺省值，命令行覆盖缺省值。
- AlphaGuard 可调参数：`--ag_topk 0.001`、`--ag_l2_lower 25`、`--ag_l2_upper 75`、`--ag_gref_strategy median`。参考策略还可选 `geometric_median`、`krum_select`。
- `src/aggregation.py` 的 `alphaguard_params` 及 AlphaGuard 方法：其他固定参数与现有防御逻辑。本次没有替换或修改。
- 此本机版本的淘汰逻辑是按本轮实际恶意参与人数加 1，按 Z 分数排名淘汰；没有套用此前服务器版本的固定 Z 阈值逻辑。
- `--num_corrupt` 是人数，`--poison_frac` 是恶意客户端本地数据投毒比例，`--beta` 是 Dirichlet 浓度。
- 保留原程序轮数行为：参数 `--rounds 150` 会被视为缺省值并替换为该数据集的轮数；例如 CIFAR10 会变成 100。运行其他指定轮数（如 10、100、200）正常生效。

## 当前命令行提供的攻击

| --attack | 方法 |
|---|---|
| badnet | BadNets |
| DBA | 分布式后门 |
| neurotoxin | Neurotoxin |
| pgd | PGD 约束攻击 |
| dynamic_trigger | 动态触发器 |
| altermin | AlterMin 交替优化攻击 |
| model_replacement | 模型替换攻击 |

## 当前命令行提供的聚合与防御

| --aggr | 方法 |
|---|---|
| avg | FedAvg 平均基线 |
| alignins | AlignIns |
| rlr | Robust Learning Rate |
| mkrum | Multi-Krum |
| mmetric | 多指标防御 |
| lockdown | Lockdown |
| foolsgold | FoolsGold |
| rfa | 几何中位数聚合 |
| alphaguard | 本机原有 AlphaGuard |
| mars | MARS |

聚合文件中另有 signguard 分支，但它没有出现在命令行 choices 中，本次未开放或修改。以上列表表示本版本的入口选项，不代表已完成全部攻击/防御组合的实验。

## 数据与环境

已复用服务器现有数据，在 `src/data` 下建立子目录软链接：CIFAR10、CIFAR100、MNIST、FashionMNIST、TinyImageNet。源数据在 BackFed-main 或 AlphaGuard-FLPoison 的 data 目录中，不要移动对应源目录。

唯一原源码改动是 `src/utils.py` 中 FashionMNIST 的数据预处理：补充 28×28 到 32×32 的 Resize，适应原 ResNet9 的池化尺寸。未修改模型、训练或防御。适配前后的 SHA256 见 `SERVER_DATA_ADAPTATION.json`；本机 40 个迁移文件的原始 SHA256 见 `LOCAL_SOURCE_SHA256.json`。

`.venv` 复用 sunPytorch 的 torch 2.6.0 / torchvision 0.21.0，单独配置 numpy 1.26.4 和 geom-median 0.1.0，避免修改旧代码来适配 NumPy 2.x。原 sunPytorch 和其他项目环境未改动。

## 结果与验证

单次运行结果沿用原程序保存到 `src/logs/<数据集>/attack_<攻击>_ar_<恶意比例>/defense_<防御>/<时间及实验信息>/`。日志内包含配置、检测、ACC 和 ASR；准确率与 ASR 输出为 0~1 比例。

服务器验证摘要见 `SERVER_CHECKS.md`。检查使用完整训练集，不代表完整多轮实验的最终效果；临时检查日志会清理，不混入正式实验结果。
