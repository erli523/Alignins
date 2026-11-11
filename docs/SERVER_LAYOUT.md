# 服务器文件说明（2026-11-11）

工作根目录：`/home/risegpu/sunjiahao`。当前使用的是本机迁移的 **Alignins**，不是 `AlignIns-AlphaGuard` 官方仓库适配副本。

## 目录按用途划分

```text
/home/risegpu/sunjiahao/
├── Alignins/                         当前实验项目，保持原位置
├── frameworks/                       对照框架与旧版本
│   ├── alignins/AlignIns-AlphaGuard/
│   ├── backfed/BackFed-main/
│   ├── backfed/BackFed-AlphaGuard/
│   ├── flpoison/AlphaGuard-FLPoison/
│   ├── flpoison/FLPoison-AlphaGuard/
│   ├── lasa/LASA-AlphaGuard/
│   └── masa/MASA-AlphaGuard/
├── records/
│   ├── history/Alignins-local/        本机历史 logs，原结构完整保留
│   └── current/Alignins -> ../../Alignins/src/logs
├── datasets/                         数据集快捷入口，不是重复副本
├── inventory/                        导入校验清单与目录迁移记录
└── SERVER_FILES.md                   本说明
```

根目录仍保留原七个框架名称的相对软链接。这些是兼容旧命令、venv 绝对路径和数据引用的快捷入口，没有复制第二份框架。正在运行的 Alignins 目录没有移动、实验没有停止。

## 当前项目

| 内容 | 位置 |
|---|---|
| 项目根目录 | `/home/risegpu/sunjiahao/Alignins` |
| 单次运行入口 | `Alignins/federated.py`，自动转入 src 执行原程序 |
| 原训练入口与参数 | `Alignins/src/federated.py` |
| AlphaGuard 与其他防御 | `Alignins/src/aggregation.py` |
| 客户端训练 | `Alignins/src/agent.py`、`agent_sparse.py` |
| 攻击实现 | `Alignins/src/agent.py`、`utils.py` 和三个 `*_attack.py` 文件 |
| 模型 | `Alignins/src/models.py`、`resnet.py`、`vgg.py` |
| 数据预处理 | `Alignins/src/utils.py` |
| 使用说明 | `Alignins/START_HERE.md` |
| 脚本用途索引 | `Alignins/docs/SCRIPT_GUIDE.md` |
| 迁移验证 | `Alignins/SERVER_CHECKS.md` |

```bash
cd ~/sunjiahao/Alignins
conda activate sunPytorch
python federated.py --data cifar10 --model resnet9 --attack badnet --aggr alphaguard --num_agents 20 --num_corrupt 4 --poison_frac 0.3 --non_iid --beta 0.5 --rounds 100 --local_ep 2 --device cuda:0
```

以上只运行一个实验。批量 sweep/ablation 脚本不是单次入口。

## 历史与当前日志

- 本机来源：`D:/pycharm/AlignIns-master/src/logs`。
- 历史日志目的地：`/home/risegpu/sunjiahao/records/history/Alignins-local`。
- 历史记录共 **2962 个文件、278,483,336 字节**，包含 **673 个 .log** 与 **2289 个运行时源码备份**，逐文件 SHA256 与大小全部核对一致。
- 服务器当前运行日志仍写入 `Alignins/src/logs`，也可以从 `records/current/Alignins` 查看。
- 两类记录分别保存，历史导入没有覆盖当前实验日志。
- 每次历史实验中的 `backup_file/` 是当时的源码版本，保留用于回溯，不是当前运行代码。

| 数据集 | 本机历史 .log 数 |
|---|---:|
| cifar10 | 342 |
| cifar100 | 121 |
| fmnist | 141 |
| mnist | 69 |

历史目录内部继续按原来的 `数据集/attack_攻击_ar_比例/defense_防御/时间及实验信息/` 保存。无需解压即可查看。根目录 `datasets/` 下的数据链接不属于实验日志。

## 对照与旧框架

| 真实目录（相对工作根目录） | 用途 |
|---|---|
| `frameworks/alignins/AlignIns-AlphaGuard` | 官方 AlignIns 新下载版本的适配副本，区别于本机旧实现 |
| `frameworks/masa/MASA-AlphaGuard` | MASA 原生攻击/防御与 AlphaGuard 适配 |
| `frameworks/lasa/LASA-AlphaGuard` | LASA 原生拜占庭攻击与 AlphaGuard 适配 |
| `frameworks/backfed/BackFed-main` | 历史 BackFed 主项目，也是已有 CIFAR/TinyImageNet 数据源 |
| `frameworks/backfed/BackFed-AlphaGuard` | BackFed AlphaGuard 适配实验 |
| `frameworks/flpoison/AlphaGuard-FLPoison` | 早期完整 FLPoison 改造版本与历史结果 |
| `frameworks/flpoison/FLPoison-AlphaGuard` | 后续较小改动的 FLPoison 适配版本 |

这些项目自己的 logs/results/checkpoints 留在对应项目内，没有混入本次本机历史导入。

## 数据集与校验记录

`datasets/CIFAR10`、`CIFAR100`、`TinyImageNet` 指向 BackFed-main 的已有数据；`datasets/MNIST`、`FashionMNIST` 指向 AlphaGuard-FLPoison 的已有数据。`Alignins/src/data` 原有软链接仍有效。此次没有再复制或移动数据集内容。

| 记录 | 位置 |
|---|---|
| 本机历史导入的逐文件 SHA256 | `inventory/history_import_20261111.json` |
| 框架移动前后位置 | `inventory/directory_moves_20261111.json` |
| 本机迁移的原代码校验 | `Alignins/LOCAL_SOURCE_SHA256.json` |
| FashionMNIST 数据适配差异 | `Alignins/SERVER_DATA_ADAPTATION.json` |

## GitHub 私有备份

本次只备份当前 **Alignins** 的代码、脚本、配置和说明。数据集、历史/当前运行日志、checkpoint、虚拟环境、缓存及本机编辑器历史不进入仓库。发布结果会写入 `Alignins/docs/GITHUB_BACKUP.md`。
