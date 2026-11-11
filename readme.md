# Alignins：本机 AlphaGuard 实验代码

从本机 AlignIns-master 迁移到 GPU 服务器的实验项目，保留本机原有 AlphaGuard、攻击和训练实现。

## 单次运行

```bash
cd ~/sunjiahao/Alignins
conda activate sunPytorch
python federated.py --data cifar10 --model resnet9 --attack badnet --aggr alphaguard --num_agents 20 --num_corrupt 4 --poison_frac 0.3 --non_iid --beta 0.5 --rounds 100 --local_ep 2 --device cuda:0
```

这是一个实验。数据和环境需先按 [START_HERE.md](START_HERE.md) 配置，GitHub 不包含数据集、日志或虚拟环境。原 src 入口的相对工作目录由根目录入口自动处理。

- [运行说明及参数](START_HERE.md)
- [服务器目录和历史日志位置](docs/SERVER_LAYOUT.md)
- [脚本按用途分类](docs/SCRIPT_GUIDE.md)
- [服务器验证摘要](SERVER_CHECKS.md)
- [上游原说明](docs/UPSTREAM_README.md)

当前服务器使用 sunPytorch，额外依赖 geom-median 0.1.0；没有为当前 sunPytorch 降级 NumPy。requirements-server.txt 是先前独立环境的版本配方，不应直接用于覆盖已有共享环境。

原有 AlphaGuard 逻辑与之前其他框架的阈值版本不同，本仓库按本机版本保存。完整训练效果需通过正式实验验证。
