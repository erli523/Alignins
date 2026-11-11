# 服务器迁移验证（2026-11-11）

本机源码来源：D:/pycharm/AlignIns-master；部署到 /home/risegpu/sunjiahao/Alignins。

40 个迁移文件中，39 个与源文件 SHA256 完全一致，包括 aggregation.py、agent.py、federated.py、所有攻击和模型文件。唯一源码改动是 utils.py 中 FashionMNIST Resize 到 32×32；详见 SERVER_DATA_ADAPTATION.json。

原单次入口 federated.py --help 正常。完整 CIFAR10 训练集 50000 张、测试集 10000 张；ResNet9、20 客户端且全选、4 恶意、BadNets、本地投毒比例 0.3、non-IID beta=0.5、AlphaGuard，两轮、每轮一个完整本地 epoch，GPU0 实际运行正常结束。耗时 74.47 秒，退出码 0。两轮只验证功能，不能说明最终收敛或防御效果。

五个数据集均实际读取并执行两张图像的模型前向计算：

- cifar10: train=50000, test=10000, output=[2, 10]
- cifar100: train=50000, test=10000, output=[2, 100]
- mnist: train=60000, test=10000, output=[2, 10]
- fmnist: train=60000, test=10000, output=[2, 10]
- tinyimagenet: train=100000, test=10000, output=[2, 200]

环境：sunPytorch 的 torch 2.6.0+cu124 / torchvision 0.21.0；独立 .venv 内 numpy 1.26.4、geom-median 0.1.0。未修改原环境。

临时检查日志和该次原生 migration_check 输出目录已清理；保留源文件校验记录和验证摘要。其他攻击/防御组合未逐一训练验证。

## 根目录入口更新

新增根目录 federated.py，转发到 src/federated.py 并处理相对工作目录。在 sunPytorch 内补齐 geom-median 0.1.0 后，根目录 python federated.py --help 退出码为 0。未改动该环境 NumPy/PyTorch 版本；未在此次入口检查中重新训练模型。
