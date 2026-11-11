#!/bin/bash

# AlterMin 攻击快速开始脚本
# 用于快速测试 AlterMin 攻击是否正常工作

echo "========================================="
echo "AlterMin 攻击快速开始测试"
echo "========================================="

cd src

echo ""
echo "测试 1: 基础功能测试（简化版，快速验证）"
echo "----------------------------------------"
python federated.py \
  --data cifar10 \
  --num_agents 10 \
  --num_corrupt 2 \
  --poison_frac 0.3 \
  --target_class 7 \
  --attack altermin \
  --aggr avg \
  --rounds 20 \
  --local_ep 2 \
  --altermin_source_label 3 \
  --altermin_boosting_factor 1.5 \
  --altermin_rho 1e-4 \
  --altermin_poisoned_cnt 50

echo ""
echo "========================================="
echo "快速测试完成！"
echo "========================================="
echo ""
echo "如果测试成功，您可以："
echo "1. 运行完整测试：bash test_altermin.sh"
echo "2. 查看详细文档：ALTERMIN_ATTACK_README.md"
echo "3. 自定义参数进行实验"
echo ""

