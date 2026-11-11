#!/bin/bash

# MNIST 快速开始脚本
# 用于快速测试 MNIST 数据集是否正常工作

echo "========================================="
echo "MNIST 数据集快速测试"
echo "========================================="

cd src

echo ""
echo "测试 1: MNIST + BadNet 攻击 + AlignIns 防御 (SimpleCNN, 200轮)"
echo "----------------------------------------"
python federated.py \
  --data mnist \
  --num_agents 10 \
  --num_corrupt 2 \
  --poison_frac 0.3 \
  --target_class 7 \
  --attack badnet \
  --aggr alignins \
  --rounds 200 \
  --local_ep 2

echo ""
echo "========================================="
echo "快速测试完成！"
echo "========================================="
echo ""
echo "其他测试示例："
echo "1. 使用 RFA 防御：--aggr rfa"
echo "2. 使用 avg（无防御）：--aggr avg"
echo "3. Non-IID 设置：添加 --non_iid --beta 0.5"
echo "4. 切换 GPU：添加 --device cuda:1"
echo ""

