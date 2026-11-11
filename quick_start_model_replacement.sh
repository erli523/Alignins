#!/bin/bash

# Model Replacement 攻击快速开始脚本
# 用于快速测试 Model Replacement 攻击是否正常工作

echo "========================================="
echo "Model Replacement 攻击快速开始测试"
echo "========================================="

cd src

echo ""
echo "测试: 基础功能测试（简化版，快速验证）"
echo "----------------------------------------"
python federated.py \
  --data cifar10 \
  --num_agents 10 \
  --num_corrupt 1 \
  --poison_frac 0.3 \
  --target_class 7 \
  --attack model_replacement \
  --aggr avg \
  --rounds 20 \
  --local_ep 2 \
  --model_replacement_scaling_factor 50 \
  --model_replacement_alpha 0.5 \
  --model_replacement_constrain_weight 0.02

echo ""
echo "========================================="
echo "快速测试完成！"
echo "========================================="
echo ""
echo "如果测试成功，您可以："
echo "1. 运行完整测试：bash test_model_replacement.sh"
echo "2. 查看详细文档：MODEL_REPLACEMENT_ATTACK_README.md"
echo "3. 自定义参数进行实验"
echo "4. 对比不同攻击方法（AlterMin, BadNet 等）"
echo ""

