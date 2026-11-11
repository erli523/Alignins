#!/bin/bash
# 动态触发器攻击测试脚本

echo "======================================"
echo "动态触发器后门攻击测试"
echo "======================================"

# 进入src目录
cd src || exit

# 测试1：动态触发器 + AlphaGuard防御
echo ""
echo "[测试1] 动态触发器 vs AlphaGuard 防御"
python federated.py \
  --data cifar10 \
  --num_agents 20 \
  --num_corrupt 4 \
  --poison_frac 0.3 \
  --attack dynamic_trigger \
  --aggr alphaguard \
  --non_iid \
  --beta 0.5 \
  --trigger_size 5 \
  --trigger_lr 0.1 \
  --trigger_outter_epochs 30 \
  --trigger_optim_interval 5 \
  --dm_adv_model_count 2

echo ""
echo "[测试完成] 查看日志了解详细结果"
echo "======================================"

