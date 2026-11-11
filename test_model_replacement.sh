#!/bin/bash

# Model Replacement 攻击测试脚本
# 测试 Model Replacement (Constrain-and-Scale) 攻击在不同防御方法下的表现

echo "========================================="
echo "Model Replacement 攻击测试脚本"
echo "========================================="

# 基础配置
DATA="cifar10"
NUM_AGENTS=20
NUM_CORRUPT=4
POISON_FRAC=0.3
TARGET_CLASS=7

# Model Replacement 攻击参数
MR_SCALING_FACTOR=50
MR_ALPHA=0.5
MR_CONSTRAIN_WEIGHT=0.02

echo ""
echo "测试配置："
echo "  数据集: $DATA"
echo "  客户端数量: $NUM_AGENTS"
echo "  恶意客户端数量: $NUM_CORRUPT"
echo "  投毒比例: $POISON_FRAC"
echo "  目标类别: $TARGET_CLASS"
echo "  Scaling Factor (γ): $MR_SCALING_FACTOR"
echo "  Alpha (分类损失权重): $MR_ALPHA"
echo "  Constrain Weight: $MR_CONSTRAIN_WEIGHT"
echo ""

# 切换到 src 目录
cd src

echo "========================================="
echo "测试 1: Model Replacement vs 平均聚合 (无防御)"
echo "========================================="
python federated.py \
  --data $DATA \
  --num_agents $NUM_AGENTS \
  --num_corrupt $NUM_CORRUPT \
  --poison_frac $POISON_FRAC \
  --target_class $TARGET_CLASS \
  --attack model_replacement \
  --aggr avg \
  --model_replacement_scaling_factor $MR_SCALING_FACTOR \
  --model_replacement_alpha $MR_ALPHA \
  --model_replacement_constrain_weight $MR_CONSTRAIN_WEIGHT

echo ""
echo "========================================="
echo "测试 2: Model Replacement vs AlignIns 防御"
echo "========================================="
python federated.py \
  --data $DATA \
  --num_agents $NUM_AGENTS \
  --num_corrupt $NUM_CORRUPT \
  --poison_frac $POISON_FRAC \
  --target_class $TARGET_CLASS \
  --attack model_replacement \
  --aggr alignins \
  --model_replacement_scaling_factor $MR_SCALING_FACTOR \
  --model_replacement_alpha $MR_ALPHA \
  --model_replacement_constrain_weight $MR_CONSTRAIN_WEIGHT

echo ""
echo "========================================="
echo "测试 3: Model Replacement vs RLR 防御"
echo "========================================="
python federated.py \
  --data $DATA \
  --num_agents $NUM_AGENTS \
  --num_corrupt $NUM_CORRUPT \
  --poison_frac $POISON_FRAC \
  --target_class $TARGET_CLASS \
  --attack model_replacement \
  --aggr rlr \
  --model_replacement_scaling_factor $MR_SCALING_FACTOR \
  --model_replacement_alpha $MR_ALPHA \
  --model_replacement_constrain_weight $MR_CONSTRAIN_WEIGHT

echo ""
echo "========================================="
echo "测试 4: 不同 Scaling Factor 的影响"
echo "========================================="
for SCALE in 20 50 100 200; do
  echo "测试 Scaling Factor = $SCALE"
  python federated.py \
    --data $DATA \
    --num_agents $NUM_AGENTS \
    --num_corrupt $NUM_CORRUPT \
    --poison_frac $POISON_FRAC \
    --target_class $TARGET_CLASS \
    --attack model_replacement \
    --aggr avg \
    --model_replacement_scaling_factor $SCALE \
    --model_replacement_alpha $MR_ALPHA \
    --model_replacement_constrain_weight $MR_CONSTRAIN_WEIGHT
  echo ""
done

echo ""
echo "========================================="
echo "测试 5: 不同 Alpha 权重的影响"
echo "========================================="
for ALPHA in 0.2 0.5 0.8; do
  echo "测试 Alpha = $ALPHA"
  python federated.py \
    --data $DATA \
    --num_agents $NUM_AGENTS \
    --num_corrupt $NUM_CORRUPT \
    --poison_frac $POISON_FRAC \
    --target_class $TARGET_CLASS \
    --attack model_replacement \
    --aggr avg \
    --model_replacement_scaling_factor $MR_SCALING_FACTOR \
    --model_replacement_alpha $ALPHA \
    --model_replacement_constrain_weight $MR_CONSTRAIN_WEIGHT
  echo ""
done

echo ""
echo "========================================="
echo "测试 6: Non-IID 场景"
echo "========================================="
python federated.py \
  --data $DATA \
  --num_agents $NUM_AGENTS \
  --num_corrupt $NUM_CORRUPT \
  --poison_frac $POISON_FRAC \
  --target_class $TARGET_CLASS \
  --attack model_replacement \
  --aggr alignins \
  --non_iid \
  --beta 0.5 \
  --model_replacement_scaling_factor $MR_SCALING_FACTOR \
  --model_replacement_alpha $MR_ALPHA \
  --model_replacement_constrain_weight $MR_CONSTRAIN_WEIGHT

echo ""
echo "========================================="
echo "所有测试完成！"
echo "========================================="

