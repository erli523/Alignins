#!/bin/bash

# AlterMin 攻击测试脚本
# 测试 AlterMin 攻击在不同防御方法下的表现

echo "========================================="
echo "AlterMin 攻击测试脚本"
echo "========================================="

# 基础配置
DATA="cifar10"
NUM_AGENTS=20
NUM_CORRUPT=4
POISON_FRAC=0.3
TARGET_CLASS=7

# AlterMin 攻击参数
ALTERMIN_SOURCE_LABEL=3
ALTERMIN_BOOSTING_FACTOR=1.5
ALTERMIN_RHO=1e-4
ALTERMIN_POISONED_CNT=100
ALTERMIN_MALICIOUS_EPOCHS=1

echo ""
echo "测试配置："
echo "  数据集: $DATA"
echo "  客户端数量: $NUM_AGENTS"
echo "  恶意客户端数量: $NUM_CORRUPT"
echo "  投毒比例: $POISON_FRAC"
echo "  目标类别: $TARGET_CLASS"
echo "  源标签: $ALTERMIN_SOURCE_LABEL"
echo "  Boosting 因子: $ALTERMIN_BOOSTING_FACTOR"
echo ""

# 切换到 src 目录
cd src

echo "========================================="
echo "测试 1: AlterMin vs 平均聚合 (无防御)"
echo "========================================="
python federated.py \
  --data $DATA \
  --num_agents $NUM_AGENTS \
  --num_corrupt $NUM_CORRUPT \
  --poison_frac $POISON_FRAC \
  --target_class $TARGET_CLASS \
  --attack altermin \
  --aggr avg \
  --altermin_source_label $ALTERMIN_SOURCE_LABEL \
  --altermin_boosting_factor $ALTERMIN_BOOSTING_FACTOR \
  --altermin_rho $ALTERMIN_RHO \
  --altermin_poisoned_cnt $ALTERMIN_POISONED_CNT \
  --altermin_malicious_epochs $ALTERMIN_MALICIOUS_EPOCHS

echo ""
echo "========================================="
echo "测试 2: AlterMin vs AlignIns 防御"
echo "========================================="
python federated.py \
  --data $DATA \
  --num_agents $NUM_AGENTS \
  --num_corrupt $NUM_CORRUPT \
  --poison_frac $POISON_FRAC \
  --target_class $TARGET_CLASS \
  --attack altermin \
  --aggr alignins \
  --altermin_source_label $ALTERMIN_SOURCE_LABEL \
  --altermin_boosting_factor $ALTERMIN_BOOSTING_FACTOR \
  --altermin_rho $ALTERMIN_RHO \
  --altermin_poisoned_cnt $ALTERMIN_POISONED_CNT \
  --altermin_malicious_epochs $ALTERMIN_MALICIOUS_EPOCHS

echo ""
echo "========================================="
echo "测试 3: AlterMin vs RLR 防御"
echo "========================================="
python federated.py \
  --data $DATA \
  --num_agents $NUM_AGENTS \
  --num_corrupt $NUM_CORRUPT \
  --poison_frac $POISON_FRAC \
  --target_class $TARGET_CLASS \
  --attack altermin \
  --aggr rlr \
  --altermin_source_label $ALTERMIN_SOURCE_LABEL \
  --altermin_boosting_factor $ALTERMIN_BOOSTING_FACTOR \
  --altermin_rho $ALTERMIN_RHO \
  --altermin_poisoned_cnt $ALTERMIN_POISONED_CNT \
  --altermin_malicious_epochs $ALTERMIN_MALICIOUS_EPOCHS

echo ""
echo "========================================="
echo "测试 4: AlterMin vs Multi-Krum 防御"
echo "========================================="
python federated.py \
  --data $DATA \
  --num_agents $NUM_AGENTS \
  --num_corrupt $NUM_CORRUPT \
  --poison_frac $POISON_FRAC \
  --target_class $TARGET_CLASS \
  --attack altermin \
  --aggr mkrum \
  --altermin_source_label $ALTERMIN_SOURCE_LABEL \
  --altermin_boosting_factor $ALTERMIN_BOOSTING_FACTOR \
  --altermin_rho $ALTERMIN_RHO \
  --altermin_poisoned_cnt $ALTERMIN_POISONED_CNT \
  --altermin_malicious_epochs $ALTERMIN_MALICIOUS_EPOCHS

echo ""
echo "========================================="
echo "测试 5: AlterMin (Non-IID) vs AlignIns"
echo "========================================="
python federated.py \
  --data $DATA \
  --num_agents $NUM_AGENTS \
  --num_corrupt $NUM_CORRUPT \
  --poison_frac $POISON_FRAC \
  --target_class $TARGET_CLASS \
  --attack altermin \
  --aggr alignins \
  --non_iid \
  --beta 0.5 \
  --altermin_source_label $ALTERMIN_SOURCE_LABEL \
  --altermin_boosting_factor $ALTERMIN_BOOSTING_FACTOR \
  --altermin_rho $ALTERMIN_RHO \
  --altermin_poisoned_cnt $ALTERMIN_POISONED_CNT \
  --altermin_malicious_epochs $ALTERMIN_MALICIOUS_EPOCHS

echo ""
echo "========================================="
echo "测试 6: 不同 Boosting 因子的影响"
echo "========================================="
for BOOST in 1.0 1.5 2.0 3.0; do
  echo "测试 Boosting 因子 = $BOOST"
  python federated.py \
    --data $DATA \
    --num_agents $NUM_AGENTS \
    --num_corrupt $NUM_CORRUPT \
    --poison_frac $POISON_FRAC \
    --target_class $TARGET_CLASS \
    --attack altermin \
    --aggr avg \
    --altermin_source_label $ALTERMIN_SOURCE_LABEL \
    --altermin_boosting_factor $BOOST \
    --altermin_rho $ALTERMIN_RHO \
    --altermin_poisoned_cnt $ALTERMIN_POISONED_CNT \
    --altermin_malicious_epochs $ALTERMIN_MALICIOUS_EPOCHS
  echo ""
done

echo ""
echo "========================================="
echo "所有测试完成！"
echo "========================================="

