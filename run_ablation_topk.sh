#!/bin/bash

# 切换到脚本所在目录，确保可以正确找到 src/federated.py
cd "$(dirname "$0")"

# AlphaGuard Top-K Ablation Study Script
# 验证不同 Top-K 比例对防御效果的影响

# 基础参数设置
DATASET="cifar10"
ROUNDS=100  # 可以根据需要减少轮数以加快实验
ATTACK="badnet"
AGGR="alphaguard"
NUM_CORRUPT=2
GPU_ID=0  # 设置使用的GPU ID
BETA=0.5  # Non-IID 程度

# 创建日志目录
mkdir -p logs/ablation

echo "=================================================================="
echo "Start AlphaGuard Top-K Ablation Study"
echo "Dataset: $DATASET | Attack: $ATTACK | Corrupt: $NUM_CORRUPT | Non-IID: $BETA"
echo "=================================================================="

# 定义要测试的 Top-K 值列表
# 0.0001: 极度关注异常 (可能噪声大)
# 0.001:  默认值 (平衡)
# 0.01:   关注较多参数
# 0.1:    关注大量参数 (可能稀释攻击特征)
# 1.0:    使用所有参数 (相当于没有特征筛选)
TOPK_VALUES=(0.0001 0.001 0.01 0.1 1.0)

for TOPK in "${TOPK_VALUES[@]}"
do
    EXP_NAME="topk_${TOPK}_noniid_${BETA}"
    echo "Running experiment with Top-K = ${TOPK}..."
    
    python src/federated.py \
        --data $DATASET \
        --aggr $AGGR \
        --attack $ATTACK \
        --num_corrupt $NUM_CORRUPT \
        --rounds $ROUNDS \
        --device "cuda:$GPU_ID" \
        --exp_name_extra $EXP_NAME \
        --ag_topk $TOPK \
        --non_iid \
        --beta $BETA \
        > "logs/ablation/${DATASET}_${ATTACK}_${EXP_NAME}.log" 2>&1
        
    echo "Experiment ${EXP_NAME} finished. Log saved to logs/ablation/${DATASET}_${ATTACK}_${EXP_NAME}.log"
    echo "------------------------------------------------------------------"
done

echo "All experiments completed."

