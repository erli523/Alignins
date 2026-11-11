#!/bin/bash

# MARS Defense Quick Start Script
# This script demonstrates the MARS defense method on CIFAR10 dataset

echo "========================================="
echo "MARS Defense Quick Start"
echo "========================================="
echo ""
echo "MARS (Model Agnostic Robust Subspace) Defense"
echo "- Uses Channel-wise Batch normalization Eigenvector (CBE) analysis"
echo "- Detects malicious clients via Wasserstein distance clustering"
echo "- Compatible with CNN and ResNet architectures"
echo ""

# Configuration
DATA="cifar10"
ATTACK="badnet"
AGGR="mars"
NUM_AGENTS=20
NUM_CORRUPT=4
POISON_FRAC=0.5
TARGET_CLASS=7
ROUNDS=100
LOCAL_EP=2
BS=64
CLIENT_LR=0.1
SERVER_LR=1.0

echo "Configuration:"
echo "  Dataset: $DATA"
echo "  Attack: $ATTACK"
echo "  Defense: $AGGR"
echo "  Total clients: $NUM_AGENTS"
echo "  Malicious clients: $NUM_CORRUPT"
echo "  Poison fraction: $POISON_FRAC"
echo "  Target class: $TARGET_CLASS"
echo ""

# Run training
echo "Starting training..."
python src/federated.py \
    --data $DATA \
    --attack $ATTACK \
    --aggr $AGGR \
    --num_agents $NUM_AGENTS \
    --num_corrupt $NUM_CORRUPT \
    --poison_frac $POISON_FRAC \
    --target_class $TARGET_CLASS \
    --rounds $ROUNDS \
    --local_ep $LOCAL_EP \
    --bs $BS \
    --client_lr $CLIENT_LR \
    --server_lr $SERVER_LR

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="







