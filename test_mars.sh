#!/bin/bash

echo "========================================="
echo "Testing MARS Defense on MNIST"
echo "========================================="

# Test 1: MARS + BadNet attack on MNIST
echo ""
echo "Test 1: MARS defense against BadNet attack on MNIST"
python src/federated.py \
    --data mnist \
    --attack badnet \
    --aggr mars \
    --num_agents 20 \
    --num_corrupt 4 \
    --poison_frac 0.5 \
    --target_class 7 \
    --rounds 50 \
    --local_ep 2 \
    --bs 64 \
    --client_lr 0.1 \
    --server_lr 1.0

echo ""
echo "Test 1 completed!"

# Test 2: MARS + DBA attack on MNIST
echo ""
echo "Test 2: MARS defense against DBA attack on MNIST"
python src/federated.py \
    --data mnist \
    --attack DBA \
    --aggr mars \
    --num_agents 20 \
    --num_corrupt 4 \
    --poison_frac 0.5 \
    --target_class 7 \
    --rounds 50 \
    --local_ep 2 \
    --bs 64 \
    --client_lr 0.1 \
    --server_lr 1.0

echo ""
echo "Test 2 completed!"

echo ""
echo "========================================="
echo "All MARS tests completed!"
echo "========================================="







