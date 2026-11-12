"""
Model Replacement Attack (Constrain-and-Scale Attack)
模型替换攻击 - 来自论文: How to Backdoor Federated Learning (AISTATS '20)

该攻击通过两个关键策略实现隐蔽且有效的后门攻击：
1. Constrain（约束）: 使用余弦距离损失保持与全局模型的接近性
2. Scale（放大）: 将恶意更新放大 γ 倍，主导全局模型聚合
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class ModelReplacementAttacker:
    """
    Model Replacement 攻击器
    
    攻击流程：
    1. Constrain 训练：使用特殊损失函数训练
       Loss = α × CE_Loss + (1-α) × Cosine_Distance_Loss
    2. Scale 放大：将模型更新放大 γ 倍
       Scaled_Update = G_t + γ × (Update - G_t)
    """
    
    def __init__(self, args, worker_id):
        """
        初始化 Model Replacement 攻击器
        
        Args:
            args: 参数对象，包含攻击配置
            worker_id: 客户端 ID
        """
        self.args = args
        self.worker_id = worker_id
        self.device = args.device
        
        # 攻击参数（从 args 中读取，如果不存在则使用默认值）
        self.scaling_factor = getattr(args, 'model_replacement_scaling_factor', 50)  # 放大因子 γ
        self.alpha = getattr(args, 'model_replacement_alpha', 0.5)  # 分类损失权重
        
        # 目标标签：如果显式设置则使用，否则使用 args.target_class
        target_label_arg = getattr(args, 'model_replacement_target_label', None)
        self.target_label = target_label_arg if target_label_arg is not None else args.target_class
        
        self.constrain_weight = getattr(args, 'model_replacement_constrain_weight', 0.1)  # 最终缩放权重
        
        # 全局模型权重向量（用于计算余弦距离）
        self.global_weights_vec = None
        
        logging.info(f"ModelReplacementAttacker initialized for client {worker_id}")
        logging.info(f"  Scaling factor (γ): {self.scaling_factor}")
        logging.info(f"  Alpha (classification weight): {self.alpha}")
        logging.info(f"  Target label: {self.target_label}")
        logging.info(f"  Constrain weight: {self.constrain_weight}")
    
    def save_global_weights(self, global_model):
        """
        保存全局模型权重向量（用于计算余弦距离）
        
        Args:
            global_model: 全局模型
        """
        # 使用 state_dict() 而不是 parameters() 以保持与 agent.py 的一致性
        self.global_weights_vec = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]
        ).detach().clone()
    
    def constrain_loss(self, model, outputs, labels, criterion):
        """
        Constrain 损失函数 = α × 分类损失 + (1-α) × 余弦距离损失
        
        注意：余弦距离的量级与 CE 损失差异很大，需要调整权重
        
        Args:
            model: 当前模型
            outputs: 模型输出
            labels: 标签
            criterion: 基础损失函数
        
        Returns:
            总损失
        """
        # 分类损失
        ce_loss = criterion(outputs, labels)
        
        # 余弦距离损失：当前模型与全局模型的余弦距离
        if self.global_weights_vec is not None:
            # 获取当前模型权重（保留梯度！）
            # 使用 state_dict() 而不是 parameters() 以保持与 agent.py 的一致性
            current_weights_vec = parameters_to_vector(
                [model.state_dict()[name] for name in model.state_dict()]
            )
            
            # 计算余弦相似度（使用 PyTorch，保留梯度）
            cos_sim = nn.functional.cosine_similarity(
                current_weights_vec.unsqueeze(0),
                self.global_weights_vec.unsqueeze(0),
                dim=1
            )
            
            # 余弦距离 = 1 - 余弦相似度（值域 [0, 2]）
            cosine_dist = 1.0 - cos_sim
            
            # 组合损失
            # 注意：余弦距离通常在 0-0.5 之间，CE loss 在 0-3 之间
            # 需要调整权重使两者量级相当
            total_loss = self.alpha * ce_loss + (1 - self.alpha) * cosine_dist.squeeze()
        else:
            total_loss = ce_loss
        
        return total_loss
    
    def constrain_train(self, model, train_loader, criterion, optimizer, epochs=None):
        """
        Constrain 训练：使用余弦距离约束的训练
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            criterion: 损失函数
            optimizer: 优化器
            epochs: 训练轮数
        
        Returns:
            平均损失
        """
        if epochs is None:
            epochs = self.args.local_ep
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # 使用 Constrain 损失
                loss = self.constrain_loss(model, outputs, labels, criterion)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def scale_update(self, model, initial_weights_vec):
        """
        Scale（放大）模型更新
        
        最终更新 = constrain_weight × scaling_factor × (X - G_t)
        
        Args:
            model: 训练后的模型
            initial_weights_vec: 训练前的全局模型权重（G_t）
        
        Returns:
            放大后的更新向量
        """
        # 当前模型权重 X
        # 使用 state_dict() 而不是 parameters() 以保持与 agent.py 的一致性
        current_weights_vec = parameters_to_vector(
            [model.state_dict()[name] for name in model.state_dict()]
        ).detach().clone()
        
        # 计算原始更新 Update = X - G_t
        update = current_weights_vec - initial_weights_vec
        
        # 计算有效放大因子
        # effective_scaling = constrain_weight × scaling_factor
        effective_scaling = self.constrain_weight * self.scaling_factor
        
        # 放大更新
        scaled_update = effective_scaling * update
        
        logging.info(f"  Effective scaling factor: {effective_scaling:.2f} (constrain_weight × scaling_factor)")
        
        return scaled_update
    
    def model_replacement_train(self, model, train_loader, criterion, optimizer, round_num=0):
        """
        Model Replacement 攻击的核心训练流程
        
        流程：
        1. Constrain 训练：使用余弦距离约束
        2. Scale 放大：放大模型更新
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            criterion: 损失函数
            optimizer: 优化器
            round_num: 当前轮次
        
        Returns:
            训练损失
        """
        logging.info(f"[Round {round_num}] Client {self.worker_id} starting Model Replacement training...")
        
        # 保存初始权重（用于后续放大）
        # 使用 state_dict() 而不是 parameters() 以保持与 agent.py 的一致性
        initial_weights_vec = parameters_to_vector(
            [model.state_dict()[name] for name in model.state_dict()]
        ).detach().clone()
        
        # 步骤 1: Constrain 训练
        constrain_loss = self.constrain_train(model, train_loader, criterion, optimizer)
        logging.info(f"  Constrain training completed, loss: {constrain_loss:.4f}")
        
        # 步骤 2: Scale 放大更新
        scaled_update = self.scale_update(model, initial_weights_vec)
        
        # 计算放大比例
        original_update = parameters_to_vector(
            [model.state_dict()[name] for name in model.state_dict()]
        ) - initial_weights_vec
        scale_ratio = torch.norm(scaled_update) / (torch.norm(original_update) + 1e-8)
        logging.info(f"  Update scaled by factor: {scale_ratio:.2f}")
        
        # 将放大后的更新加载回模型
        # 注意：这里我们需要将 scaled_update 转换为模型更新的形式
        # scaled_update 本身就是新的权重向量（对于 FedAvg）或更新向量（对于 FedOpt）
        
        return constrain_loss, scaled_update

