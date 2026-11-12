"""
AlterMin Attack (Alternating Minimization Attack)
交替最小化攻击 - 来自论文: Analyzing Federated Learning Through an Adversarial Lens (ICML '19)

该攻击通过交替优化两个目标来实现隐蔽的后门攻击：
1. 隐蔽性目标 (Stealth Objective): 在良性数据上训练，同时最小化与全局模型的距离
2. 攻击目标 (Attack Objective): 在恶意数据上训练，提高后门攻击成功率
3. Boosting: 放大恶意更新，增强攻击效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import logging
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class AlterMinAttacker:
    """
    AlterMin 攻击器
    
    攻击流程：
    1. 良性训练：在干净数据上训练，同时最小化与全局模型的距离（隐蔽性）
    2. 恶意训练：在投毒数据上训练，最大化攻击成功率
    3. Boosting：放大恶意更新
    """
    
    def __init__(self, args, worker_id):
        """
        初始化 AlterMin 攻击器
        
        Args:
            args: 参数对象，包含攻击配置
            worker_id: 客户端 ID
        """
        self.args = args
        self.worker_id = worker_id
        self.device = args.device
        
        # 攻击参数（从 args 中读取，如果不存在则使用默认值）
        self.attack_mode = getattr(args, 'altermin_attack_mode', 'targeted')  # 攻击模式
        self.source_label = getattr(args, 'altermin_source_label', 3)  # 源标签
        
        # 目标标签：如果显式设置则使用，否则使用 args.target_class
        target_label_arg = getattr(args, 'altermin_target_label', None)
        self.target_label = target_label_arg if target_label_arg is not None else args.target_class
        
        self.poisoned_sample_cnt = getattr(args, 'altermin_poisoned_cnt', 100)  # 投毒样本数量
        self.boosting_factor = getattr(args, 'altermin_boosting_factor', 1.5)  # Boosting 因子
        self.rho = getattr(args, 'altermin_rho', 1e-4)  # 距离损失权重
        
        # 良性训练轮数：如果显式设置则使用，否则使用 args.local_ep
        benign_epochs_arg = getattr(args, 'altermin_benign_epochs', None)
        self.benign_epochs = benign_epochs_arg if benign_epochs_arg is not None else args.local_ep
        
        self.malicious_epochs = getattr(args, 'altermin_malicious_epochs', 1)  # 恶意训练轮数
        
        # 全局模型权重向量（用于计算距离损失）
        self.global_weights_vec = None
        
        # 投毒数据加载器
        self.poisoned_loader = None
        
        logging.info(f"AlterMinAttacker initialized for client {worker_id}")
        logging.info(f"  Attack mode: {self.attack_mode}")
        logging.info(f"  Source label: {self.source_label} -> Target label: {self.target_label}")
        logging.info(f"  Boosting factor: {self.boosting_factor}, Rho: {self.rho}")
        logging.info(f"  Benign epochs: {self.benign_epochs}, Malicious epochs: {self.malicious_epochs}")
    
    def init_poisoned_loader(self, test_dataset):
        """
        初始化投毒数据加载器
        从测试集中采样一小部分数据作为投毒数据
        
        Args:
            test_dataset: 测试数据集
        """
        # 从测试集中选择源标签的样本
        if self.attack_mode == 'targeted':
            # 有目标攻击：只选择特定源标签的样本
            source_indices = [i for i, (_, label) in enumerate(test_dataset) 
                            if label == self.source_label]
        else:
            # 无目标攻击：从所有样本中选择
            source_indices = list(range(len(test_dataset)))
        
        # 🔧 FIX: 基于 worker_id 和 poisoned_sample_cnt 生成唯一的随机种子
        # 这样不同的 client 和不同的 poisoning_frac 会有不同的投毒数据
        seed = hash((self.worker_id, self.poisoned_sample_cnt, self.target_label)) % (2**32)
        rng = np.random.RandomState(seed)
        
        # 随机采样
        if len(source_indices) > self.poisoned_sample_cnt:
            selected_indices = rng.choice(source_indices, self.poisoned_sample_cnt, replace=False)
        else:
            selected_indices = source_indices
        
        # 创建投毒数据集（标签翻转）
        poisoned_data = []
        for idx in selected_indices:
            img, _ = test_dataset[idx]
            poisoned_data.append((img, self.target_label))
        
        # 创建 DataLoader
        from torch.utils.data import TensorDataset, DataLoader
        images = torch.stack([x[0] for x in poisoned_data])
        labels = torch.tensor([x[1] for x in poisoned_data], dtype=torch.long)
        poisoned_dataset = TensorDataset(images, labels)
        
        self.poisoned_loader = DataLoader(
            poisoned_dataset, 
            batch_size=self.args.bs, 
            shuffle=True, 
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        logging.info(f"  Initialized poisoned loader with {len(poisoned_data)} samples")
    
    def save_global_weights(self, global_model):
        """
        保存全局模型权重向量（用于计算距离损失）
        
        Args:
            global_model: 全局模型
        """
        self.global_weights_vec = parameters_to_vector(global_model.parameters()).detach().clone()
    
    def stealth_loss(self, model, outputs, labels, criterion):
        """
        隐蔽性损失 = 标准交叉熵损失 + 距离损失
        
        Args:
            model: 当前模型
            outputs: 模型输出
            labels: 标签
            criterion: 损失函数
        
        Returns:
            总损失
        """
        # 标准交叉熵损失
        ce_loss = criterion(outputs, labels)
        
        # 距离损失：当前模型与全局模型的 L2 距离
        if self.global_weights_vec is not None:
            current_weights_vec = parameters_to_vector(model.parameters())
            distance_loss = torch.norm(current_weights_vec - self.global_weights_vec, p=2)
            total_loss = ce_loss + self.rho * distance_loss
        else:
            total_loss = ce_loss
        
        return total_loss
    
    def benign_train(self, model, train_loader, criterion, optimizer, epochs=None):
        """
        良性训练：在干净数据上训练，同时保持与全局模型的接近性（隐蔽性目标）
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            criterion: 损失函数
            optimizer: 优化器
            epochs: 训练轮数（如果为 None 则使用 self.benign_epochs）
        
        Returns:
            平均损失
        """
        if epochs is None:
            epochs = self.benign_epochs
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # 使用隐蔽性损失
                loss = self.stealth_loss(model, outputs, labels, criterion)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def malicious_train(self, model, criterion, optimizer, epochs=None):
        """
        恶意训练：在投毒数据上训练（攻击目标）
        
        Args:
            model: 模型
            criterion: 损失函数
            optimizer: 优化器
            epochs: 训练轮数（如果为 None 则使用 self.malicious_epochs）
        
        Returns:
            平均损失
        """
        if epochs is None:
            epochs = self.malicious_epochs
        
        if self.poisoned_loader is None:
            logging.warning("Poisoned loader not initialized, skipping malicious training")
            return 0.0
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for inputs, labels in self.poisoned_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # 只使用标准交叉熵损失（不加距离损失）
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate_asr(self, model):
        """
        验证攻击成功率 (ASR)
        
        Args:
            model: 模型
        
        Returns:
            asr: 攻击成功率
            loss: 平均损失
        """
        if self.poisoned_loader is None:
            return 0.0, 0.0
        
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in self.poisoned_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        asr = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.poisoned_loader) if len(self.poisoned_loader) > 0 else 0.0
        
        return asr, avg_loss
    
    def apply_boosting(self, model, pre_train_weights_vec):
        """
        应用 Boosting 增强恶意更新
        
        新权重 = 旧权重 + boosting_factor * (新权重 - 旧权重)
        
        Args:
            model: 当前模型
            pre_train_weights_vec: 训练前的权重向量
        """
        post_train_weights_vec = parameters_to_vector(model.parameters()).detach().clone()
        
        # 计算增强后的权重
        boosted_weights_vec = pre_train_weights_vec + self.boosting_factor * (post_train_weights_vec - pre_train_weights_vec)
        
        # 加载回模型
        vector_to_parameters(boosted_weights_vec, model.parameters())
        
        logging.info(f"  Applied boosting with factor {self.boosting_factor}")
    
    def altermin_train(self, model, train_loader, criterion, optimizer, round_num=0):
        """
        AlterMin 攻击的核心训练流程
        
        流程：
        1. 良性训练（隐蔽性目标）
        2. 验证 ASR
        3. 如果 ASR 未达到 100%，进行恶意训练（攻击目标）
        4. 应用 Boosting 增强
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            criterion: 损失函数
            optimizer: 优化器
            round_num: 当前轮次
        
        Returns:
            benign_loss: 良性训练损失
            asr: 攻击成功率
        """
        logging.info(f"[Round {round_num}] Client {self.worker_id} starting AlterMin training...")
        
        # 步骤 1: 良性训练（隐蔽性目标）
        benign_loss = self.benign_train(model, train_loader, criterion, optimizer)
        logging.info(f"  Benign training completed, loss: {benign_loss:.4f}")
        
        # 步骤 2: 验证当前 ASR
        asr, asr_loss = self.validate_asr(model)
        logging.info(f"  Current ASR: {asr:.4f}, ASR Loss: {asr_loss:.4f}")
        
        # 步骤 3 & 4: 如果 ASR < 1.0，进行恶意训练并应用 Boosting
        if asr_loss > 0.0 and asr < 1.0:
            # 保存训练前的权重
            pre_train_weights_vec = parameters_to_vector(model.parameters()).detach().clone()
            
            # 恶意训练
            mal_loss = self.malicious_train(model, criterion, optimizer)
            logging.info(f"  Malicious training completed, loss: {mal_loss:.4f}")
            
            # 应用 Boosting
            self.apply_boosting(model, pre_train_weights_vec)
            
            # 再次验证 ASR
            final_asr, final_asr_loss = self.validate_asr(model)
            logging.info(f"  After boosting - ASR: {final_asr:.4f}, ASR Loss: {final_asr_loss:.4f}")
        else:
            logging.info(f"  ASR already high ({asr:.4f}), skipping malicious training")
        
        return benign_loss, asr

