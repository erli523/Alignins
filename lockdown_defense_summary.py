"""
===============================================================================
Lockdown 防御方法完整代码整理
===============================================================================

论文来源: Lockdown: Backdoor Defense for Federated Learning with Isolated Subspace Training
GitHub: https://github.com/git-disl/Lockdown

===============================================================================
Lockdown 防御方法的核心思路与思想总结
===============================================================================

1. **核心理念 - 孤立子空间训练 (Isolated Subspace Training)**
   
   Lockdown的核心思想是通过稀疏训练（Sparse Training）和动态掩码机制，
   让每个客户端在"孤立的参数子空间"中进行训练，从而限制恶意客户端的影响范围。
   
   主要包含以下几个关键机制：
   
   a) **稀疏掩码初始化 (Sparse Mask Initialization)**
      - 使用 ERK (Erdős-Rényi Kernel) 分布初始化参数的稀疏性
      - 每个客户端只激活模型的一部分参数（由掩码mask控制）
      - 不同层根据其形状动态分配不同的稀疏度
   
   b) **动态掩码更新 (Dynamic Mask Update)**
      - Fire操作：每轮训练后，丢弃权重绝对值最小的参数
      - Regrow操作：根据梯度信息，激活具有最大梯度的新参数
      - 通过余弦退火策略逐渐减少更新的参数数量
   
   c) **梯度筛选机制 (Gradient Screening)**
      - 在掩码更新前，先对本地数据进行梯度采样
      - 选择梯度绝对值最大的参数位置进行激活
      - 可选择禁用梯度检查（dis_check_gradient），改为随机选择
   
   d) **协同过滤投票 (Collaborative Filtering at Test Time)**
      - 在测试阶段，统计每个参数被多少个客户端激活
      - 只保留被至少θ个客户端激活的参数（θ_ld阈值）
      - 通过投票机制过滤掉恶意客户端可能污染的参数

2. **防御原理分析**

   a) **为什么能防御后门攻击？**
      - 恶意客户端只能影响其激活的参数子空间
      - 如果恶意客户端是少数，其独特激活的参数会被投票过滤掉
      - 后门触发器需要特定的参数组合，稀疏训练破坏了这种组合的完整性
   
   b) **关键假设**
      - 恶意客户端数量是少数（小于总客户端数的某个比例）
      - 良性客户端的参数激活模式具有一致性
      - 恶意客户端的参数激活模式会偏离正常分布
   
   c) **优势**
      - 不需要检测恶意客户端（无需异常检测）
      - 通过结构化的稀疏性自然实现防御
      - 同时可以减少通信开销（只传输非零参数）
   
   d) **局限性**
      - 需要足够多的良性客户端形成"多数投票"
      - 如果恶意客户端能学习到良性客户端的激活模式，可能绕过防御
      - 稀疏训练可能影响模型的最终性能

3. **与其他防御方法的区别**

   - AlignIns/RLR: 基于梯度的统计检测
   - Krum/MultiKrum: 基于距离的客户端选择
   - Lockdown: 基于参数空间隔离和投票机制
   
   Lockdown的独特之处在于它不依赖于检测恶意更新，而是通过结构化约束
   （稀疏掩码）和最终的投票机制来自然地抑制恶意影响。

===============================================================================
"""

import copy
import math
import torch
import numpy as np
import logging
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
import torch.nn as nn


# ============================================================================
# 第一部分：稀疏性计算与掩码初始化
# ============================================================================

def calculate_sparsities(args, params, tabu=[], distribution="ERK"):
    """
    计算每一层参数的稀疏度（稀疏性分布）
    
    参数:
        args: 包含 dense_ratio 等配置的参数对象
        params: 模型参数字典 {name: tensor}
        tabu: 不进行稀疏化的层名称列表
        distribution: 稀疏性分布策略，"uniform" 或 "ERK"
    
    返回:
        spasities: 每层的稀疏度字典 {name: sparsity_value}
    
    核心思想：
        - uniform: 所有层使用相同的稀疏度
        - ERK (Erdős-Rényi Kernel): 根据层的形状动态分配稀疏度
          * 较大的层（如全连接层）会有更高的稀疏度
          * 较小的层（如卷积层）会有较低的稀疏度
          * 这样可以在保持总参数量的同时，更好地保留模型表达能力
    """
    spasities = {}
    
    if distribution == "uniform":
        # 统一稀疏度：所有层使用相同的稀疏比例
        for name in params:
            if name not in tabu:
                spasities[name] = 1 - args.dense_ratio
            else:
                spasities[name] = 0
                
    elif distribution == "ERK":
        # ERK分布：根据参数形状动态分配稀疏度
        logging.info('initialize by ERK')
        total_params = 0
        for name in params:
            total_params += params[name].numel()
            
        is_epsilon_valid = False
        dense_layers = set()
        density = args.dense_ratio
        
        # 迭代计算epsilon，直到找到有效的稀疏度分配
        while not is_epsilon_valid:
            # ERK的核心公式：
            # 每层的密度 = epsilon * (sum(shape) / prod(shape))
            # 其中 sum(shape)/prod(shape) 是层的"连接比例"
            # 例如：100x100的层 -> (100+100)/(100*100) = 0.02
            #       10x10的层   -> (10+10)/(10*10) = 0.2
            # 所以小层会保留更多参数
            
            divisor = 0
            rhs = 0
            raw_probabilities = {}
            
            for name in params:
                # 跳过BatchNorm的统计参数
                if name in tabu or "running" in name or "track" in name:
                    dense_layers.add(name)
                    
                n_param = np.prod(params[name].shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density
                
                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    # ERK概率：(sum of dimensions) / (product of dimensions)
                    raw_probabilities[name] = (
                        np.sum(params[name].shape) / np.prod(params[name].shape)
                    ) ** 1
                    divisor += raw_probabilities[name] * n_param
                    
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            
            # 检查是否有层的概率超过1（即需要100%密集）
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True
        
        # 使用有效的epsilon计算最终的稀疏度
        for name in params:
            if name in dense_layers:
                spasities[name] = 0  # 完全密集
            else:
                spasities[name] = (1 - epsilon * raw_probabilities[name])
                
    return spasities


def init_masks(params, sparsities):
    """
    根据稀疏度初始化掩码
    
    参数:
        params: 模型参数字典
        sparsities: 每层的稀疏度字典
    
    返回:
        masks: 初始化的二值掩码字典 {name: tensor}
               1表示参数激活，0表示参数被屏蔽
    
    核心思想：
        - 随机初始化掩码，确保每层有正确数量的激活参数
        - 使用随机排列选择激活的参数位置
        - 这种随机性确保不同客户端激活不同的参数子空间
    """
    masks = {}
    for name in params:
        # 初始化为全0掩码
        masks[name] = torch.zeros_like(params[name])
        
        # 计算需要激活的参数数量
        dense_numel = int((1 - sparsities[name]) * torch.numel(masks[name]))
        
        if dense_numel > 0:
            # 展平为一维
            temp = masks[name].view(-1)
            # 随机排列所有位置
            perm = torch.randperm(len(temp))
            # 选择前dense_numel个位置激活
            perm = perm[:dense_numel]
            temp[perm] = 1
            
        masks[name] = masks[name].to("cpu")
        
    return masks


# ============================================================================
# 第二部分：稀疏Agent类 - Lockdown的核心实现
# ============================================================================

class LockdownAgent:
    """
    Lockdown防御方法中的稀疏训练Agent
    
    核心功能：
        1. 维护动态更新的稀疏掩码
        2. 在本地训练时只更新被掩码激活的参数
        3. 使用Fire-Regrow机制动态调整掩码
        4. 使用梯度信息指导掩码更新
    """
    
    def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None):
        """
        初始化Lockdown Agent
        
        参数:
            id: 客户端ID
            args: 配置参数
            train_dataset: 训练数据集
            data_idxs: 该客户端的数据索引
            mask: 初始化的稀疏掩码
        """
        self.id = id
        self.args = args
        self.is_malicious = (id < args.num_corrupt)
        
        # 创建本地数据集和加载器
        self.train_dataset = train_dataset  # 简化示例，实际需要DatasetSplit
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=args.bs, 
            shuffle=True,
            drop_last=True
        )
        self.n_data = len(self.train_dataset)
        
        # 初始化掩码（每个客户端维护自己的掩码）
        self.mask = copy.deepcopy(mask)
        self.num_remove = None  # 记录每次需要移除的参数数量
    
    
    def screen_gradients(self, model):
        """
        梯度筛选：在本地数据上采样梯度
        
        目的：
            - 计算每个参数位置的梯度大小
            - 用于后续的Regrow操作（选择梯度大的位置激活）
        
        返回:
            gradient: 每个参数的累积梯度字典
        """
        model.train()
        criterion = nn.CrossEntropyLoss()
        
        # 初始化梯度累积器
        gradient = {name: 0 for name, param in model.named_parameters()}
        
        # 在多个batch上采样梯度
        for _, (x, labels) in enumerate(self.train_loader):
            model.zero_grad()
            x, labels = x.to(self.args.device), labels.to(self.args.device)
            
            # 前向传播
            log_probs = model(x)
            loss = criterion(log_probs, labels.long())
            
            # 反向传播
            loss.backward()
            
            # 累积梯度
            for name, param in model.named_parameters():
                gradient[name] += param.grad.data
                
        return gradient
    
    
    def fire_mask(self, weights, masks, round):
        """
        Fire操作：移除权重绝对值最小的参数
        
        核心思想：
            - 每轮训练后，丢弃一部分权重值最小的参数
            - 丢弃的数量通过余弦退火逐渐减少：
              drop_ratio = (anneal_factor/2) * (1 + cos(round*π/total_rounds))
            - 这样在训练初期更新较多，后期趋于稳定
        
        参数:
            weights: 当前模型权重
            masks: 当前掩码
            round: 当前训练轮次
        
        返回:
            masks: 更新后的掩码
            num_remove: 每层需要移除的参数数量
        """
        # 余弦退火计算drop_ratio
        drop_ratio = self.args.anneal_factor / 2 * (
            1 + np.cos((round * np.pi) / self.args.rounds)
        )
        
        # 计算每层需要移除的参数数量
        num_remove = {}
        for name in masks:
            num_non_zeros = torch.sum(masks[name].to(self.args.device))
            num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
        
        # 对每层执行Fire操作
        for name in masks:
            if num_remove[name] > 0 and "track" not in name and "running" not in name:
                # 将激活参数的权重保留，未激活的设为大值（确保不被选中）
                temp_weights = torch.where(
                    masks[name].to(self.args.device) > 0, 
                    torch.abs(weights[name]),
                    100000 * torch.ones_like(weights[name])
                )
                
                # 排序，找到权重最小的num_remove[name]个参数
                x, idx = torch.sort(temp_weights.view(-1).to(self.args.device))
                
                # 将这些参数的掩码设为0（移除）
                masks[name].view(-1)[idx[:num_remove[name]]] = 0
                
        return masks, num_remove
    
    
    def update_mask(self, masks, num_remove, gradient=None):
        """
        Regrow操作：根据梯度信息激活新参数
        
        核心思想：
            - 在未激活的参数中，选择梯度绝对值最大的参数激活
            - 激活数量等于Fire操作移除的数量，保持稀疏度不变
            - 如果禁用梯度检查（dis_check_gradient），则随机选择
        
        参数:
            masks: 当前掩码
            num_remove: 需要激活的参数数量（等于Fire移除的数量）
            gradient: 梯度信息
        
        返回:
            masks: 更新后的掩码
        """
        for name in gradient:
            if self.args.dis_check_gradient:
                # 方案1：随机选择未激活的参数
                # 反转掩码：0变1，1变0
                temp = torch.where(
                    masks[name] == 0, 
                    torch.ones_like(masks[name]), 
                    torch.zeros_like(masks[name])
                )
                # 随机抽样
                idx = torch.multinomial(
                    temp.flatten().to(self.args.device), 
                    num_remove[name], 
                    replacement=False
                )
                masks[name].view(-1)[idx] = 1
            else:
                # 方案2：根据梯度选择（默认方案）
                # 在未激活的参数中保留梯度值，激活的参数设为很小的负值
                temp = torch.where(
                    masks[name].to(self.args.device) == 0, 
                    torch.abs(gradient[name]),
                    -100000 * torch.ones_like(gradient[name])
                )
                
                # 排序，选择梯度最大的num_remove[name]个参数
                sort_temp, idx = torch.sort(temp.view(-1), descending=True)
                masks[name].view(-1)[idx[:num_remove[name]]] = 1
                
        return masks
    
    
    def local_train(self, global_model, criterion, round=None, 
                   global_mask=None, neurotoxin_mask=None, updates_dict=None):
        """
        Lockdown的本地训练流程
        
        关键步骤：
            1. 应用当前掩码到全局模型（只激活部分参数）
            2. 如果不是第一轮，先Regrow（激活新参数）
            3. 执行本地SGD训练（梯度也被掩码过滤）
            4. 训练后执行Fire操作（移除小权重参数）
            5. 返回稀疏更新（只包含激活参数的更新）
        
        返回:
            update: 参数更新向量（已应用掩码）
        """
        # 记录初始全局模型参数（用于计算更新）
        initial_global_model_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]
        ).detach()
        
        # 步骤1：应用掩码到模型参数
        # 只保留被掩码激活的参数，其余置零
        global_model.to(self.args.device)
        for name, param in global_model.named_parameters():
            self.mask[name] = self.mask[name].to(self.args.device)
            param.data = param.data * self.mask[name]
        
        # 步骤2：Regrow操作（如果不是第一轮）
        if self.num_remove is not None:
            # 先采样梯度
            gradient = self.screen_gradients(global_model)
            # 根据梯度更新掩码（激活新参数）
            self.mask = self.update_mask(self.mask, self.num_remove, gradient)
        
        # 步骤3：本地训练
        global_model.train()
        optimizer = torch.optim.SGD(
            global_model.parameters(), 
            lr=self.args.client_lr * (self.args.lr_decay) ** round,
            weight_decay=self.args.wd
        )
        
        for local_epoch in range(self.args.local_ep):
            for i, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs = inputs.to(device=self.args.device)
                labels = labels.to(device=self.args.device)
                
                # 前向传播
                outputs = global_model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                
                # 关键：梯度也需要应用掩码！
                # 只更新被激活的参数
                for name, param in global_model.named_parameters():
                    param.grad.data = self.mask[name].to(self.args.device) * param.grad.data
                
                # 优化器更新
                optimizer.step()
        
        # 步骤4：Fire操作（移除小权重参数）
        self.mask, self.num_remove = self.fire_mask(
            global_model.state_dict(), 
            self.mask, 
            round
        )
        
        # 步骤5：计算并返回稀疏更新
        with torch.no_grad():
            # 训练后的参数
            after_train = parameters_to_vector(
                [global_model.state_dict()[name] for name in global_model.state_dict()]
            ).detach()
            
            # 掩码向量
            array_mask = parameters_to_vector(
                [self.mask[name].to(self.args.device) for name in global_model.state_dict()]
            ).detach()
            
            # 更新 = 掩码 * (训练后参数 - 初始参数)
            # 只传输激活参数的更新
            self.update = array_mask * (after_train - initial_global_model_params)
            
        return self.update


# ============================================================================
# 第三部分：Lockdown的服务器端聚合与测试
# ============================================================================

class LockdownServer:
    """
    Lockdown的服务器端实现
    
    核心功能：
        1. 聚合客户端更新（使用FedAvg）
        2. 在测试时应用协同过滤（Collaborative Filtering）
    """
    
    def __init__(self, args, agent_data_sizes):
        """
        初始化Lockdown服务器
        
        参数:
            args: 配置参数
            agent_data_sizes: 每个客户端的数据量
        """
        self.args = args
        self.agent_data_sizes = agent_data_sizes
    
    
    def aggregate_updates(self, agent_updates_dict):
        """
        聚合客户端更新
        
        Lockdown使用标准的FedAvg聚合：
            aggregated_update = Σ(n_k/n_total * update_k)
        
        其中：
            n_k: 客户端k的数据量
            n_total: 总数据量
            update_k: 客户端k的更新（已经是稀疏的）
        
        注意：
            - 每个客户端的更新已经通过掩码变成稀疏的了
            - 聚合后的更新会在不同客户端激活的位置上进行加权平均
        """
        sm_updates = 0
        total_data = 0
        
        # 加权平均
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates += n_agent_data * update
            total_data += n_agent_data
        
        aggregated_update = sm_updates / total_data
        return aggregated_update
    
    
    def apply_collaborative_filtering(self, global_model, agents, theta_ld=10):
        """
        协同过滤 (Collaborative Filtering) - Lockdown的核心防御机制
        
        核心思想：
            在测试阶段，对每个参数位置统计有多少客户端激活了它。
            只保留被至少 theta_ld 个客户端激活的参数，其余置零。
        
        防御原理：
            - 良性客户端训练正常任务，其激活模式会有相似性
            - 恶意客户端植入后门，可能需要激活特殊的参数位置
            - 如果恶意客户端是少数，其特殊激活的参数不会达到阈值
            - 因此后门相关的参数会被过滤掉
        
        参数:
            global_model: 全局模型
            agents: 所有客户端（需要访问其掩码）
            theta_ld: 投票阈值（至少多少个客户端激活才保留）
        
        返回:
            test_model: 应用了协同过滤的测试模型
        """
        test_model = copy.deepcopy(global_model)
        
        # 对每个参数，统计有多少客户端激活了它
        for name, param in test_model.named_parameters():
            # 累加所有客户端的掩码
            mask_sum = 0
            for agent in agents:
                mask_sum += agent.mask[name].to(self.args.device)
            
            # 只保留被至少theta_ld个客户端激活的参数
            # mask_sum >= theta_ld 的位置保留，其余置零
            param.data = torch.where(
                mask_sum >= theta_ld,
                param,
                torch.zeros_like(param)
            )
        
        return test_model


# ============================================================================
# 第四部分：Lockdown完整训练流程示例
# ============================================================================

def lockdown_federated_learning(args, global_model, train_dataset, 
                                 test_loader, user_groups):
    """
    Lockdown联邦学习完整流程
    
    参数:
        args: 配置参数
        global_model: 初始化的全局模型
        train_dataset: 训练数据集
        test_loader: 测试数据加载器
        user_groups: 客户端数据划分
    """
    
    # ========== 步骤1：初始化稀疏掩码 ==========
    logging.info("Initializing sparse masks with ERK distribution...")
    
    # 获取模型参数
    params = {
        name: copy.deepcopy(global_model.state_dict()[name]) 
        for name in global_model.state_dict()
    }
    
    # 计算每层的稀疏度
    sparsity = calculate_sparsities(
        args, params, 
        distribution=args.mask_init  # 通常使用 "ERK"
    )
    
    # 初始化掩码
    if args.same_mask == 1:
        # 所有客户端共享同一个初始掩码
        shared_mask = init_masks(params, sparsity)
    
    
    # ========== 步骤2：创建Lockdown客户端 ==========
    logging.info("Creating Lockdown agents...")
    
    agents = []
    agent_data_sizes = {}
    
    for client_id in range(args.num_agents):
        if args.same_mask == 1:
            # 使用共享掩码
            mask = shared_mask
        else:
            # 每个客户端独立初始化掩码
            mask = init_masks(params, sparsity)
        
        agent = LockdownAgent(
            id=client_id,
            args=args,
            train_dataset=train_dataset,  # 需要根据user_groups进行划分
            data_idxs=user_groups[client_id],
            mask=mask
        )
        
        agents.append(agent)
        agent_data_sizes[client_id] = agent.n_data
    
    
    # ========== 步骤3：创建服务器 ==========
    server = LockdownServer(args, agent_data_sizes)
    
    
    # ========== 步骤4：联邦学习训练循环 ==========
    logging.info("Starting Lockdown federated learning...")
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    
    for round_num in range(1, args.rounds + 1):
        logging.info(f"========== Round {round_num} ==========")
        
        # 4.1 选择参与的客户端
        num_selected = max(1, int(args.agent_frac * args.num_agents))
        selected_clients = np.random.choice(
            args.num_agents, 
            num_selected, 
            replace=False
        )
        
        # 4.2 客户端本地训练
        agent_updates_dict = {}
        initial_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]
        ).detach()
        
        for client_id in selected_clients:
            logging.info(f"Client {client_id} training...")
            
            # 复制全局模型给客户端
            local_model = copy.deepcopy(global_model)
            
            # 本地训练（包含Fire-Regrow机制）
            update = agents[client_id].local_train(
                local_model, 
                criterion, 
                round=round_num
            )
            
            agent_updates_dict[client_id] = update
            
            # 恢复全局模型（准备给下一个客户端）
            # vector_to_model(initial_params, global_model)
        
        # 4.3 服务器聚合更新
        logging.info("Aggregating updates...")
        aggregated_update = server.aggregate_updates(agent_updates_dict)
        
        # 4.4 更新全局模型
        new_params = initial_params + args.server_lr * aggregated_update
        # vector_to_model(new_params, global_model)
        
        # 4.5 测试（应用协同过滤）
        if round_num % args.snap == 0:
            logging.info("Testing with Collaborative Filtering...")
            
            # 创建带协同过滤的测试模型
            test_model = server.apply_collaborative_filtering(
                global_model, 
                agents, 
                theta_ld=args.theta_ld  # 通常设置为10
            )
            
            # 在测试集上评估
            test_accuracy = evaluate_model(test_model, test_loader, criterion, args)
            logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    
    logging.info("Lockdown federated learning completed!")
    return global_model, agents


def evaluate_model(model, data_loader, criterion, args):
    """
    评估模型性能
    
    参数:
        model: 待评估的模型
        data_loader: 测试数据加载器
        criterion: 损失函数
        args: 配置参数
    
    返回:
        accuracy: 准确率
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy


# ============================================================================
# 第五部分：关键参数配置说明
# ============================================================================

"""
Lockdown的关键超参数：

1. dense_ratio (默认: 0.25)
   - 参数的密集度（激活比例）
   - 值越小，稀疏度越高，防御效果可能越好，但模型性能可能下降
   - 通常设置为 0.1 ~ 0.5

2. mask_init (默认: "ERK")
   - 掩码初始化方式
   - "ERK": Erdős-Rényi Kernel，根据层形状动态分配
   - "uniform": 所有层统一稀疏度

3. anneal_factor (默认: 0.0001)
   - 控制Fire操作的退火速度
   - 值越大，每轮更新的参数越多
   - 值越小，掩码越稳定

4. theta_ld (默认: 10)
   - 协同过滤的投票阈值
   - 需要根据总客户端数量和恶意客户端比例设置
   - 一般设置为: num_agents * 0.4 ~ 0.6

5. same_mask (默认: 1)
   - 是否所有客户端使用相同的初始掩码
   - 1: 共享初始掩码（推荐）
   - 0: 每个客户端独立初始化掩码

6. dis_check_gradient (默认: False)
   - 是否禁用梯度检查
   - False: 使用梯度信息指导Regrow（推荐）
   - True: 随机选择Regrow位置

7. local_ep (默认: 2)
   - 本地训练轮数
   - Lockdown中不宜过大，避免掩码变化过快

8. client_lr (默认: 0.1)
   - 客户端学习率
   - 稀疏训练可能需要较大的学习率

9. server_lr (默认: 1)
   - 服务器学习率
   - 控制聚合更新的应用程度
"""


# ============================================================================
# 第六部分：Lockdown与其他防御方法的对比
# ============================================================================

"""
防御方法对比：

1. FedAvg (无防御)
   - 简单加权平均
   - 无防御能力
   - 性能基线

2. Krum / MultiKrum
   - 基于欧式距离的客户端选择
   - 假设恶意更新与良性更新距离较远
   - 需要准确的拜占庭客户端数量估计
   
3. RLR (Robust Learning Rate)
   - 基于符号一致性的学习率调整
   - 对抗符号翻转攻击
   - 只调整学习率，不过滤客户端

4. AlignIns
   - 基于多个统计指标（TDA, MPSA）检测异常
   - 需要设置多个阈值
   - 计算开销较大

5. FoolsGold
   - 基于历史更新相似度降权
   - 假设恶意客户端更新相似
   - 需要维护历史信息

6. Lockdown (本方法)
   优势：
   - 不需要检测恶意客户端
   - 通过结构化约束自然防御
   - 同时减少通信开销
   - 对多种攻击有效
   
   劣势：
   - 需要足够多的良性客户端
   - 稀疏训练可能影响性能
   - 需要仔细调整稀疏度和阈值
"""


# ============================================================================
# 附录：向量与模型参数转换工具函数
# ============================================================================

def vector_to_model(vec, model):
    """
    将一维向量转换回模型参数
    
    参数:
        vec: 参数向量
        model: 目标模型
    
    返回:
        state_dict: 更新后的状态字典
    """
    state_dict = model.state_dict()
    pointer = 0
    
    for name in state_dict:
        num_param = state_dict[name].numel()
        state_dict[name].data = vec[pointer:pointer + num_param].view_as(
            state_dict[name]
        ).data
        pointer += num_param
    
    model.load_state_dict(state_dict)
    return state_dict


def name_param_to_array(param):
    """
    将参数字典转换为一维向量
    
    参数:
        param: 参数字典 {name: tensor}
    
    返回:
        vec: 一维参数向量
    """
    vec = []
    for name in param:
        vec.append(param[name].view(-1))
    return torch.cat(vec)


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    """
    Lockdown使用示例
    
    注意：这只是示意代码，实际运行需要完整的数据集和模型定义
    """
    
    # 配置参数
    class Args:
        # 联邦学习基本参数
        num_agents = 20          # 客户端总数
        num_corrupt = 2          # 恶意客户端数
        rounds = 100             # 训练轮数
        agent_frac = 1.0         # 每轮参与的客户端比例
        local_ep = 2             # 本地训练轮数
        bs = 64                  # batch size
        
        # Lockdown特定参数
        dense_ratio = 0.25       # 参数密集度（25%激活）
        mask_init = "ERK"        # ERK初始化
        anneal_factor = 0.0001   # 退火因子
        theta_ld = 10            # 协同过滤阈值
        same_mask = 1            # 使用共享初始掩码
        dis_check_gradient = False  # 启用梯度检查
        
        # 优化器参数
        client_lr = 0.1          # 客户端学习率
        server_lr = 1.0          # 服务器学习率
        lr_decay = 0.99          # 学习率衰减
        wd = 1e-4                # 权重衰减
        momentum = 0.0           # 动量
        
        # 其他
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        snap = 1                 # 每几轮测试一次
    
    args = Args()
    
    print("=" * 80)
    print("Lockdown防御方法示例")
    print("=" * 80)
    print(f"\n配置参数:")
    print(f"  总客户端数: {args.num_agents}")
    print(f"  恶意客户端数: {args.num_corrupt}")
    print(f"  参数密集度: {args.dense_ratio}")
    print(f"  协同过滤阈值: {args.theta_ld}")
    print(f"  掩码初始化: {args.mask_init}")
    print("\n")
    
    # 这里需要：
    # 1. 加载数据集
    # 2. 创建模型
    # 3. 调用 lockdown_federated_learning()
    
    print("提示：完整运行需要：")
    print("  1. 准备训练和测试数据集")
    print("  2. 定义神经网络模型")
    print("  3. 调用 lockdown_federated_learning() 函数")
    print("\n")
    print("=" * 80)
    print("代码整理完成！")
    print("=" * 80)





