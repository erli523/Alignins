"""
Dynamic Trigger Attack with Adversarial Model Enhancement
动态触发器后门攻击（带对抗模型增强）
"""
import torch
import torch.nn as nn
import copy
import time
import logging


class DynamicTriggerAttacker:
    """
    动态触发器攻击器
    通过梯度优化寻找最优触发器模式，并使用对抗模型增强
    """
    
    def __init__(self, args, trigger_size=5):
        self.args = args
        self.trigger_size = trigger_size
        self.device = args.device
        
        # 初始化触发器和掩码
        if args.data == 'cifar10' or args.data == 'cifar100':
            img_size = 32
            channels = 3
        elif args.data == 'fmnist' or args.data == 'mnist':
            img_size = 28
            channels = 1
        else:
            img_size = 64  # tinyimagenet
            channels = 3
        
        # 🔧 FIX: 触发器初始化为随机值（更容易优化）
        self.trigger = torch.rand((1, channels, img_size, img_size), 
                                 requires_grad=False, device=self.device)
        
        # 掩码：指定触发器在图像上的位置（左上角）
        self.mask = torch.zeros_like(self.trigger)
        self.mask[:, :, 2:2+self.trigger_size, 2:2+self.trigger_size] = 1
        self.mask = self.mask.to(self.device)
        
        # 备份初始触发器
        self.trigger0 = self.trigger.clone()
        
        # 🔧 FIX: 存储测试集数据加载器（用于准确的ASR测试）
        self.test_loader = None
        
        logging.info(f"DynamicTriggerAttacker initialized: trigger_size={trigger_size}, img_size={img_size}x{img_size}")
    
    def init_badnets_trigger(self):
        """初始化为 BadNet 风格的触发器（用于对比实验）"""
        self.trigger[:, 0, :, :] = 1  # 红色通道全白
        logging.info("Initialized BadNet-style trigger")
    
    def get_adv_model(self, model, dataloader):
        """
        创建对抗模型：在带触发器的数据上对全局模型进行对抗训练
        返回对抗模型和相似度权重
        """
        adv_model = copy.deepcopy(model)
        adv_model.train()
        
        ce_loss = nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), 
                                 lr=0.1, momentum=0.9, weight_decay=5e-4)
        
        # 对抗训练轮数
        adv_epochs = getattr(self.args, 'dm_adv_epochs', 2)
        
        for _ in range(adv_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 添加触发器
                inputs = self.trigger * self.mask + (1 - self.mask) * inputs
                
                outputs = adv_model(inputs)
                loss = ce_loss(outputs, labels)
                
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()
        
        # 🔧 FIX: 计算对抗模型与原模型的参数差异（而不是梯度）
        # 因为 model.eval() 状态下没有梯度
        param_diff_sum = 0.0
        param_count = 0
        
        for (name, adv_param), (_, orig_param) in zip(adv_model.named_parameters(), 
                                                       model.named_parameters()):
            if 'weight' in name:  # 只计算权重层
                param_count += 1
                # 计算参数的 L2 距离，归一化后作为权重
                diff = torch.norm(adv_param - orig_param) / torch.norm(orig_param)
                param_diff_sum += diff.item()
        
        # 平均差异越大，权重越高（说明对抗模型变化越大）
        avg_diff = param_diff_sum / param_count if param_count > 0 else 1.0
        # 将差异转换为权重（范围 [0.5, 1.5]）
        adv_weight = min(max(avg_diff, 0.5), 1.5)
        
        return adv_model, adv_weight
    
    def validate_asr(self, model, dataloader, trigger=None):
        """
        验证当前触发器的攻击成功率 (ASR)
        
        Args:
            model: 模型
            dataloader: 数据加载器
            trigger: 要测试的触发器（如果为 None 则使用 self.trigger）
        """
        model.eval()
        ce_loss = nn.CrossEntropyLoss(label_smoothing=0.001)
        
        # 🔧 FIX: 如果没有指定触发器，使用 self.trigger
        if trigger is None:
            trigger = self.trigger
        
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 🔧 FIX: 使用指定的触发器（而不是固定使用 self.trigger）
                inputs = trigger * self.mask + (1 - self.mask) * inputs
                # 所有标签改为目标类别
                labels[:] = self.args.target_class
                
                outputs = model(inputs)
                loss = ce_loss(outputs, labels)
                total_loss += loss.item()
                
                pred = outputs.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        asr = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        
        return asr, avg_loss
    
    def search_trigger(self, model, dataloader, round_num=0):
        """
        核心方法：通过梯度优化搜索最优触发器
        
        Args:
            model: 当前的全局模型
            dataloader: 恶意客户端的数据加载器（用于优化触发器）
            round_num: 当前联邦学习轮数
        
        Note:
            - 优化过程使用 dataloader（训练集）
            - 最终验证使用 self.test_loader（测试集，如果有）
        """
        start_time = time.time()
        logging.info(f"[Round {round_num}] Starting trigger optimization...")
        
        model.eval()  # 模型设为评估模式（不更新模型参数）
        
        # 触发器优化参数
        trigger_lr = getattr(self.args, 'trigger_lr', 0.01)  # 🔧 FIX: Adam 需要更小的学习率
        K = getattr(self.args, 'trigger_outter_epochs', 50)  # 外层优化轮数
        adv_K = getattr(self.args, 'dm_adv_K', 10)  # 每隔多少轮创建对抗模型
        adv_model_count = getattr(self.args, 'dm_adv_model_count', 2)  # 对抗模型数量
        noise_loss_lambda = getattr(self.args, 'noise_loss_lambda', 0.1)  # 对抗损失权重
        
        ce_loss = nn.CrossEntropyLoss()
        
        # 克隆触发器用于优化
        t = self.trigger.clone().detach().to(self.device)
        t.requires_grad = True  # 🔧 FIX: 一开始就设置 requires_grad
        m = self.mask.clone().to(self.device)
        
        # 🔧 FIX: 使用优化器管理触发器更新
        trigger_optimizer = torch.optim.Adam([t], lr=trigger_lr)
        
        adv_models = []
        adv_weights = []
        
        # 🔧 FIX: 在开始时就创建对抗模型（不要等到 iter=10）
        if adv_model_count > 0:
            for _ in range(adv_model_count):
                adv_model, adv_w = self.get_adv_model(model, dataloader)
                adv_models.append(adv_model)
                adv_weights.append(adv_w)
            logging.info(f"  Created {adv_model_count} initial adversarial models")
        
        for iter in range(K):
            # 🔧 FIX: 每隔一定轮数验证 ASR（使用正在优化的触发器 t）
            if iter % 10 == 0:
                # 在训练集上快速验证（监控优化进度）
                asr_train, loss_train = self.validate_asr(model, dataloader, trigger=t)
                logging.info(f"  Iter {iter}/{K}: Train ASR={asr_train:.4f}, Loss={loss_train:.4f}")
            
            # 🔧 FIX: 每隔 adv_K 轮更新对抗模型（从 iter=adv_K 开始）
            if iter > 0 and iter % adv_K == 0 and adv_model_count > 0:
                # 清理旧的对抗模型
                for adv_model in adv_models:
                    del adv_model
                adv_models = []
                adv_weights = []
                
                # 创建新的对抗模型
                for _ in range(adv_model_count):
                    adv_model, adv_w = self.get_adv_model(model, dataloader)
                    adv_models.append(adv_model)
                    adv_weights.append(adv_w)
                
                logging.info(f"  Updated {adv_model_count} adversarial models at iter {iter}")
            
            # 🔧 FIX: 遍历多个 batch 进行优化（而不是只用一个）
            batch_count = 0
            max_batches = min(3, len(dataloader))  # 最多用 3 个 batch
            
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 🔧 FIX: 清零梯度（非常重要！）
                trigger_optimizer.zero_grad()
                
                # 添加当前触发器
                poisoned_inputs = t * m + (1 - m) * inputs
                # 🔧 FIX: 标签转换为 long 类型
                target_labels = torch.full_like(labels, self.args.target_class, dtype=torch.long)
                
                # 在原模型上计算损失
                outputs = model(poisoned_inputs)
                loss = ce_loss(outputs, target_labels)
                
                # 如果有对抗模型，加入对抗损失
                if len(adv_models) > 0:
                    for idx, adv_model in enumerate(adv_models):
                        adv_outputs = adv_model(poisoned_inputs)
                        adv_loss = ce_loss(adv_outputs, target_labels)
                        adv_w = adv_weights[idx]
                        # 加权对抗损失
                        loss += noise_loss_lambda * adv_w * adv_loss / adv_model_count
                
                # 反向传播计算触发器梯度
                loss.backward()
                
                # 🔧 FIX: 使用优化器更新触发器
                trigger_optimizer.step()
                
                # 🔧 FIX: 裁剪到 [0, 1] 范围（正常像素值）
                with torch.no_grad():
                    t.data = torch.clamp(t.data, min=0.0, max=1.0)
                
                batch_count += 1
                if batch_count >= max_batches:
                    break
        
        # 最终验证
        with torch.no_grad():
            self.trigger = t.detach().clone()  # 🔧 FIX: 确保触发器正确保存
        
        elapsed = time.time() - start_time
        
        # 🔧 FIX: 在测试集上验证（如果有），否则使用训练集
        if self.test_loader is not None:
            final_asr, final_loss = self.validate_asr(model, self.test_loader, trigger=self.trigger)
            logging.info(f"[Round {round_num}] Trigger optimization completed in {elapsed:.2f}s")
            logging.info(f"  Test ASR: {final_asr:.4f}, Test Loss: {final_loss:.4f}")
        else:
            # 如果没有测试集，使用训练集（但会打印警告）
            final_asr, final_loss = self.validate_asr(model, dataloader, trigger=self.trigger)
            logging.warning(f"[Round {round_num}] No test_loader set, using training data for validation")
            logging.info(f"[Round {round_num}] Trigger optimization completed in {elapsed:.2f}s")
            logging.info(f"  Train ASR: {final_asr:.4f}, Train Loss: {final_loss:.4f}")
        
        # 清理对抗模型
        for adv_model in adv_models:
            del adv_model
    
    def poison_batch(self, inputs, labels, poison_ratio=None):
        """
        毒化一个 batch 的数据
        
        Args:
            inputs: 输入图像 [batch_size, C, H, W]
            labels: 标签 [batch_size]
            poison_ratio: 毒化比例（如果为 None 则使用 args.poison_frac）
        
        Returns:
            poisoned_inputs, poisoned_labels
        """
        if poison_ratio is None:
            poison_ratio = self.args.poison_frac
        
        batch_size = inputs.size(0)
        num_poison = int(poison_ratio * batch_size)
        
        if num_poison > 0:
            # 前 num_poison 个样本添加触发器
            inputs[:num_poison] = self.trigger * self.mask + inputs[:num_poison] * (1 - self.mask)
            labels[:num_poison] = self.args.target_class
        
        return inputs, labels
    
    def poison_all(self, inputs, labels):
        """
        毒化所有数据（用于测试 ASR）
        """
        inputs = self.trigger * self.mask + inputs * (1 - self.mask)
        labels[:] = self.args.target_class
        return inputs, labels

