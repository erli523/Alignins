import copy
import time

import torch
import utils
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from dynamic_trigger_attack import DynamicTriggerAttacker
from altermin_attack import AlterMinAttacker
from model_replacement_attack import ModelReplacementAttacker

class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None, backdoor_train_dataset=None):
        self.id = id
        self.args = args
        self.error = 0
        self.hessian_metrix = []
        self.dynamic_trigger_attacker = None  # 动态触发器攻击器
        self.altermin_attacker = None  # AlterMin 攻击器
        self.model_replacement_attacker = None  # Model Replacement 攻击器
        
        # get datasets, fedemnist is handled differently as it doesn't come with pytorch
        if self.args.data != "tinyimagenet":
            # for backdoor attack, agent poisons his local dataset
            if self.id < args.num_corrupt and self.args.attack != 'non' and self.args.data != 'sen140':
                self.clean_backup_dataset = copy.deepcopy(train_dataset)
                self.data_idxs = data_idxs
                
                # 如果是动态触发器攻击，不预先毒化数据集，而是运行时毒化
                if self.args.attack == 'dynamic_trigger':
                    trigger_size = getattr(self.args, 'trigger_size', 5)
                    self.dynamic_trigger_attacker = DynamicTriggerAttacker(args, trigger_size=trigger_size)
                    # 使用干净数据进行训练（运行时毒化）
                elif self.args.attack == 'altermin':
                    # AlterMin 攻击：不预先毒化训练数据集，使用干净数据训练
                    self.altermin_attacker = AlterMinAttacker(args, worker_id=self.id)
                    # 使用干净数据进行训练，攻击器会在训练过程中使用单独的投毒数据
                elif self.args.attack == 'model_replacement':
                    # Model Replacement 攻击：使用余弦距离约束 + Scaling
                    self.model_replacement_attacker = ModelReplacementAttacker(args, worker_id=self.id)
                    # 先毒化数据集（使用触发器）
                    utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
                else:
                    # 其他攻击：预先毒化数据集
                    utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
            
            # 创建或重新创建 DatasetSplit
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
                    
            if self.id < args.num_corrupt and self.args.attack != 'non' and self.args.data == 'sen140':
                self.clean_backup_dataset = copy.deepcopy(train_dataset)
                self.data_idxs = data_idxs
                benign_part = data_idxs[:int(len(data_idxs) * (1 - self.args.poison_frac))]
                malicious_part = data_idxs[int(len(data_idxs) * (1 - self.args.poison_frac)):]

                self.train_dataset = utils.DatasetSplit_new(train_dataset, backdoor_train_dataset, benign_part, malicious_part, data_idxs)
        else:
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs, runtime_poison=True, args=args,
                                                        client_id=id)
        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                       num_workers=args.num_workers, pin_memory=False, drop_last=True)
        # size of local dataset
        self.n_data = len(self.train_dataset)

    def check_poison_timing(self, round):
        if round > self.args.cease_poison:
            self.train_dataset = utils.DatasetSplit(self.clean_backup_dataset, self.data_idxs)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                           num_workers=self.args.num_workers, pin_memory=False, drop_last=True)

    def local_train(self, global_model, criterion, round=None, neurotoxin_mask=None):
        # print(len(self.train_dataset))
        """ Do a local training over the received global model, return the update """
        # start = time.time()
        initial_global_model_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        
        if self.id < self.args.num_corrupt:
            self.check_poison_timing(round)
            
            # 如果是动态触发器攻击，先优化触发器
            if self.args.attack == 'dynamic_trigger' and self.dynamic_trigger_attacker is not None:
                # 每隔一定轮数重新优化触发器
                trigger_optim_interval = getattr(self.args, 'trigger_optim_interval', 5)
                if round % trigger_optim_interval == 0 or round == 1:
                    self.dynamic_trigger_attacker.search_trigger(global_model, self.train_loader, round_num=round)
            
            # 如果是 AlterMin 攻击，使用特殊的训练流程
            if self.args.attack == 'altermin' and self.altermin_attacker is not None:
                # 保存全局模型权重用于计算距离损失
                self.altermin_attacker.save_global_weights(global_model)
                
                # 初始化投毒数据加载器（第一轮时）
                if self.altermin_attacker.poisoned_loader is None:
                    # 需要获取测试集来初始化投毒数据
                    # 从 federated.py 传入的 test_dataset 通过其他方式访问
                    # 这里我们使用一个简单的方法：从训练集中选择样本
                    pass  # 将在 federated.py 中初始化
                
                # 创建优化器
                optimizer = torch.optim.SGD(global_model.parameters(), 
                                          lr=self.args.client_lr * (self.args.lr_decay) ** round,
                                          weight_decay=self.args.wd, 
                                          momentum=self.args.momentum)
                
                # 执行 AlterMin 训练（交替优化）
                self.altermin_attacker.altermin_train(
                    global_model, self.train_loader, criterion, optimizer, round_num=round
                )
                
                # 计算并返回更新
                with torch.no_grad():
                    after_train = parameters_to_vector(
                        [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
                    self.update = after_train - initial_global_model_params
                    return self.update
            
            # 如果是 Model Replacement 攻击，使用特殊的训练流程
            if self.args.attack == 'model_replacement' and self.model_replacement_attacker is not None:
                # 保存全局模型权重用于计算余弦距离
                self.model_replacement_attacker.save_global_weights(global_model)
                
                # 创建优化器
                optimizer = torch.optim.SGD(global_model.parameters(), 
                                          lr=self.args.client_lr * (self.args.lr_decay) ** round,
                                          weight_decay=self.args.wd, 
                                          momentum=self.args.momentum)
                
                # 执行 Model Replacement 训练（Constrain + Scale）
                _, scaled_update = self.model_replacement_attacker.model_replacement_train(
                    global_model, self.train_loader, criterion, optimizer, round_num=round
                )
                
                # 返回放大后的更新
                self.update = scaled_update
                return self.update
        
        global_model.train()
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr * (self.args.lr_decay) ** round,
                                    weight_decay=self.args.wd, momentum=self.args.momentum)

        regular_loss = 0.0
        for local_epoch in range(self.args.local_ep):
            start = time.time()
            old_gradient = {}
            old_gradient_mine = {}
            old_params = {}
            for i, (inputs, labels) in enumerate(self.train_loader):
                # if i == 0 and self.is_malicious:
                #     save_image(torch.cat([inputs[labels == self.args.target_class][:10]]), '%s_image.png' % self.id, normalize=True, nrow=10)
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                                 labels.to(device=self.args.device, non_blocking=True)
                
                # 如果是动态触发器攻击，运行时毒化数据
                if self.id < self.args.num_corrupt and self.args.attack == 'dynamic_trigger' and self.dynamic_trigger_attacker is not None:
                    inputs, labels = self.dynamic_trigger_attacker.poison_batch(inputs, labels)
                
                outputs = global_model(inputs)
                # outputs = outputs[:, :]
                minibatch_loss = criterion(outputs, labels)
                # print(minibatch_loss)
                minibatch_loss.backward()
                if self.args.attack == "neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
                    for name, param in global_model.named_parameters():
                        param.grad.data = neurotoxin_mask[name].to(self.args.device) * param.grad.data
                if self.args.attack == "r_neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
                    for name, param in global_model.named_parameters():
                        param.grad.data = (torch.ones_like(neurotoxin_mask[name].to(self.args.device))-neurotoxin_mask[name].to(self.args.device) ) * param.grad.data
                optimizer.step()

                if self.args.attack == 'pgd' and self.id < self.args.num_corrupt and (i == len(self.train_loader) - 1):
                    if self.args.data == 'cifar10':
                        eps = torch.norm(initial_global_model_params) * 0.1
                    else:
                        eps = torch.norm(initial_global_model_params)

                    current_local_model_params = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
                    norm_diff = torch.norm(current_local_model_params - initial_global_model_params)
                    print('clip before: ', norm_diff)
                    if norm_diff > eps:
                        w_proj_vec = eps * (current_local_model_params - initial_global_model_params) / norm_diff + initial_global_model_params

                        print('clip after: ', torch.norm(w_proj_vec - initial_global_model_params))

                        new_state_dict = utils.vector_to_model_wo_load(w_proj_vec, global_model)    
                        global_model.load_state_dict(new_state_dict)

            end = time.time()
            train_time = end - start
            print("local epoch %d \t client: %d \t mal: %s \t loss: %.8f \t time: %.2f" % (local_epoch, self.id, str(self.is_malicious),
                                                                     minibatch_loss, train_time))

        with torch.no_grad():
            after_train = parameters_to_vector(
                [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            self.update = after_train - initial_global_model_params

            return self.update
