import copy
import math
import torch
import utils
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import logging
import time
from dynamic_trigger_attack import DynamicTriggerAttacker
from altermin_attack import AlterMinAttacker
from model_replacement_attack import ModelReplacementAttacker

class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None, backdoor_train_dataset=None):
        self.id = id
        self.args = args
        self.error = 0
        self.is_malicious = (id < args.num_corrupt)  # 标记是否为恶意客户端
        self.dynamic_trigger_attacker = None  # 动态触发器攻击器
        self.altermin_attacker = None  # AlterMin 攻击器
        self.model_replacement_attacker = None  # Model Replacement 攻击器
        
        # poisoned datasets, tinyimagenet is handled differently as the dataset is not loaded into memory
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
                
                # 创建数据集（所有恶意客户端都需要）
                self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
            elif self.id < args.num_corrupt and self.args.attack != 'non' and self.args.data == 'sen140':
                self.clean_backup_dataset = copy.deepcopy(train_dataset)
                self.data_idxs = data_idxs
                benign_part = data_idxs[:int(len(data_idxs) * (1 - self.args.poison_frac))]
                malicious_part = data_idxs[int(len(data_idxs) * (1 - self.args.poison_frac)):]
                self.train_dataset = utils.DatasetSplit_new(train_dataset, backdoor_train_dataset, benign_part, malicious_part, data_idxs)
            else:
                # 正常客户端或其他情况
                self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
        else:
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs, runtime_poison=True, args=args,
                                                        client_id=id)

        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                       num_workers=args.num_workers, pin_memory=False, drop_last=True)
        # size of local dataset
        self.n_data = len(self.train_dataset)

        self.mask = copy.deepcopy(mask)
        self.num_remove= None

    def check_poison_timing(self, round):
        if round> self.args.cease_poison:
            self.train_dataset = utils.DatasetSplit(self.clean_backup_dataset, self.data_idxs)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                       num_workers=self.args.num_workers, pin_memory=False, drop_last=True)


    def screen_gradients(self, model):
        model.train()
        # # # train and update
        criterion = nn.CrossEntropyLoss()
        gradient = {name: 0 for name, param in model.named_parameters()}
        # # sample 10 batch  of data
        batch_num = 0
        for _, (x, labels) in enumerate(self.train_loader):
            batch_num+=1
            model.zero_grad()
            x, labels = x.to(self.args.device), labels.to(self.args.device)
            log_probs = model.forward(x)
            minibatch_loss = criterion(log_probs, labels.long())
            loss = minibatch_loss
            loss.backward()
            for name, param in model.named_parameters():
                gradient[name] += param.grad.data
        return gradient

    def update_mask(self, masks, num_remove, gradient=None):
        for name in gradient:
            if self.args.dis_check_gradient:
                temp = torch.where(masks[name] == 0, torch.ones_like(masks[name]), torch.zeros_like(masks[name]))
                idx = torch.multinomial(temp.flatten().to(self.args.device), num_remove[name], replacement=False)
                masks[name].view(-1)[idx] = 1
            else:
                temp = torch.where(masks[name].to(self.args.device) == 0, torch.abs(gradient[name]),
                                    -100000 * torch.ones_like(gradient[name]))
                sort_temp, idx = torch.sort(temp.view(-1), descending=True)
                masks[name].view(-1)[idx[:num_remove[name]]] = 1
        return masks
    
    # def init_mask(self,  gradient=None):
    #     for name in self.mask:
    #         num_init = torch.count_nonzero(self.mask[name])
    #         self.mask[name] = torch.zeros_like(self.mask[name])
    #         sort_temp, idx = torch.sort(torch.abs(gradient[name]).view(-1), descending=True)
    #         self.mask[name].view(-1)[idx[:num_init]] = 1
             

    def fire_mask(self, weights, masks, round):
        
        drop_ratio = self.args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / (self.args.rounds)))
    
        # logging.info(drop_ratio)
        num_remove = {}
        for name in masks:
                num_non_zeros = torch.sum(masks[name].to(self.args.device))
                num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
     
        for name in masks:
            if num_remove[name]>0 and  "track" not in name and "running" not in name: 
                temp_weights = torch.where(masks[name].to(self.args.device) > 0, torch.abs(weights[name]),
                                        100000 * torch.ones_like(weights[name]))
                x, idx = torch.sort(temp_weights.view(-1).to(self.args.device))
                masks[name].view(-1)[idx[:num_remove[name]]] = 0
        return masks, num_remove



    def local_train(self, global_model, criterion, round=None, temparature=10, alpha=0.3, global_mask= None, neurotoxin_mask =None, updates_dict =None):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.id  <  self.args.num_corrupt:
            self.check_poison_timing(round)
            
            # 如果是动态触发器攻击，先优化触发器
            if self.args.attack == 'dynamic_trigger' and self.dynamic_trigger_attacker is not None:
                # 每隔一定轮数重新优化触发器
                trigger_optim_interval = getattr(self.args, 'trigger_optim_interval', 2)
                if round % trigger_optim_interval == 0 or round == 1:
                    self.dynamic_trigger_attacker.search_trigger(global_model, self.train_loader, round_num=round)
            
            # 如果是 AlterMin 攻击，使用特殊的训练流程
            if self.args.attack == 'altermin' and self.altermin_attacker is not None:
                # 保存全局模型权重用于计算距离损失
                self.altermin_attacker.save_global_weights(global_model)
                
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
                
        global_model.to(self.args.device)
        for name, param in global_model.named_parameters():
            self.mask[name] =self.mask[name].to(self.args.device)
            param.data = param.data * self.mask[name]
        if self.num_remove!=None:
            if self.id>=  self.args.num_corrupt or self.args.attack!="fix_mask" :
                gradient = self.screen_gradients(global_model)
                self.mask = self.update_mask(self.mask, self.num_remove, gradient)
        
        global_model.train()
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr * (self.args.lr_decay) ** round,
                                    weight_decay=self.args.wd)

        for local_epoch in range(self.args.local_ep):
            start = time.time()
            for i, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device), \
                                 labels.to(device=self.args.device)
                
                # 如果是动态触发器攻击，运行时毒化数据
                if self.id < self.args.num_corrupt and self.args.attack == 'dynamic_trigger' and self.dynamic_trigger_attacker is not None:
                    inputs, labels = self.dynamic_trigger_attacker.poison_batch(inputs, labels)
                
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                loss = minibatch_loss
                loss.backward()
                for name, param in global_model.named_parameters():
                    param.grad.data = self.mask[name].to(self.args.device) * param.grad.data
                optimizer.step()

            end = time.time()
            train_time = end - start
            print("local epoch %d \t client: %d \t mal: %s \t loss: %.8f \t time: %.2f" % (local_epoch, self.id, str(self.is_malicious),
                                                                     minibatch_loss, train_time))
           
        

            if self.args.attack == 'pgd' and self.id < self.args.num_corrupt and (i == len(self.train_loader) - 1):
                eps = torch.norm(initial_global_model_params)
                current_local_model_params = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
                norm_diff = torch.norm(current_local_model_params - initial_global_model_params)
                # print('clip before: ', norm_diff)
                if norm_diff > eps:
                    w_proj_vec = eps * (current_local_model_params - initial_global_model_params) / norm_diff + initial_global_model_params

                # print('clip after: ', torch.norm(w_proj_vec - initial_global_model_params))

                    new_state_dict = utils.vector_to_model_wo_load(w_proj_vec, global_model)    
                    global_model.load_state_dict(new_state_dict)
        if self.id<  self.args.num_corrupt:
            if self.args.attack=="fix_mask":
                self.mask = self.mask 

            elif self.args.attack == "omniscient":
                if len(global_mask):
                    self.mask = copy.deepcopy(global_mask)
                else:
                    self.mask = self.mask 
            elif self.args.attack == "neurotoxin":
                if len(neurotoxin_mask):
                    self.mask = neurotoxin_mask
                else:
                    self.mask = self.mask
            else:
                self.mask, self.num_remove = self.fire_mask(global_model.state_dict(), self.mask, round)

        else:
            self.mask, self.num_remove = self.fire_mask(global_model.state_dict(), self.mask, round) 
            
        with torch.no_grad():
            after_train = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            array_mask = parameters_to_vector([ self.mask[name].to(self.args.device) for name in global_model.state_dict()]).detach()
            self.update = ( array_mask *(after_train - initial_global_model_params))
            if "scale" in self.args.attack:
                logging.info("scale update for" + self.args.attack.split("_",1)[1] + " times")
                if self.id<  self.args.num_corrupt:
                    self.update=  int(self.args.attack.split("_",1)[1]) * self.update
        return self.update

