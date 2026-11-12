# # import utils
# # import models
# # import math
# # import copy
# # import numpy as np
# # from agent import Agent
# # from agent_sparse import Agent as Agent_s
# # from aggregation import Aggregation
# # import torch
# # import random
# # from torch.utils.data import DataLoader
# # import torch.nn as nn
# # from torch.nn.utils import parameters_to_vector
# # import logging
# # import argparse
# # import os
# # import warnings

# # warnings.filterwarnings("ignore")

# # if __name__ == "__main__":
# #     print("程序开始运行...")
# #     torch.backends.cudnn.enabled = True
# #     torch.backends.cudnn.benchmark = False
# #     torch.manual_seed(0)
# #     torch.cuda.manual_seed_all(0)
# #     np.random.seed(0)
# #     random.seed(0)
# #     torch.backends.cudnn.deterministic = True

# #     print("初始化参数解析器...")
# #     parser = argparse.ArgumentParser(description="pass in a parameter")

# #     parser.add_argument(
# #         "--data", type=str, default="cifar10", help="dataset we want to train on"
# #     )
# #     parser.add_argument("--num_agents", type=int, default=20, help="number of agents:K")
# #     parser.add_argument(
# #         "--agent_frac", type=float, default=1.0, help="fraction of agents per round:C"
# #     )
# #     parser.add_argument(
# #         "--num_corrupt", type=int, default=2, help="number of corrupt agents"
# #     )
# #     parser.add_argument(
# #         "--rounds", type=int, default=150, help="number of communication rounds:R"
# #     )
# #     parser.add_argument(
# #         "--local_ep", type=int, default=2, help="number of local epochs:E"
# #     )
# #     parser.add_argument("--bs", type=int, default=64, help="local batch size: B")
# #     parser.add_argument(
# #         "--client_lr", type=float, default=0.1, help="clients learning rate"
# #     )
# #     parser.add_argument(
# #         "--server_lr", type=float, default=1, help="servers learning rate"
# #     )
# #     parser.add_argument(
# #         "--target_class", type=int, default=7, help="target class for backdoor attack"
# #     )
# #     parser.add_argument(
# #         "--poison_frac",
# #         type=float,
# #         default=0.5,
# #         help="fraction of dataset to corrupt for backdoor attack",
# #     )
# #     parser.add_argument(
# #         "--pattern_type", type=str, default="plus", help="shape of bd pattern"
# #     )
# #     parser.add_argument(
# #         "--theta", type=int, default=8, help="break ties when votes sum to 0"
# #     )
# #     parser.add_argument(
# #         "--theta_ld", type=int, default=10, help="break ties when votes sum to 0"
# #     )
# #     parser.add_argument(
# #         "--snap", type=int, default=1, help="do inference in every num of snap rounds"
# #     )
# #     parser.add_argument(
# #         "--device",
# #         default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
# #         help="To use cuda, set to a specific GPU ID.",
# #     )
# #     parser.add_argument(
# #         "--num_workers", type=int, default=0, help="num of workers for multithreading"
# #     )
# #     parser.add_argument(
# #         "--dense_ratio",
# #         type=float,
# #         default=0.25,
# #         help="num of workers for multithreading",
# #     )
# #     parser.add_argument(
# #         "--anneal_factor",
# #         type=float,
# #         default=0.0001,
# #         help="num of workers for multithreading",
# #     )
# #     parser.add_argument(
# #         "--se_threshold",
# #         type=float,
# #         default=1e-4,
# #         help="num of workers for multithreading",
# #     )
# #     parser.add_argument("--non_iid", action="store_true", default=False)
# #     parser.add_argument("--debug", action="store_true", default=False)
# #     parser.add_argument("--beta", type=float, default=0.5)
# #     parser.add_argument(
# #         "--attack",
# #         type=str,
# #         default="badnet",
# #         choices=["badnet", "DBA", "neurotoxin", "pgd", "dynamic_trigger", "altermin"],
# #     )
# #     parser.add_argument(
# #         "--aggr",
# #         type=str,
# #         default="avg",
# #         choices=[
# #             "avg",
# #             "alignins",
# #             "rlr",
# #             "mkrum",
# #             "mmetric",
# #             "lockdown",
# #             "foolsgold",
# #             "rfa",
# #             "alphaguard"
# #         ],
# #         help="aggregation function to aggregate agents' local weights",
# #     )
# #     parser.add_argument("--lr_decay", type=float, default=0.99)
# #     parser.add_argument("--momentum", type=float, default=0.0)
# #     parser.add_argument("--mask_init", type=str, default="ERK")
# #     parser.add_argument("--wd", type=float, default=1e-4)
# #     parser.add_argument("--same_mask", type=int, default=1)
# #     parser.add_argument("--cease_poison", type=float, default=100000)
# #     parser.add_argument("--exp_name_extra", type=str, help="defence name", default="")
# #     parser.add_argument("--super_power", action="store_true")
# #     parser.add_argument("--clean", action="store_true")
# #     parser.add_argument("--sparsity", type=float, default=0.3)
# #     parser.add_argument("--lambda_s", type=float, default=1.0)
# #     parser.add_argument("--lambda_c", type=float, default=1.0)
    
# #     # Dynamic Trigger Attack parameters
# #     parser.add_argument("--trigger_size", type=int, default=5, help="size of trigger pattern")
# #     parser.add_argument("--trigger_lr", type=float, default=0.1, help="learning rate for trigger optimization")
# #     parser.add_argument("--trigger_outter_epochs", type=int, default=50, help="outer epochs for trigger optimization")
# #     parser.add_argument("--trigger_optim_interval", type=int, default=5, help="optimize trigger every N rounds")
# #     parser.add_argument("--dm_adv_epochs", type=int, default=2, help="adversarial model training epochs")
# #     parser.add_argument("--dm_adv_K", type=int, default=10, help="create adversarial models every K iterations")
# #     parser.add_argument("--dm_adv_model_count", type=int, default=2, help="number of adversarial models")
# #     parser.add_argument("--noise_loss_lambda", type=float, default=0.1, help="weight for adversarial loss")
    
# #     # AlterMin Attack parameters
# #     parser.add_argument("--altermin_attack_mode", type=str, default="targeted", choices=["targeted", "untargeted"], 
# #                        help="AlterMin attack mode: targeted or untargeted")
# #     parser.add_argument("--altermin_source_label", type=int, default=3, help="source label for AlterMin attack")
# #     parser.add_argument("--altermin_target_label", type=int, default=None, help="target label (default: args.target_class)")
# #     parser.add_argument("--altermin_poisoned_cnt", type=int, default=100, help="number of poisoned samples for AlterMin")
# #     parser.add_argument("--altermin_boosting_factor", type=float, default=1.5, help="boosting factor for AlterMin")
# #     parser.add_argument("--altermin_rho", type=float, default=1e-4, help="distance loss weight for AlterMin")
# #     parser.add_argument("--altermin_benign_epochs", type=int, default=None, help="benign training epochs (default: local_ep)")
# #     parser.add_argument("--altermin_malicious_epochs", type=int, default=1, help="malicious training epochs for AlterMin")

# #     args = parser.parse_args()
# #     print("参数解析完成！")
# #     print(f"使用的参数: data={args.data}, aggr={args.aggr}, attack={args.attack}")

# #     if args.clean:
# #         args.num_corrupt = 0
# #         args.exp_name_extra = "clean"

# #     if args.super_power:
# #         args.exp_name_extra = "sp"

# #     per_data_dict = {
# #         "rounds": {"fmnist": 50, "cifar10": 100, "cifar100": 100, "tinyimagenet": 50},
# #         "num_target": {"fmnist": 10, "cifar10": 10, "cifar100": 100, "tinyimagenet": 200,},
# #     }

# #     args.rounds = per_data_dict["rounds"][args.data]
# #     args.num_target = per_data_dict["num_target"][args.data]

# #     print("设置日志系统...")
# #     args.log_dir = utils.setup_logging(args)

# #     print(f"加载 {args.data} 数据集...")
# #     train_dataset, val_dataset = utils.get_datasets(args.data)
# #     print("数据集加载完成！")
# #     backdoor_train_dataset = None

# #     val_loader = DataLoader(
# #         val_dataset,
# #         batch_size=args.bs,
# #         shuffle=False,
# #         num_workers=args.num_workers,
# #         pin_memory=False,
# #     )
# #     if args.non_iid:
# #         user_groups = utils.distribute_data_dirichlet(train_dataset, args)
# #     else:
# #         user_groups = utils.distribute_data(
# #             train_dataset, args, n_classes=args.num_target
# #         )

# #     idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()

# #     if args.data != "tinyimagenet":
# #         poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
# #         utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
# #     else:
# #         poisoned_val_set = utils.DatasetSplit(
# #             copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args
# #         )

# #     poisoned_val_loader = DataLoader(
# #         poisoned_val_set,
# #         batch_size=args.bs,
# #         shuffle=False,
# #         num_workers=args.num_workers,
# #         pin_memory=False,
# #     )
# #     if args.data != "tinyimagenet":
# #         idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
# #         poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
# #         utils.poison_dataset(
# #             poisoned_val_set_only_x.dataset,
# #             args,
# #             idxs,
# #             poison_all=True,
# #             modify_label=False,
# #         )
# #     else:
# #         poisoned_val_set_only_x = utils.DatasetSplit(
# #             copy.deepcopy(val_dataset),
# #             idxs,
# #             runtime_poison=True,
# #             args=args,
# #             modify_label=False,
# #         )

# #     poisoned_val_only_x_loader = DataLoader(
# #         poisoned_val_set_only_x,
# #         batch_size=args.bs,
# #         shuffle=False,
# #         num_workers=args.num_workers,
# #         pin_memory=False,
# #     )

# #     # initialize a model, and the agents
# #     print("初始化全局模型...")
# #     global_model = models.get_model(args.data, args).to(args.device)
# #     print("模型初始化完成！")

# #     global_mask = {}
# #     neurotoxin_mask = {}
# #     updates_dict = {}
# #     n_model_params = len(
# #         parameters_to_vector(
# #             [global_model.state_dict()[name] for name in global_model.state_dict()]
# #         )
# #     )
# #     params = {
# #         name: copy.deepcopy(global_model.state_dict()[name])
# #         for name in global_model.state_dict()
# #     }

# #     if args.aggr == "lockdown":
# #         sparsity = utils.calculate_sparsities(args, params, distribution=args.mask_init)
# #         mask = utils.init_masks(params, sparsity)

# #     agents, agent_data_sizes = [], {}
# #     print(f"创建 {args.num_agents} 个客户端代理...")
# #     for _id in range(0, args.num_agents):
# #         if args.aggr == "lockdown":
# #             if args.same_mask == 0:
# #                 agent = Agent_s(
# #                     _id,
# #                     args,
# #                     train_dataset,
# #                     user_groups[_id],
# #                     mask=utils.init_masks(params, sparsity),
# #                     backdoor_train_dataset=backdoor_train_dataset,
# #                 )
# #             else:
# #                 agent = Agent_s(
# #                     _id,
# #                     args,
# #                     train_dataset,
# #                     user_groups[_id],
# #                     mask=mask,
# #                     backdoor_train_dataset=backdoor_train_dataset,
# #                 )
# #         else:
# #             agent = Agent(
# #                 _id,
# #                 args,
# #                 train_dataset,
# #                 user_groups[_id],
# #                 backdoor_train_dataset=backdoor_train_dataset,
# #             )
# #         agent.is_malicious = 1 if _id < args.num_corrupt else 0
# #         agent_data_sizes[_id] = agent.n_data
# #         agents.append(agent)

# #         logging.info(
# #             "build client:{} mal:{} data_num:{}".format(
# #                 _id, agent.is_malicious, agent.n_data
# #             )
# #         )
    
# #     # 如果是 AlterMin 攻击，需要为恶意客户端初始化投毒数据加载器
# #     if args.attack == 'altermin':
# #         print("初始化 AlterMin 攻击器的投毒数据...")
# #         for _id in range(args.num_corrupt):
# #             if agents[_id].altermin_attacker is not None:
# #                 agents[_id].altermin_attacker.init_poisoned_loader(val_dataset)
# #         print("AlterMin 投毒数据初始化完成！")

# #     aggregator = Aggregation(agent_data_sizes, n_model_params, args)

# #     criterion = nn.CrossEntropyLoss().to(args.device)
# #     agent_updates_dict = {}

# #     best_acc = -1

# #     for rnd in range(1, args.rounds + 1):
# #         logging.info("--------round {} ------------".format(rnd))
# #         rnd_global_params = parameters_to_vector(
# #             [
# #                 copy.deepcopy(global_model.state_dict()[name])
# #                 for name in global_model.state_dict()
# #             ]
# #         )
# #         agent_updates_dict = {}
# #         chosen = np.random.choice(
# #             args.num_agents,
# #             math.floor(args.num_agents * args.agent_frac),
# #             replace=False,
# #         )
# #         chosen = sorted(chosen)
# #         if args.aggr == "lockdown":
# #             old_mask = [copy.deepcopy(agent.mask) for agent in agents]

# #         for agent_id in chosen:
# #             if agents[agent_id].is_malicious and args.super_power:
# #                 continue
# #             global_model = global_model.to(args.device)

# #             if args.aggr == "lockdown":
# #                 update = agents[agent_id].local_train(
# #                     global_model,
# #                     criterion,
# #                     rnd,
# #                     global_mask=global_mask,
# #                     neurotoxin_mask=neurotoxin_mask,
# #                     updates_dict=updates_dict,
# #                 )
# #             else:
# #                 update = agents[agent_id].local_train(
# #                     global_model, criterion, rnd, neurotoxin_mask=neurotoxin_mask
# #                 )
# #             agent_updates_dict[agent_id] = update
# #             utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)

# #         # aggregate params obtained by agents and update the global params
# #         updates_dict, neurotoxin_mask = aggregator.aggregate_updates(
# #             global_model, agent_updates_dict
# #         )

# #         # inference in every args.snap rounds
# #         logging.info("---------Test {} ------------".format(rnd))
# #         if rnd % args.snap == 0:
# #             if args.aggr != "lockdown":
# #                 val_acc = utils.get_loss_n_accuracy(
# #                     global_model, criterion, val_loader, args, rnd, args.num_target
# #                 )
# #                 asr = utils.get_loss_n_accuracy(
# #                     global_model,
# #                     criterion,
# #                     poisoned_val_loader,
# #                     args,
# #                     rnd,
# #                     num_classes=args.num_target,
# #                 )
# #                 poison_acc = utils.get_loss_n_accuracy(
# #                     global_model,
# #                     criterion,
# #                     poisoned_val_only_x_loader,
# #                     args,
# #                     rnd,
# #                     args.num_target,
# #                 )
# #             else:
# #                 test_model = copy.deepcopy(global_model)

# #                 # CF
# #                 for name, param in test_model.named_parameters():
# #                     mask = 0
# #                     for id, agent in enumerate(agents):
# #                         mask += old_mask[id][name].to(args.device)
# #                     param.data = torch.where(
# #                         mask.to(args.device) >= args.theta_ld,
# #                         param,
# #                         torch.zeros_like(param),
# #                     )
# #                 val_acc = utils.get_loss_n_accuracy(
# #                     test_model, criterion, val_loader, args, rnd, args.num_target
# #                 )
# #                 asr = utils.get_loss_n_accuracy(
# #                     test_model,
# #                     criterion,
# #                     poisoned_val_loader,
# #                     args,
# #                     rnd,
# #                     args.num_target,
# #                 )
# #                 poison_acc = utils.get_loss_n_accuracy(
# #                     test_model,
# #                     criterion,
# #                     poisoned_val_only_x_loader,
# #                     args,
# #                     rnd,
# #                     args.num_target,
# #                 )
# #                 del test_model

# #             logging.info("Clean ACC:              %.4f" % val_acc)
# #             logging.info("Attack Success Ratio:   %.4f" % asr)
# #             logging.info("Backdoor ACC:           %.4f" % poison_acc)

# #             if val_acc > best_acc:
# #                 best_acc = val_acc
# #                 best_asr = asr
# #                 best_bcdr_acc = poison_acc

# #         logging.info("------------------------------".format(rnd))

# #     logging.info("Best results:")
# #     logging.info("Clean ACC:              %.4f" % best_acc)
# #     logging.info("Attack Success Ratio:   %.4f" % best_asr)
# #     logging.info("Backdoor ACC:           %.4f" % best_bcdr_acc)
# #     logging.info("Training has finished!")


# import utils
# import models
# import math
# import copy
# import numpy as np
# from agent import Agent
# from agent_sparse import Agent as Agent_s
# from aggregation import Aggregation
# import torch
# import random
# from torch.utils.data import DataLoader
# import torch.nn as nn
# from torch.nn.utils import parameters_to_vector
# import logging
# import argparse
# import os
# import warnings

# warnings.filterwarnings("ignore")

# if __name__ == "__main__":
#     print("程序开始运行...")
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = False
#     torch.manual_seed(0)
#     torch.cuda.manual_seed_all(0)
#     np.random.seed(0)
#     random.seed(0)
#     torch.backends.cudnn.deterministic = True

#     print("初始化参数解析器...")
#     parser = argparse.ArgumentParser(description="pass in a parameter")

#     parser.add_argument(
#         "--data", type=str, default="cifar10", help="dataset we want to train on"
#     )
#     parser.add_argument("--num_agents", type=int, default=20, help="number of agents:K")
#     parser.add_argument(
#         "--agent_frac", type=float, default=1.0, help="fraction of agents per round:C"
#     )
#     parser.add_argument(
#         "--num_corrupt", type=int, default=2, help="number of corrupt agents"
#     )
#     parser.add_argument(
#         "--rounds", type=int, default=150, help="number of communication rounds:R"
#     )
#     parser.add_argument(
#         "--local_ep", type=int, default=2, help="number of local epochs:E"
#     )
#     parser.add_argument("--bs", type=int, default=64, help="local batch size: B")
#     parser.add_argument(
#         "--client_lr", type=float, default=0.1, help="clients learning rate"
#     )
#     parser.add_argument(
#         "--server_lr", type=float, default=1, help="servers learning rate"
#     )
#     parser.add_argument(
#         "--target_class", type=int, default=7, help="target class for backdoor attack"
#     )
#     parser.add_argument(
#         "--poison_frac",
#         type=float,
#         default=0.5,
#         help="fraction of dataset to corrupt for backdoor attack",
#     )
#     parser.add_argument(
#         "--pattern_type", type=str, default="plus", help="shape of bd pattern"
#     )
#     parser.add_argument(
#         "--theta", type=int, default=8, help="break ties when votes sum to 0"
#     )
#     parser.add_argument(
#         "--theta_ld", type=int, default=10, help="break ties when votes sum to 0"
#     )
#     parser.add_argument(
#         "--snap", type=int, default=1, help="do inference in every num of snap rounds"
#     )
#     parser.add_argument(
#         "--device",
#         default=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
#         help="To use cuda, set to a specific GPU ID.",
#     )
#     parser.add_argument(
#         "--num_workers", type=int, default=0, help="num of workers for multithreading"
#     )
#     parser.add_argument(
#         "--dense_ratio",
#         type=float,
#         default=0.25,
#         help="num of workers for multithreading",
#     )
#     parser.add_argument(
#         "--anneal_factor",
#         type=float,
#         default=0.0001,
#         help="num of workers for multithreading",
#     )
#     parser.add_argument(
#         "--se_threshold",
#         type=float,
#         default=1e-4,
#         help="num of workers for multithreading",
#     )
#     parser.add_argument("--non_iid", action="store_true", default=False)
#     parser.add_argument("--debug", action="store_true", default=False)
#     parser.add_argument("--beta", type=float, default=0.5)
#     parser.add_argument(
#         "--attack",
#         type=str,
#         default="badnet",
#         choices=["badnet", "DBA", "neurotoxin", "pgd", "dynamic_trigger", "altermin"],
#     )
#     parser.add_argument(
#         "--aggr",
#         type=str,
#         default="avg",
#         choices=[
#             "avg",
#             "alignins",
#             "rlr",
#             "mkrum",
#             "mmetric",
#             "lockdown",
#             "foolsgold",
#             "rfa",
#             "alphaguard"
#         ],
#         help="aggregation function to aggregate agents' local weights",
#     )
#     parser.add_argument("--lr_decay", type=float, default=0.99)
#     parser.add_argument("--momentum", type=float, default=0.0)
#     parser.add_argument("--mask_init", type=str, default="ERK")
#     parser.add_argument("--wd", type=float, default=1e-4)
#     parser.add_argument("--same_mask", type=int, default=1)
#     parser.add_argument("--cease_poison", type=float, default=100000)
#     parser.add_argument("--exp_name_extra", type=str, help="defence name", default="")
#     parser.add_argument("--super_power", action="store_true")
#     parser.add_argument("--clean", action="store_true")
#     parser.add_argument("--sparsity", type=float, default=0.3)
#     parser.add_argument("--lambda_s", type=float, default=1.0)
#     parser.add_argument("--lambda_c", type=float, default=1.0)
    
#     # Dynamic Trigger Attack parameters
#     parser.add_argument("--trigger_size", type=int, default=5, help="size of trigger pattern")
#     parser.add_argument("--trigger_lr", type=float, default=0.1, help="learning rate for trigger optimization")
#     parser.add_argument("--trigger_outter_epochs", type=int, default=50, help="outer epochs for trigger optimization")
#     parser.add_argument("--trigger_optim_interval", type=int, default=5, help="optimize trigger every N rounds")
#     parser.add_argument("--dm_adv_epochs", type=int, default=2, help="adversarial model training epochs")
#     parser.add_argument("--dm_adv_K", type=int, default=10, help="create adversarial models every K iterations")
#     parser.add_argument("--dm_adv_model_count", type=int, default=2, help="number of adversarial models")
#     parser.add_argument("--noise_loss_lambda", type=float, default=0.1, help="weight for adversarial loss")
    
#     # AlterMin Attack parameters
#     parser.add_argument("--altermin_attack_mode", type=str, default="targeted", choices=["targeted", "untargeted"], 
#                        help="AlterMin attack mode: targeted or untargeted")
#     parser.add_argument("--altermin_source_label", type=int, default=3, help="source label for AlterMin attack")
#     parser.add_argument("--altermin_target_label", type=int, default=None, help="target label (default: args.target_class)")
#     parser.add_argument("--altermin_poisoned_cnt", type=int, default=100, help="number of poisoned samples for AlterMin")
#     parser.add_argument("--altermin_boosting_factor", type=float, default=1.5, help="boosting factor for AlterMin")
#     parser.add_argument("--altermin_rho", type=float, default=1e-4, help="distance loss weight for AlterMin")
#     parser.add_argument("--altermin_benign_epochs", type=int, default=None, help="benign training epochs (default: local_ep)")
#     parser.add_argument("--altermin_malicious_epochs", type=int, default=1, help="malicious training epochs for AlterMin")

#     args = parser.parse_args()
#     print("参数解析完成！")
#     print(f"使用的参数: data={args.data}, aggr={args.aggr}, attack={args.attack}")

#     if args.clean:
#         args.num_corrupt = 0
#         args.exp_name_extra = "clean"

#     if args.super_power:
#         args.exp_name_extra = "sp"

#     per_data_dict = {
#         "rounds": {"fmnist": 50, "cifar10": 100, "cifar100": 100, "tinyimagenet": 50},
#         "num_target": {"fmnist": 10, "cifar10": 10, "cifar100": 100, "tinyimagenet": 200,},
#     }

#     args.rounds = per_data_dict["rounds"][args.data]
#     args.num_target = per_data_dict["num_target"][args.data]

#     print("设置日志系统...")
#     args.log_dir = utils.setup_logging(args)

#     print(f"加载 {args.data} 数据集...")
#     train_dataset, val_dataset = utils.get_datasets(args.data)
#     print("数据集加载完成！")
#     backdoor_train_dataset = None

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=args.bs,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=False,
#     )
#     if args.non_iid:
#         user_groups = utils.distribute_data_dirichlet(train_dataset, args)
#     else:
#         user_groups = utils.distribute_data(
#             train_dataset, args, n_classes=args.num_target
#         )

#     # 根据攻击类型选择不同的测试数据
#     if args.attack == 'altermin':
#         # AlterMin: 只用源标签的样本测试 ASR
#         altermin_source = getattr(args, 'altermin_source_label', 3)
#         idxs = [i for i, label in enumerate(val_dataset.targets) if label == altermin_source]
#         poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
#         # 将所有源标签样本的标签改为目标标签
#         for idx in idxs:
#             poisoned_val_set.dataset.targets[idx] = args.target_class
#     else:
#         # 其他攻击：使用所有非目标类别的样本
#         idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
#         if args.data != "tinyimagenet":
#             poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
#             utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
#         else:
#             poisoned_val_set = utils.DatasetSplit(
#                 copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args
#             )

#     poisoned_val_loader = DataLoader(
#         poisoned_val_set,
#         batch_size=args.bs,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=False,
#     )
#     # 创建 poisoned_val_only_x（用于测试 Backdoor ACC）
#     # 对所有攻击：测试所有非目标类别的样本，标签保持不变
#     idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
    
#     if args.attack == 'altermin':
#         # AlterMin: 干净图像 + 原标签
#         # Backdoor ACC 测试模型在所有非目标类别上的准确率（检测模型整体是否被污染）
#         poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
#         # 不修改图像和标签，保持原样
#     else:
#         # BadNet/DBA 等: 带触发器图像 + 原标签  
#         # Backdoor ACC 测试模型能否抵抗触发器，仍然正确识别原类别
#         if args.data != "tinyimagenet":
#             poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
#             utils.poison_dataset(
#                 poisoned_val_set_only_x.dataset,
#                 args,
#                 idxs,
#                 poison_all=True,
#                 modify_label=False,
#             )
#         else:
#             poisoned_val_set_only_x = utils.DatasetSplit(
#                 copy.deepcopy(val_dataset),
#                 idxs,
#                 runtime_poison=True,
#                 args=args,
#                 modify_label=False,
#             )

#     poisoned_val_only_x_loader = DataLoader(
#         poisoned_val_set_only_x,
#         batch_size=args.bs,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=False,
#     )

#     # initialize a model, and the agents
#     print("初始化全局模型...")
#     global_model = models.get_model(args.data, args).to(args.device)
#     print("模型初始化完成！")

#     global_mask = {}
#     neurotoxin_mask = {}
#     updates_dict = {}
#     n_model_params = len(
#         parameters_to_vector(
#             [global_model.state_dict()[name] for name in global_model.state_dict()]
#         )
#     )
#     params = {
#         name: copy.deepcopy(global_model.state_dict()[name])
#         for name in global_model.state_dict()
#     }

#     if args.aggr == "lockdown":
#         sparsity = utils.calculate_sparsities(args, params, distribution=args.mask_init)
#         mask = utils.init_masks(params, sparsity)

#     agents, agent_data_sizes = [], {}
#     print(f"创建 {args.num_agents} 个客户端代理...")
#     for _id in range(0, args.num_agents):
#         if args.aggr == "lockdown":
#             if args.same_mask == 0:
#                 agent = Agent_s(
#                     _id,
#                     args,
#                     train_dataset,
#                     user_groups[_id],
#                     mask=utils.init_masks(params, sparsity),
#                     backdoor_train_dataset=backdoor_train_dataset,
#                 )
#             else:
#                 agent = Agent_s(
#                     _id,
#                     args,
#                     train_dataset,
#                     user_groups[_id],
#                     mask=mask,
#                     backdoor_train_dataset=backdoor_train_dataset,
#                 )
#         else:
#             agent = Agent(
#                 _id,
#                 args,
#                 train_dataset,
#                 user_groups[_id],
#                 backdoor_train_dataset=backdoor_train_dataset,
#             )
#         agent.is_malicious = 1 if _id < args.num_corrupt else 0
#         agent_data_sizes[_id] = agent.n_data
#         agents.append(agent)

#         logging.info(
#             "build client:{} mal:{} data_num:{}".format(
#                 _id, agent.is_malicious, agent.n_data
#             )
#         )
    
#     # 如果是 AlterMin 攻击，需要为恶意客户端初始化投毒数据加载器
#     if args.attack == 'altermin':
#         print("初始化 AlterMin 攻击器的投毒数据...")
#         for _id in range(args.num_corrupt):
#             if agents[_id].altermin_attacker is not None:
#                 agents[_id].altermin_attacker.init_poisoned_loader(val_dataset)
#         print("AlterMin 投毒数据初始化完成！")

#     aggregator = Aggregation(agent_data_sizes, n_model_params, args)

#     criterion = nn.CrossEntropyLoss().to(args.device)
#     agent_updates_dict = {}

#     best_acc = -1

#     for rnd in range(1, args.rounds + 1):
#         logging.info("--------round {} ------------".format(rnd))
#         rnd_global_params = parameters_to_vector(
#             [
#                 copy.deepcopy(global_model.state_dict()[name])
#                 for name in global_model.state_dict()
#             ]
#         )
#         agent_updates_dict = {}
#         chosen = np.random.choice(
#             args.num_agents,
#             math.floor(args.num_agents * args.agent_frac),
#             replace=False,
#         )
#         chosen = sorted(chosen)
#         if args.aggr == "lockdown":
#             old_mask = [copy.deepcopy(agent.mask) for agent in agents]

#         for agent_id in chosen:
#             if agents[agent_id].is_malicious and args.super_power:
#                 continue
#             global_model = global_model.to(args.device)

#             if args.aggr == "lockdown":
#                 update = agents[agent_id].local_train(
#                     global_model,
#                     criterion,
#                     rnd,
#                     global_mask=global_mask,
#                     neurotoxin_mask=neurotoxin_mask,
#                     updates_dict=updates_dict,
#                 )
#             else:
#                 update = agents[agent_id].local_train(
#                     global_model, criterion, rnd, neurotoxin_mask=neurotoxin_mask
#                 )
#             agent_updates_dict[agent_id] = update
#             utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)

#         # aggregate params obtained by agents and update the global params
#         updates_dict, neurotoxin_mask = aggregator.aggregate_updates(
#             global_model, agent_updates_dict
#         )

#         # inference in every args.snap rounds
#         logging.info("---------Test {} ------------".format(rnd))
#         if rnd % args.snap == 0:
#             if args.aggr != "lockdown":
#                 val_acc = utils.get_loss_n_accuracy(
#                     global_model, criterion, val_loader, args, rnd, args.num_target
#                 )
#                 asr = utils.get_loss_n_accuracy(
#                     global_model,
#                     criterion,
#                     poisoned_val_loader,
#                     args,
#                     rnd,
#                     num_classes=args.num_target,
#                 )
#                 poison_acc = utils.get_loss_n_accuracy(
#                     global_model,
#                     criterion,
#                     poisoned_val_only_x_loader,
#                     args,
#                     rnd,
#                     args.num_target,
#                 )
#             else:
#                 test_model = copy.deepcopy(global_model)

#                 # CF
#                 for name, param in test_model.named_parameters():
#                     mask = 0
#                     for id, agent in enumerate(agents):
#                         mask += old_mask[id][name].to(args.device)
#                     param.data = torch.where(
#                         mask.to(args.device) >= args.theta_ld,
#                         param,
#                         torch.zeros_like(param),
#                     )
#                 val_acc = utils.get_loss_n_accuracy(
#                     test_model, criterion, val_loader, args, rnd, args.num_target
#                 )
#                 asr = utils.get_loss_n_accuracy(
#                     test_model,
#                     criterion,
#                     poisoned_val_loader,
#                     args,
#                     rnd,
#                     args.num_target,
#                 )
#                 poison_acc = utils.get_loss_n_accuracy(
#                     test_model,
#                     criterion,
#                     poisoned_val_only_x_loader,
#                     args,
#                     rnd,
#                     args.num_target,
#                 )
#                 del test_model

#             logging.info("Clean ACC:              %.4f" % val_acc)
#             logging.info("Attack Success Ratio:   %.4f" % asr)
#             logging.info("Backdoor ACC:           %.4f" % poison_acc)

#             if val_acc > best_acc:
#                 best_acc = val_acc
#                 best_asr = asr
#                 best_bcdr_acc = poison_acc

#         logging.info("------------------------------".format(rnd))

#     logging.info("Best results:")
#     logging.info("Clean ACC:              %.4f" % best_acc)
#     logging.info("Attack Success Ratio:   %.4f" % best_asr)
#     logging.info("Backdoor ACC:           %.4f" % best_bcdr_acc)
#     logging.info("Training has finished!")


import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from agent_sparse import Agent as Agent_s
from aggregation import Aggregation
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import logging
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("程序开始运行...")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    print("初始化参数解析器...")
    parser = argparse.ArgumentParser(description="pass in a parameter")

    parser.add_argument(
        "--data", type=str, default="cifar10", help="dataset we want to train on"
    )
    parser.add_argument("--num_agents", type=int, default=20, help="number of agents:K")
    parser.add_argument(
        "--agent_frac", type=float, default=1.0, help="fraction of agents per round:C"
    )
    parser.add_argument(
        "--num_corrupt", type=int, default=2, help="number of corrupt agents"
    )
    parser.add_argument(
        "--rounds", type=int, default=150, help="number of communication rounds:R"
    )
    parser.add_argument(
        "--local_ep", type=int, default=2, help="number of local epochs:E"
    )
    parser.add_argument("--bs", type=int, default=64, help="local batch size: B")
    parser.add_argument(
        "--client_lr", type=float, default=0.1, help="clients learning rate"
    )
    parser.add_argument(
        "--server_lr", type=float, default=1, help="servers learning rate"
    )
    parser.add_argument(
        "--target_class", type=int, default=7, help="target class for backdoor attack"
    )
    parser.add_argument(
        "--poison_frac",
        type=float,
        default=0.5,
        help="fraction of dataset to corrupt for backdoor attack",
    )
    parser.add_argument(
        "--pattern_type", type=str, default="plus", help="shape of bd pattern"
    )
    parser.add_argument(
        "--theta", type=int, default=8, help="break ties when votes sum to 0"
    )
    parser.add_argument(
        "--theta_ld", type=int, default=10, help="break ties when votes sum to 0"
    )
    parser.add_argument(
        "--snap", type=int, default=1, help="do inference in every num of snap rounds"
    )
    parser.add_argument(
        "--device",
        default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="To use cuda, set to a specific GPU ID.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="num of workers for multithreading"
    )
    parser.add_argument(
        "--dense_ratio",
        type=float,
        default=0.25,
        help="num of workers for multithreading",
    )
    parser.add_argument(
        "--anneal_factor",
        type=float,
        default=0.0001,
        help="num of workers for multithreading",
    )
    parser.add_argument(
        "--se_threshold",
        type=float,
        default=1e-4,
        help="num of workers for multithreading",
    )
    parser.add_argument("--non_iid", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--attack",
        type=str,
        default="badnet",
        choices=["badnet", "DBA", "neurotoxin", "pgd", "dynamic_trigger", "altermin", "model_replacement"],
    )
    parser.add_argument(
        "--aggr",
        type=str,
        default="avg",
        choices=[
            "avg",
            "alignins",
            "rlr",
            "mkrum",
            "mmetric",
            "lockdown",
            "foolsgold",
            "rfa",
            "alphaguard"
        ],
        help="aggregation function to aggregate agents' local weights",
    )
    parser.add_argument("--lr_decay", type=float, default=0.99)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--mask_init", type=str, default="ERK")
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--same_mask", type=int, default=1)
    parser.add_argument("--cease_poison", type=float, default=100000)
    parser.add_argument("--exp_name_extra", type=str, help="defence name", default="")
    parser.add_argument("--super_power", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--lambda_s", type=float, default=1.0)
    parser.add_argument("--lambda_c", type=float, default=1.0)
    
    # Dynamic Trigger Attack parameters
    parser.add_argument("--trigger_size", type=int, default=5, help="size of trigger pattern")
    parser.add_argument("--trigger_lr", type=float, default=0.1, help="learning rate for trigger optimization")
    parser.add_argument("--trigger_outter_epochs", type=int, default=50, help="outer epochs for trigger optimization")
    parser.add_argument("--trigger_optim_interval", type=int, default=5, help="optimize trigger every N rounds")
    parser.add_argument("--dm_adv_epochs", type=int, default=2, help="adversarial model training epochs")
    parser.add_argument("--dm_adv_K", type=int, default=10, help="create adversarial models every K iterations")
    parser.add_argument("--dm_adv_model_count", type=int, default=2, help="number of adversarial models")
    parser.add_argument("--noise_loss_lambda", type=float, default=0.1, help="weight for adversarial loss")
    
    # AlterMin Attack parameters
    parser.add_argument("--altermin_attack_mode", type=str, default="targeted", choices=["targeted", "untargeted"], 
                       help="AlterMin attack mode: targeted or untargeted")
    parser.add_argument("--altermin_source_label", type=int, default=3, help="source label for AlterMin attack")
    parser.add_argument("--altermin_target_label", type=int, default=None, help="target label (default: args.target_class)")
    parser.add_argument("--altermin_poisoned_cnt", type=int, default=100, help="number of poisoned samples for AlterMin")
    parser.add_argument("--altermin_boosting_factor", type=float, default=1.5, help="boosting factor for AlterMin")
    parser.add_argument("--altermin_rho", type=float, default=1e-4, help="distance loss weight for AlterMin")
    parser.add_argument("--altermin_benign_epochs", type=int, default=None, help="benign training epochs (default: local_ep)")
    parser.add_argument("--altermin_malicious_epochs", type=int, default=1, help="malicious training epochs for AlterMin")
    
    # Model Replacement Attack parameters
    parser.add_argument("--model_replacement_scaling_factor", type=float, default=50, 
                       help="scaling factor γ for Model Replacement attack")
    parser.add_argument("--model_replacement_alpha", type=float, default=0.5, 
                       help="alpha weight for classification loss in Model Replacement")
    parser.add_argument("--model_replacement_target_label", type=int, default=None, 
                       help="target label for Model Replacement (default: args.target_class)")
    parser.add_argument("--model_replacement_constrain_weight", type=float, default=0.02, 
                       help="final scaling weight for Model Replacement update")

    args = parser.parse_args()
    print("参数解析完成！")
    print(f"使用的参数: data={args.data}, aggr={args.aggr}, attack={args.attack}")

    if args.clean:
        args.num_corrupt = 0
        args.exp_name_extra = "clean"

    if args.super_power:
        args.exp_name_extra = "sp"

    per_data_dict = {
        "rounds": {"fmnist": 50, "cifar10": 100, "cifar100": 100, "tinyimagenet": 50},
        "num_target": {"fmnist": 10, "cifar10": 10, "cifar100": 100, "tinyimagenet": 200,},
    }

    args.rounds = per_data_dict["rounds"][args.data]
    args.num_target = per_data_dict["num_target"][args.data]

    print("设置日志系统...")
    args.log_dir = utils.setup_logging(args)

    print(f"加载 {args.data} 数据集...")
    train_dataset, val_dataset = utils.get_datasets(args.data)
    print("数据集加载完成！")
    backdoor_train_dataset = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    if args.non_iid:
        user_groups = utils.distribute_data_dirichlet(train_dataset, args)
    else:
        user_groups = utils.distribute_data(
            train_dataset, args, n_classes=args.num_target
        )

    # 根据攻击类型选择不同的测试数据
    if args.attack == 'altermin':
        # AlterMin: 只用源标签的样本测试 ASR
        altermin_source = getattr(args, 'altermin_source_label', 3)
        idxs = [i for i, label in enumerate(val_dataset.targets) if label == altermin_source]
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        # 将所有源标签样本的标签改为目标标签
        for idx in idxs:
            poisoned_val_set.dataset.targets[idx] = args.target_class
    else:
        # 其他攻击：使用所有非目标类别的样本
        idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
        if args.data != "tinyimagenet":
            poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
            utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
        else:
            poisoned_val_set = utils.DatasetSplit(
                copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args
            )

    poisoned_val_loader = DataLoader(
        poisoned_val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    # 创建 poisoned_val_only_x（用于测试 Backdoor ACC）
    # 对所有攻击：测试所有非目标类别的样本，标签保持不变
    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
    
    if args.attack == 'altermin':
        # AlterMin: 干净图像 + 原标签
        # Backdoor ACC 测试模型在所有非目标类别上的准确率（检测模型整体是否被污染）
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        # 不修改图像和标签，保持原样
    else:
        # BadNet/DBA 等: 带触发器图像 + 原标签  
        # Backdoor ACC 测试模型能否抵抗触发器，仍然正确识别原类别
        if args.data != "tinyimagenet":
            poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
            utils.poison_dataset(
                poisoned_val_set_only_x.dataset,
                args,
                idxs,
                poison_all=True,
                modify_label=False,
            )
        else:
            poisoned_val_set_only_x = utils.DatasetSplit(
                copy.deepcopy(val_dataset),
                idxs,
                runtime_poison=True,
                args=args,
                modify_label=False,
            )

    poisoned_val_only_x_loader = DataLoader(
        poisoned_val_set_only_x,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # initialize a model, and the agents
    print("初始化全局模型...")
    global_model = models.get_model(args.data, args).to(args.device)
    print("模型初始化完成！")

    global_mask = {}
    neurotoxin_mask = {}
    updates_dict = {}
    n_model_params = len(
        parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]
        )
    )
    params = {
        name: copy.deepcopy(global_model.state_dict()[name])
        for name in global_model.state_dict()
    }

    if args.aggr == "lockdown":
        sparsity = utils.calculate_sparsities(args, params, distribution=args.mask_init)
        mask = utils.init_masks(params, sparsity)

    agents, agent_data_sizes = [], {}
    print(f"创建 {args.num_agents} 个客户端代理...")
    for _id in range(0, args.num_agents):
        if args.aggr == "lockdown":
            if args.same_mask == 0:
                agent = Agent_s(
                    _id,
                    args,
                    train_dataset,
                    user_groups[_id],
                    mask=utils.init_masks(params, sparsity),
                    backdoor_train_dataset=backdoor_train_dataset,
                )
            else:
                agent = Agent_s(
                    _id,
                    args,
                    train_dataset,
                    user_groups[_id],
                    mask=mask,
                    backdoor_train_dataset=backdoor_train_dataset,
                )
        else:
            agent = Agent(
                _id,
                args,
                train_dataset,
                user_groups[_id],
                backdoor_train_dataset=backdoor_train_dataset,
            )
        agent.is_malicious = 1 if _id < args.num_corrupt else 0
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

        logging.info(
            "build client:{} mal:{} data_num:{}".format(
                _id, agent.is_malicious, agent.n_data
            )
        )
    
    # 如果是 AlterMin 攻击，需要为恶意客户端初始化投毒数据加载器
    if args.attack == 'altermin':
        print("初始化 AlterMin 攻击器的投毒数据...")
        for _id in range(args.num_corrupt):
            if agents[_id].altermin_attacker is not None:
                agents[_id].altermin_attacker.init_poisoned_loader(val_dataset)
        print("AlterMin 投毒数据初始化完成！")

    aggregator = Aggregation(agent_data_sizes, n_model_params, args)

    criterion = nn.CrossEntropyLoss().to(args.device)
    agent_updates_dict = {}

    best_acc = -1

    for rnd in range(1, args.rounds + 1):
        logging.info("--------round {} ------------".format(rnd))
        rnd_global_params = parameters_to_vector(
            [
                copy.deepcopy(global_model.state_dict()[name])
                for name in global_model.state_dict()
            ]
        )
        agent_updates_dict = {}
        chosen = np.random.choice(
            args.num_agents,
            math.floor(args.num_agents * args.agent_frac),
            replace=False,
        )
        chosen = sorted(chosen)
        if args.aggr == "lockdown":
            old_mask = [copy.deepcopy(agent.mask) for agent in agents]

        for agent_id in chosen:
            if agents[agent_id].is_malicious and args.super_power:
                continue
            global_model = global_model.to(args.device)

            if args.aggr == "lockdown":
                update = agents[agent_id].local_train(
                    global_model,
                    criterion,
                    rnd,
                    global_mask=global_mask,
                    neurotoxin_mask=neurotoxin_mask,
                    updates_dict=updates_dict,
                )
            else:
                update = agents[agent_id].local_train(
                    global_model, criterion, rnd, neurotoxin_mask=neurotoxin_mask
                )
            agent_updates_dict[agent_id] = update
            utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)

        # aggregate params obtained by agents and update the global params
        updates_dict, neurotoxin_mask = aggregator.aggregate_updates(
            global_model, agent_updates_dict
        )

        # inference in every args.snap rounds
        logging.info("---------Test {} ------------".format(rnd))
        if rnd % args.snap == 0:
            if args.aggr != "lockdown":
                val_acc = utils.get_loss_n_accuracy(
                    global_model, criterion, val_loader, args, rnd, args.num_target
                )
                asr = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    poisoned_val_loader,
                    args,
                    rnd,
                    num_classes=args.num_target,
                )
                poison_acc = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    poisoned_val_only_x_loader,
                    args,
                    rnd,
                    args.num_target,
                )
            else:
                test_model = copy.deepcopy(global_model)

                # CF
                for name, param in test_model.named_parameters():
                    mask = 0
                    for id, agent in enumerate(agents):
                        mask += old_mask[id][name].to(args.device)
                    param.data = torch.where(
                        mask.to(args.device) >= args.theta_ld,
                        param,
                        torch.zeros_like(param),
                    )
                val_acc = utils.get_loss_n_accuracy(
                    test_model, criterion, val_loader, args, rnd, args.num_target
                )
                asr = utils.get_loss_n_accuracy(
                    test_model,
                    criterion,
                    poisoned_val_loader,
                    args,
                    rnd,
                    args.num_target,
                )
                poison_acc = utils.get_loss_n_accuracy(
                    test_model,
                    criterion,
                    poisoned_val_only_x_loader,
                    args,
                    rnd,
                    args.num_target,
                )
                del test_model

            logging.info("Clean ACC:              %.4f" % val_acc)
            logging.info("Attack Success Ratio:   %.4f" % asr)
            logging.info("Backdoor ACC:           %.4f" % poison_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_asr = asr
                best_bcdr_acc = poison_acc

        logging.info("------------------------------".format(rnd))

    logging.info("Best results:")
    logging.info("Clean ACC:              %.4f" % best_acc)
    logging.info("Attack Success Ratio:   %.4f" % best_asr)
    logging.info("Backdoor ACC:           %.4f" % best_bcdr_acc)
    logging.info("Training has finished!")
