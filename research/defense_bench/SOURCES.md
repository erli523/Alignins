# 论文、实现来源与忠实性边界

检索日期：2026-11-17。优先级为作者论文／正式会议页面、作者代码、第三方复现。源码仅下载到临时目录用于核对，不将没有明确许可的作者仓库整体复制进本项目。新增公式实现与项目适配代码有独立来源说明；没有改写已有作者文件。

## RLR — AAAI 2021

- 论文：[Defending against Backdoors in Federated Learning with Robust Learning Rate](https://ojs.aaai.org/index.php/AAAI/article/view/17118)。AAAI 直连遇到访问拦截；同时核对 [作者预印本](https://arxiv.org/abs/2007.03767) 和作者代码。
- [作者仓库](https://github.com/TinfoilHat0/Defending-Against-Backdoors-with-Robust-Learning-Rate)，固定提交 `20b8f845264c5c2f78bdb61e4e8d0729de211fe2`。
- 核对 `src/aggregation.py::compute_robustLR`、`agg_avg`、`aggregate_updates`。
- 实现保留符号计数不加权、更新按样本数加权及低一致性坐标反向学习率。没有添加 AlphaGuard 检测、客户端排除、RAMP 幅值筛选。
- 适配差异：数据、模型、攻击与优化器来自本项目；不启用作者代码里独立可选的 DP/裁剪开关。

## FLAME — USENIX Security 2022

- [会议页面](https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen)、[正式全文](https://www.usenix.org/system/files/sec22-nguyen.pdf)。已读取 Algorithm 1、§4.3、附录噪声设置与 Appendix E。
- 搜索到的 [zhmzm/FLAME](https://github.com/zhmzm/FLAME) 自述为 **Unofficial implementation**；不标为作者官方代码，当前代码依据论文重新实现。
- 聚类对象是 `W_i = G_previous + delta_i` 的**本地完整模型**，不是直接把 delta 向量用于余弦聚类。
- 剪裁半径为**全部**客户端 delta 的 L2 范数中位数，偶数样本取中间两值的均值；不是聚类后的中位数，也不是 `torch.median` 的下中位数。
- 保留模型等权平均；高斯噪声标准差为 `lambda * median_norm`。论文图像分类采用的 `lambda=0.001` 被保留为可配置默认值。
- HDBSCAN 使用预计算余弦距离、`min_samples=1` 和严格多数的最小簇大小；开启单簇许可，以允许只有一个良性群体。
- 无多数簇、零范数本地模型或非有限更新时明确报错。这是显式失败策略，不伪造“全部良性”回退结果。未实现论文的私有 STPC 聚合版本，也不作差分隐私保证。

## Lockdown — NeurIPS 2023

- [会议页面](https://neurips.cc/virtual/2023/poster/71476)、[作者仓库与全文](https://github.com/git-disl/Lockdown)。
- 固定提交 `147ec87642a4f61dea87a2e1685947efaa7fdc44`；核对 `src/agent_sparse.py`、`src/federated.py`、`src/aggregation.py`、`src/options.py`。
- 本项目已经有该稀疏客户端的演进实现。独立入口直接复用它，而不是在聚合端写一个同名简化版。
- 保留 ERK 掩码初始化、梯度受掩码约束、余弦退火剪枝、后续梯度恢复、稀疏更新、样本数加权聚合和共识融合。
- 共识使用本轮本地训练**开始之前**保存的掩码，在模型副本上测试；不把清理后的模型反馈为下一轮训练模型。
- 为避免已有扩展攻击在特殊分支提前返回而绕开隔离子空间，目前仅开放 BadNet/DBA；不是宣称原论文只支持这两种攻击。
- 本地与上游实现有攻击扩展与接口差异，记录为 `native_protocol_adapter`，而非字节级官方复现。稀疏性、退火和共识门限需要在独立验证集/预先规定网格下选择，不能按最终测试集挑最好结果。

## CRFL — ICML 2021

- [PMLR 论文与全文](https://proceedings.mlr.press/v139/xie21a.html)、[作者代码](https://github.com/AI-secure/CRFL)。
- 固定提交 `aaad6072282f9c36b9a5506014593149dd9cd040`；核对 `main.py`、`utils/helper.py`、`smooth_mnist.py` 和 `certify_mnist.py`。
- 依据论文 Algorithm 1：FedAvg 后裁剪模型参数 `w / max(1, ||w||/rho)`，不是裁剪每个客户端 delta。非最终轮加高斯噪声用于后续广播；最终轮保留裁剪模型。
- 作者代码保存/评估裁剪模型后再加下一轮噪声；本适配区分广播模型与评估模型，不把两次噪声误叠加到同一评估基模型。
- 依据 Algorithm 2 的分类器部分进行参数扰动模型的多数投票；固定一组抽样模型跨评估 batch 使用，不把它改成 softmax 概率均值。
- **尚未实现认证概率置信界、认证半径和 certified accuracy。** ACC/ASR 只是经验平滑指标，文件中固定 `certified=false`。
- 原始仓库实验为 MNIST、EMNIST、LOAN，使用不同的动态模型范数日程；没有核实到作者 CIFAR/ResNet9 的现成日程。故要求用户明确设置裁剪界和训练／测试噪声，本版本使用常数界，绝不标为“论文 CIFAR 默认参数”。
- 连续每轮攻击、当前优化器以及框架数据增强等设置不自动满足原文认证定理假设。不能从这个适配结果外推理论认证。

## Prodigal — Knowledge-Based Systems 2026

- [论文 DOI](https://doi.org/10.1016/j.knosys.2025.114903)、[出版页面](https://www.sciencedirect.com/science/article/pii/S0950705125019410)、[论文同名作者代码](https://github.com/Lslland/Prodigal)。
- 固定提交 `6eab315a61453f95703219e9a5857c04d170e3f5`；核对 `fl_api/standalone/Isolation/` 下的 trainer、aggregation、api，以及运行参数文件。
- 完整论文方法部分未通过出版页面取得；实现依据公开作者代码，状态明确标为 `author_code_algorithm_adapter`。
- 必须保留的细节：
  1. `get_topk_mask` 按每个参数层的**最后一个 minibatch 梯度绝对值**选 `int(numel * topk)` 个坐标，不以本地训练总 delta 替代。
  2. 共识为 `count > theta`，不是 `>=`；只比较支持集，不增加符号一致性检测。
  3. 公共坐标值来自所有参与者**完整本地模型的等权平均**，不是只在支持该坐标的人之间平均。
  4. 客户端下轮接收“公共模型＋自己的 Top-K 非公共参数”；既不按真实恶意身份选择下发对象，也不把所有人重置到统一公共模型。
  5. 非公共且不属于该客户端 Top-K 的参数置零，而不是保留旧全局值。服务器评估仅公共部分。
- 作者代码还提供消融 `TopKnoPer`，此处不运行该消融。
- 攻击实现使用本项目原生 Agent。例如 Neurotoxin 在梯度更新前应用 mask，而上游训练器有在 `optimizer.step()` 后才改 gradient 的实现；不把这个攻击实现差异隐瞒为完整原作复现。
- 目前限定全参与，因此没有猜测客户端缺席时个性化状态和共识计票的策略。

### PGD 接口兼容（全部密集方法共用）

本项目 `Agent.local_train` 的 PGD 投影分支依赖 `utils.vector_to_model_wo_load` 返回 state dict，但当前工具函数没有返回值。`native_compat.py` 仅为本实验的 Agent 实例绑定同一训练函数和独立工具代理，返回投影向量对应的字典；不猴子补丁全局模块，不修改源文件，也不调整 PGD 攻击定义。测试以高本地学习率强制触发投影，检查 CIFAR10 更新范数在 `0.1 * ||initial_model||` 边界，且原类和原工具函数身份保持不变。

## APR-FL — IEEE TCE 2025：暂缺复现依据

- [DOI](https://doi.org/10.1109/TCE.2025.3612464)、[作者机构记录](https://napier-repository.worktribe.com/output/4911171/apr-fl-defending-against-hidden-backdoor-attacks-with-adaptive-parameter-replacement)、[另一作者机构记录](https://pure.kfupm.edu.sa/en/publications/apr-fl-defending-against-hidden-backdoor-attacks-with-adaptive-pa/)。
- 核实题名为 *APR-FL: Defending Against Hidden Backdoor Attacks With Adaptive Parameter Replacement*；记录为 2025 年 9 月 22 日在线发表，2025 年 11 月卷期。
- 公开摘要描述的是识别本地更新中变化较小的关键参数、用少量干净数据重训，再通过多任务损失处理更新。不能仅实现一个服务器 clean fine-tuning 就声称为 APR-FL。
- 搜索了题名、简称、DOI、PDF、GitHub 仓库与代码。未取得足够的完整方法／作者代码，不代表作者一定未开源。
- 继续实现需要：全文或作者代码；关键参数选择公式；干净数据的来源、规模、持有位置；MTL 两项损失和权重；训练日程与替换时机。测试集不得充当干净训练集。
- `--method apr_fl` 明确报错，不生成虚假的对比性能。

## FedTOF — Knowledge-Based Systems 2026：暂缺复现依据

- [出版页面](https://www.sciencedirect.com/science/article/pii/S0950705126012025)、[DOI](https://doi.org/10.1016/j.knosys.2026.116476)。
- 核实题名为 *FedTOF: Mitigating poisoning attacks in Federated Learning via tail orthogonal forgetting*，卷 350、文章号 116476，**卷期日期为 2026 年 9 月 27 日**；不将搜索抓取日期视为发表日期。
- 公开摘要支持“尾子空间检测＋正交遗忘，不直接删除整个客户端”的描述。但是摘要不足以确定历史梯度的具体组织方式、ATSD/FOF 的全部公式、秩选择和时间调度。
- 搜索题名、FedTOF、code、GitHub、预印本／PDF，未取得可核验的完整方法或作者实现；不宣称网上不存在源码。
- 继续实现需要完整论文或作者代码，尤其是历史窗口、梯度矩阵构造/中心化、tail rank、自适应阈值、何时遗忘以及遗忘强度。
- `--method fedtof` 明确报错。没有实现一个随意 SVD 投影后将其冠名 FedTOF，也不能仅凭摘要下结论说 RAMP 已经没有创新空间。

## 研究表述建议

把结果写作“在统一 Alignins 训练/攻击设置下，对原方法核心机制进行独立适配的对比实验”，并逐项列明原作与适配差异。CRFL 必须附“经验防御，不含认证”，FLAME 必须附“依据论文重实现”，两项未实现方法不能出现在实验结果表中假装已跑完。
