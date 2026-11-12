# GitHub 代码与日志备份

仓库：https://github.com/erli523/Alignins

可见性：公开；按用户当前设置保留。

备份时间：2026-11-12T12:55:15+08:00（Asia/Shanghai）。

本次包含当前 Alignins 全部源码、攻击/防御、独立 src/ramp_b.py、实验脚本/配置/说明，以及当前和历史日志、逐轮指标、报告、图片、坐标索引和源码快照。

## 日志位置

- logs/history/Alignins-local/：此前从本机导入的历史日志及对应 backup_file 源码。服务器原始归档仍在 /home/risegpu/sunjiahao/records/history/Alignins-local。
- src/logs/：现有框架训练日志。
- research_results/RAMP_topk/：已完成的 Top-K 诊断、随机对照和中文分析。
- research_results/RAMP_oracle/：四组 Oracle 实验的现有记录。
- research_results 内原有 validation 子目录：现存校验记录，沿用原名，与正式实验分开。

运行中的 B/D 实验不会被中断；本次日志是采集时刻的快照，后续新增轮次需要再次同步。采集状态见 [GITHUB_BACKUP_MANIFEST.json](GITHUB_BACKUP_MANIFEST.json)。不修改原始日志或伪造 completed 状态。

数据集、虚拟环境/缓存、模型权重和完整配对更新快照不进入 Git。统计坐标索引 .npz 可复核且不包含原始训练图片，随实验记录上传。原服务器上述文件均保留。
