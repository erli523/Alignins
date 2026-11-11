# 当前 Alignins 文件用途

当前正在运行的项目目录与原脚本位置保持原样；此索引按用途分类，避免运行错入口。

## 单次入口

- `federated.py`
- `src/federated.py`

## 批量扫描与消融

- `run_ablation_gref_strategy.py`
- `run_ablation_l2_filter.py`
- `run_ablation_model.py`
- `run_ablation_topk.py`
- `run_ablation_topk.sh`
- `run_noniid_sweep.py`
- `run_poison_frac_sweep.py`
- `run_simple_sweep.py`
- `run_sweep.py`

## 分析与绘图

- `analyze_alphaguard_logs copy 2.py`
- `analyze_alphaguard_logs copy.py`
- `analyze_alphaguard_logs.py`
- `lockdown_defense_summary.py`
- `testplot.py`

## 快捷命令示例

- `quick_start_altermin.sh`
- `quick_start_mars.sh`
- `quick_start_mnist.sh`
- `quick_start_model_replacement.sh`

## 测试与调试

- `debug_model_replacement.py`
- `test_altermin.sh`
- `test_dynamic_trigger.sh`
- `test_mars.sh`
- `test_mars_core.py`
- `test_model_replacement.sh`

## 旧代码修改工具

- `add_tpr_fpr_to_krum_mmetric.py`

单次实验只使用根目录 `python federated.py ...`。旧代码修改工具会修改源码，不要作为训练入口运行。历史 backup_file 下的源码只是记录。
