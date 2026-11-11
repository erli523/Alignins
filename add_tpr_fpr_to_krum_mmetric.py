"""
为 Multi-Krum 和 Multi-Metric 添加 TPR/FPR 计算的代码补丁

使用方法：
1. 阅读下面的代码修改说明
2. 手动将代码添加到 src/aggregation.py 对应位置
3. 或者运行此脚本自动添加（备份原文件）

作者：AI Assistant
日期：2024
"""

import os
import shutil

# ============================================================================
# 第一部分：Multi-Krum 的 TPR/FPR 计算代码
# ============================================================================

KRUM_TPR_FPR_CODE = """
        # ========== 添加 TPR/FPR 计算 ==========
        # 原有代码：score_index = score_index[0: krum_param_m]
        # print('%d clients are selected' % len(score_index))
        
        # 计算检测指标
        selected_clients = [list(agent_updates_dict.keys())[i] for i in score_index]
        all_clients = list(agent_updates_dict.keys())
        
        # 被排除的客户端
        removed_clients = set(all_clients) - set(selected_clients)
        
        # 统计恶意和良性客户端
        removed_malicious = sum(1 for idx in removed_clients if idx < self.args.num_corrupt)
        removed_benign = sum(1 for idx in removed_clients if idx >= self.args.num_corrupt)
        
        selected_malicious = sum(1 for idx in selected_clients if idx < self.args.num_corrupt)
        selected_benign = sum(1 for idx in selected_clients if idx >= self.args.num_corrupt)
        
        # 计算 TPR 和 FPR
        if self.args.num_corrupt > 0:
            TPR = removed_malicious / self.args.num_corrupt
        else:
            TPR = 0.0
        
        total_benign = len(all_clients) - self.args.num_corrupt
        if total_benign > 0:
            FPR = removed_benign / total_benign
        else:
            FPR = 0.0
        
        # 输出检测结果
        logging.info('========== Multi-Krum Detection Metrics ==========')
        logging.info('Total clients: %d (Malicious: %d, Benign: %d)' % 
                    (len(all_clients), self.args.num_corrupt, total_benign))
        logging.info('Selected clients: %d' % len(selected_clients))
        logging.info('  - Malicious in selection: %d/%d' % (selected_malicious, self.args.num_corrupt))
        logging.info('  - Benign in selection: %d/%d' % (selected_benign, total_benign))
        logging.info('Removed clients: %d' % len(removed_clients))
        logging.info('  - Malicious removed: %d/%d' % (removed_malicious, self.args.num_corrupt))
        logging.info('  - Benign removed: %d/%d' % (removed_benign, total_benign))
        logging.info('TPR (Detection Rate):    %.4f' % TPR)
        logging.info('FPR (False Positive):    %.4f' % FPR)
        logging.info('==================================================')
        
        print('%d clients are selected (TPR: %.4f, FPR: %.4f)' % (len(score_index), TPR, FPR))
        # ========== TPR/FPR 计算结束 ==========
"""

# ============================================================================
# 第二部分：Multi-Metric 的 TPR/FPR 计算代码
# ============================================================================

MMETRIC_TPR_FPR_CODE = """
        # ========== 添加 TPR/FPR 计算 ==========
        # 原有代码：print(topk_ind)
        # current_dict = {}
        
        # 计算检测指标
        all_client_indices = list(range(len(chosen_clients)))
        selected_indices = topk_ind.tolist()
        removed_indices = set(all_client_indices) - set(selected_indices)
        
        # 统计恶意和良性客户端（注意：chosen_clients的顺序是 malicious + benign）
        removed_malicious = 0
        removed_benign = 0
        selected_malicious = 0
        selected_benign = 0
        
        for idx in removed_indices:
            if chosen_clients[idx] < self.args.num_corrupt:
                removed_malicious += 1
            else:
                removed_benign += 1
        
        for idx in selected_indices:
            if chosen_clients[idx] < self.args.num_corrupt:
                selected_malicious += 1
            else:
                selected_benign += 1
        
        # 计算 TPR 和 FPR
        if self.args.num_corrupt > 0:
            TPR = removed_malicious / self.args.num_corrupt
        else:
            TPR = 0.0
        
        total_benign = len(chosen_clients) - self.args.num_corrupt
        if total_benign > 0:
            FPR = removed_benign / total_benign
        else:
            FPR = 0.0
        
        # 输出检测结果
        logging.info('========== Multi-Metric Detection Metrics ==========')
        logging.info('Total clients: %d (Malicious: %d, Benign: %d)' % 
                    (len(chosen_clients), self.args.num_corrupt, total_benign))
        logging.info('Selection ratio: %.2f%%' % (p * 100))
        logging.info('Selected clients: %d' % len(selected_indices))
        logging.info('  - Malicious in selection: %d/%d' % (selected_malicious, self.args.num_corrupt))
        logging.info('  - Benign in selection: %d/%d' % (selected_benign, total_benign))
        logging.info('Removed clients: %d' % len(removed_indices))
        logging.info('  - Malicious removed: %d/%d' % (removed_malicious, self.args.num_corrupt))
        logging.info('  - Benign removed: %d/%d' % (removed_benign, total_benign))
        logging.info('TPR (Detection Rate):    %.4f' % TPR)
        logging.info('FPR (False Positive):    %.4f' % FPR)
        logging.info('Mahalanobis distances: %s' % str([round(x, 4) for x in scores[:5]]))
        logging.info('====================================================')
        
        print('Multi-Metric: Selected %d clients (TPR: %.4f, FPR: %.4f)' % (len(topk_ind), TPR, FPR))
        # ========== TPR/FPR 计算结束 ==========
"""

# ============================================================================
# 第三部分：修改说明
# ============================================================================

def print_modification_instructions():
    """打印手动修改说明"""
    
    print("=" * 80)
    print("Multi-Krum 和 Multi-Metric 添加 TPR/FPR 计算 - 修改指南")
    print("=" * 80)
    print()
    
    print("【方法1：手动修改】")
    print("-" * 80)
    print()
    
    print("1️⃣  修改 Multi-Krum (src/aggregation.py, 约 Line 552)")
    print()
    print("   找到以下代码：")
    print("   " + "-" * 70)
    print("   score_index = score_index[0: krum_param_m]")
    print("   ")
    print("   print('%d clients are selected' % len(score_index))")
    print("   return_updates = [agent_updates_dict[i] for i in score_index]")
    print("   " + "-" * 70)
    print()
    print("   在 'print' 语句和 'return_updates' 之间插入以下代码：")
    print()
    print(KRUM_TPR_FPR_CODE)
    print()
    
    print("2️⃣  修改 Multi-Metric (src/aggregation.py, 约 Line 623)")
    print()
    print("   找到以下代码：")
    print("   " + "-" * 70)
    print("   topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]")
    print("   ")
    print("   print(topk_ind)")
    print("   current_dict = {}")
    print("   " + "-" * 70)
    print()
    print("   在 'print(topk_ind)' 和 'current_dict = {}' 之间插入以下代码：")
    print()
    print(MMETRIC_TPR_FPR_CODE)
    print()
    
    print("=" * 80)
    print()
    
    print("【方法2：使用此脚本自动修改】")
    print("-" * 80)
    print()
    print("运行以下命令：")
    print("   python add_tpr_fpr_to_krum_mmetric.py --auto-apply")
    print()
    print("注意：此脚本会自动备份原文件到 src/aggregation.py.backup")
    print()
    
    print("=" * 80)


def auto_apply_modifications():
    """自动应用修改"""
    
    aggregation_file = "src/aggregation.py"
    backup_file = "src/aggregation.py.backup"
    
    # 检查文件是否存在
    if not os.path.exists(aggregation_file):
        print(f"❌ 错误：找不到文件 {aggregation_file}")
        return False
    
    # 备份原文件
    print(f"📋 备份原文件到 {backup_file}...")
    shutil.copy2(aggregation_file, backup_file)
    
    # 读取原文件
    with open(aggregation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 标记是否成功修改
    krum_modified = False
    mmetric_modified = False
    
    # 修改Multi-Krum
    print("🔧 修改 Multi-Krum...")
    for i, line in enumerate(lines):
        if "print('%d clients are selected' % len(score_index))" in line:
            # 在这行之前插入TPR/FPR计算代码
            indent = "        "  # 8个空格
            tpr_lines = [indent + l + "\n" for l in KRUM_TPR_FPR_CODE.strip().split('\n')]
            lines = lines[:i] + tpr_lines + ["\n"] + lines[i:]
            krum_modified = True
            print("   ✅ Multi-Krum TPR/FPR 代码已添加")
            break
    
    if not krum_modified:
        print("   ⚠️  未找到 Multi-Krum 的插入位置")
    
    # 修改Multi-Metric
    print("🔧 修改 Multi-Metric...")
    for i, line in enumerate(lines):
        if "print(topk_ind)" in line and i < len(lines) - 1:
            # 检查下一行是否是 current_dict = {}
            if "current_dict = {}" in lines[i+1]:
                # 在这两行之间插入TPR/FPR计算代码
                indent = "        "  # 8个空格
                tpr_lines = [indent + l + "\n" for l in MMETRIC_TPR_FPR_CODE.strip().split('\n')]
                lines = lines[:i+1] + ["\n"] + tpr_lines + ["\n"] + lines[i+1:]
                mmetric_modified = True
                print("   ✅ Multi-Metric TPR/FPR 代码已添加")
                break
    
    if not mmetric_modified:
        print("   ⚠️  未找到 Multi-Metric 的插入位置")
    
    # 写回文件
    if krum_modified or mmetric_modified:
        print("💾 保存修改...")
        with open(aggregation_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print()
        print("=" * 80)
        print("✅ 修改完成！")
        print("=" * 80)
        print()
        print(f"修改内容：")
        if krum_modified:
            print("  ✅ Multi-Krum: 已添加 TPR/FPR 计算")
        if mmetric_modified:
            print("  ✅ Multi-Metric: 已添加 TPR/FPR 计算")
        print()
        print(f"原文件已备份到: {backup_file}")
        print()
        print("测试方法：")
        print("  python federated.py --aggr mkrum")
        print("  python federated.py --aggr mmetric")
        print()
        return True
    else:
        print()
        print("❌ 修改失败：未找到合适的插入位置")
        print("   请手动修改或检查代码结构是否发生变化")
        return False


def create_example_script():
    """创建测试脚本"""
    
    test_script = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
测试 Multi-Krum 和 Multi-Metric 的 TPR/FPR 计算

用法：
    python test_tpr_fpr.py
\"\"\"

import subprocess
import sys

def run_experiment(aggr_method, extra_args=""):
    \"\"\"运行单个实验\"\"\"
    print("=" * 80)
    print(f"Testing {aggr_method.upper()}")
    print("=" * 80)
    
    cmd = f"python federated.py --aggr {aggr_method} --rounds 10 {extra_args}"
    print(f"Command: {cmd}")
    print()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 提取TPR和FPR
        output = result.stdout + result.stderr
        
        for line in output.split('\\n'):
            if 'TPR' in line or 'FPR' in line or 'Detection' in line:
                print(line)
        
        print()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("\\n" + "=" * 80)
    print("TPR/FPR 测试脚本")
    print("=" * 80 + "\\n")
    
    # 测试配置
    experiments = [
        ("mkrum", ""),
        ("mmetric", ""),
        ("alignins", "--lambda_s 1.0 --lambda_c 1.0"),
        ("mars", ""),
    ]
    
    results = {}
    
    for aggr, args in experiments:
        success = run_experiment(aggr, args)
        results[aggr] = "✅" if success else "❌"
    
    # 汇总结果
    print("=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    for aggr, status in results.items():
        print(f"{status} {aggr}")
    print()
"""
    
    with open("test_tpr_fpr.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print("📝 已创建测试脚本: test_tpr_fpr.py")
    print("   运行方法: python test_tpr_fpr.py")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--auto-apply":
        print()
        print("🚀 自动应用修改模式")
        print()
        auto_apply_modifications()
        create_example_script()
    else:
        print_modification_instructions()
        print()
        print("💡 提示：")
        print("   如果你想自动应用修改，请运行：")
        print("   python add_tpr_fpr_to_krum_mmetric.py --auto-apply")
        print()





