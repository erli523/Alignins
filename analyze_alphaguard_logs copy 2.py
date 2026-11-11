import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# --- 1. 全局样式升级 (学术论文风格) ---
sns.set_theme(style="ticks", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42  # 确保字体嵌入，方便后期编辑
plt.rcParams['ps.fonttype'] = 42

# 高级配色：朱红 (Malicious) vs 矢车菊蓝 (Benign)
COLOR_PALETTE = {'Malicious': '#EE4B2B', 'Benign': '#3B719F'} 

def parse_log_file(file_path):
    """
    解析日志文件 (保持原逻辑不变)
    """
    data = []
    current_round = 0
    round_pattern = re.compile(r'--------round\s+(\d+)\s+------------')
    metric_pattern = re.compile(
        r'Client\s+(\d+)(?:\s+\((MAL)\))?:\s+Top-K=([\d\.]+),\s+Deviation=([\d\.]+).*?Z-score=([\d\.]+)'
    )
    
    attack_name = "Unknown"
    dataset = "Unknown"
    
    filename = os.path.basename(file_path)
    if "attack_" in filename:
        try:
            parts = filename.split("attack_")[1]
            if "_" in parts:
                attack_name = parts.split("_")[0]
            else:
                attack_name = parts.split(".")[0]
        except:
            pass
            
    path_parts = os.path.normpath(file_path).split(os.sep)
    for part in path_parts:
        if attack_name == "Unknown" and part.startswith("attack_"):
            attack_name = part.replace("attack_", "").split("_")[0]
        if part in ["cifar10", "cifar100", "mnist", "fmnist"]:
            dataset = part
            
    print(f"解析日志: {attack_name} | {dataset}")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            round_match = round_pattern.search(line)
            if round_match:
                current_round = int(round_match.group(1))
                continue
            metric_match = metric_pattern.search(line)
            if metric_match:
                data.append({
                    'Round': current_round,
                    'Client ID': int(metric_match.group(1)),
                    'Type': 'Malicious' if metric_match.group(2) == 'MAL' else 'Benign',
                    'Top-K': float(metric_match.group(3)),
                    'Deviation': float(metric_match.group(4)),
                    'Z-score': float(metric_match.group(5)),
                    'Attack': attack_name,
                    'Dataset': dataset
                })
    return pd.DataFrame(data)

# --- 核心绘图函数 ---
def plot_multi_attack_comparison(all_data_df, metric='Z-score', output_dir='.'):
    """
    生成美观的横向小提琴图，聚焦于阈值附近的分布分离情况。
    """
    if 'Attack' not in all_data_df.columns or all_data_df['Attack'].nunique() < 1:
        return

    # 1. 数据清洗与美化
    plot_data = all_data_df.copy().dropna(subset=[metric])
    
    # 将攻击名称首字母大写，显得更规范 (例如 badnet -> BadNet)
    plot_data['Attack'] = plot_data['Attack'].apply(lambda x: x.capitalize() if x.islower() else x)
    # 特殊处理缩写 (可选)
    plot_data['Attack'] = plot_data['Attack'].replace({'Dba': 'DBA', 'Pgd': 'PGD', 'F3ba': 'F3BA'})

    # 2. 智能截断 (Smart Clipping)
    # 我们只关心 0 到 15 (或20) 之间的分布。超过 20 的值在视觉上都是“极右”，没必要画到 100。
    VISUAL_LIMIT = 15.0
    
    # 将超过 Limit 的值压扁到 Limit，这样它们会在最右边形成一个“包”，而不是消失
    plot_data[metric] = plot_data[metric].clip(upper=VISUAL_LIMIT)

    # 3. 计算画布高度
    n_attacks = plot_data['Attack'].nunique()
    plt.figure(figsize=(10, n_attacks * 1.1 + 1.5)) # 动态高度

    # 4. 绘图
    ax = sns.violinplot(
        data=plot_data,
        y='Attack',
        x=metric,
        hue='Type',
        split=True,       # 左右分开
        inner="quart",    # 显示四分位线 (比 box 更简洁)
        orient='h',       # 横向
        palette=COLOR_PALETTE,
        linewidth=1.2,    # 边框线宽
        cut=0,            # 严格限制在数据范围内，不外推
        scale='width',    # 宽度归一化
        density_norm='width', # 新版 seaborn 参数，等同于 scale='width'
        saturation=0.85   # 稍微降低饱和度，更有质感
    )

    # 5. 绘制阈值线 (画龙点睛)
    THRESHOLD = 1.3  # 这里修改你的 AlphaGuard 阈值
    plt.axvline(x=THRESHOLD, color='#555555', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)
    
    # 在阈值线旁边加标注
    plt.text(x=THRESHOLD + 0.3, y=-0.7, s=r'Decision Boundary ($\tau \approx 1.3$)', 
             color='#555555', fontsize=11, style='italic', va='center')

    # 6. 细节美化
    # 去掉上、左、右边框，只留底部的轴
    sns.despine(left=True, bottom=False, top=True, right=True)
    
    # Y轴设置
    plt.ylabel('') # 不需要 Label，攻击名很直观
    plt.yticks(fontsize=13, fontweight='bold', color='#333333')
    
    # X轴设置
    plt.xlabel(f'Anomaly Score ({metric})', fontsize=13, labelpad=10)
    # 手动设置 X 轴刻度，并在最后显示 ">15"
    ticks = [0, 1.3, 5, 10, VISUAL_LIMIT]
    tick_labels = ['0', r'$\tau$', '5', '10', f'>{int(VISUAL_LIMIT)}']
    plt.xticks(ticks, tick_labels, fontsize=11)
    
    # 添加垂直网格，方便看分布位置
    ax.xaxis.grid(True, linestyle=':', alpha=0.4, color='gray')

    # 7. 图例设置 (放在顶部居中)
    plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.18), 
        ncol=2, 
        frameon=False, 
        fontsize=12,
        title=None,
        handletextpad=0.5
    )

    # 标题
    plt.title('Distribution Separation of AlphaGuard Under Different Attacks', 
              y=1.2, fontsize=15, fontweight='bold', color='#222222')

    # 8. 保存
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'Comparison_Violin_{metric}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> 图表已生成: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='src/logs')
    parser.add_argument('--output', type=str, default='analysis_results')
    args = parser.parse_args()
    
    if not os.path.exists(args.output): os.makedirs(args.output)
        
    log_files = glob.glob(os.path.join(args.log_dir, "**/*.log"), recursive=True)
    if not log_files:
        print("未找到日志文件。")
        return

    all_data = []
    print(f"找到 {len(log_files)} 个日志文件...")
    
    for log_file in log_files:
        df = parse_log_file(log_file)
        if not df.empty:
            all_data.append(df)
            
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        print("\n正在生成对比图...")
        # 只画 Z-score，这是最重要的指标
        plot_multi_attack_comparison(final_df, metric='Z-score', output_dir=args.output)
        print(f"\n完成！请查看 {args.output} 目录下的 PDF 文件。")
    else:
        print("数据为空，请检查日志格式。")

if __name__ == "__main__":
    main()