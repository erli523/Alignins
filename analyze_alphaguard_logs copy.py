import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker

# 设置绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False

def parse_log_file(file_path):
    """
    解析单个日志文件，提取 Top-K, Deviation, Z-score 等指标。
    """
    data = []
    current_round = 0
    
    # 正则表达式
    round_pattern = re.compile(r'--------round\s+(\d+)\s+------------')
    # 匹配格式: Client 0 (MAL): Top-K=3.1955, Deviation=1.1747 (HIGH), Z-score=4.6320
    # 同时也兼容没有 (MAL) 的情况
    metric_pattern = re.compile(
        r'Client\s+(\d+)(?:\s+\((MAL)\))?:\s+Top-K=([\d\.]+),\s+Deviation=([\d\.]+).*?Z-score=([\d\.]+)'
    )
    
    # 初始化变量
    attack_name = "Unknown"
    dataset = "Unknown"
    
    # 1. 从文件名尝试提取 (优先级高，适合手动整理的场景)
    filename = os.path.basename(file_path)
    if "attack_" in filename:
        # 匹配 attack_DBA.log 或 attack_DBA_xxx.log
        try:
            # 找到 attack_ 后面的部分，直到遇到 _ 或 .
            parts = filename.split("attack_")[1]
            # 假设攻击名后面紧跟 _ 或者 .
            if "_" in parts:
                attack_name = parts.split("_")[0]
            else:
                attack_name = parts.split(".")[0]
        except:
            pass
            
    # 2. 尝试从路径提取更多信息 (兼容原始 logs 目录结构)
    path_parts = os.path.normpath(file_path).split(os.sep)
    for part in path_parts:
        if attack_name == "Unknown" and part.startswith("attack_"):
            attack_name = part.replace("attack_", "").split("_")[0]
        if part in ["cifar10", "cifar100", "mnist", "fmnist"]:
            dataset = part
            
    print(f"正在解析日志: {file_path}")
    print(f"  - 数据集: {dataset}")
    print(f"  - 攻击: {attack_name}")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 检测 Round
            round_match = round_pattern.search(line)
            if round_match:
                current_round = int(round_match.group(1))
                continue
            
            # 检测指标行
            metric_match = metric_pattern.search(line)
            if metric_match:
                client_id = int(metric_match.group(1))
                is_malicious = metric_match.group(2) == 'MAL'
                top_k = float(metric_match.group(3))
                deviation = float(metric_match.group(4))
                z_score = float(metric_match.group(5))
                
                data.append({
                    'Round': current_round,
                    'Client ID': client_id,
                    'Type': 'Malicious' if is_malicious else 'Benign',
                    'Top-K': top_k,
                    'Deviation': deviation,
                    'Z-score': z_score,
                    'Attack': attack_name,
                    'Dataset': dataset,
                    'File': os.path.basename(file_path)
                })
    
    return pd.DataFrame(data)

def plot_metric_evolution(df, metric='Z-score', output_pdf=None):
    """
    绘制指标随时间的变化趋势 (Line Plot with CI)
    """
    plt.figure(figsize=(14, 7))
    
    # 使用 lineplot 自动计算均值和置信区间
    sns.lineplot(
        data=df, 
        x='Round', 
        y=metric, 
        hue='Type', 
        style='Type',
        palette={'Malicious': '#d62728', 'Benign': '#1f77b4'}, # 使用更专业的配色
        markers=False,
        dashes=False,
        linewidth=2,
        alpha=0.9
    )
    
    plt.title(f'{metric} Evolution over Rounds', fontsize=16, fontweight='bold')
    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.legend(title='Client Type', loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_pdf:
        output_pdf.savefig()
        plt.close()
    else:
        plt.show()

def plot_heatmap(df, metric='Z-score', output_pdf=None):
    """
    绘制热力图: X轴=Round, Y轴=Client ID, 颜色=Metric
    """
    # Pivot table: 行=Client ID, 列=Round, 值=Metric
    pivot_df = df.pivot_table(index='Client ID', columns='Round', values=metric)
    
    plt.figure(figsize=(20, 10))
    
    # 动态调整颜色范围，避免极端值影响整体显示
    vmax = np.percentile(df[metric], 98)
    
    sns.heatmap(
        pivot_df, 
        cmap='RdYlBu_r', # 红蓝渐变，红色代表高值(异常)
        vmax=vmax,
        center=pivot_df.values.mean(), # 设置中间值
        cbar_kws={'label': metric}
    )
    
    # 标记恶意客户端（在Y轴标签上用红色显示）
    malicious_ids = df[df['Type'] == 'Malicious']['Client ID'].unique()
    ax = plt.gca()
    for label in ax.get_yticklabels():
        if int(label.get_text()) in malicious_ids:
            label.set_color('#d62728')
            label.set_fontweight('bold')
            
    plt.title(f'{metric} Heatmap (Red Y-labels = Malicious Clients)', fontsize=18, fontweight='bold')
    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel('Client ID', fontsize=14)
    
    if output_pdf:
        output_pdf.savefig()
        plt.close()
    else:
        plt.show()

def plot_distribution_boxplot(df, metric='Z-score', output_pdf=None):
    """
    绘制分布箱线图: 对比 Malicious 和 Benign 的指标分布
    """
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(
        data=df, 
        x='Type', 
        y=metric, 
        palette={'Malicious': '#ff9999', 'Benign': '#99ccff'},
        showfliers=False,
        width=0.5
    )
    # 叠加抖动散点，展示具体密度
    sns.stripplot(
        data=df,
        x='Type',
        y=metric,
        color='black',
        alpha=0.1,
        jitter=0.2,
        size=2
    )
    
    plt.title(f'{metric} Distribution Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Client Type', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if output_pdf:
        output_pdf.savefig()
        plt.close()
    else:
        plt.show()

# --- 1. 全局样式升级 ---
# 使用 Seaborn 的 paper 上下文，但手动调整字体和线宽以适应学术发表
sns.set_context("paper", rc={"font.size": 12,"axes.titlesize": 16,"axes.labelsize": 14})
sns.set_style("ticks") # 使用 ticks 风格，比 whitegrid 更干净，适合后期加网格

# 定义一套专业的配色方案 (Colorblind Friendly)
# Malicious: 深红 (强调), Benign: 钢蓝 (冷静)
COLOR_PALETTE = {'Malicious': '#D62728', 'Benign': '#2C7BB6'} 

def plot_multi_attack_comparison(all_data_df, metric='Z-score', output_dir='.'):
    """
    生成美观的横向对比图：Horizontal Split Violin Plot
    自动截断极端值，聚焦于区分度。
    """
    if 'Attack' not in all_data_df.columns or all_data_df['Attack'].nunique() < 1:
        return

    # --- 1. 数据预处理：视觉截断 (Visual Clipping) ---
    # 目的：为了不让 Z-score=1000 的点毁掉整张图，我们将过大的值“压”在一个视觉上限内
    # 例如：如果阈值是 3，那么展示到 15 就足够证明分开了，不需要展示到 1000
    
    plot_data = all_data_df.copy().dropna(subset=[metric])
    
    # 自动计算一个合理的视觉上限 (Visual Max)
    # 逻辑：取良性用户最大值的 5 倍，或者恶意用户中位数的 1.5 倍，取较小者，避免被极端值拉伸
    benign_max = plot_data[plot_data['Type'] == 'Benign'][metric].max()
    malicious_median = plot_data[plot_data['Type'] == 'Malicious'][metric].median()
    
    # 如果是 Z-score，通常良性在 3 以内。我们给一个硬上限建议，比如 20
    # 你也可以手动指定，例如 visual_limit = 15
    visual_limit = max(benign_max * 4, 10) 
    if metric == 'Z-score':
         # 限制在 20 以内，足以展示分离效果
        visual_limit = min(visual_limit, 20) 
    
    print(f"  [绘图提示] 为了美观，{metric} 大于 {visual_limit:.2f} 的值已被截断显示。")
    plot_data[metric] = plot_data[metric].clip(upper=visual_limit)

    # --- 2. 设置画布 ---
    # 高度根据攻击数量自动调整，宽度固定
    n_attacks = plot_data['Attack'].nunique()
    plt.figure(figsize=(10, n_attacks * 1.2 + 2)) 
    
    # --- 3. 核心绘图: 横向小提琴 ---
    ax = sns.violinplot(
        data=plot_data,
        y='Attack',      # 攻击在 Y 轴
        x=metric,        # 指标在 X 轴
        hue='Type',      # 分组颜色
        split=True,      # 左右(上下)分开
        inner="quart",   # 内部显示四分位数
        orient='h',      # <--- 关键：横向
        palette={'Malicious': '#E64B35', 'Benign': '#4DBBD5'}, # 更加高级的红蓝配色 (Nature 风格)
        linewidth=1,
        cut=0,           # 不延伸超过数据范围
        scale='width'    # 宽度标准化，让每个攻击看起来一样宽
    )

    # --- 4. 添加阈值线 (垂直线) ---
    # 假设你的 AlphaGuard 阈值 tau ≈ 1.3 或 3
    # 这条线是证明防御有效的关键：所有蓝色的都在线左边，红色的在线右边
    threshold_val = 3.0 # 如果你的阈值是 1.3，请改为 1.3
    if metric == 'Z-score':
        plt.axvline(x=threshold_val, color='gray', linestyle='--', linewidth=2, alpha=0.8)
        plt.text(x=threshold_val + 0.5, y=-0.6, s=r'Threshold $\tau$', color='gray', fontsize=12, fontweight='bold')

    # --- 5. 美化坐标轴和去除边框 ---
    # 去掉所有的边框，只保留底部的轴线
    sns.despine(left=True, bottom=False, top=True, right=True)
    
    # Y轴：攻击名称字体加大
    plt.yticks(fontsize=14, fontweight='bold')
    plt.ylabel('') # 去掉 Y 轴 Label，因为攻击名已经很显然了
    
    # X轴：美化
    plt.xlabel(f'{metric} Value (Clipped at {visual_limit})', fontsize=14)
    plt.xticks(fontsize=12)
    
    # 增加垂直网格线，方便对比位置
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='lightgray')
    
    # --- 6. 调整图例位置 ---
    # 放在图的上方，横向排列
    plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.15), 
        ncol=2, 
        frameon=False, 
        fontsize=14,
        title=None
    )
    
    plt.title(f'Distribution of {metric} by Attack Strategy', y=1.15, fontsize=18, fontweight='bold')
    
    # --- 7. 保存 ---
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'horizontal_comparison_{metric}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> 已保存横向对比图: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="AlphaGuard 日志分析与可视化工具")
    parser.add_argument('--log_dir', type=str, default='src/logs', help='日志根目录')
    parser.add_argument('--file', type=str, help='指定单个日志文件路径 (可选)')
    parser.add_argument('--output', type=str, default='analysis_results', help='结果输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    # 收集文件
    log_files = []
    if args.file:
        if os.path.exists(args.file):
            log_files.append(args.file)
        else:
            print(f"错误: 文件未找到 {args.file}")
            return
    else:
        # 递归查找所有 .log 文件
        print(f"正在扫描目录: {args.log_dir} ...")
        # 简单的 glob 可能会漏掉深层目录，这里使用 walk
        for root, dirs, files in os.walk(args.log_dir):
            for file in files:
                if file.endswith(".log"):
                    full_path = os.path.join(root, file)
                    log_files.append(full_path)
    
    if not log_files:
        print("未找到日志文件。")
        return

    all_data = []
    
    print(f"找到 {len(log_files)} 个日志文件。开始解析...")
    
    for log_file in log_files:
        df = parse_log_file(log_file)
        
        if df.empty:
            continue
            
        all_data.append(df)
        
        # 为每个有效日志生成单独的详细 PDF 报告
        log_name = os.path.splitext(os.path.basename(log_file))[0]
        attack_name = df['Attack'].iloc[0]
        pdf_path = os.path.join(args.output, f"report_{attack_name}_{log_name}.pdf")
        
        print(f"  -> 生成报告: {pdf_path}")
        with PdfPages(pdf_path) as pdf:
            # 1. Z-score 趋势
            plot_metric_evolution(df, metric='Z-score', output_pdf=pdf)
            # 2. Top-K 趋势
            plot_metric_evolution(df, metric='Top-K', output_pdf=pdf)
            # 3. Z-score 热力图
            plot_heatmap(df, metric='Z-score', output_pdf=pdf)
            # 4. Z-score 分布对比
            plot_distribution_boxplot(df, metric='Z-score', output_pdf=pdf)
            
            # 还可以添加文本页显示一些统计信息
            plt.figure(figsize=(10, 6))
            plt.axis('off')
            stats_text = f"Analysis Report for {log_name}\n\n"
            stats_text += f"Attack: {attack_name}\n"
            stats_text += f"Dataset: {df['Dataset'].iloc[0]}\n"
            stats_text += f"Total Rounds: {df['Round'].max()}\n"
            stats_text += f"Total Clients: {df['Client ID'].nunique()}\n"
            
            mal_df = df[df['Type'] == 'Malicious']
            ben_df = df[df['Type'] == 'Benign']
            
            stats_text += f"\nAvg Z-score (Malicious): {mal_df['Z-score'].mean():.4f}\n"
            stats_text += f"Avg Z-score (Benign):    {ben_df['Z-score'].mean():.4f}\n"
            
            plt.text(0.1, 0.5, stats_text, fontsize=14)
            pdf.savefig()
            plt.close()

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # 如果有多种攻击，生成对比图
        if final_df['Attack'].nunique() > 1:
            print("\n生成多攻击对比图...")
            plot_multi_attack_comparison(final_df, metric='Z-score', output_dir=args.output)
            plot_multi_attack_comparison(final_df, metric='Top-K', output_dir=args.output)
            
        print(f"\n分析完成！所有结果已保存至: {args.output}")
    else:
        print("没有提取到有效的 AlphaGuard 数据。请确认日志中包含 'Client X (MAL): Top-K=...' 等信息。")

if __name__ == "__main__":
    main()
