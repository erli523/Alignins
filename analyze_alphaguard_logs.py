# # import os
# # import re
# # import glob
# # import argparse
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from matplotlib.backends.backend_pdf import PdfPages

# # # --- 全局样式配置 ---
# # sns.set_theme(style="ticks", context="paper", font_scale=1.6)
# # plt.rcParams['font.family'] = 'serif'
# # plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
# # plt.rcParams['axes.unicode_minus'] = False
# # plt.rcParams['pdf.fonttype'] = 42
# # plt.rcParams['ps.fonttype'] = 42

# # # 颜色：深红 vs 矢车菊蓝
# # COLOR_PALETTE = {'Malicious': '#D64045', 'Benign': '#2E86AB'} 

# # def parse_log_file(file_path):
# #     """解析日志文件 (逻辑不变)"""
# #     data = []
# #     current_round = 0
    
# #     attack_name = "Unknown"
# #     dataset = "Unknown"
    
# #     filename = os.path.basename(file_path)
# #     if "attack_" in filename:
# #         try:
# #             parts = filename.split("attack_")[1]
# #             if "_" in parts: attack_name = parts.split("_")[0]
# #             else: attack_name = parts.split(".")[0]
# #         except: pass
            
# #     path_parts = os.path.normpath(file_path).split(os.sep)
# #     for part in path_parts:
# #         if attack_name == "Unknown" and part.startswith("attack_"):
# #             attack_name = part.replace("attack_", "").split("_")[0]
# #         if part in ["cifar10", "cifar100", "mnist", "fmnist"]:
# #             dataset = part
    
# #     # 特殊处理：ff 攻击使用不同的日志格式
# #     is_ff_attack = "ff" in attack_name.lower() or "ff" in filename.lower()
    
# #     if is_ff_attack:
# #         # FF 格式：Round 0 (大写R，从0开始，无横线)
# #         round_pattern = re.compile(r'Round\s+(\d+)')
# #         # FF 格式：没有 Deviation 字段
# #         metric_pattern = re.compile(
# #             r'Client\s+(\d+)(?:\s+\((MAL)\))?\s*:\s*Top-K=\s*([\d\.]+).*?Z-score=\s*([\d\.]+)'
# #         )
# #     else:
# #         # 标准格式：--------round 1 ------------ (小写r，有横线)
# #         round_pattern = re.compile(r'--------round\s+(\d+)\s+------------')
# #         # 标准格式：有 Deviation 字段
# #         metric_pattern = re.compile(
# #             r'Client\s+(\d+)(?:\s+\((MAL)\))?:\s+Top-K=([\d\.]+),\s+Deviation=([\d\.]+).*?Z-score=([\d\.]+)'
# #         )
            
# #     print(f"解析日志: {attack_name} | {dataset} | FF模式={is_ff_attack}")

# #     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
# #         for line in f:
# #             round_match = round_pattern.search(line)
# #             if round_match:
# #                 current_round = int(round_match.group(1))
# #                 continue
# #             metric_match = metric_pattern.search(line)
# #             if metric_match:
# #                 if is_ff_attack:
# #                     # FF 格式：只有 4 个匹配组（没有 Deviation）
# #                     data.append({
# #                         'Round': current_round,
# #                         'Client ID': int(metric_match.group(1)),
# #                         'Type': 'Malicious' if metric_match.group(2) == 'MAL' else 'Benign',
# #                         'Top-K': float(metric_match.group(3)),
# #                         'Z-score': float(metric_match.group(4)),
# #                         'Attack': attack_name,
# #                         'Dataset': dataset
# #                     })
# #                 else:
# #                     # 标准格式：有 5 个匹配组（包含 Deviation）
# #                     data.append({
# #                         'Round': current_round,
# #                         'Client ID': int(metric_match.group(1)),
# #                         'Type': 'Malicious' if metric_match.group(2) == 'MAL' else 'Benign',
# #                         'Top-K': float(metric_match.group(3)),
# #                         'Deviation': float(metric_match.group(4)),
# #                         'Z-score': float(metric_match.group(5)),
# #                         'Attack': attack_name,
# #                         'Dataset': dataset
# #                     })
# #     return pd.DataFrame(data)

# # def plot_multi_attack_comparison(all_data_df, metric='Z-score', output_dir='.', start_round=10):
# #     """
# #     使用【平方根坐标轴】绘制横向小提琴图。
# #     """
# #     if 'Attack' not in all_data_df.columns or all_data_df['Attack'].nunique() < 1:
# #         return

# #     # 1. 数据过滤
# #     print(f"  [数据过滤] 仅保留 Round >= {start_round} 的数据...")
# #     plot_data = all_data_df[all_data_df['Round'] >= start_round].copy()
# #     plot_data = plot_data.dropna(subset=[metric])
    
# #     # 美化名称
# #     plot_data['Attack'] = plot_data['Attack'].apply(lambda x: x.upper() if len(x) <= 3 else x.capitalize())
# #     plot_data['Attack'] = plot_data['Attack'].replace({'Badnet': 'BadNet', 'F3ba': 'F3BA'})

# #     # --- 2. 关键步骤：平方根变换 ---
# #     # 相比直接截断，平方根能自然地压缩大值，放大小值
# #     # 我们在一个新的列上画图，但在轴上显示原始值
# #     plot_data['Sqrt_Metric'] = np.sqrt(plot_data[metric])

# #     # 设定一个视觉上限 (Visual Limit)
# #     # 我们希望显示到原始值 25 或 36 左右
# #     # sqrt(25) = 5, sqrt(36) = 6.
# #     # 这样良性区域 (0-1.3) 的 sqrt 值是 (0-1.14)，占据了整个画幅的约 1/5，非常清晰
# #     ORIGINAL_LIMIT = 30.0
# #     SQRT_LIMIT = np.sqrt(ORIGINAL_LIMIT) # ≈ 5.47

# #     # 截断处理 (针对平方根后的值)
# #     plot_data['Sqrt_Metric'] = plot_data['Sqrt_Metric'].clip(upper=SQRT_LIMIT)

# #     # 3. 画布设置
# #     n_attacks = plot_data['Attack'].nunique()
# #     plt.figure(figsize=(11, n_attacks * 1.2 + 1)) 

# #     # 4. 绘图 (注意：X轴使用变换后的 Sqrt_Metric)
# #     ax = sns.violinplot(
# #         data=plot_data,
# #         y='Attack',
# #         x='Sqrt_Metric',  # <--- 画的是平方根数据
# #         hue='Type',
# #         split=True,       
# #         inner="quart",    
# #         orient='h',       
# #         palette=COLOR_PALETTE,
# #         linewidth=1.2,    
# #         cut=0,            
# #         scale='width',    
# #         density_norm='width', 
# #         saturation=0.9    
# #     )

# #     # 5. 绘制阈值线 (也要做平方根变换)
# #     THRESHOLD = 1.3
# #     SQRT_THRESHOLD = np.sqrt(THRESHOLD) # ≈ 1.14
    
# #     plt.axvline(x=SQRT_THRESHOLD, color='#444444', linestyle='--', linewidth=2, alpha=0.8, zorder=0)
# #     plt.text(x=SQRT_THRESHOLD + 0.15, y=-0.7, s=r'$\tau \approx 1.3$', 
# #              color='#444444', fontsize=14, fontweight='bold', va='center')

# #     # 6. 坐标轴魔法：将 X 轴刻度还原为原始值
# #     # 我们选择一些有代表性的原始值：0, 1.3(阈值), 5, 10, 20, 30
# #     tick_vals_original = [0, THRESHOLD, 5, 10, 20, ORIGINAL_LIMIT]
# #     # 计算它们在图上的位置 (平方根)
# #     tick_locs = [np.sqrt(v) for v in tick_vals_original]
# #     # 生成标签
# #     tick_labels = ['0', r'$\tau$', '5', '10', '20', f'>{int(ORIGINAL_LIMIT)}']
    
# #     plt.xticks(tick_locs, tick_labels, fontsize=13)
# #     plt.xlabel(f'Anomaly Score ({metric}) - Sqrt Scale', fontsize=14, labelpad=10)

# #     # 7. 其他美化
# #     sns.despine(left=True, bottom=False, top=True, right=True)
# #     plt.ylabel('') 
# #     plt.yticks(fontsize=14, fontweight='bold', color='#222222')
    
# #     # 网格线 (根据刻度位置画)
# #     ax.xaxis.grid(True, linestyle=':', alpha=0.5, color='gray')

# #     plt.legend(
# #         loc='upper center', 
# #         bbox_to_anchor=(0.5, 1.15), 
# #         ncol=2, 
# #         frameon=False, 
# #         fontsize=14,
# #         title=None,
# #         markerscale=1.5
# #     )

# #     plt.title(f'Distribution Separation )', 
# #               y=1.15, fontsize=16, fontweight='bold', color='#111111')

# #     # 8. 保存
# #     plt.tight_layout()
# #     save_path = os.path.join(output_dir, f'Violin_SqrtScale_{metric}.pdf')
# #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #     plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
# #     plt.close()
# #     print(f"  -> 图表已生成: {save_path}")

# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--log_dir', type=str, default='src/logs')
# #     parser.add_argument('--output', type=str, default='analysis_results')
# #     parser.add_argument('--start_round', type=int, default=10)
# #     args = parser.parse_args()
    
# #     if not os.path.exists(args.output): os.makedirs(args.output)
        
# #     log_files = glob.glob(os.path.join(args.log_dir, "**/*.log"), recursive=True)
# #     if not log_files:
# #         print("未找到日志文件。")
# #         return

# #     all_data = []
# #     print(f"找到 {len(log_files)} 个日志文件...")
# #     for log_file in log_files:
# #         df = parse_log_file(log_file)
# #         if not df.empty: all_data.append(df)
            
# #     if all_data:
# #         final_df = pd.concat(all_data, ignore_index=True)
# #         print("\n正在生成对比图...")
# #         plot_multi_attack_comparison(final_df, metric='Z-score', output_dir=args.output, start_round=args.start_round)
# #         print(f"\n完成！")
# #     else:
# #         print("数据为空。")

# # if __name__ == "__main__":
# #     main()
# #mport os
# # import re
# # import glob
# # import argparse
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # # --- 1. 学术单栏风格配置 ---
# # # 单栏图通常要求字体紧凑，线条清晰
# # sns.set_theme(style="ticks", context="paper", font_scale=1.1) 
# # plt.rcParams['font.family'] = 'serif'
# # plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
# # plt.rcParams['axes.unicode_minus'] = False
# # plt.rcParams['pdf.fonttype'] = 42
# # plt.rcParams['ps.fonttype'] = 42

# # # 颜色：保持你之前的一致性
# # COLOR_PALETTE = {'Malicious': '#D64045', 'Benign': '#2E86AB'} 

# # def parse_log_file(file_path):
# #     """解析日志文件 (保持不变)"""
# #     data = []
# #     current_round = 0
# #     round_pattern = re.compile(r'--------round\s+(\d+)\s+------------')
# #     metric_pattern = re.compile(r'Client\s+(\d+)(?:\s+\((MAL)\))?:\s+Top-K=([\d\.]+),\s+Deviation=([\d\.]+).*?Z-score=([\d\.]+)')
    
# #     attack_name = "Unknown"
    
# #     filename = os.path.basename(file_path)
# #     if "attack_" in filename:
# #         try:
# #             parts = filename.split("attack_")[1]
# #             if "_" in parts: attack_name = parts.split("_")[0]
# #             else: attack_name = parts.split(".")[0]
# #         except: pass
            
# #     # 这里可以根据你的实际情况增加数据集判断，或者写死
# #     print(f"解析日志: {attack_name}")

# #     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
# #         for line in f:
# #             round_match = round_pattern.search(line)
# #             if round_match:
# #                 current_round = int(round_match.group(1))
# #                 continue
# #             metric_match = metric_pattern.search(line)
# #             if metric_match:
# #                 data.append({
# #                     'Round': current_round,
# #                     'Type': 'Malicious' if metric_match.group(2) == 'MAL' else 'Benign',
# #                     'Z-score': float(metric_match.group(5)),
# #                     'Attack': attack_name
# #                 })
# #     return pd.DataFrame(data)

# # def plot_single_column_violin(all_data_df, metric='Z-score', output_dir='.', start_round=10):
# #     if 'Attack' not in all_data_df.columns: return

# #     # 1. 数据过滤
# #     plot_data = all_data_df[all_data_df['Round'] >= start_round].copy()
# #     plot_data = plot_data.dropna(subset=[metric])
    
# #     # 缩写攻击名称以适应单栏 (可选)
# #     # plot_data['Attack'] = plot_data['Attack'].replace({'Neurotoxin': 'Neuro.', 'ModelReplacement': 'MR'})
# #     plot_data['Attack'] = plot_data['Attack'].apply(lambda x: x.capitalize())
# #     plot_data['Attack'] = plot_data['Attack'].replace({'Badnet': 'BadNet', 'F3ba': 'F3BA', 'Dba': 'DBA', 'Pgd':'PGD'})

# #     # 2. 平方根变换 (保持你的设计)
# #     plot_data['Sqrt_Metric'] = np.sqrt(plot_data[metric])
    
# #     # 视觉截断
# #     ORIGINAL_LIMIT = 30.0
# #     SQRT_LIMIT = np.sqrt(ORIGINAL_LIMIT)
# #     plot_data['Sqrt_Metric'] = plot_data['Sqrt_Metric'].clip(upper=SQRT_LIMIT)

# #     # 3. 画布设置 (关键：单栏比例)
# #     # 宽度设为 5 英寸 (插入论文时缩放为 3.3 英寸会很清晰)
# #     n_attacks = plot_data['Attack'].nunique()
# #     plt.figure(figsize=(5.5, n_attacks * 0.8 + 1.2)) 

# #     # 4. 绘图
# #     ax = sns.violinplot(
# #         data=plot_data,
# #         y='Attack',
# #         x='Sqrt_Metric',
# #         hue='Type',
# #         split=True,       
# #         inner="quart",    
# #         orient='h',       
# #         palette=COLOR_PALETTE,
# #         linewidth=1.0,    
# #         cut=0,            
# #         scale='width',    
# #         density_norm='width', 
# #         saturation=0.9    
# #     )

# #     # 5. 阈值线
# #     THRESHOLD = 1.3
# #     SQRT_THRESHOLD = np.sqrt(THRESHOLD)
# #     plt.axvline(x=SQRT_THRESHOLD, color='#333333', linestyle='--', linewidth=1.5, alpha=0.9, zorder=0)
    
# #     # 阈值文字 (放在图内，节省空间)
# #     # 找一个合适的位置，比如第一行攻击的上方
# #     # plt.text(x=SQRT_THRESHOLD + 0.2, y=-0.6, s=r'$\tau \approx 1.3$', color='#333333', fontsize=10, fontweight='bold')

# #     # 6. 坐标轴与刻度
# #     tick_vals_original = [0, THRESHOLD, 5, 10, ORIGINAL_LIMIT]
# #     tick_locs = [np.sqrt(v) for v in tick_vals_original]
# #     tick_labels = ['0', r'$\tau$', '5', '10', f'>{int(ORIGINAL_LIMIT)}']
    
# #     plt.xticks(tick_locs, tick_labels, fontsize=10)
# #     plt.xlabel(f'Anomaly Score ($Z_i$) - Sqrt Scale', fontsize=11, labelpad=5)
    
# #     # 移除 Y 轴 Label，保留刻度文字
# #     plt.ylabel('')
# #     plt.yticks(fontsize=11, fontweight='bold', color='#222222')
    
# #     # 7. 极致美化
# #     sns.despine(left=True, bottom=False, top=True, right=True)
# #     ax.xaxis.grid(True, linestyle=':', alpha=0.5, color='gray')

# #     # 图例 (放在顶部，单行)
# #     plt.legend(
# #         loc='lower left', 
# #         bbox_to_anchor=(0, 1.02), # 放在图的左上角外
# #         ncol=2, 
# #         frameon=False, 
# #         fontsize=10,
# #         title=None,
# #         handletextpad=0.3,
# #         borderaxespad=0
# #     )

# #     # 紧凑布局
# #     plt.tight_layout()
# #     save_path = os.path.join(output_dir, f'SingleCol_Violin_{metric}.pdf')
# #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #     print(f"生成单栏图: {save_path}")

# # # --- 主程序逻辑 (复制你原来的 main 即可，调用上面的 plot 函数) ---
# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--log_dir', type=str, default='src/logs')
# #     parser.add_argument('--output', type=str, default='analysis_results')
# #     args = parser.parse_args()
# #     if not os.path.exists(args.output): os.makedirs(args.output)
    
# #     # 这里模拟读取所有日志，请确保你的路径正确
# #     log_files = glob.glob(os.path.join(args.log_dir, "**/*.log"), recursive=True)
# #     all_data = []
# #     for f in log_files:
# #         df = parse_log_file(f)
# #         if not df.empty: all_data.append(df)
        
# #     if all_data:
# #         final_df = pd.concat(all_data)
# #         plot_single_column_violin(final_df, output_dir=args.output)

# # if __name__ == "__main__":
# #     main()


# import os
# import re
# import glob
# import argparse
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # --- 学术单栏风格配置 ---
# sns.set_theme(style="ticks", context="paper", font_scale=1.1) 
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['ps.fonttype'] = 42

# # 配色：红蓝对比
# COLOR_PALETTE = {'Malicious': '#D64045', 'Benign': '#2E86AB'} 

# def parse_log_file(file_path):
#     """解析日志文件"""
#     data = []
#     current_round = 0
#     round_pattern = re.compile(r'--------round\s+(\d+)\s+------------')
#     metric_pattern = re.compile(r'Client\s+(\d+)(?:\s+\((MAL)\))?:\s+Top-K=([\d\.]+),\s+Deviation=([\d\.]+).*?Z-score=([\d\.]+)')
    
#     attack_name = "Unknown"
#     filename = os.path.basename(file_path)
#     if "attack_" in filename:
#         try:
#             parts = filename.split("attack_")[1]
#             if "_" in parts: attack_name = parts.split("_")[0]
#             else: attack_name = parts.split(".")[0]
#         except: pass

#     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#         for line in f:
#             round_match = round_pattern.search(line)
#             if round_match:
#                 current_round = int(round_match.group(1))
#                 continue
#             metric_match = metric_pattern.search(line)
#             if metric_match:
#                 data.append({
#                     'Round': current_round,
#                     'Type': 'Malicious' if metric_match.group(2) == 'MAL' else 'Benign',
#                     'Z-score': float(metric_match.group(5)),
#                     'Attack': attack_name
#                 })
#     return pd.DataFrame(data)

# def plot_single_column_violin(all_data_df, metric='Z-score', output_dir='.', start_round=10):
#     if 'Attack' not in all_data_df.columns: return

#     # 1. 数据过滤
#     plot_data = all_data_df[all_data_df['Round'] >= start_round].copy()
#     plot_data = plot_data.dropna(subset=[metric])
    
#     # 格式化攻击名称
#     plot_data['Attack'] = plot_data['Attack'].apply(lambda x: x.capitalize())
#     plot_data['Attack'] = plot_data['Attack'].replace({'Badnet': 'BadNet', 'F3ba': 'F3BA', 'Dba': 'DBA', 'Pgd':'PGD'})

#     # 2. 平方根变换 & 截断到 25
#     plot_data['Sqrt_Metric'] = np.sqrt(plot_data[metric])
    
#     ORIGINAL_LIMIT = 25.0  # <--- 修改为 25
#     SQRT_LIMIT = np.sqrt(ORIGINAL_LIMIT)
#     plot_data['Sqrt_Metric'] = plot_data['Sqrt_Metric'].clip(upper=SQRT_LIMIT)

#     # 3. 画布设置 (单栏宽度约 3.5 inch, 稍微设大一点 5.0 以保证清晰度，插入 latex 时缩放)
#     n_attacks = plot_data['Attack'].nunique()
#     plt.figure(figsize=(5.0, n_attacks * 0.8 + 1.0)) 

#     # 4. 绘图
#     ax = sns.violinplot(
#         data=plot_data,
#         y='Attack',
#         x='Sqrt_Metric',
#         hue='Type',
#         split=True,       
#         inner="quart",    
#         orient='h',       
#         palette=COLOR_PALETTE,
#         linewidth=1.0,    
#         cut=0,            
#         scale='width',    
#         density_norm='width', 
#         saturation=0.9    
#     )

#     # 5. 阈值线
#     THRESHOLD = 1.3
#     SQRT_THRESHOLD = np.sqrt(THRESHOLD)
#     plt.axvline(x=SQRT_THRESHOLD, color='#333333', linestyle='--', linewidth=1.5, alpha=0.9, zorder=0)
    
#     # 6. 坐标轴与刻度 (0, 1.3, 5, 10, 25)
#     tick_vals_original = [0, THRESHOLD, 5, 10, ORIGINAL_LIMIT]
#     tick_locs = [np.sqrt(v) for v in tick_vals_original]
#     tick_labels = ['0', r'$\tau$', '5', '10', f'>{int(ORIGINAL_LIMIT)}']
    
#     plt.xticks(tick_locs, tick_labels, fontsize=10)
#     plt.xlabel(f'Anomaly Score ($Z_i$) - Sqrt Scale', fontsize=11, labelpad=5)
    
#     plt.ylabel('')
#     plt.yticks(fontsize=11, fontweight='bold', color='#222222')
    
#     # 7. 美化
#     sns.despine(left=True, bottom=False, top=True, right=True)
#     ax.xaxis.grid(True, linestyle=':', alpha=0.5, color='gray')

#     # --- 关键修改：图例居中 (Top Center) ---
#     plt.legend(
#         loc='lower center',       # 以该点为基准的 "下中"
#         bbox_to_anchor=(0.5, 1.02), # 坐标 (0.5, 1.02) 即图表正上方
#         ncol=2, 
#         frameon=False, 
#         fontsize=10,
#         title=None,
#         handletextpad=0.3,
#         columnspacing=1.5  # 增加两个图例之间的间距
#     )

#     plt.tight_layout()
#     save_path = os.path.join(output_dir, f'SingleCol_Violin_{metric}_Limit25.pdf')
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"生成单栏图 (Limit=25): {save_path}")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--log_dir', type=str, default='src/logs')
#     parser.add_argument('--output', type=str, default='analysis_results')
#     args = parser.parse_args()
#     if not os.path.exists(args.output): os.makedirs(args.output)
    
#     log_files = glob.glob(os.path.join(args.log_dir, "**/*.log"), recursive=True)
#     all_data = []
#     print(f"扫描到 {len(log_files)} 个日志文件...")
#     for f in log_files:
#         df = parse_log_file(f)
#         if not df.empty: all_data.append(df)
        
#     if all_data:
#         final_df = pd.concat(all_data)
#         plot_single_column_violin(final_df, output_dir=args.output)
#     else:
#         print("未找到有效数据。")

# if __name__ == "__main__":
#     main()
# import os
# import re
# import glob
# import argparse
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # --- 学术单栏风格配置 ---
# sns.set_theme(style="ticks", context="paper", font_scale=1.2) 
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['ps.fonttype'] = 42

# # 配色：红蓝对比
# COLOR_PALETTE = {'Malicious': '#D64045', 'Benign': '#2E86AB'} 

# def parse_log_file(file_path):
#     """解析日志文件 (保持不变)"""
#     data = []
#     current_round = 0
#     round_pattern = re.compile(r'--------round\s+(\d+)\s+------------')
#     metric_pattern = re.compile(r'Client\s+(\d+)(?:\s+\((MAL)\))?:\s+Top-K=([\d\.]+),\s+Deviation=([\d\.]+).*?Z-score=([\d\.]+)')
    
#     attack_name = "Unknown"
#     filename = os.path.basename(file_path)
#     if "attack_" in filename:
#         try:
#             parts = filename.split("attack_")[1]
#             if "_" in parts: attack_name = parts.split("_")[0]
#             else: attack_name = parts.split(".")[0]
#         except: pass

#     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#         for line in f:
#             round_match = round_pattern.search(line)
#             if round_match:
#                 current_round = int(round_match.group(1))
#                 continue
#             metric_match = metric_pattern.search(line)
#             if metric_match:
#                 data.append({
#                     'Round': current_round,
#                     'Type': 'Malicious' if metric_match.group(2) == 'MAL' else 'Benign',
#                     'Z-score': float(metric_match.group(5)),
#                     'Attack': attack_name
#                 })
#     return pd.DataFrame(data)

# def plot_single_column_violin(all_data_df, metric='Z-score', output_dir='.', start_round=10):
#     if 'Attack' not in all_data_df.columns: return

#     # 1. 数据过滤
#     plot_data = all_data_df[all_data_df['Round'] >= start_round].copy()
#     plot_data = plot_data.dropna(subset=[metric])
    
#     # 格式化攻击名称
#     plot_data['Attack'] = plot_data['Attack'].apply(lambda x: x.capitalize())
#     plot_data['Attack'] = plot_data['Attack'].replace({'Badnet': 'BadNet', 'F3ba': 'F3BA', 'Dba': 'DBA', 'Pgd':'PGD'})

#     # 2. 平方根变换 & 截断到 25
#     plot_data['Sqrt_Metric'] = np.sqrt(plot_data[metric])
    
#     ORIGINAL_LIMIT = 25.0  
#     SQRT_LIMIT = np.sqrt(ORIGINAL_LIMIT)
#     plot_data['Sqrt_Metric'] = plot_data['Sqrt_Metric'].clip(upper=SQRT_LIMIT)

#     # 3. 画布设置
#     n_attacks = plot_data['Attack'].nunique()
#     plt.figure(figsize=(5.0, n_attacks * 0.8 + 1.0)) 

#     # 4. 绘图
#     ax = sns.violinplot(
#         data=plot_data,
#         y='Attack',
#         x='Sqrt_Metric',
#         hue='Type',
#         split=True,       
#         inner="quart",    
#         orient='h',       
#         palette=COLOR_PALETTE,
#         linewidth=1.0,    
#         cut=0,            
#         scale='width',    
#         density_norm='width', 
#         saturation=0.9    
#     )

#     # 5. 阈值线 (虚线)
#     THRESHOLD = 1.3
#     SQRT_THRESHOLD = np.sqrt(THRESHOLD)
#     plt.axvline(x=SQRT_THRESHOLD, color='#333333', linestyle='--', linewidth=1.5, alpha=1.0, zorder=0)

#     # 6. 坐标轴与刻度 (0, 1.3(τ), 5, 10, >25)
#     tick_vals_original = [0, THRESHOLD, 5, 10, ORIGINAL_LIMIT]
#     tick_locs = [np.sqrt(v) for v in tick_vals_original]
    
#     # --- 关键修改：在标签中直接注明 (τ) ---
#     tick_labels = ['0', r'1.3 ($\tau$)', '5', '10', '>25']
    
#     plt.xticks(tick_locs, tick_labels, fontsize=11)
#     plt.xlabel(f'Anomaly Score ($Z_i$) - Sqrt Scale', fontsize=12, labelpad=5)
    
#     plt.ylabel('')
#     plt.yticks(fontsize=12, fontweight='bold', color='#222222')
    
#     # 7. 美化
#     sns.despine(left=True, bottom=False, top=True, right=True)
#     ax.xaxis.grid(True, linestyle=':', alpha=0.5, color='gray')

#     # 图例居中
#     plt.legend(
#         loc='lower center',       
#         bbox_to_anchor=(0.5, 1.02), 
#         ncol=2, 
#         frameon=False, 
#         fontsize=11,
#         title=None,
#         handletextpad=0.4,
#         columnspacing=1.5  
#     )

#     plt.tight_layout()
#     save_path = os.path.join(output_dir, f'SingleCol_Labeled_{metric}.pdf')
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"生成最终图表: {save_path}")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--log_dir', type=str, default='src/logs')
#     parser.add_argument('--output', type=str, default='analysis_results')
#     args = parser.parse_args()
#     if not os.path.exists(args.output): os.makedirs(args.output)
    
#     log_files = glob.glob(os.path.join(args.log_dir, "**/*.log"), recursive=True)
#     all_data = []
#     print(f"扫描到 {len(log_files)} 个日志文件...")
#     for f in log_files:
#         df = parse_log_file(f)
#         if not df.empty: all_data.append(df)
        
#     if all_data:
#         final_df = pd.concat(all_data)
#         plot_single_column_violin(final_df, output_dir=args.output)
#     else:
#         print("未找到有效数据。")

# if __name__ == "__main__":
#     main()


import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 样式设置 (完全复刻 Line Chart 的配置)
# ==========================================
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 强制黑色字体
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2 # 与 Line Chart 一致

COLOR_PALETTE = {'Malicious': '#D62728', 'Benign': '#1F77B4'} # 使用 Line Chart 中 AlphaGuard 和 Lockdown 的配色风格

def parse_log_file(file_path):
    """(保持解析逻辑不变)"""
    data = []
    current_round = 0
    
    attack_name = "Unknown"
    filename = os.path.basename(file_path)
    if "attack_" in filename:
        try:
            parts = filename.split("attack_")[1]
            if "_" in parts: attack_name = parts.split("_")[0]
            else: attack_name = parts.split(".")[0]
        except: pass
    
    # 特殊处理：ff 攻击使用不同的日志格式
    is_ff_attack = "ff" in attack_name.lower() or "ff" in filename.lower()
    
    if is_ff_attack:
        # FF 格式：Round 0 (大写R，从0开始，无横线)
        round_pattern = re.compile(r'Round\s+(\d+)')
        # FF 格式：没有 Deviation 字段
        metric_pattern = re.compile(
            r'Client\s+(\d+)(?:\s+\((MAL)\))?\s*:\s*Top-K=\s*([\d\.]+).*?Z-score=\s*([\d\.]+)'
        )
    else:
        # 标准格式：--------round 1 ------------ (小写r，有横线)
        round_pattern = re.compile(r'--------round\s+(\d+)\s+------------')
        # 标准格式：有 Deviation 字段
        metric_pattern = re.compile(
            r'Client\s+(\d+)(?:\s+\((MAL)\))?:\s+Top-K=([\d\.]+),\s+Deviation=([\d\.]+).*?Z-score=([\d\.]+)'
        )
    
    print(f"解析日志: {attack_name} | FF模式={is_ff_attack}")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            round_match = round_pattern.search(line)
            if round_match:
                current_round = int(round_match.group(1))
                continue
            metric_match = metric_pattern.search(line)
            if metric_match:
                if is_ff_attack:
                    # FF 格式：只有 4 个匹配组（没有 Deviation）
                    data.append({
                        'Round': current_round,
                        'Type': 'Malicious' if metric_match.group(2) == 'MAL' else 'Benign',
                        'Z-score': float(metric_match.group(4)),  # 组号 4
                        'Attack': attack_name
                    })
                else:
                    # 标准格式：有 5 个匹配组（包含 Deviation）
                    data.append({
                        'Round': current_round,
                        'Type': 'Malicious' if metric_match.group(2) == 'MAL' else 'Benign',
                        'Z-score': float(metric_match.group(5)),  # 组号 5
                        'Attack': attack_name
                    })
    return pd.DataFrame(data)

def plot_compact_violin(all_data_df, metric='Z-score', output_dir='.', start_round=10):
    if 'Attack' not in all_data_df.columns: return

    # 数据过滤与转换
    plot_data = all_data_df[all_data_df['Round'] >= start_round].copy()
    plot_data = plot_data.dropna(subset=[metric])
    plot_data['Attack'] = plot_data['Attack'].apply(lambda x: x.capitalize())
    plot_data['Attack'] = plot_data['Attack'].replace({
        'Badnet': 'BadNet', 
        'F3ba': 'F3BA', 
        'Dba': 'DBA', 
        'Pgd': 'PGD',
        'Model': 'Model Rep',
        'Ff': 'F3BA',
        'Altermin': 'AlterMin'
    })
    
    # 自定义攻击顺序（从下到上）：BadNet, DBA, Neurotoxin, PGD, AlterMin, Model Rep, FF
    attack_order = ['BadNet', 'DBA', 'Neurotoxin', 'PGD', 'AlterMin', 'Model Rep', 'F3BA']
    # 只保留实际存在的攻击
    existing_attacks = [att for att in attack_order if att in plot_data['Attack'].unique()]
    plot_data['Attack'] = pd.Categorical(plot_data['Attack'], categories=existing_attacks, ordered=True)

    plot_data['Sqrt_Metric'] = np.sqrt(plot_data[metric])
    ORIGINAL_LIMIT = 25.0  
    SQRT_LIMIT = np.sqrt(ORIGINAL_LIMIT)
    plot_data['Sqrt_Metric'] = plot_data['Sqrt_Metric'].clip(upper=SQRT_LIMIT)

    # --- 关键修改：画布尺寸 (扁平化) ---
    n_attacks = plot_data['Attack'].nunique()
    # 宽度 6.5 (与 Line Chart 的 figsize=(6.5, 6) 宽度匹配)
    plt.figure(figsize=(6.5, n_attacks * 0.5 + 1.2)) 

    # 绘图
    ax = sns.violinplot(
        data=plot_data,
        y='Attack',
        x='Sqrt_Metric',
        hue='Type',
        split=True,       
        inner="quart",    
        orient='h',       
        palette=COLOR_PALETTE,
        linewidth=1.2,    # 线宽与边框一致
        cut=0,            
        scale='width',    
        density_norm='width', 
        saturation=1.0    # 饱和度设为1，颜色更实
    )

    # 阈值线
    THRESHOLD = 1.3
    SQRT_THRESHOLD = np.sqrt(THRESHOLD)
    plt.axvline(x=SQRT_THRESHOLD, color='#333333', linestyle='--', linewidth=1.5, alpha=1.0, zorder=0)

    # 坐标轴刻度与字体 (FontSize = 11, FontWeight = normal/bold 根据需要)
    tick_vals_original = [0, THRESHOLD, 5, 10, ORIGINAL_LIMIT]
    tick_locs = [np.sqrt(v) for v in tick_vals_original]
    tick_labels = ['0', r'1.3($\tau$)', '5', '10', '>25']
    
    plt.xticks(tick_locs, tick_labels, fontsize=11)
    # X轴标签 (FontSize = 12, Bold)
    plt.xlabel(f'Anomaly Score ($Z_i$) - Sqrt Scale', fontsize=12, fontweight='bold', labelpad=6)
    
    plt.ylabel('')
    # Y轴刻度 (攻击名称, FontSize = 12, Bold)
    plt.yticks(fontsize=12, fontweight='bold')
    
    # --- 关键修改：Spines (去上右，留左下，加粗) ---
    sns.despine(left=False, bottom=False, top=True, right=True) # 这里设为 False 是为了手动控制
    ax.spines['left'].set_visible(False)   # Violin图通常不需要左边框
    ax.spines['bottom'].set_visible(True)  # 需要底边框
    ax.spines['bottom'].set_linewidth(1.2) # 加粗
    
    # 网格线
    ax.xaxis.grid(True, linestyle='--', linewidth=0.8, color='#D3D3D3', alpha=0.8)

    # --- 关键修改：图例 (FontSize = 12, MarkerScale = 1.5) ---
    plt.legend(
        loc='upper center',       
        bbox_to_anchor=(0.5, 1.25), # 放在顶部
        ncol=2, 
        frameon=False, 
        fontsize=12,         # 字体放大
        markerscale=1.5,     # 图标放大
        title=None,
        handletextpad=0.4,
        columnspacing=1.5
    )

    # 紧凑布局
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'Compact_Violin_Styled.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"生成风格统一的图表: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='src/logs')
    parser.add_argument('--output', type=str, default='analysis_results')
    args = parser.parse_args()
    if not os.path.exists(args.output): os.makedirs(args.output)
    
    log_files = glob.glob(os.path.join(args.log_dir, "**/*.log"), recursive=True)
    all_data = []
    for f in log_files:
        df = parse_log_file(f)
        if not df.empty: all_data.append(df)
        
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        plot_compact_violin(final_df, output_dir=args.output)

if __name__ == "__main__":
    main()