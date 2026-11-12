"""Regenerable CSV-based reports. No access to training tensors is required."""
import csv
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROLES = ('benign', 'malicious')
COLORS = dict(benign='#2878b5', malicious='#c82423')


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def mean(rows, key):
    values = [float(r[key]) for r in rows if r.get(key, '') != '']
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float('nan')


def grouped_curve(rows, key, role=None):
    grouped = defaultdict(list)
    for row in rows:
        if role is None or row['role'] == role:
            grouped[int(row['round'])].append(row)
    rounds = sorted(grouped)
    return rounds, [mean(grouped[r], key) for r in rounds]


def save_figure(fig, path):
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def enrichment(rows, layers, role, set_name, round_number=None):
    selected = defaultdict(list)
    for row in rows:
        if (row['role'] == role and row['set_name'] == set_name
                and (round_number is None or int(row['round']) == round_number)):
            selected[row['layer']].append(row)
    dimension = sum(layer['numel'] for layer in layers)
    values = []
    for layer in layers:
        observations = selected[layer['name']]
        count = sum(int(r['count']) for r in observations)
        total = sum(int(r['set_size']) for r in observations)
        share = count / total if total else float('nan')
        values.append(dict(layer=layer['name'], count=count, share=share,
                           enrichment=share / (layer['numel'] / dimension)))
    return values


def role_summary(clients, metrics, metadata):
    results = []
    for role in ROLES:
        selected = [r for r in clients if r['role'] == role]
        results.append(dict(
            attack=metadata.get('training', {}).get('attack', '?'), role=role,
            status=metadata['status'], rounds=metadata.get('observed_rounds', 0),
            client_rounds=len(selected), overlap=mean(selected, 'raw_rel_overlap'),
            promoted_fraction=mean(selected, 'promoted_fraction'),
            raw_temporal_jaccard=mean(selected, 'raw_temporal_jaccard'),
            rel_temporal_jaccard=mean(selected, 'rel_temporal_jaccard'),
            promoted_temporal_jaccard=mean(selected, 'promoted_temporal_jaccard'),
            final_clean_acc=metrics[-1]['clean_acc'] if metrics else '',
            final_asr=metrics[-1]['asr'] if metrics else '',
        ))
    return results


def make_report(output):
    output = Path(output)
    meta = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    clients = read_csv(output / 'tables' / 'client_round.csv')
    rounds = read_csv(output / 'tables' / 'round.csv')
    metrics = read_csv(output / 'tables' / 'metrics.csv')
    layers = meta.get('parameter_layout', [])
    layer_rows = read_csv(output / 'tables' / 'layer_round.csv')
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    training = meta.get('training', {})
    title = f'{training.get("data", "?")} / {training.get("model", "?")} / {training.get("attack", "?")} / native {training.get("aggr", "?")}'
    generated = []
    if metrics:
        fig, ax = plt.subplots(figsize=(9, 4))
        for key, label in [('clean_acc', 'Clean ACC'), ('asr', 'ASR'), ('backdoor_acc', 'Triggered original-label ACC')]:
            ax.plot([int(r['round']) for r in metrics], [float(r[key]) for r in metrics], label=label)
        ax.set(xlabel='Global round', ylabel='Fraction', ylim=(-0.02, 1.02), title=title)
        ax.legend(); ax.grid(alpha=.2)
        name = '01_training.png'; save_figure(fig, figures / name); generated.append(name)
    if clients:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for role in ROLES:
            color = COLORS[role]
            x, y = grouped_curve(clients, 'promoted_fraction', role)
            axes[0].plot(x, y, color=color, label=role)
            for key, style, label in [('raw_temporal_jaccard', '--', 'raw'),
                                      ('rel_temporal_jaccard', '-', 'relative')]:
                x, y = grouped_curve(clients, key, role)
                axes[1].plot(x, y, style, color=color, label=f'{role}: {label}')
            x, y = grouped_curve(clients, 'promoted_temporal_jaccard', role)
            axes[2].plot(x, y, color=color, label=role)
        for ax, heading in zip(axes, ['Promoted fraction = 1 - raw/relative overlap',
                                      'Raw / relative temporal Jaccard', 'Promoted-set temporal Jaccard']):
            ax.set(xlabel='Global round', ylabel='Fraction', ylim=(-.02, 1.02), title=heading)
            ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.suptitle(title); fig.tight_layout()
        name = '02_overlap_and_stability.png'; save_figure(fig, figures / name); generated.append(name)
    if layers and layer_rows:
        fig, axes = plt.subplots(2, 3, figsize=(15, max(7, len(layers) * .3)))
        for i, role in enumerate(ROLES):
            for j, set_name in enumerate(['raw', 'relative', 'promoted']):
                values = enrichment(layer_rows, layers, role, set_name)
                image = np.asarray([r['enrichment'] for r in values])[:, None]
                image = np.log2(np.maximum(image, 1 / 16))
                ax = axes[i, j]
                im = ax.imshow(image, aspect='auto', vmin=-4, vmax=4, cmap='RdBu_r')
                ax.set(yticks=range(len(layers)), yticklabels=[r['name'] for r in layers],
                       xticks=[], title=f'{role}: {set_name}')
                ax.tick_params(axis='y', labelsize=7)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.55,
                     label='log2 enrichment; clipped to [-4, 4]; zero -> -4')
        fig.suptitle(title + '\nPooled layer concentration normalized by parameter count')
        name = '03_layer_distribution.png'; save_figure(fig, figures / name); generated.append(name)
        round_ids = sorted({int(r['round']) for r in layer_rows})
        # One grouping pass keeps report generation linear in the number of CSV rows.
        by_round = defaultdict(list)
        for row in layer_rows:
            if row['set_name'] == 'promoted':
                by_round[int(row['round'])].append(row)
        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(layers) * .28)))
        for ax, role in zip(axes, ROLES):
            values = np.asarray([[r['enrichment'] for r in enrichment(
                by_round[n], layers, role, 'promoted')] for n in round_ids]).T
            im = ax.imshow(np.log2(np.maximum(values, 1 / 16)), aspect='auto',
                           vmin=-4, vmax=4, cmap='RdBu_r', interpolation='nearest')
            ticks = np.unique(np.linspace(0, len(round_ids) - 1, min(6, len(round_ids))).astype(int))
            ax.set(yticks=range(len(layers)), yticklabels=[r['name'] for r in layers],
                   xticks=ticks, xticklabels=[round_ids[t] for t in ticks],
                   xlabel='Global round', title=role + ': promoted coordinates')
            ax.tick_params(axis='y', labelsize=7)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.6, label='log2 layer enrichment (clipped)')
        fig.suptitle(title)
        name = '04_layer_over_rounds.png'; save_figure(fig, figures / name); generated.append(name)
    if rounds:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        x = [int(r['round']) for r in rounds]
        axes[0].plot(x, [float(r['reference_abs_below_epsilon_fraction']) for r in rounds], label='abs(reference) < epsilon')
        axes[0].plot(x, [float(r['reference_abs_zero_fraction']) for r in rounds], label='reference = 0')
        axes[0].set(ylim=(-.02, 1.02), ylabel='Fraction', title='Denominator diagnostics')
        axes[0].legend(fontsize=8)
        axes[1].plot(x, [float(r['analysis_seconds']) for r in rounds])
        axes[1].set(ylabel='Seconds', title='Extra pre-aggregation analysis time')
        for ax in axes:
            ax.set_xlabel('Global round'); ax.grid(alpha=.2)
        fig.suptitle(title); fig.tight_layout()
        name = '05_reference_and_cost.png'; save_figure(fig, figures / name); generated.append(name)
    report = [f'# RAMP Top-K 诊断：{training.get("attack", "?")}', '',
              f'- 状态：**{meta["status"]}**；已观察全局轮数：{meta.get("observed_rounds", 0)}。',
              f'- 训练：`{title}`；客户端 {training.get("num_agents", "?")}，恶意 {training.get("num_corrupt", "?")}，'
              f'poison_frac={training.get("poison_frac", "?")}，Non-IID beta={training.get("beta", "?")}，'
              f'local_ep={training.get("local_ep", "?")}。',
              f'- 实际配置、公式、参数索引映射、版本及源码 SHA256：`run.json`；源码快照：`source_snapshot/`。',
              '- 观察过滤前的全部上传更新；原生 AlphaGuard 评分、过滤及聚合保持原样。',
              '- 本次相对特征使用 `abs(delta)/(abs(reference)+epsilon)`，不是原生 AlphaGuard 的内部相除公式。',
              '- 范围：可训练参数的本地模型差值，不是某个 batch 的梯度；BatchNorm 状态不参与本诊断。', '',
              '## 如何读这些结果', '',
              '- 原始集合：`TopK(abs(delta))`；相对集合：`TopK(abs(delta)/(abs(reference)+epsilon))`。',
              '- 差集（promoted）：相对集合减去原始集合；两个 Top-K 使用同一个 K。',
              '- overlap=`交集大小/K`；promoted fraction=`差集大小/K`，二者之和为 1。',
              '- temporal Jaccard：同一客户端相邻两次参与时集合的交并比；首轮和两个空集记为空值。',
              '- 层富集度=`该层占选中集合的比例/该层占全部可训练参数的比例`；1 为按参数量均匀分布。',
              '- 层图汇总使用坐标计数汇总后的比例；空差集为缺失值，零计数与缺失值不同。',
              '- 层指一个具名参数张量（如卷积权重或 BN 权重），索引映射见配置。', '',
              '## 良性与恶意客户端对比', '',
              '|角色|客户端-轮次观测数|原始/相对重合率|差集比例|相对集合时间 Jaccard|差集时间 Jaccard|',
              '|---|---:|---:|---:|---:|---:|']
    for row in role_summary(clients, metrics, meta):
        report.append(f'|{row["role"]}|{row["client_rounds"]}|{row["overlap"]:.4f}|{row["promoted_fraction"]:.4f}|'
                      f'{row["rel_temporal_jaccard"]:.4f}|{row["promoted_temporal_jaccard"]:.4f}|')
    if metrics:
        last = metrics[-1]
        report.extend(['', f'最后一次评估（Global {last["round"]}）：Clean ACC={last["clean_acc"]}，'
                       f'ASR={last["asr"]}，触发图片原标签准确率={last["backdoor_acc"]}。'])
    if layers and layer_rows:
        report.extend(['', '## 差集主要集中在哪里', '', '|角色|参数张量|累计选中次数|差集坐标占比|层富集度|',
                       '|---|---|---:|---:|---:|'])
        for role in ROLES:
            values = enrichment(layer_rows, layers, role, 'promoted')
            values = sorted([r for r in values if np.isfinite(r['enrichment']) and r['count']],
                            key=lambda r: r['enrichment'], reverse=True)[:5]
            for row in values:
                report.append(f'|{role}|{row["layer"]}|{row["count"]}|{row["share"]:.4f}|{row["enrichment"]:.3f}|')
    if rounds:
        report.extend(['', f'平均参考分量低于 epsilon 的比例：{mean(rounds, "reference_abs_below_epsilon_fraction"):.2%}；'
                       f'每轮诊断平均额外耗时：{mean(rounds, "analysis_seconds"):.3f}s。'])
    report.extend(['', '## 图片', ''])
    report.extend([f'![{name}](figures/{name})' for name in generated])
    report.extend(['', '## 解释边界', '',
                   '- 本实验描述坐标结构与时间稳定性，不证明这些坐标具有后门因果作用；下一步需要 Oracle 恢复。',
                   '- 良性/恶意身份仅用于分组统计。参考候选可能包含恶意客户端，其比例记录在 round.csv。',
                   '- 参考候选不足时回退到全部参与者；回退标志已记录，没有使用真实良性身份。',
                   '- 若参考幅值普遍低于 epsilon，应结合分母诊断理解相对排序，不能直接认定为低活动后门坐标。',
                   '- 大量边界并列值会影响 Top-K 身份；client_round.csv 保存并列计数，暂不解释为严格稳定性证据。',
                   '- 当前为单种子描述实验；跨攻击训练轨迹不同，不构成严格的配对因果比较。',
                   '- 如果只运行 1 个全局轮次，时间稳定性图为空是正常的；至少 2 轮才有相邻轮次比较。',
                   '- 训练指标来自原生日志的四位小数；状态 failed/interrupted 的结果属于部分实验。', '',
                   '## 文件', '',
                   '- `tables/client_round.csv`：每客户端每轮集合、分数、角色、参考身份及时间稳定性。',
                   '- `tables/layer_round.csv`：每客户端每轮每参数张量的集合数量和归一化富集度。',
                   '- `tables/round.csv`：参考候选、分母与耗时诊断。',
                   '- `tables/metrics.csv`：逐轮训练评估；不会重复计入末尾 best-results 汇总。',
                   '- `indices/`：默认保存 Global 1、每 25 轮及最后一轮的原始/相对坐标索引；差集可直接重算。',
                   '- `console.log`：完整控制台输出；不另写重复 training.log，不保存完整客户端更新向量。', ''])
    (output / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    print(f'Report generated: {output / "report.md"}', flush=True)


def compare_batch(batch):
    batch = Path(batch)
    output = batch / 'comparison'
    output.mkdir(exist_ok=True)
    experiments = []
    for path in sorted(batch.iterdir()):
        if path.is_dir() and (path / 'run.json').exists():
            meta = json.loads((path / 'run.json').read_text(encoding='utf-8'))
            experiments.append((path, meta, read_csv(path / 'tables' / 'client_round.csv'),
                                read_csv(path / 'tables' / 'metrics.csv')))
    if not experiments:
        raise ValueError('No experiment run.json found immediately below batch directory')
    summaries = [row for _, meta, clients, metrics in experiments
                 for row in role_summary(clients, metrics, meta)]
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for path, meta, _, metrics in experiments:
        if metrics:
            for ax, key in zip(axes, ['clean_acc', 'asr']):
                ax.plot([int(r['round']) for r in metrics], [float(r[key]) for r in metrics], label=path.name)
    for ax, name in zip(axes, ['Clean ACC', 'ASR']):
        ax.set(xlabel='Global round', ylabel=name, ylim=(-.02, 1.02), title=name)
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Different attacks under the same native aggregation (descriptive comparison)')
    fig.tight_layout(); save_figure(fig, output / '01_training_comparison.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = [path.name for path, *_ in experiments]
    positions = np.arange(len(experiments))
    for ax, key, label in zip(axes, ['promoted_fraction', 'promoted_temporal_jaccard'],
                              ['Promoted fraction', 'Promoted temporal Jaccard']):
        for offset, role in [(-.18, 'benign'), (.18, 'malicious')]:
            values = [next(r[key] for r in summaries if r['attack'] == meta['training']['attack']
                           and r['role'] == role) for _, meta, _, _ in experiments]
            ax.bar(positions + offset, values, width=.35, color=COLORS[role], label=role)
        ax.set(xticks=positions, xticklabels=labels, ylabel=label, ylim=(0, 1.05), title=label)
        ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); save_figure(fig, output / '02_sets_comparison.png')
    rows = ['# RAMP Top-K 独立实验批次', '',
            '这是不同攻击在同一原生聚合算法下的结构诊断，不是不同防御算法的效果比较。', '',
            '|攻击|状态|已观察轮数|详细报告|', '|---|---|---:|---|']
    for path, meta, _, _ in experiments:
        rows.append(f'|{path.name}|{meta["status"]}|{meta.get("observed_rounds", 0)}|[查看]({path.name}/report.md)|')
    rows.extend(['', '![训练曲线](comparison/01_training_comparison.png)', '',
                 '![集合对比](comparison/02_sets_comparison.png)', '',
                 '汇总数值：[summary.csv](comparison/summary.csv)。首轮没有时间稳定性数据；'
                 '单种子结果仅供描述分析。每项实验的配置及源码快照位于其独立目录。', ''])
    (batch / 'README.md').write_text('\n'.join(rows), encoding='utf-8')
    print(f'Batch comparison generated: {batch / "README.md"}', flush=True)
