"""Atomic, concurrent-safe reporting for the four independent Oracle trajectories."""
import csv
import fcntl
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from research.oracle_core import GROUPS


LABELS = dict(zip(GROUPS, ['A: No repair', 'B: Promoted repair', 'C: Layer-matched random', 'D: Full clean update']))
COLORS = ['#c44e52', '#4c72b0', '#dd8452', '#55a868']


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def atomic_text(path, content):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(content, encoding='utf-8')
    os.replace(temp, path)


def make_report(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    rows = read_csv(directory / 'tables' / 'metrics.csv')
    if not rows:
        atomic_text(directory / 'report.md', '# Oracle experiment\n\nNo completed evaluation yet. Inspect run.json and console.log.\n')
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    rounds = [int(r['round']) for r in rows]
    for ax, names in zip(axes, [('clean_acc', 'triggered_original_acc'), ('asr', 'clean_target_rate')]):
        for name in names:
            ax.plot(rounds, [100 * float(r[name]) for r in rows], label=name)
        ax.set(xlabel='Global round', ylabel='Percent', ylim=(0, 100))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(LABELS[metadata['group']])
    (directory / 'figures').mkdir(exist_ok=True)
    fig.savefig(directory / 'figures' / 'training.png', dpi=170); plt.close(fig)
    last = rows[-1]
    text = f'''# {LABELS[metadata['group']]}

状态：{metadata['status']}；已完成评估 {len(rows)} / {metadata['settings']['rounds']} 轮。

- 最后一轮 Clean ACC：{float(last['clean_acc']):.2%}
- 最后一轮 ASR：{float(last['asr']):.2%}
- 同一非目标干净测试集合预测为目标类：{float(last['clean_target_rate']):.2%}
- 带触发器图片原标签准确率：{float(last['triggered_original_acc']):.2%}

![Training](figures/training.png)

这是使用真实恶意身份和配对正常更新的 Oracle 实验，不是可部署防御。所有参与客户端均按数据量加权 FedAvg，不执行 AlphaGuard 淘汰或重加权。差集定位沿用前批诊断公式 abs(delta)/(abs(reference)+epsilon)，不是调用原生 AlphaGuard 的过滤流程。

每轮均进行干预，每组有独立全局模型。A/B/C/D 从相同初始模型和数据划分出发，后续轨迹会分叉。随机组每个客户端每轮一次独立抽样；并非多种子统计验证。poison_frac=0.3 沿用原生“非目标类候选集的 30%”语义。

## 文件

- run.json：设置、状态、源代码哈希、实际轮数、起始模型和数据划分指纹。
- console.log：本组完整控制台输出；Global 轮次、参与/恶意/良性名单、定位分数及修复数量。
- tables/metrics.csv：逐轮全局指标。
- tables/client_round.csv：逐客户端定位和替换幅度、聚合权重、攻击/正常更新差异。
- tables/layer_round.csv：恶意客户端各层差集与随机集合数量。
- snapshots/：指定轮次的聚合前全局模型、恶意更新、配对正常更新、修复索引和参与客户端权重。
- final_model.pt：最后一轮模型。
- source_snapshot/：本次执行源码快照。

默认只保存最后一轮完整配对快照，控制服务器磁盘占用。中间轮的统计仍逐轮保存。
'''
    atomic_text(directory / 'report.md', text)


def compare_batch(batch):
    batch = Path(batch)
    batch.mkdir(parents=True, exist_ok=True)
    with (batch / '.report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        available, statuses, metas = {}, {}, {}
        for group in GROUPS:
            path = batch / group
            if not (path / 'run.json').exists():
                statuses[group] = 'not_started'; continue
            meta = json.loads((path / 'run.json').read_text(encoding='utf-8'))
            metas[group] = meta
            statuses[group] = meta['status']
            # Only include closed, completed trajectories in cross-group plots.
            if meta['status'] == 'completed':
                rows = read_csv(path / 'tables' / 'metrics.csv')
                if len(rows) != meta['settings']['rounds']:
                    raise ValueError(f'Incomplete metrics for completed group {group}')
                available[group] = rows
        fingerprints = {(m.get('initial_model_sha256'), m.get('partition_sha256'),
                         m.get('poison_plan_sha256'), m['signature'])
                        for m in metas.values() if m.get('initial_model_sha256') and m.get('partition_sha256')}
        if len(fingerprints) > 1:
            raise ValueError('Group model/data/poison/config fingerprints differ; refuse comparison')
        first_updates = {(json.dumps(m.get('first_round_attack_sha256'), sort_keys=True),
                          json.dumps(m.get('first_round_clean_sha256'), sort_keys=True))
                         for group, m in metas.items() if group in available}
        if len(first_updates) > 1:
            raise ValueError('First-round attack/clean pairing differs across groups; refuse comparison')
        comparison = batch / 'comparison'
        comparison.mkdir(exist_ok=True)
        summary = []
        for group, rows in available.items():
            summary.append(dict(group=group, rounds=len(rows), final_acc=float(rows[-1]['clean_acc']),
                                final_asr=float(rows[-1]['asr']),
                                last10_acc=np.mean([float(r['clean_acc']) for r in rows[-10:]]),
                                last10_asr=np.mean([float(r['asr']) for r in rows[-10:]])))
        if summary:
            tmp = comparison / f'summary.{os.getpid()}.tmp'
            with tmp.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
                writer.writeheader(); writer.writerows(summary)
            os.replace(tmp, comparison / 'summary.csv')
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
            for group, color in zip(GROUPS, COLORS):
                if group not in available:
                    continue
                rows = available[group]
                for ax, metric in zip(axes, ['clean_acc', 'asr', 'clean_target_rate']):
                    ax.plot([int(r['round']) for r in rows], [100 * float(r[metric]) for r in rows],
                            label=LABELS[group], color=color)
                    ax.set(title=metric, xlabel='Global round', ylabel='Percent', ylim=(0, 100))
                    ax.grid(alpha=.2)
            axes[0].legend(fontsize=7)
            fig.suptitle('Oracle coordinate repair: independent FedAvg training trajectories')
            temp_image = comparison / f'curves.{os.getpid()}.tmp.png'
            fig.savefig(temp_image, dpi=170); plt.close(fig)
            os.replace(temp_image, comparison / 'curves.png')
            fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
            for group, color in zip(GROUPS, COLORS):
                if group in available:
                    rows = available[group][-10:]
                    ax.scatter(100*np.mean([float(r['asr']) for r in rows]),
                               100*np.mean([float(r['clean_acc']) for r in rows]),
                               color=color, s=70, label=LABELS[group])
            ax.set(xlabel='ASR (%) - lower is better', ylabel='Clean ACC (%) - higher is better',
                   title='Last 10 global rounds: descriptive means')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
            temp_image = comparison / f'tradeoff.{os.getpid()}.tmp.png'
            fig.savefig(temp_image, dpi=170); plt.close(fig)
            os.replace(temp_image, comparison / 'acc_asr.png')
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
            for group, color in [(GROUPS[1], COLORS[1]), (GROUPS[2], COLORS[2])]:
                if group not in available:
                    continue
                client_rows = read_csv(batch / group / 'tables/client_round.csv')
                by_round = {}
                for row in client_rows:
                    if row['role'] == 'malicious':
                        by_round.setdefault(int(row['round']), []).append(row)
                x = sorted(by_round)
                dimension = metas[group]['parameter_count']
                budgets = [100*np.mean([int(r['selected_count'])/dimension for r in by_round[n]]) for n in x]
                energies = [100*np.mean([float(r['oracle_error_energy_removed_fraction']) for r in by_round[n]]) for n in x]
                for ax, values in zip(axes, [budgets, energies]):
                    ax.plot(x, values, label=LABELS[group], color=color)
                    ax.set(xlabel='Global round', ylabel='Percent')
                    ax.grid(alpha=.2); ax.legend(fontsize=8)
            axes[0].set_title('Selected coordinates / model parameters')
            axes[1].set_title('Paired attack-clean error energy removed')
            fig.suptitle('Malicious-client means; each group uses its own evolving model')
            temp_image = comparison / f'budget.{os.getpid()}.tmp.png'
            fig.savefig(temp_image, dpi=170); plt.close(fig)
            os.replace(temp_image, comparison / 'repair_budget.png')
        lines = ['# RAMP Oracle 四组实验', '',
                 '四组独立训练，每轮干预，所有客户端按数据量加权 FedAvg；不执行 AlphaGuard 淘汰。', '',
                 '|组别|状态|报告|', '|---|---|---|']
        for group in GROUPS:
            lines.append(f'|{LABELS[group]}|{statuses[group]}|[本组报告]({group}/report.md)|')
        lines.extend(['', f'已完成 {len(available)}/4 组。只有 completed 组进入对比图；运行中可先查看各组 tables/metrics.csv。', '',
                      '|组别|最终 ACC|最终 ASR|末10轮 ACC|末10轮 ASR|', '|---|---:|---:|---:|---:|'])
        for r in summary:
            lines.append(f'|{r["group"]}|{r["final_acc"]:.2%}|{r["final_asr"]:.2%}|{r["last10_acc"]:.2%}|{r["last10_asr"]:.2%}|')
        if summary:
            lines += ['', '![逐轮指标](comparison/curves.png)', '', '![ACC-ASR](comparison/acc_asr.png)',
                      '', '![替换预算和正常更新差异](comparison/repair_budget.png)',
                      '', '替换预算/差异能量图是几何诊断，不等价于 ASR 修复效果。B/C 后期轨迹不同，预算也可能自然分叉。']
        lines += ['', '## 解释边界', '',
                  '优先比较 B 对 A 的 ASR 降幅与 ACC 代价，再比较 B 对 C 的额外价值，D 是完整正常更新参考。'
                  '本批不包含“原始幅值修复”第五组。随机组只有一条训练轨迹，不是随机重复置信区间。',
                  '真实恶意身份用于选取修复对象，配对正常更新用于替换，因此这是 Oracle 机制验证。'
                  '各组后续全局模型不同，最终差异包含训练反馈；不能只凭终局曲线推断单坐标因果效应。',
                  '第一轮快照指纹可验证同起点；各组末轮完整配对快照可用于后续局部反事实分析。', '']
        atomic_text(batch / 'README.md', '\n'.join(lines))
        print(f'Batch comparison: {batch / "README.md"}', flush=True)
