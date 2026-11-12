"""Post-hoc interpretation of the completed RAMP Top-K batch; never trains."""
import ast
import collections
import csv
import json
import math
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'analysis'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(BASE / 'badnet' / 'source_snapshot'))
from research.topk_report import random_role_curve

ATTACKS = ['badnet', 'pgd', 'neurotoxin']
ROLES = ['benign', 'malicious']
PHASES = [('all', 1, 100), ('early', 1, 20), ('middle', 21, 60), ('late', 61, 100)]


def read(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def average(rows, key):
    values = [float(r[key]) for r in rows if r.get(key, '') != '']
    return float(np.mean([v for v in values if np.isfinite(v)]))


def write(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


phases, layer_summary, metrics_summary, audits = [], [], [], []
curves = {}
layer_stability = {}
poison_details = []
for attack in ATTACKS:
    path = BASE / attack
    meta = json.loads((path / 'run.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'completed' and meta['observed_rounds'] == 100
    clients = read(path / 'tables/client_round.csv')
    metrics = read(path / 'tables/metrics.csv')
    rounds = read(path / 'tables/round.csv')
    layers = read(path / 'tables/layer_round.csv')
    random_rows = read(path / 'tables/random_control.csv')
    assert len(clients) == 2000 and len(metrics) == 100 and len(rounds) == 100
    assert len(random_rows) == 2000 * meta['diagnostics']['random_repeats']
    assert len(layers) == 2000 * len(meta['parameter_layout']) * 3
    text = (path / 'console.log').read_text(encoding='utf-8')
    detections = re.findall(r'Malicious detected:\s*(\d+)/(\d+)', text)
    before = text.count('clip before:')
    after = text.count('clip after:')
    active = collections.Counter()
    current_round = 0
    for line in text.splitlines():
        match = re.search(r'--------round (\d+)', line)
        if match:
            current_round = int(match.group(1))
        if 'clip after:' in line:
            active[current_round] += 1
    audits.append(dict(attack=attack, clients=len(clients), random_records=len(random_rows),
                       all_four_removed_rounds=sum(a == b == '4' for a, b in detections),
                       detection_rounds=len(detections), projection_checks=before,
                       projection_activated=after, projection_rounds=json.dumps(dict(active)),
                       elapsed_seconds=meta['elapsed_seconds'],
                       mean_analysis_seconds=average(rounds, 'analysis_seconds')))
    metrics_summary.append(dict(attack=attack, final_acc=float(metrics[-1]['clean_acc']),
                                final_asr=float(metrics[-1]['asr']),
                                last10_acc=average(metrics[-10:], 'clean_acc'),
                                last10_asr=average(metrics[-10:], 'asr'),
                                max_asr=max(float(r['asr']) for r in metrics)))
    layout = meta['parameter_layout']
    index = {r['name']: i for i, r in enumerate(layout)}
    counts = collections.defaultdict(lambda: np.zeros(len(layout)))
    for row in layers:
        if row['set_name'] == 'promoted':
            counts[(row['role'], int(row['round']))][index[row['layer']]] += int(row['count'])
    for role in ROLES:
        curve = random_role_curve(random_rows, role)
        curves[(attack, role)] = curve
        for phase, lo, hi in PHASES:
            selected = [r for r in clients if r['role'] == role and lo <= int(r['round']) <= hi]
            round_rows = [r for r in rounds if lo <= int(r['round']) <= hi]
            curve_phase = [r for r in curve if lo <= r['round'] <= hi and np.isfinite(r['actual'])]
            row = dict(attack=attack, role=role, phase=phase, start=lo, end=hi,
                       promoted_fraction=average(selected, 'promoted_fraction'),
                       actual_jaccard=average(selected, 'promoted_temporal_jaccard'),
                       random_jaccard=average(selected, 'random_temporal_jaccard_mean'),
                       excess=average(selected, 'temporal_jaccard_excess'),
                       raw_jaccard=average(selected, 'raw_temporal_jaccard'),
                       relative_jaccard=average(selected, 'rel_temporal_jaccard'),
                       mean_epsilon_fraction=average(round_rows, 'reference_abs_below_epsilon_fraction'),
                       above_mc_envelope=sum(r['actual'] > r['q975'] for r in curve_phase),
                       defined_rounds=len(curve_phase))
            phases.append(row)
            arrays = np.stack([counts[(role, n)] for n in range(lo, hi + 1)])
            probabilities = arrays / arrays.sum(axis=1, keepdims=True)
            cosine = np.sum(probabilities[1:] * probabilities[:-1], axis=1) / (
                np.linalg.norm(probabilities[1:], axis=1) * np.linalg.norm(probabilities[:-1], axis=1))
            variation = np.abs(probabilities[1:] - probabilities[:-1]).sum(axis=1) / 2
            layer_stability[(attack, role, phase)] = (float(cosine.mean()), float(variation.mean()))
            share = arrays.sum(axis=0) / arrays.sum()
            for layer, fraction in zip(layout, share):
                layer_summary.append(dict(attack=attack, role=role, phase=phase, layer=layer['name'],
                                          share=float(fraction), enrichment=float(fraction) /
                                          (layer['numel'] / meta['analysis_parameter_count'])))
    if attack == 'badnet':
        for cid, values in re.findall(r'client:(\d+), distribution: (\[[^\]]+\])', text):
            if int(cid) < 4:
                values = ast.literal_eval(values)
                total = sum(values)
                eligible = total - values[meta['training']['target_class']]
                poisoned = math.floor(meta['training']['poison_frac'] * eligible)
                poison_details.append(dict(client=int(cid), local_samples=total,
                                           eligible_samples=eligible, poisoned_samples=poisoned,
                                           local_fraction=poisoned / total))

write('phase_summary.csv', phases)
write('layer_summary.csv', layer_summary)
write('metrics_summary.csv', metrics_summary)
write('implementation_audit.csv', audits)
write('poison_fraction_audit.csv', poison_details)

fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey='row', layout='constrained')
for i, role in enumerate(ROLES):
    upper = max(r['actual'] for attack in ATTACKS for r in curves[(attack, role)] if np.isfinite(r['actual'])) * 1.15
    for j, attack in enumerate(ATTACKS):
        ax = axes[i, j]
        curve = curves[(attack, role)]
        x = [r['round'] for r in curve]
        ax.plot(x, [r['actual'] for r in curve], color='#c82423' if i else '#2878b5', label='Actual promoted')
        ax.plot(x, [r['random'] for r in curve], '--', color='#555555', label='Matched random')
        ax.fill_between(x, [r['q025'] for r in curve], [r['q975'] for r in curve], color='gray', alpha=.25)
        ax.set(title=f'{attack}: {role}', ylim=(0, upper), xlabel='Global round')
        ax.grid(alpha=.2)
        if j == 0:
            ax.set_ylabel('Temporal Jaccard')
            ax.legend(fontsize=8)
fig.suptitle('Actual vs layer-matched random; shading = conditional Monte Carlo range')
fig.savefig(OUT / 'stability_overview.png', dpi=180)
plt.close(fig)

late = {(r['attack'], r['role']): r for r in phases if r['phase'] == 'late'}
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, layout='constrained')
for ax, role in zip(axes, ROLES):
    x = np.arange(3)
    for offset, key, color, label in [(-.18, 'actual_jaccard', '#2878b5' if role == 'benign' else '#c82423', 'Actual'),
                                     (.18, 'random_jaccard', '#999999', 'Matched random')]:
        values = [late[(attack, role)][key] for attack in ATTACKS]
        bars = ax.bar(x + offset, values, width=.35, color=color, label=label)
        ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
    ax.set(xticks=x, xticklabels=ATTACKS, title=role, ylabel='Mean temporal Jaccard', ylim=(0, .045))
    ax.legend(); ax.grid(axis='y', alpha=.2)
fig.suptitle('Late training: global rounds 61-100 (descriptive, one seed)')
fig.savefig(OUT / 'late_phase_stability.png', dpi=180)
plt.close(fig)

lines = ['# 本次 RAMP Top-K 实验结果分析', '',
         '结论：差集具有超过按层匹配随机基线的时间结构；BadNets 下恶意客户端更稳定，'
         'Neurotoxin 的角色区分较弱。这支持继续做 Oracle 恢复，但还不证明差集是后门功能坐标。', '',
         '三项实验均完整完成 100 个全局轮次，共 6,000 个客户端轮次观测和 180,000 条随机重复记录。'
         '本分析只读取已有 CSV、日志及源码快照，不重新训练、不修改原始结果。', '',
         '## 1. 训练状态', '', '|攻击|最终 Clean ACC|最终 ASR|末10轮平均 ACC|末10轮平均 ASR|',
         '|---|---:|---:|---:|---:|']
for r in metrics_summary:
    lines.append(f'|{r["attack"]}|{r["final_acc"]:.2%}|{r["final_asr"]:.2%}|{r["last10_acc"]:.2%}|{r["last10_asr"]:.2%}|')
lines.extend(['', '全局准确率约 83%，没有全局模型坍塌的迹象；ASR 较低。'
              '这些是现有 AlphaGuard 下的结果，RAMP 修复尚未启用，不能称为 RAMP 的防御效果。'
              '未记录同样非目标干净图片预测为目标类的比例，也没有本批无攻击基线，因此不能量化残余 ASR 中的后门成分。', '',
              '## 2. 差集稳定性与随机对照', '',
              '以下先看后期 Global 61—100，减少将初期变化与后期状态混在一起。', '',
              '|攻击|良性实际 Jaccard|良性随机|恶意实际 Jaccard|恶意随机|', '|---|---:|---:|---:|---:|'])
for attack in ATTACKS:
    b, m = late[(attack, 'benign')], late[(attack, 'malicious')]
    lines.append(f'|{attack}|{b["actual_jaccard"]:.5f}|{b["random_jaccard"]:.5f}|'
                 f'{m["actual_jaccard"]:.5f}|{m["random_jaccard"]:.5f}|')
lines.extend(['', '- 后期 BadNets 和原生 PGD 的恶意差集实际 Jaccard 约为良性的 5 倍；Neurotoxin 约为 1.7 倍，绝对区分弱一些。',
              '- 良性差集也长期高于随机，所以“高于随机”不能直接作为恶意判据。',
              '- 全程中同角色均值超过模拟 97.5% 分位数的轮次如下；轮次不独立，不将计数解释为正式显著性检验。', '',
              '|攻击|良性超过模拟上界|恶意超过模拟上界|', '|---|---:|---:|'])
for attack in ATTACKS:
    rows = {r['role']: r for r in phases if r['attack'] == attack and r['phase'] == 'all'}
    lines.append(f'|{attack}|{rows["benign"]["above_mc_envelope"]}/99|{rows["malicious"]["above_mc_envelope"]}/99|')
lines.extend(['', '具体坐标的绝对 Jaccard 仍较低，不能描述成一批基本不变的固定坐标。'
              '原始 Top-K 的时间 Jaccard 普遍高于相对 Top-K，也不支持“相对排序让坐标更稳定”这一说法。'
              '当前可支持的是差集存在非随机的持续成分，而非其稳定性优于所有其他选法。', '',
              '![跨轮稳定性](stability_overview.png)', '', '![后期稳定性](late_phase_stability.png)', '',
              '## 3. 集中在哪些层', '',
              '全程主要集中于 conv2、conv3、conv4 的卷积权重和 res1 两个卷积权重；良性、恶意有较多共同区域。'
              '下面为各攻击恶意客户端的全程主要参数张量，按实际差集份额排序。', '',
              '|攻击|参数张量|差集份额|按参数量归一富集度|', '|---|---|---:|---:|'])
for attack in ATTACKS:
    rows = sorted([r for r in layer_summary if r['attack'] == attack and r['role'] == 'malicious'
                   and r['phase'] == 'all'], key=lambda r: r['share'], reverse=True)[:5]
    for r in rows:
        lines.append(f'|{attack}|{r["layer"]}|{r["share"]:.2%}|{r["enrichment"]:.2f}×|')
lines.extend(['', '角色内按轮汇总的层分布相邻余弦相似度全程约 0.981—0.995，后期约 0.993—1.000。'
              '层分布比具体坐标稳定，但这也是良性更新具有的结构特征；余弦还会受大层主导。'
              '该指标是角色汇总层分布的稳定性，不代表每个客户端或具体坐标都不变。',
              '后期分类头在 BadNets/PGD 恶意差集约占 3.3%，良性约 1.8%；Neurotoxin 恶意仅约 0.67%，'
              '说明攻击机制不同，不能把某一层直接固定为通用后门区域。差集约 96%—99% 来自卷积权重，'
              '不是由 BN 可训练仿射参数主导，BN running 状态原本就未进入本诊断。', '',
              '## 4. 两个影响解释的重要实现事实', '',
              '### PGD 投影后期未生效', ''])
pgd = next(r for r in audits if r['attack'] == 'pgd')
lines.extend([f'日志有 {pgd["projection_checks"]} 次投影前检查，实际投影 {pgd["projection_activated"]} 次，'
              f'触发轮次及次数为 {pgd["projection_rounds"]}。后 98 轮未触发投影。',
              '本次源码快照 src/agent.py 以 0.1×全局模型 L2 为 CIFAR-10 投影半径，'
              '只在本地更新超出该半径时投影。因而不能将这里与 BadNets 相近的结果视为两种充分独立攻击机制的重复验证。'
              '这是本次原生实现和参数下的事实，不是所有 PGD 方法的一般结论。', '',
              '### poison_frac=0.3 的实际含义', '',
              '源码快照 src/utils.py::poison_dataset 对非目标类候选样本取 30%，不是对整个本地数据集取 30%。'
              '根据日志中的原始客户端类别分布及源码规则，实际数量推导如下（不是额外保存的投毒清单）。', '',
              '|恶意客户端|本地样本数|非目标类候选|推导投毒数|占全部本地样本|', '|---|---:|---:|---:|---:|'])
for r in poison_details:
    lines.append(f'|{r["client"]}|{r["local_samples"]}|{r["eligible_samples"]}|{r["poisoned_samples"]}|{r["local_fraction"]:.2%}|')
weighted_poison = sum(r['poisoned_samples'] for r in poison_details) / sum(r['local_samples'] for r in poison_details)
lines.extend(['', f'四个恶意客户端按样本数合计约 {weighted_poison:.2%}。这属于框架原生配置语义，'
              '论文应明确写成候选集比例，或后续另设“占全部本地数据 30%”实验；本次未修改代码。', '',
              '## 5. epsilon 与参考', '',
              '后期全模型约 99.8% 的参考坐标绝对值小于 epsilon=0.001；epsilon 在这些分母中至少与参考同等重要。'
              '这不是除零错误，也不能据此断定相对排序失效；但不能忽略其正则化尺度对排序的影响。'
              '现有比例统计针对全模型，不是入选差集坐标的比例；本次未保存相应坐标值，因此不能进一步推断其选中分母分布。', '',
              '## 6. 对 RAMP 后续设计的判断', '',
              '1. 值得继续做小范围 Oracle 坐标恢复，优先 BadNets，再检查 Neurotoxin 的适用边界。',
              '2. 差集比例本身不能当恶意分数：良性全程约 49%，BadNets/PGD 恶意约 34%，Neurotoxin 恶意约 21%。',
              '3. 不应只凭坐标属于差集就衰减；活动范围阈值 C 和客户端强度 omega 仍需验证，并报告良性误修复。',
              '4. Oracle 应比较实际集合与同数量同层随机集合、原始 Top-K，以及全部坐标恢复上限；需要同客户端反事实正常更新。',
              '5. 当前只有单种子、单数据集、单模型、固定恶意客户端身份。Non-IID 客户端差异可能混入角色差异，'
              '不能由当前结构统计直接推出后门因果作用。',
              '6. 本次未保存完整模型或更新，CSV 和稀疏索引不能独立完成 Oracle；后续需要在新训练的选定轮次采集配对数据。', '',
              '阶段划分和角色层分布分析为事后描述，不是预注册检验；模拟范围不是置信区间。', '',
              '## 可复核文件', '',
              '- phase_summary.csv：全程、前期、中期、后期的角色统计。',
              '- layer_summary.csv：各阶段、各角色的层份额和富集度。',
              '- metrics_summary.csv：训练指标汇总。',
              '- implementation_audit.csv：过滤记录、PGD 投影计数、耗时。',
              '- poison_fraction_audit.csv：基于原始类别分布与源码规则的实际投毒比例推导。',
              '- build_interpretation.py：本分析脚本；只处理已有结果。', ''])
(OUT / 'RESULT_ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')
print('Analysis saved:', OUT / 'RESULT_ANALYSIS.md')
print('Final metrics:', metrics_summary)
print('PGD projection audit:', pgd)
