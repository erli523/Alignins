import matplotlib.pyplot as plt
import pandas as pd

# 创建数据 (仅取 CIFAR10 部分作为演示)
columns = ['Defense', 'BadNet', 'DBA', 'Neurotoxin', 'PDG', 'AlterMin', 'ModelRep', 'ff']
data = [
    ['Fedavg', '0.8497/0.3458/0.5599', '0.8510/0.4444/0.4759', '0.8342/0.3060/0.5949', '0.8489/0.5647/0.3817', '0.7435/0.9030/0.6813', '0.8513/0.7146/0.2644', '0.8016/0.8849'],
    ['AlignIns', '0.7954/0.0102/0.7693', '0.8071/0.0310/0.7577', '0.8230/0.0208/0.7993', '0.8177/0.1697/0.6470', '0.8091/0.0150/0.8289', '0.8236/0.0130/0.8064', '0.8133/0.1939'],
    ['RLR', '0.7943/0.2322/0.5640', '0.8017/0.3443/0.5180', '0.7670/0.0460/0.7322', '0.7961/0.2347/0.5393', '0.6982/0.8330/0.6359', '0.8118/0.4680/0.4552', '0.8130/0.4166'],
    ['Multikrum', '0.7989/0.7579/0.2144', '0.7833/0.0510/0.7483', '0.7818/0.0351/0.7411', '0.8049/0.8169/0.1673', '0.7821/0.0540/0.7798', '0.7786/0.0363/0.7490', '0.7261/0.0872'],
    ['RFA', '0.7772/0.6546/0.2661', '0.7670/0.5088/0.3752', '0.7834/0.5427/0.3790', '0.7780/0.4579/0.3530', '0.7383/0.1070/0.7240', '0.6875/0.0362/0.6398', '0.5115/0.0171'],
    ['FoolsGold', '0.8462/0.3453/0.5577', '0.8492/0.3079/0.5621', '0.8172/0.0203/0.7909', '0.8519/0.5070/0.4249', '0.7368/0.9010/0.6747', '0.8546/0.7314/0.2418', '0.8151/0.9902'],
    ['Multi-Metirc', '0.7636/0.5994/0.3392', '0.7800/0.5043/0.4173', '0.7692/0.8161/0.1581', '0.7362/0.2999/0.5177', '0.7334/0.2000/0.7599', '0.6928/0.7320/0.2046', '0.7372/0.7792'],
    ['LockDown', '0.8239/0.2266/0.6293', '0.8303/0.1271/0.7186', '0.7537/0.0079/0.7534', '0.8195/0.1880/0.6547', '0.7968/0.0360/0.8041', '0.8059/0.0096/0.7799', '0.8004/0.2438'],
    ['AlphaGuard', '0.8230/0.0172/0.7988', '0.8312/0.0242/0.8049', '0.8230/0.0208/0.7993', '0.8282/0.0194/0.7981', '0.8351/0.0360/0.8358', '0.8252/0.0194/0.8092', '0.8127/0.0938']
]

fig, ax = plt.subplots(figsize=(16, 6))

# 隐藏坐标轴
ax.axis('tight')
ax.axis('off')

# 绘制表格
the_table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')

# 模拟 LaTeX Booktabs 风格 (调整字体大小和行高)
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1, 1.8)

# 获取所有单元格对象
cells = the_table.get_celld()

# 样式调整循环
for (row, col), cell in cells.items():
    cell.set_edgecolor('white') # 默认去掉所有边框
    cell.set_text_props(fontfamily='serif') # 使用衬线字体模拟 LaTeX
    
    # 表头 (Toprule & Midrule)
    if row == 0:
        cell.set_text_props(weight='bold')
        # 模拟上横线 (通过画线实现，这里简化处理)
        cell.visible_edges = 'TB' 
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5) # 粗线
        
    # 底部 (Bottomrule)
    if row == len(data):
        cell.visible_edges = 'B'
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)

    # AlphaGuard 标红 (最后一行的第一列)
    if row == len(data) and col == 0:
        cell.set_text_props(color='red', fontfamily='serif')

# 手动添加左侧的 "CIFAR10" 标签 (模拟 multirow)
plt.text(-0.08, 0.5, "CIFAR10\nnon-iid 0.5\npr 0.3", 
         transform=ax.transAxes, fontsize=11, 
         verticalalignment='center', horizontalalignment='center', 
         fontfamily='serif', color='black')

# 给左侧标签的部分文本标红 (通过覆盖绘制模拟，matplotlib直接混色很难)
# 为了演示效果简单起见，我们在左侧画一个红色的注释
plt.text(-0.08, 0.45, "non-iid 0.5\npr 0.3", 
         transform=ax.transAxes, fontsize=10, 
         verticalalignment='center', horizontalalignment='center', 
         fontfamily='serif', color='red')

plt.title("LaTeX Demo Preview (CIFAR10 Subset)", pad=20, fontsize=14, fontfamily='serif', weight='bold')
plt.show()