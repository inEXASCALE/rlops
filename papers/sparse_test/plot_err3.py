import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 美学设置：白色背景、专业高对比配色
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")  # 新颜色方案：柔和但区分度高的定性配色（绿色、橙色、蓝绿色、粉红），非常适合论文

dir = 'results_random_sparse3'
df = pd.read_csv(f'{dir}/test_results.csv')

# 只保留有数据的 4 个组
bins = [101, 201, 301, 401, 501]
labels = ['101-200', '201-300', '301-400', '401-500']
df['size_group'] = pd.cut(df['matrix_size'], bins=bins, labels=labels, include_lowest=True)
df = df.dropna(subset=['size_group']).copy()

# 点大小反映样本数
group_counts = df['size_group'].value_counts()
df['point_size'] = df['size_group'].map(group_counts).astype(float) * 200
df['point_size'] = df['point_size'].clip(lower=300)

# 迭代次数轻微 jitter，避免重叠
jitter = 0.12
df['rl_gmres_jitter'] = df['rl_gmres_iterations'] + np.random.uniform(-jitter, jitter, size=len(df))
df['fp64_gmres_jitter'] = df['fp64_gmres_iterations'] + np.random.uniform(-jitter, jitter, size=len(df))

# 输出目录
output_dir = dir
os.makedirs(output_dir, exist_ok=True)

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), dpi=300)

# ====================== 左图：误差对比 ======================
sns.scatterplot(
    ax=axes[0],
    x='rl_error', y='fp64_error',
    data=df,
    hue='size_group',
    size='point_size',
    sizes=(300, 1200),
    alpha=0.75,
    edgecolor='white',
    linewidth=1.2,
    legend=False
)

min_val = min(df['rl_error'].min(), df['fp64_error'].min()) * 0.5
max_val = max(df['rl_error'].max(), df['fp64_error'].max()) * 2
axes[0].plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', linewidth=2.2)

axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel('RL Relative Forward Error', fontsize=14)
axes[0].set_ylabel('FP64 Relative Forward Error', fontsize=14)
axes[0].set_title('Error Comparison', fontsize=16, fontweight='bold')

# 增大坐标轴刻度标签大小
axes[0].tick_params(axis='both', which='major', labelsize=12)
axes[0].tick_params(axis='both', which='minor', labelsize=10)

# ====================== 右图：迭代次数对比 ======================
scatter = sns.scatterplot(
    ax=axes[1],
    x='rl_gmres_jitter', y='fp64_gmres_jitter',
    data=df,
    hue='size_group',
    size='point_size',
    sizes=(300, 1200),
    alpha=0.75,
    edgecolor='white',
    linewidth=1.2,
    legend='brief'
)

min_iter = df[['rl_gmres_iterations', 'fp64_gmres_iterations']].min().min() - 1
max_iter = df[['rl_gmres_iterations', 'fp64_gmres_iterations']].max().max() + 1
axes[1].plot([min_iter, max_iter], [min_iter, max_iter], color='gray', linestyle='--', linewidth=2.2)

axes[1].set_xlabel('RL GMRES Iterations', fontsize=14)
axes[1].set_ylabel('FP64 GMRES Iterations', fontsize=14)
axes[1].set_title('GMRES Iterations Comparison', fontsize=16, fontweight='bold')

# 增大坐标轴刻度标签大小
axes[1].tick_params(axis='both', which='major', labelsize=12)
axes[1].tick_params(axis='both', which='minor', labelsize=10)

# ============ 精美半透明图例 ============
handles, labels_legend = axes[1].get_legend_handles_labels()
hue_handles = handles[1:5]
hue_labels = labels_legend[1:5]



legend = axes[1].legend(
    hue_handles, hue_labels,
    title='Matrix Size',
    fontsize=12, title_fontsize=13,
    loc='upper left',
    frameon=True,
    fancybox=True,
    framealpha=0.85,      # 半透明背景
    facecolor='white',
    edgecolor='none',
    labelspacing=0.8,
    handletextpad=0.8,
    bbox_to_anchor=(1, 1)
)
# 整体标题
# plt.suptitle('RL vs FP64: Error and Iteration Comparison', fontsize=19, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 保存更高分辨率
plt.savefig(f'{dir}/error_iterations_comparison.pdf', bbox_inches='tight', dpi=400)
plt.savefig(f'{dir}/error_iterations_comparison.png', bbox_inches='tight', dpi=400)

plt.show()