"""
precision_usage_visualizer.py
====================================
Automatically visualizes BF16/TF32/FP32/FP64 usage across Low/Medium/High
condition ranges for all four RL experiments.

UPDATED:
  • All numbers (segment labels + total labels) are now BLACK
  • Pure white background + uniform larger fonts kept
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('bmh')

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 19,
})

cmap = plt.get_cmap('Set3')
colors = cmap(np.linspace(0, 1, 4))

precisions = ['bf16', 'tf32', 'fp32', 'fp64']
precision_labels = ['BF16', 'TF32', 'FP32', 'FP64']

coarse_ranges = {
    'Low ($10^0$--$10^3$)':  ['10^0 to 10^1', '10^1 to 10^2', '10^2 to 10^3'],
    'Medium ($10^3$--$10^6$)': ['10^3 to 10^4', '10^4 to 10^5', '10^5 to 10^6'],
    'High ($10^6$--$10^9$)':  ['10^6 to 10^7', '10^7 to 10^8', '10^8 to 10^9'],
}

# ====================== DATA LOADING & AGGREGATION ======================
def aggregate_to_coarse(df: pd.DataFrame) -> dict:
    agg = {}
    for coarse_name, fine_list in coarse_ranges.items():
        sub = df[df['Condition Range'].isin(fine_list)].copy()
        if len(sub) == 0:
            continue
        total_matrices = sub['Num Matrices'].sum()
        if total_matrices == 0:
            continue
        group_data = {}
        for p in precisions:
            weighted = (sub[p] * sub['Num Matrices']).sum() / total_matrices
            group_data[p] = weighted
        agg[coarse_name] = group_data
    return agg


# Load all four experiments
experiments = [
    {"folder": "results_random_dense1", "label": "RL(W₁) τ = 10⁻⁶"},
    {"folder": "results_random_dense2", "label": "RL(W₂) τ = 10⁻⁶"},
    {"folder": "results_random_dense3", "label": "RL(W₁) τ = 10⁻⁸"},
    {"folder": "results_random_dense4", "label": "RL(W₂) τ = 10⁻⁸"},
]

all_agg = []
for exp in experiments:
    csv_path = Path(exp["folder"]) / "precision_usage_by_condition.csv"
    if not csv_path.exists():
        print(f"⚠️  File not found: {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    agg_data = aggregate_to_coarse(df)
    all_agg.append((exp["label"], agg_data))

# ====================== BEAUTIFUL PLOTTING ======================
fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
axes = axes.flatten()

for idx, (exp_label, agg_data) in enumerate(all_agg):
    ax = axes[idx]
    range_names = list(agg_data.keys())
    bottom = np.zeros(len(range_names))
    
    for i, p in enumerate(precisions):
        values = [agg_data[r].get(p, 0.0) for r in range_names]
        
        bars = ax.bar(range_names, values, bottom=bottom,
                      label=precision_labels[i] if idx == 0 else "",
                      color=colors[i], edgecolor='white', linewidth=1.5)
        
        # BLACK numbers inside segments
        for j, v in enumerate(values):
            if v > 0.03:  
                total_bar = sum(agg_data[range_names[j]].get(p2, 0.0) for p2 in precisions)
                if total_bar > 0 and v > 0.05 * total_bar:
                    ax.text(j, bottom[j] + v/2,
                            f'{v:.2f}',
                            ha='center', va='center',
                            fontsize=11.5, color='black', fontweight='bold')
        
        bottom += np.array(values)
    
    # BLACK total height labels on top
    for j, total in enumerate(bottom):
        ax.text(j, total + 0.03, f'{total:.2f}',
                ha='center', va='bottom', fontsize=12.5, fontweight='semibold', color='black')
    
    # Explicit ticks
    ax.set_xticks(np.arange(len(range_names)))
    ax.set_xticklabels(range_names, rotation=15, ha='right')
    
    ax.set_title(exp_label, pad=15)
    ax.set_ylabel('Average Precision Usage per Matrix\n'
                  '(total height = avg outer iterations)')
    ax.set_ylim(0, max(bottom) * 1.15 if len(bottom) > 0 else 5)

# Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Precision', title_fontsize=14,
           loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=4)

fig.suptitle('Precision Selection Distribution by Condition Number Range\n'
             'RL Mixed-Precision GMRES-IR',
             y=1.01)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save
plt.savefig('precision_usage_by_condition_range.png', dpi=400, bbox_inches='tight')
plt.savefig('precision_usage_by_condition_range.pdf', bbox_inches='tight')

print(" Visualization completed! (ALL numbers now BLACK)")
print("   -> precision_usage_by_condition_range.png (400 DPI)")
print("   -> precision_usage_by_condition_range.pdf")
plt.show()