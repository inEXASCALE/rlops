import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# ===========================
# CONFIG: Font Sizes
# ===========================
FONT_SIZES = {
    'title': 16,
    'axis_label': 14,
    'tick_label': 12,
    'legend': 12,
    'pie_text': 12
}

# ===========================
# 1. Load Data
# ===========================
print("Loading data...")
metrics_df = pd.read_csv("test_metrics.csv")
results_df = pd.read_csv("test_results.csv")

base_threshold = 1e-9  # Tolerance from solver (tol=1e-7)

# ===========================
# 2. Merge & Parse Precision
# ===========================
df = metrics_df.merge(results_df, on='matrix_id', suffixes=('_metric', '_result'))

def parse_precision(precision_str):
    try:
        d = ast.literal_eval(precision_str)
        return pd.Series({
            'fp32_count': d.get('fp32', 0),
            'fp64_count': d.get('fp64', 0),
            'bf16_count': d.get('bf16', 0),
            'tf32_count': d.get('tf32', 0),
            'total_precisions': sum(d.values()),
            'uses_mixed': any(k in d and d[k] > 0 for k in ['bf16', 'tf32', 'fp32'])
        })
    except Exception as e:
        print(f"Error parsing: {precision_str} -> {e}")
        return pd.Series({'fp32_count': 0, 'fp64_count': 0, 'bf16_count': 0, 'tf32_count': 0,
                          'total_precisions': 0, 'uses_mixed': False})

precision_parsed = df['precision_usage'].apply(parse_precision)
df = pd.concat([df, precision_parsed], axis=1)

# ===========================
# 3. Summary Statistics
# ===========================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

metrics_stats = metrics_df[['condition_number', 'sparsity', 'matrix_size']].describe()
print("\n1. Test Matrices (test_metrics.csv):")
print(metrics_stats.loc[['mean', 'min', 'max']].round(6))

rl_cols = ['rl_error', 'rl_iterations', 'rl_gmres_iterations', 'fp32_count', 'fp64_count']
rl_stats = df[rl_cols].describe().loc[['mean', 'min', 'max']]
fp64_cols = ['fp64_error', 'fp64_iterations', 'fp64_gmres_iterations']
fp64_stats = df[fp64_cols].describe().loc[['mean', 'min', 'max']]

max_errors = df[['rl_error', 'rl_nbe']].max(axis=1)
median_cond = np.median(df['condition_number'])
threshold = base_threshold * median_cond
success_count = (max_errors < threshold).sum()
success_rate = success_count / len(df) if len(df) > 0 else 0

print("\n2. RL Performance:")
print(rl_stats.round(6))

print("\n3. FP64 Baseline Performance:")
print(fp64_stats.round(6))

# ===========================
# 4. Create Figure with LARGE FONTS
# ===========================
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': FONT_SIZES['tick_label'],
    'axes.titlesize': FONT_SIZES['title'],
    'axes.labelsize': FONT_SIZES['axis_label'],
    'xtick.labelsize': FONT_SIZES['tick_label'],
    'ytick.labelsize': FONT_SIZES['tick_label'],
    'legend.fontsize': FONT_SIZES['legend'],
    'figure.figsize': (12, 6)  # Larger figure
})

fig = plt.figure()

# --- Plot 1: RL vs FP64 Error ---
ax1 = plt.subplot(1, 2, 1)
df['error_ratio'] = df['rl_error'] / df['fp64_error']
sns.scatterplot(data=df, x='fp64_error', y='rl_error', hue='uses_mixed', palette='deep', ax=ax1, s=80)
ax1.plot([1e-12, 1e-6], [1e-12, 1e-6], 'r--', label='y = x', linewidth=2)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('FP64 Error', fontsize=FONT_SIZES['axis_label'])
ax1.set_ylabel('RL Error', fontsize=FONT_SIZES['axis_label'])
ax1.set_title('RL vs FP64 Error', fontsize=FONT_SIZES['title'], fontweight='bold', pad=15)
legend1 = ax1.legend(title='Mixed Precision', loc='lower right', fontsize=FONT_SIZES['legend'], title_fontsize=FONT_SIZES['legend'], fancybox=True, framealpha=0.2)
legend1.get_title().set_fontweight('bold')

# --- Plot 2: Error Ratio vs Condition Number ---
ax2 = plt.subplot(1, 2, 2)
df['log_cond'] = np.log10(df['condition_number'])
sns.scatterplot(data=df, x='log_cond', y='error_ratio', hue='uses_mixed', ax=ax2, s=80)
ax2.axhline(1, color='red', linestyle='--', linewidth=2, label='Equal Error')
ax2.set_yscale('log')
ax2.set_xlabel('$\\log_{10}(\\kappa(A))$', fontsize=FONT_SIZES['axis_label'] + 2)
ax2.set_ylabel('RL Error / FP64 Error', fontsize=FONT_SIZES['axis_label'])
ax2.set_title('Accuracy vs Matrix Conditioning', fontsize=FONT_SIZES['title'], fontweight='bold', pad=15)
legend2 = ax2.legend(title='Mixed Precision', fontsize=FONT_SIZES['legend'], title_fontsize=FONT_SIZES['legend'], fancybox=True, framealpha=0.2)
legend2.get_title().set_fontweight('bold')

plt.tight_layout(pad=4.0, w_pad=3.0, h_pad=3.0)

output_plot = "rl_vs_fp64_comparison.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nFigure saved as: {output_plot}")

# Optional: Show
plt.show()

print("\n" + "="*60)
print("FINAL COMPARISON SUMMARY")
print("="*60)

summary = pd.DataFrame({
    'Metric': [
        'Success Rate',
        'Mean RL Error',
        'Mean RL NBE',
        'Mean FP64 Error',
        'Mean FP64 NBE',
        'Mean Error Ratio (RL/FP64)',
        'RL Worse than FP64 (%)',
        'Uses Mixed Precision (%)',
        'Avg RL Outer Iter',
        'Avg FP64 Outer Iter',
    ],
    'Value': [
        f"{success_rate * 100:.2f}%",
        f"{df['rl_error'].mean():.2e}",
        f"{df['rl_nbe'].mean():.2e}",
        f"{df['fp64_error'].mean():.2e}",
        f"{df['fp64_nbe'].mean():.2e}",
        f"{df['error_ratio'].mean():.3f}",
        f"{100 * (df['rl_error'] > df['fp64_error']).mean():.1f}%",
        f"{100 * df['uses_mixed'].mean():.1f}%",
        f"{df['rl_iterations'].mean():.2f}",
        f"{df['fp64_iterations'].mean():.2f}",
        #f"{df['rl_gmres_iterations'].mean():.1f}",
        #f"{df['fp64_gmres_iterations'].mean():.1f}"
    ]
})

print(summary.to_string(index=False))