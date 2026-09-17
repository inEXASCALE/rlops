import pandas as pd
import numpy as np
import ast

# Read the CSV files
metrics_df = pd.read_csv('test_metrics.csv')
results_df = pd.read_csv('test_results.csv')

# Filter out any 'average' row and convert matrix_id to int64
results_df = results_df[results_df['matrix_id'] != 'average']
results_df['matrix_id'] = results_df['matrix_id'].astype('int64')

# Merge the dataframes on matrix_id and matrix_size
merged_df = pd.merge(metrics_df, results_df, on=['matrix_id', 'matrix_size'])

# Define condition number ranges (aligned with training bucketing)
condition_ranges = [
    (1e0, 1e3, '10^0 to 10^3 (Low)'),
    (1e3, 1e6, '10^3 to 10^6 (Medium)'),
    (1e6, 1e10, 'results')
]



# Initialize results storage
precision_types = ['bf16', 'tf32', 'fp32', 'fp64']  # Match solver's precisions
results = []
# Success threshold: relative to fp64 unit roundoff (2^-53 ≈ 1.11e-16) scaled by condition number
base_threshold = 1e-6  # Tolerance from solver (tol=1e-7)

# Process each condition number range
for lower, upper, label in condition_ranges:
    # Filter matrices in the current condition number range
    range_df = merged_df[(merged_df['condition_number'] >= lower) & (merged_df['condition_number'] < upper)]
    
    if not range_df.empty:
        # Parse precision_usage dictionaries
        precision_counts = {prec: [] for prec in precision_types}
        for prec_str in range_df['precision_usage']:
            prec_dict = ast.literal_eval(prec_str)
            for prec in precision_types:
                precision_counts[prec].append(prec_dict.get(prec, 0))
        
        # Compute average precision usage
        avg_precision = {prec: np.mean(counts) if counts else 0 for prec, counts in precision_counts.items()}
        
        # Compute success rate (threshold scaled by condition number)
        max_errors = range_df[['rl_error', 'rl_nbe']].max(axis=1)
        # Scale threshold by median condition number in the range
        median_cond = np.median(range_df['condition_number'])
        threshold = base_threshold * median_cond
            
        # success mask
        success_mask = max_errors < threshold
        success_count = success_mask.sum()
        success_rate = success_count / len(range_df) if len(range_df) > 0 else 0

        # ===== 关键修改：仅对成功样本取 mean =====
        success_df = range_df#[success_mask]

        # 防止无成功样本时报错
        if success_df.empty:
            avg_metrics = {
                'Condition Range': label,
                'Num Matrices': len(range_df),
                'Success Rate': success_rate,
                'Avg RL Error': np.nan,
                'Avg RL NBE': np.nan,
                'Avg RL Rho': np.nan,
                'Avg RL CBE': np.nan,
                'Avg RL Iterations': np.nan,
                'Avg RL GMRES Iterations': np.nan,
                'Avg fp64 Error': np.nan,
                'Avg fp64 NBE': np.nan,
                'Avg fp64 Rho': np.nan,
                'Avg fp64 CBE': np.nan,
                'Avg fp64 Iterations': np.nan,
                'Avg fp64 GMRES Iterations': np.nan,
            }
        else:
            avg_metrics = {
                'Condition Range': label,
                'Num Matrices': len(range_df),
                'Success Rate': success_rate,
                'Avg RL Error': success_df['rl_error'].mean(),
                'Avg RL NBE': success_df['rl_nbe'].mean(),
                'Avg RL Rho': success_df['rl_rho'].mean(),
                'Avg RL CBE': success_df['rl_cbe'].mean(),
                'Avg RL Iterations': success_df['rl_iterations'].mean(),
                'Avg RL GMRES Iterations': success_df['rl_gmres_iterations'].mean(),
                'Avg fp64 Error': range_df['fp64_error'].mean(),
                'Avg fp64 NBE': range_df['fp64_nbe'].mean(),
                'Avg fp64 Rho': range_df['fp64_rho'].mean(),
                'Avg fp64 CBE': range_df['fp64_cbe'].mean(),
                'Avg fp64 Iterations': range_df['fp64_iterations'].mean(),
                'Avg fp64 GMRES Iterations': range_df['fp64_gmres_iterations'].mean(),
            }

            
        # Combine precision usage and metrics
        avg_metrics.update(avg_precision)
        results.append(avg_metrics)

# Create results DataFrame
columns = ['Condition Range', 'Num Matrices', 'Success Rate',
           'Avg RL Error', 'Avg RL NBE', 'Avg RL Rho', 'Avg RL CBE',
           'Avg RL Iterations', 'Avg RL GMRES Iterations',
           'Avg fp64 Error', 'Avg fp64 NBE', 'Avg fp64 Rho', 'Avg fp64 CBE',
           'Avg fp64 Iterations', 'Avg fp64 GMRES Iterations'] + precision_types
results_df = pd.DataFrame(results, columns=columns)

# Print the table with formatted floating-point values
print("\nAverage Precision Usage and Metrics by Condition Number Range:")
pd.set_option('display.float_format', '{:.4e}'.format)
print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv('precision_usage_and_metrics_by_condition.csv', index=False)