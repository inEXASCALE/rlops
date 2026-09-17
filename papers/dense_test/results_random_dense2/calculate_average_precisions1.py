import pandas as pd
import numpy as np
import ast

# Read the CSV files
metrics_df = pd.read_csv('test_metrics.csv')
results_df = pd.read_csv('test_results.csv')

# Filter out the 'average' row and convert matrix_id to int64
results_df = results_df[results_df['matrix_id'] != 'average']
results_df['matrix_id'] = results_df['matrix_id'].astype('int64')

# Merge the dataframes on matrix_id and matrix_size
merged_df = pd.merge(metrics_df, results_df, on=['matrix_id', 'matrix_size'])

# Define condition number ranges
condition_ranges = [
    (1e0, 1e1, '10^0 to 10^1'),
    (1e1, 1e2, '10^1 to 10^2'),
    (1e2, 1e3, '10^2 to 10^3'),
    (1e3, 1e4, '10^3 to 10^4'),
    (1e4, 1e5, '10^4 to 10^5'),
    (1e5, 1e6, '10^5 to 10^6'),
    (1e6, 1e7, '10^6 to 10^7'),
    (1e7, 1e8, '10^7 to 10^8'),
    (1e8, 1e9, '10^8 to 10^9'),
    (1e9, 1e10, '10^9 to 10^10')
]


# Initialize results storage
precision_types = ['bf16', 'tf32', 'fp32', 'fp64']
results = []
threshold = 1e-6  # Success threshold for max(rl_error, rl_nbe)

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
        
        # Compute success rate
        max_errors = range_df[['rl_error', 'rl_nbe']].max(axis=1)
        success_count = (max_errors < threshold).sum()
        success_rate = success_count / len(range_df) if len(range_df) > 0 else 0
        
        # Store results
        avg_precision['Condition Range'] = label
        avg_precision['Num Matrices'] = len(range_df)
        avg_precision['Success Rate'] = success_rate
        results.append(avg_precision)

# Create results DataFrame
results_df = pd.DataFrame(results, columns=['Condition Range', 'Num Matrices', 'Success Rate'] + precision_types)

# Print the table
print("\nAverage Precision Usage and Success Rate by Condition Number Range:")
print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv('precision_usage_by_condition.csv', index=False)