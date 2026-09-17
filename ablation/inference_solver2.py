import numpy as np
import torch
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import logging
from collections import Counter

# 导入必要的模块
from linear_system_utils import *
from gmresir_solver1 import GMRESIR5Solver, load_data

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局字体大小设置
FONTSIZE_TITLE = 16
FONTSIZE_LABELS = 16
FONTSIZE_TICKS = 16
FONTSIZE_LEGEND = 16
# 关键修改：换成更通用的字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 条件数分组定义
CONDITION_RANGES = [
    (1e0, 1e3, 'Low\n(10⁰-10³)'),
    (1e3, 1e6, 'Medium\n(10³-10⁶)'),
    (1e6, 1e9, 'High\n(10⁶-10⁹)')
]

def categorize_condition_number(cond):
    """根据条件数范围分类"""
    for lower, upper, label in CONDITION_RANGES:
        if lower <= cond < upper:
            return label
    # 处理边界情况
    if cond >= 1e9:
        return 'High\n(10⁶-10⁹)'  # 超高条件数也归入High
    return 'Low\n(10⁰-10³)'  # 极小条件数归入Low

def load_trained_model(solver, model_path):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return False
    
    success = solver.load_model(model_path)
    if success:
        logging.info(f"Successfully loaded model from {model_path}")
        logging.info(f"Agent has {len(solver.agent.valid_combinations)} valid precision combinations")
        logging.info(f"State space size: {solver.agent.state_space_size}")
    return success

def inference_test(solver, test_data, save_dir='inference_results'):
    """
    仅推理模式测试（不训练）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    results = []
    precision_usage_all = []  # 存储所有4个精度的使用情况
    
    logging.info(f"Starting inference on {len(test_data)} test matrices...")
    
    for i, (A_scipy, b_np, x_true_np, metadata) in enumerate(tqdm(test_data, desc="Inference")):
        # 准备数据
        A_torch = scipy_to_torch_sparse(A_scipy).coalesce()
        b = torch.tensor(b_np, dtype=torch.float64, device=solver.device).view(-1, 1)
        x_true = torch.tensor(x_true_np, dtype=torch.float64, device=solver.device).view(-1, 1)
        
        # 获取状态并选择精度（phase="test"，epsilon=0.1，基本都是exploitation）
        state = solver.agent.get_state(A_scipy, b_np)
        precisions, action_idx = solver.agent.choose_action(state, phase="test")
        
        # 记录选择的精度组合 [uf, ug, u, ur]
        precision_usage_all.append({
            'matrix_id': i,
            'uf': precisions[0],   # Factorization precision
            'ug': precisions[1],   # GMRES precision
            'u': precisions[2],    # Working precision
            'ur': precisions[3],   # Residual precision
            'condition_number': metadata['condition_number'],
            'matrix_size': metadata['matrix_size']
        })
        
        # 执行混合精度求解
        x_rl, iters_rl, iters_rl_gmres, errors_rl, log_rl, \
        final_nbe_rl, final_ferr_rl, final_rho_rl, final_cbe_rl, \
        converged, stagnation, converged_gmres, lu_failed, init_failed = \
            solver.ir5_gmres_mixed_precision(A_torch, b, x_true, 
                                            matrix_id=i, phase="test", 
                                            precisions=precisions)
        
        # 计算误差
        error_rl = torch.norm(x_rl - x_true, p=float('inf')) / torch.norm(x_true, p=float('inf'))
        
        # 存储结果
        result = {
            'matrix_id': i,
            'matrix_size': A_scipy.shape[0],
            'condition_number': metadata['condition_number'],
            'uf': precisions[0],
            'ug': precisions[1],
            'u': precisions[2],
            'ur': precisions[3],
            'error': float(error_rl),
            'nbe': float(final_nbe_rl),
            'rho': float(final_rho_rl),
            'cbe': float(final_cbe_rl),
            'iterations': int(iters_rl),
            'gmres_iterations': sum(iters_rl_gmres),
            'converged': converged,
            'lu_failed': lu_failed,
            'init_failed': init_failed
        }
        results.append(result)
        
        if (i + 1) % 10 == 0:
            logging.info(f"Processed {i+1}/{len(test_data)} matrices")
    
    # 保存结果
    df_results = pd.DataFrame(results)
    df_precision_usage = pd.DataFrame(precision_usage_all)
    
    csv_path = os.path.join(save_dir, 'inference_results.csv')
    precision_path = os.path.join(save_dir, 'precision_usage.csv')
    
    df_results.to_csv(csv_path, index=False)
    df_precision_usage.to_csv(precision_path, index=False)
    
    logging.info(f"Saved results to {csv_path}")
    logging.info(f"Saved precision usage to {precision_path}")
    
    return df_results, df_precision_usage

def visualize_precision_usage(df_precision, save_dir='inference_results'):
    """
    可视化四个精度步骤的使用比例
    """
    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 精度标签映射：小写转大写
    precision_label_map = {
        'e5m2': 'E5M2',
        'e4m3': 'E4M3',
        'bf16': 'BF16',
        'fp16': 'FP16',
        'tf32': 'TF32',
        'fp32': 'FP32',
        'fp64': 'FP64'
    }
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Precision Usage Distribution Across Four IR5 Stages', 
                 fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.995)
    
    precision_stages = ['uf', 'ug', 'u', 'ur']
    stage_names = [
        'Factorization ($u_f$)',
        'GMRES Solve ($u_g$)',
        'Working Precision ($u$)',
        'Residual Computation ($u_r$)'
    ]
    
    # 定义颜色映射
    precision_colors = {
        'E5M2': "#46bbd5",
        'E4M3': "#e68530",
        'BF16': "#43d143",
        'FP16': "#8D9F1D",
        'TF32': "#ea6247",
        'FP32': "#ef74ce",
        'FP64': "#d6f189"
    }
    
    for idx, (stage, name) in enumerate(zip(precision_stages, stage_names)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # 统计每个精度的使用次数并转换为大写标签
        precision_counts = df_precision[stage].value_counts()
        precision_counts.index = precision_counts.index.map(lambda x: precision_label_map.get(x, x.upper()))
        
        # 按预定义顺序排序（大写）
        precision_order = ['E5M2', 'E4M3', 'BF16', 'FP16', 'TF32', 'FP32', 'FP64']
        precision_counts = precision_counts.reindex(precision_order, fill_value=0)
        precision_counts = precision_counts[precision_counts > 0]  # 移除未使用的精度
        
        # 绘制饼图
        colors = [precision_colors.get(p, '#cccccc') for p in precision_counts.index]
        wedges, texts, autotexts = ax.pie(
            precision_counts.values,
            labels=precision_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': FONTSIZE_TICKS}
        )
        
        # 设置百分比文本样式
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(FONTSIZE_TICKS)
        
        # 设置标签文本样式
        for text in texts:
            text.set_fontsize(FONTSIZE_LEGEND)
            text.set_fontweight('bold')
        
        ax.set_title(name, fontsize=FONTSIZE_LABELS, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, 'precision_usage_pies_2_ab.jpg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='jpg')
    logging.info(f"Saved pie charts to {save_path}")
    plt.close()
    
    # === 第二个图：堆叠条形图 ===
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 准备数据（转换为大写标签）
    precision_order = ['E5M2', 'E4M3', 'BF16', 'FP16', 'TF32', 'FP32', 'FP64']
    stage_data = {}
    for stage in precision_stages:
        counts = df_precision[stage].value_counts()
        counts.index = counts.index.map(lambda x: precision_label_map.get(x, x.upper()))
        stage_data[stage] = counts.reindex(precision_order, fill_value=0)
    
    df_plot = pd.DataFrame(stage_data, index=precision_order).T
    df_plot = df_plot.loc[:, (df_plot != 0).any(axis=0)]  # 移除全0列
    
    # 绘制堆叠条形图
    df_plot.plot(kind='bar', stacked=True, ax=ax, 
                 color=[precision_colors.get(p, '#cccccc') for p in df_plot.columns],
                 width=0.5, edgecolor='black', linewidth=1.2)
    
    ax.set_title('Precision Distribution Across IR5 Stages (Stacked)', 
                 fontsize=FONTSIZE_TITLE, fontweight='bold', pad=20)
    ax.set_xlabel('IR5 Stage', fontsize=FONTSIZE_LABELS, fontweight='bold')
    ax.set_ylabel('Count', fontsize=FONTSIZE_LABELS, fontweight='bold')
    ax.set_xticklabels(stage_names, rotation=15, ha='right', fontsize=FONTSIZE_TICKS)
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICKS)
    
    # 透明图例
    legend = ax.legend(title='Precision', fontsize=FONTSIZE_LEGEND, title_fontsize=FONTSIZE_LEGEND,
                      loc='upper right', frameon=True, shadow=False)
    frame = legend.get_frame()
    frame.set_alpha(0.7)
    frame.set_facecolor('white')
    frame.set_edgecolor('gray')
    frame.set_linewidth(0.5)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'precision_usage_stacked_2_ab.jpg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='jpg')
    logging.info(f"Saved stacked bar chart to {save_path}")
    plt.close()
    
    # === 第三个图：按条件数分组的精度使用（共享图例）===
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 使用自定义分组函数
    df_precision['cond_bin'] = df_precision['condition_number'].apply(categorize_condition_number)
    
    # 定义分组顺序
    cond_bin_order = ['Low\n(10⁰-10³)', 'Medium\n(10³-10⁶)', 'High\n(10⁶-10⁹)']
    
    # 用于收集图例句柄和标签（只从第一个子图获取）
    handles_for_legend = None
    labels_for_legend = None
    
    for idx, (stage, name) in enumerate(zip(precision_stages, stage_names)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # 创建交叉表
        ct = pd.crosstab(df_precision['cond_bin'], df_precision[stage], normalize='index') * 100
        
        # 转换列名为大写
        ct.columns = ct.columns.map(lambda x: precision_label_map.get(x, x.upper()))
        
        # 确保所有分组都存在（即使某些分组没有数据）
        for bin_label in cond_bin_order:
            if bin_label not in ct.index:
                ct.loc[bin_label] = 0
        
        # 按指定顺序重排
        ct = ct.reindex(cond_bin_order)
        ct = ct.reindex(columns=precision_order, fill_value=0)
        ct = ct.loc[:, (ct != 0).any(axis=0)]
        
        # 绘制堆叠条形图
        ct.plot(kind='bar', stacked=True, ax=ax,
                color=[precision_colors.get(p, '#cccccc') for p in ct.columns],
                width=0.5, edgecolor='black', linewidth=0.8,
                legend=False)  # 关键：不在子图中显示图例
        
        # 只从第一个子图收集图例信息
        if idx == 0:
            handles_for_legend, labels_for_legend = ax.get_legend_handles_labels()
        
        ax.set_title(name, fontsize=FONTSIZE_LABELS, fontweight='bold', pad=10)
        
        # 只在底部两个子图显示 x 轴标签
        if row == 1:  # 底部行
            ax.set_xlabel('Condition Number Range', fontsize=FONTSIZE_LABELS, fontweight='bold')
        else:
            ax.set_xlabel('')
        
        # 只在左侧两个子图显示 y 轴标签
        if col == 0:  # 左侧列
            ax.set_ylabel('Percentage (%)', fontsize=FONTSIZE_LABELS, fontweight='bold')
        else:
            ax.set_ylabel('')
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=FONTSIZE_TICKS)
        ax.tick_params(axis='y', labelsize=FONTSIZE_TICKS)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 105])  # 设置y轴范围
    
    # 创建共享的透明图例（放在图的右侧中央）
    fig.legend(handles_for_legend, labels_for_legend,
               title='Precision', 
               fontsize=FONTSIZE_LEGEND-2,
               title_fontsize=FONTSIZE_LEGEND-2,
               loc='center right',
               bbox_to_anchor=(1.03, 0.5),
               frameon=False,
               fancybox=True,
               shadow=False,
               ncol=1)
    
    # 设置共享图例的透明度
    legend = fig.legends[0]  # 获取刚创建的图例
    frame = legend.get_frame()
    frame.set_alpha(0.7)
    frame.set_facecolor('white')
    frame.set_edgecolor('gray')
    frame.set_linewidth(0.5)
    
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])  # 为共享图例留出空间
    save_path = os.path.join(save_dir, 'precision_vs_condition_2_ab.jpg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='jpg')
    logging.info(f"Saved condition number analysis to {save_path}")
    plt.close()
    
    # === 第四个图：精度组合频率热图 ===
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 创建精度组合字符串（使用数学公式和大写精度）
    df_precision['combo'] = df_precision.apply(
        lambda row: f"[$u_f$={precision_label_map.get(row['uf'], row['uf'].upper())}, "
                    f"$u_g$={precision_label_map.get(row['ug'], row['ug'].upper())}, "
                    f"$u$={precision_label_map.get(row['u'], row['u'].upper())}, "
                    f"$u_r$={precision_label_map.get(row['ur'], row['ur'].upper())}]", 
        axis=1
    )
    
    # 统计前15个最常用的组合
    top_combos = df_precision['combo'].value_counts().head(15)
    
    # 绘制条形图
    bars = ax.barh(range(len(top_combos)), top_combos.values, 
                   color='steelblue', edgecolor='black', linewidth=1.2)
    
    # 为每个条形添加数值标签
    for i, (bar, val) in enumerate(zip(bars, top_combos.values)):
        ax.text(val + 0.5, i, f'{val}', va='center', fontsize=FONTSIZE_TICKS, fontweight='bold')
    
    ax.set_yticks(range(len(top_combos)))
    ax.set_yticklabels(top_combos.index, fontsize=FONTSIZE_TICKS-1)
    ax.set_xlabel('Frequency', fontsize=FONTSIZE_LABELS, fontweight='bold')
    ax.set_ylabel('Precision Combination', fontsize=FONTSIZE_LABELS, fontweight='bold')
    ax.set_title('Top 15 Most Frequently Used Precision Combinations', 
                 fontsize=FONTSIZE_TITLE, fontweight='bold', pad=10)
    ax.tick_params(axis='x', labelsize=FONTSIZE_TICKS)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'precision_combo_frequency_2_ab.jpg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='jpg')
    logging.info(f"Saved combination frequency to {save_path}")
    plt.close()

def print_summary_statistics(df_results, df_precision):
    """打印汇总统计信息"""
    print("\n" + "="*80)
    print("INFERENCE SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal test matrices: {len(df_results)}")
    print(f"Converged: {df_results['converged'].sum()} ({df_results['converged'].mean()*100:.1f}%)")
    print(f"LU Failed: {df_results['lu_failed'].sum()}")
    print(f"Init Failed: {df_results['init_failed'].sum()}")
    
    print(f"\n{'Metric':<20} {'Mean':<15} {'Median':<15} {'Std':<15}")
    print("-"*65)
    
    metrics = ['error', 'nbe', 'rho', 'iterations', 'gmres_iterations']
    for metric in metrics:
        values = df_results[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) > 0:
            print(f"{metric:<20} {values.mean():<15.4e} {values.median():<15.4e} {values.std():<15.4e}")
    
    print("\n" + "="*80)
    print("PRECISION USAGE STATISTICS")
    print("="*80)
    
    # 精度标签映射
    precision_label_map = {
        'e5m2': 'E5M2',
        'e4m3': 'E4M3',
        'bf16': 'BF16',
        'fp16': 'FP16',
        'tf32': 'TF32',
        'fp32': 'FP32',
        'fp64': 'FP64'
    }
    
    for stage in ['uf', 'ug', 'u', 'ur']:
        stage_full_names = {
            'uf': 'Factorization',
            'ug': 'GMRES',
            'u': 'Working',
            'ur': 'Residual'
        }
        print(f"\n{stage.upper()} (Stage: {stage_full_names[stage]}):")
        counts = df_precision[stage].value_counts()
        for prec, count in counts.items():
            prec_label = precision_label_map.get(prec, prec.upper())
            percentage = (count / len(df_precision)) * 100
            print(f"  {prec_label:<8}: {count:>4} ({percentage:>5.1f}%)")
    
    print("\n" + "="*80)
    print("TOP 10 PRECISION COMBINATIONS")
    print("="*80)
    
    df_precision['combo'] = df_precision.apply(
        lambda row: f"[{precision_label_map.get(row['uf'], row['uf'].upper())}, "
                    f"{precision_label_map.get(row['ug'], row['ug'].upper())}, "
                    f"{precision_label_map.get(row['u'], row['u'].upper())}, "
                    f"{precision_label_map.get(row['ur'], row['ur'].upper())}]", 
        axis=1
    )
    top_combos = df_precision['combo'].value_counts().head(10)
    for combo, count in top_combos.items():
        percentage = (count / len(df_precision)) * 100
        print(f"{combo:<40}: {count:>4} ({percentage:>5.1f}%)")
    
    # 添加条件数分组统计
    print("\n" + "="*80)
    print("CONDITION NUMBER DISTRIBUTION")
    print("="*80)
    
    df_precision['cond_bin'] = df_precision['condition_number'].apply(categorize_condition_number)
    cond_dist = df_precision['cond_bin'].value_counts()
    
    cond_bin_order = ['Low\n(10⁰-10³)', 'Medium\n(10³-10⁶)', 'High\n(10⁶-10⁹)']
    for bin_label in cond_bin_order:
        count = cond_dist.get(bin_label, 0)
        percentage = (count / len(df_precision)) * 100 if len(df_precision) > 0 else 0
        print(f"{bin_label.replace(chr(10), ' '):<30}: {count:>4} ({percentage:>5.1f}%)")
    
    print("\n" + "="*80)

def main():
    """主函数"""
    # 配置参数
    num_train = 100
    num_test = 100
    data_dir = 'data_random_dense2'
    model_path = 'results_random_dense2/q_tables.pkl'
    save_dir = 'inference_results2'
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 加载测试数据
    logging.info("Loading test data...")
    _, test_data, _, test_metrics = load_data(num_train, num_test, data_dir)
    
    if test_data is None:
        logging.error("Failed to load test data. Please check data directory.")
        return
    
    logging.info(f"Loaded {len(test_data)} test matrices")
    
    # 2. 初始化求解器（参数需与训练时一致）
    logging.info("Initializing solver...")
    solver = GMRESIR5Solver(
        max_iter=9999,
        tol=1e-6,
        precisions=['bf16', 'tf32', 'fp32', 'fp64'],
        max_inner_iter=1,
        max_stagnation=3,
        c1=1,
        c2=1
    )
    
    # 3. 加载训练好的模型
    logging.info(f"Loading trained model from {model_path}...")
    if not load_trained_model(solver, model_path):
        logging.error("Failed to load model. Exiting.")
        return
    
    # 4. 执行推理测试
    logging.info("Starting inference testing...")
    df_results, df_precision = inference_test(solver, test_data, save_dir)
    
    # 5. 可视化精度使用情况
    logging.info("Generating visualizations...")
    visualize_precision_usage(df_precision, save_dir)
    
    # 6. 打印汇总统计
    print_summary_statistics(df_results, df_precision)
    
    logging.info(f"\nAll results saved to: {save_dir}/")
    logging.info("Generated files:")
    logging.info("  - inference_results.csv: Detailed results for each matrix")
    logging.info("  - precision_usage.csv: Precision combinations used")
    logging.info("  - precision_usage_pies_2_ab.jpg: Pie charts for each stage")
    logging.info("  - precision_usage_stacked_2_ab.jpg: Stacked bar chart")
    logging.info("  - precision_vs_condition_2_ab.jpg: Precision vs condition number")
    logging.info("  - precision_combo_frequency_2_ab.jpg: Top combination frequencies")

if __name__ == "__main__":
    main()