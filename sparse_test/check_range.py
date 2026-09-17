import pandas as pd

dir = "results_random_sparse1"

# 科学计数法格式化函数
def sci(x):
    return f"{x:.3e}".replace("e+0", " × 10^").replace("e+", " × 10^").replace("e-0", " × 10^-").replace("e-", " × 10^-")

def format_and_print(path, title):
    data = pd.read_csv(path)
    stats = pd.DataFrame({
        "min": data.min(),
        "max": data.max(),
        # "range": data.max() - data.min()
    }).round(3)

    print(f"\n{title}")
    print("-" * 60)

    for idx in stats.index:
        if idx == "condition_number":           # 科学计数法
            print(f"{idx:16s} : "
                  f"min={sci(stats.loc[idx,'min'])}, "
                  f"max={sci(stats.loc[idx,'max'])}, "
                  #f"range={sci(stats.loc[idx,'range'])}"
                  )
        else:
            print(f"{idx:16s} : "
                  f"min={stats.loc[idx,'min']:.3f}, "
                  f"max={stats.loc[idx,'max']:.3f}, "
                  #f"range={stats.loc[idx,'range']:.3f}"
                  )

format_and_print(f"{dir}/train_metrics.csv", "Train metrics stats")
format_and_print(f"{dir}/test_metrics.csv",  "Test metrics stats")


"""
train metrics stats:
                           min           max         range
condition_number  9.925603e+07  1.559453e+10  1.549528e+10
sparsity          1.846154e-02  5.058400e-02  3.212246e-02
matrix_size       1.030000e+02  5.000000e+02  3.970000e+02
test metrics stats:
                           min           max         range
condition_number  1.015548e+08  7.863479e+09  7.761925e+09
sparsity          1.750000e-02  5.047499e-02  3.297499e-02
matrix_size       1.000000e+02  4.980000e+02  3.980000e+02


============================================================


Train metrics stats
------------------------------------------------------------
condition_number : min=9.926 × 10^7, max=1.559 × 10^10, range=1.550 × 10^10
sparsity         : min=0.018, max=0.051, range=0.032
matrix_size      : min=103.000, max=500.000, range=397.000

Test metrics stats
------------------------------------------------------------
condition_number : min=1.016 × 10^8, max=7.863 × 10^9, range=7.762 × 10^9
sparsity         : min=0.018, max=0.050, range=0.033
matrix_size      : min=100.000, max=498.000, range=398.000


Train metrics stats
------------------------------------------------------------
condition_number : min=9.926 × 10^7, max=1.559 × 10^10, 
sparsity         : min=0.018, max=0.051

Test metrics stats
------------------------------------------------------------
condition_number : min=1.016 × 10^8, max=7.863 × 10^9, 
sparsity         : min=0.018, max=0.050

"""