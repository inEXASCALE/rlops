import torch
import numpy as np
import os
# 导入你的主程序文件，假设文件名是 main.py
from svd_tests_last.test_check.gmresir_solver_demo import GMRESIR5Solver, generate_datasets, load_data

def train_and_save_demo():
    print("=== Phase 1: Training & Saving ===")
    # 1. 准备少量数据用于演示
    train_data, _ = generate_datasets(num_train=10, num_test=0, size_range=(50, 60))
    
    # 2. 初始化 Solver
    solver = GMRESIR5Solver(max_iter=50, tol=1e-6)
    
    # 3. 训练 (这里用很少的 episodes 演示)
    solver.train(train_data, episodes=2, results_dir='demo_results')
    
    # 4. 保存模型 (Train函数里其实已经保存了，但这里显式调用演示用法)
    model_path = 'results_random_dense5/final_model.pkl'
    solver.save_model(model_path)
    print(f"Model manually saved to {model_path}")

def load_and_test_demo():
    print("\n=== Phase 2: Loading & Testing (Inference) ===")
    model_path = 'results_random_dense5/final_model.pkl'
    
    if not os.path.exists(model_path):
        print("Model file not found, please run training phase first.")
        return

    # 1. 生成测试数据
    _, test_data = generate_datasets(num_train=0, num_test=5, size_range=(50, 60))
    
    # 2. 初始化一个新的 Solver 实例
    # 注意：这里不需要传入 train_data，因为我们要加载模型
    new_solver = GMRESIR5Solver(max_iter=50, tol=1e-6)
    
    # 3. 加载模型
    # 这会创建 solver.agent 并填充训练好的 Q表 和 参数
    success = new_solver.load_model(model_path)
    
    if success:
        print("Model loaded successfully. Agent Q-table shape:", new_solver.agent.q_tables[0].shape)
        
        # 4. 执行测试
        # solver.test 会使用加载进来的 agent 进行决策 (Exploitation)
        results, _, averages = new_solver.test(test_data)
        
        print("\nTest Results Summary:")
        print(f"Avg RL Iterations: {averages['avg_rl_iterations']}")
        print(f"Avg FP64 Iterations: {averages['avg_fp64_iterations']}")
    else:
        print("Failed to load model.")

if __name__ == "__main__":
    # 第一步：训练模型并保存文件
    train_and_save_demo()
    
    # 第二步：模拟在另一个时间或脚本中加载模型进行推理
    load_and_test_demo()