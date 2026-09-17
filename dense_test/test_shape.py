import numpy as np

for i in range(20):
    # 加载 npz 文件
    data = np.load(f"test_matrix_{i}.npz")
    print(f"{data['shape']}")