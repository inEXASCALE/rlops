import numpy as np
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix
import logging
import os
import random
from scipy.linalg import qr
from scipy import sparse
from scipy.sparse.linalg import splu

torch.set_printoptions(precision=8)

import pychop
from pychop import Chop
pychop.backend('torch', 1)

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

device = 'cpu' 
precision_configs = {
    'e5m2': {'exp_bits': 5, 'sig_bits': 2, 'rmode': 1},
    'e4m3': {'exp_bits': 4, 'sig_bits': 3, 'rmode': 1},
    'bf16': {'exp_bits': 8, 'sig_bits': 7, 'rmode': 1},
    'fp16': {'exp_bits': 5, 'sig_bits': 10, 'rmode': 1},
    'tf32': {'exp_bits': 8, 'sig_bits': 10, 'rmode': 1},
    'fp32': {'exp_bits': 8, 'sig_bits': 23, 'rmode': 1},
    'fp64': {'exp_bits': 11, 'sig_bits': 52, 'rmode': 1}
}

# Define precision order based on sig_bits
precision_order = sorted(precision_configs.keys(), key=lambda p: (precision_configs[p]['sig_bits'], p))

def gallery_randsvd(n, kappa=None, mode=3, kl=None, ku=None, method=0, random_state=42):
    np.random.seed(random_state)
    if kappa is None:
        kappa = np.sqrt(1 / np.finfo(float).eps)

    if isinstance(n, (list, tuple, np.ndarray)):
        if len(n) != 2:
            raise ValueError("n as a vector must have exactly two elements [m n]")
        m, n = map(int, n)
        if m < 1 or n < 1:
            raise ValueError("Matrix dimensions must be positive integers")
    else:
        m = n = int(n)
        if m < 1:
            raise ValueError("n must be a positive integer")

    if kl is None:
        kl = m - 1
    if ku is None:
        ku = kl
    if not (isinstance(kl, int) and isinstance(ku, int) and kl >= 0 and ku >= 0):
        raise ValueError("kl and ku must be non-negative integers")
    if kl >= m or ku >= n:
        raise ValueError("kl and ku must be less than matrix dimensions")

    if mode not in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
        raise ValueError("Mode must be an integer from -5 to -1 or 1 to 5")
    if method not in [0, 1]:
        raise ValueError("Method must be 0 or 1")

    if kappa <= 1:
        if m != n:
            raise ValueError("For kappa <= 1, matrix must be square (m == n)")
        lambda_min = abs(kappa)
        lambda_max = 1.0
        p = n
        if mode in [1, -1]:
            sigma = np.ones(p) * lambda_min
            sigma[0] = lambda_max
        elif mode in [2, -2]:
            sigma = np.ones(p) * lambda_max
            sigma[-1] = lambda_min
        elif mode in [3, -3]:
            k = np.arange(p)
            sigma = lambda_max * (lambda_min / lambda_max) ** (k / max(p - 1, 1))
        elif mode in [4, -4]:
            k = np.arange(p)
            sigma = lambda_max - (k / max(p - 1, 1)) * (lambda_max - lambda_min)
        elif mode in [5, -5]:
            r = np.random.rand(max(p - 2, 0))
            sigma = np.zeros(p)
            sigma[0] = lambda_max
            if p > 1:
                sigma[-1] = lambda_min
            if p > 2:
                sigma[1:-1] = lambda_max * np.exp(np.log(lambda_min / lambda_max) * r)
        if mode < 0:
            sigma = np.sort(sigma)
        else:
            sigma = np.sort(sigma)[::-1]
        if np.any(sigma <= 0):
            raise ValueError("Eigenvalues must be positive for symmetric positive definite matrix")
        X = np.random.randn(n, n)
        Q, _ = qr(X, mode='economic' if method == 0 else 'full')
        A = Q @ np.diag(sigma) @ Q.T
        return sp.csr_matrix(A)

    if abs(kappa) < 1:
        raise ValueError("For non-symmetric case, abs(kappa) must be >= 1")
    
    sigma_max = 1.0
    sigma_min = sigma_max / abs(kappa)
    p = min(m, n)

    if mode in [1, -1]:
        sigma = np.ones(p) * sigma_min
        sigma[0] = sigma_max
    elif mode in [2, -2]:
        sigma = np.ones(p) * sigma_max
        sigma[-1] = sigma_min
    elif mode in [3, -3]:
        k = np.arange(p)
        sigma = sigma_max * (sigma_min / sigma_max) ** (k / max(p - 1, 1))
    elif mode in [4, -4]:
        k = np.arange(p)
        sigma = sigma_max - (k / max(p - 1, 1)) * (sigma_max - sigma_min)
    elif mode in [5, -5]:
        r = np.random.rand(max(p - 2, 0))
        sigma = np.zeros(p)
        sigma[0] = sigma_max
        if p > 1:
            sigma[-1] = sigma_min
        if p > 2:
            sigma[1:-1] = sigma_max * np.exp(np.log(sigma_min / sigma_max) * r)

    if mode < 0:
        sigma = np.sort(sigma)
    else:
        sigma = np.sort(sigma)[::-1]

    Sigma = np.zeros((m, n))
    for i in range(p):
        Sigma[i, i] = sigma[i]

    X = np.random.randn(m, m)
    Y = np.random.randn(n, n)
    U, _ = qr(X, mode='economic' if method == 0 else 'full')
    V, _ = qr(Y, mode='economic' if method == 0 else 'full')

    A = U @ Sigma @ V.T
    return sp.csr_matrix(A)

def compute_sparsity(A):
    return A.nnz / (A.shape[0] * A.shape[1])

def hager_higham_one_norm(A, t=2, maxiter=5, tol=1e-3):
    if not sparse.isspmatrix_csc(A):
        A = sparse.csc_matrix(A)
    
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix A must be square")
    
    dtype = A.dtype
    B = np.sign(np.random.randn(n, t)).astype(dtype)
    est = np.array(0.0, dtype=dtype)
    lu = splu(A)
    for _ in range(maxiter):
        X = lu.solve(B)
        if not np.isfinite(X).all():
            return np.array(np.inf, dtype=dtype)
        X_norm1 = np.sum(np.abs(X), axis=0)
        new_est = np.max(X_norm1)
        if new_est > est:
            est = new_est
        if new_est > 0 and abs(new_est - est) / max(new_est, np.array(1e-10, dtype=dtype)) < tol:
            break
        X_norm = X_norm1.reshape(-1, 1)
        X_norm = np.where(X_norm == 0, np.array(1e-10, dtype=dtype), X_norm)
        X = X / X_norm
        B = A @ X
    a_norm = sparse.linalg.norm(A, 1)
    return a_norm * est if np.isfinite(est) else np.array(np.inf, dtype=dtype)

def estimate_condition_number_approx(A):
    return hager_higham_one_norm(A)

def estimate_condition_number(A):
    try:
        cond_number = np.linalg.cond(A.toarray() if sp.issparse(A) else A)
        return cond_number if np.isfinite(cond_number) else float('inf')
    except Exception:
        return float('inf')

def check_singularity(A):
    if sp.issparse(A):
        if np.any(A.diagonal() == 0):
            return True
        try:
            rank = np.linalg.matrix_rank(A.toarray())
            return rank < A.shape[0]
        except Exception:
            return True
    else:
        if np.any(np.diag(A) == 0):
            return True
        try:
            rank = np.linalg.matrix_rank(A)
            return rank < A.shape[0]
        except Exception:
            return True

def generate_random_linear_system(n=500, kappa=1000, random_state=42):
    np.random.seed(random_state)
    random.seed(random_state)
    A = gallery_randsvd(n, kappa=kappa, mode=2, random_state=random_state)
    x_true = np.random.randn(n)
    b = A @ x_true
    cond_number = estimate_condition_number(A)
    metadata = {
        'condition_number': cond_number,
        'matrix_size': n,
        'kappa': kappa,
        'sparsity': compute_sparsity(A)
    }
    return A, b, x_true, metadata

def generate_datasets(num_train, num_test, size_range=(800, 1500), kappa_range=(3, 12), random_state=42, verbose=True):
    np.random.seed(random_state)
    random.seed(random_state)
    train_set, test_set = [], []
    np.random.seed(random_state)
    random.seed(random_state)
    
    for i in range(num_train + num_test):
        n = np.random.randint(size_range[0], size_range[1] + 1)
        kappa = 10**np.random.randint(kappa_range[0], kappa_range[1])
        print("kappa:", kappa, kappa_range)
        try:
            A, b, x_true, metadata = generate_random_linear_system(
                n=n, kappa=kappa, random_state=random_state + i
            )
        except Exception as e:
            print(f"Error generating system {i}: {e}")
            continue
        if check_singularity(A):
            print(f"System {i}: Singular or zero-diagonal matrix detected, skipping")
            continue
        if verbose:
            print(f"System {i}: matrix_size {n}x{n}, sparsity {metadata['sparsity']:.3f}, cond {metadata['condition_number']:.3e}, A_norm {spla.norm(A, ord=float('inf')):.3e}, b_norm {np.linalg.norm(b, ord=2):.3e}")
        if len(train_set) < num_train:
            train_set.append((A, b, x_true, metadata))
        else:
            test_set.append((A, b, x_true, metadata))

    return train_set, test_set


def compute_diagonal_dominance(A):
    A_csr = A.tocsr()
    diag = np.abs(A_csr.diagonal())
    off_diag_sums = np.array([np.sum(np.abs(A_csr.getrow(i).toarray())) - np.abs(A_csr[i,i]) for i in range(A.shape[0])])
    dominance = np.min(diag / (off_diag_sums + 1e-10))
    return dominance if np.isfinite(dominance) else 0.0

def scipy_to_torch_sparse(A):
    A_coo = A.tocoo()
    indices = torch.LongTensor(np.vstack((A_coo.row, A_coo.col)))
    values = torch.DoubleTensor(A_coo.data)
    shape = A_coo.shape
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float64, device=device).coalesce()

def calculate_unit_roundoff(configs):
    results = {}
    for name, config in configs.items():
        p = config['sig_bits'] + 1
        unit_roundoff = 2 ** (-p)
        results[name] = unit_roundoff
    return results

def compute_unit_roundoff(precision):
    sig_bits = precision_configs[precision]['sig_bits']
    t = sig_bits + 1
    return 2**(1 - t) / 2

reference_config = 'fp64'
reference_bits = precision_configs[reference_config]['sig_bits']
precision_types_errors = {}
for config_name, config in precision_configs.items():
    sig_bits = config['sig_bits']
    precision_types_errors[config_name] = reference_bits / sig_bits
    print(f"Precision {config_name}: sig_bits={sig_bits}, roundoff errors={precision_types_errors[config_name]:.4f}")

def to_float64(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(torch.float64)
    elif isinstance(x, torch.Tensor):
        return x.to(torch.float64)
    else:
        raise TypeError("Input must be a NumPy array or a Torch tensor")

def chop(x, precision, phase="test"):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float64, device=device)

    if precision == 'fp64':
        return to_float64(x)
    
    ch = Chop(**precision_configs[precision])

    # 处理缩放逻辑
    if precision in ['e5m2', 'e4m3', 'fp16']:
        scale = torch.max(torch.abs(x)) # 加上 abs 更稳健
        
        # 准备输入
        if scale != 0:
            op_input = x / scale
        else:
            op_input = x
            
        # 执行截断
        result = ch(op_input)
        
        # --- 关键修复：确保 result 回到 Tensor 进行后续计算 ---
        if isinstance(result, np.ndarray):
            result = torch.from_numpy(result).to(device)
            
        if scale != 0:
            result = result * scale
    else:
        result = ch(x)
        # --- 关键修复 ---
        if isinstance(result, np.ndarray):
            result = torch.from_numpy(result).to(device)

    if not torch.any(torch.isnan(result)) and not torch.any(torch.isinf(result)):
        return result.to(torch.float64).to(device)
    
    logging.debug(f"Chop: Precision {precision} produced NaN/inf in {phase}, returning None")
    return None

def rounding(x, precision, phase="test"):
    return chop(x, precision, phase=phase)

def mixed_precision_op(op, x, precision, y=None, phase="test"):
    if precision == "fp64":
        if y is None:
            return op(x.to(device))
        else:
            return op(x.to(device), y.to(device))
    
    x_rounded = rounding(x, precision, phase)
    if x_rounded is None:
        logging.debug(f"mixed_precision_op: x_rounded is None for precision {precision} in {phase}")
        return None
    
    if y is None:
        unrounded = op(x_rounded)
    else:
        y_rounded = rounding(y, precision, phase)
        if y_rounded is None:
            logging.debug(f"mixed_precision_op: y_rounded is None for precision {precision} in {phase}")
            return None
        unrounded = op(x_rounded, y_rounded)
    
    if unrounded is None:
        logging.debug(f"mixed_precision_op: Operation result is None for precision {precision} in {phase}")
        return None
    
    unrounded = unrounded.clone().detach()
    result = rounding(unrounded, precision, phase)
    if result is None:
        logging.debug(f"mixed_precision_op: Final rounding produced None for precision {precision} in {phase}")
    return result.to(device) if result is not None else None

def round_sparse_matrix(A, precision, phase="test"):
    if precision == 'fp64':
        return A
    A_coo = A.tocoo()
    # 保持输入为 Tensor，pychop 通常能处理 Tensor 输入即使后端是 numpy
    data = torch.tensor(A_coo.data, dtype=torch.float64, device=device)
    
    ch = Chop(**precision_configs[precision])
    try:
        rounded_data = ch(data)
        
        # --- 修复开始: 兼容 Numpy 和 Tensor 返回值 ---
        if isinstance(rounded_data, np.ndarray):
            # 如果 backend('numpy')，返回的是 numpy 数组
            nan_count = np.sum(np.isnan(rounded_data))
            inf_count = np.sum(np.isinf(rounded_data))
            final_data_np = rounded_data # 已经是 numpy，无需转换
        else:
            # 如果 backend('torch')，返回的是 tensor
            nan_count = torch.sum(torch.isnan(rounded_data)).item()
            inf_count = torch.sum(torch.isinf(rounded_data)).item()
            final_data_np = rounded_data.cpu().numpy()
        # --- 修复结束 ---

        if nan_count > 0 or inf_count > 0:
            logging.debug(f"round_sparse_matrix: Precision {precision} produced {nan_count} NaN and {inf_count} inf values in {phase}, returning None")
            return None
        
        return csc_matrix((final_data_np, (A_coo.row, A_coo.col)), shape=A.shape)
    except Exception as e:
        logging.debug(f"round_sparse_matrix: Exception in {phase}: {type(e).__name__}: {str(e)}, returning None")
        return None

def round_sparse_tensor(sparse_tensor, precision, phase="test"):
    if precision == 'fp64':
        return sparse_tensor
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    rounded_values = rounding(values, precision, phase=phase)
    if rounded_values is None:
        logging.debug(f"round_sparse_tensor: Rounding values failed for precision {precision} in {phase}, returning None")
        return None
    return torch.sparse_coo_tensor(
        indices,
        rounded_values,
        sparse_tensor.shape,
        dtype=rounded_values.dtype,
        device=rounded_values.device
    ).coalesce()

class IterationCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1

class BanditAgent:
    def __init__(self, precisions=['e5m2', 'e4m3', 'bf16', 'fp16', 'tf32', 'fp32', 'fp64'], 
                 c1=2.5, c2=1, top_k=20, train_data=None, seed=42):
        self.precisions = precisions
        self.precision_to_index = {p: i for i, p in enumerate(self.precisions)}
        # Generate valid combinations respecting uf <= u <= up <= ur
        self.valid_combinations = []
        for uf in self.precisions:
            for u in self.precisions:
                if precision_order.index(u) < precision_order.index(uf):
                    continue
                for up in self.precisions:
                    if precision_order.index(up) < precision_order.index(u):
                        continue
                    for ur in self.precisions:
                        if precision_order.index(ur) < precision_order.index(up):
                            continue
                        self.valid_combinations.append([uf, u, up, ur])
        
        self.valid_combinations = self.valid_combinations[-top_k:][::-1]

        self.cond_bins = 10
        self.norm_bins = 10
        self.state_space_size = self.cond_bins * self.norm_bins
        self.q_tables = [np.ones((self.state_space_size, len(self.valid_combinations))) * -.0]
        self.epsilon = 1.0
        self.alpha = 0.5
        # self.gamma = 0.9
        self.c1 = c1
        self.c2 = c2
        self.training_losses = []
        self.rng = random.Random(seed)

        # Compute dynamic bin ranges based on training data
        self.cond_min, self.cond_max = 0, 10  # Default log10 ranges
        self.norm_min, self.norm_max = -2, 2
        if train_data is not None:
            cond_numbers = []
            a_norms = []
            for A, _, _, metadata in train_data:
                cond_numbers.append(np.log10(max(estimate_condition_number(A), 1)))
                a_norms.append(np.log10(max(spla.norm(A, ord=float('inf')), 1)))
            self.cond_min = min(cond_numbers) if cond_numbers else 0
            self.cond_max = max(cond_numbers) if cond_numbers else 10
            self.norm_min = min(a_norms) if a_norms else -2
            self.norm_max = max(a_norms) if a_norms else 2
            logging.info(f"Computed bin ranges: cond [{self.cond_min:.2f}, {self.cond_max:.2f}], A_norm [{self.norm_min:.2f}, {self.norm_max:.2f}]")

    def discretize_state(self, state):
        condition_number, A_norm = state
        # Dynamic binning based on computed ranges
        cond_log = np.log10(max(condition_number, 1))
        norm_log = np.log10(max(A_norm, 1))
        cond_range = self.cond_max - self.cond_min if self.cond_max > self.cond_min else 1
        norm_range = self.norm_max - self.norm_min if self.norm_max > self.norm_min else 1
        cond_idx = np.clip(int((cond_log - self.cond_min) / cond_range * (self.cond_bins - 1)), 0, self.cond_bins - 1)
        norm_idx = np.clip(int((norm_log - self.norm_min) / norm_range * (self.norm_bins - 1)), 0, self.norm_bins - 1)
        return cond_idx * self.norm_bins + norm_idx

    def choose_action(self, state, phase="train", step=0, total_steps=1000):
        # --- 改动 1: 基于 Step 的线性衰减 ---
        # 这样每处理一个矩阵，epsilon 都会微小下降
        epsilon = max(0.1, 1.0 - step / total_steps)
        
        # 使用这一行来替代原来的 episode 逻辑
        # epsilon = max(0.1, 1.0 - episode / total_episodes) 

        # 使用独立的随机数生成器
        random_num = self.rng.random()
        
        if phase == "train" and random_num < epsilon:
            # --- 改动 2: 再次强调，绝对删掉这里的 random.seed() ---
            # random.seed(int(episode)+1000)  <-- DELETE THIS LINE
            
            logging.info(f"Exploration (epsilon={epsilon:.4f}): Choosing random action")
            action_idx = self.rng.randint(0, len(self.valid_combinations) - 1)
            precisions = self.valid_combinations[action_idx]
            logging.info(f"Selected action: {precisions}")
            return precisions, action_idx
        else:
            state_idx = self.discretize_state(state)
            action_idx = np.argmax(self.q_tables[0][state_idx])
            precisions = self.valid_combinations[action_idx]
            logging.info(f"Exploitation (epsilon={epsilon:.4f}): Choosing action {precisions} for state {state_idx}")
            return precisions, action_idx

    def update_model(self, state, action, reward, next_state):
        state_idx = self.discretize_state(state)
        # next_state_idx = self.discretize_state(next_state)
        action_idx = action
        current_q = self.q_tables[0][state_idx, action_idx]
        # next_max_q = np.max(self.q_tables[0][next_state_idx])
        new_q = current_q + self.alpha * (reward - current_q) 
        # new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_tables[0][state_idx, action_idx] = new_q

        print(f"Updated Q-value for state {state_idx}, action {action_idx}: {current_q:.4f} -> {new_q:.4f} (reward: {reward:.4f})\n")
        self.training_losses.append(abs(new_q - current_q))

    def get_state(self, frustrum, b):
        condition_number = estimate_condition_number(frustrum)
        A_norm = spla.norm(frustrum, ord=float('inf'))
        return np.array([condition_number, A_norm])

    def print_top_actions(self):
        for state_idx in range(self.state_space_size):
            top_action_idx = np.argmax(self.q_tables[0][state_idx])
            top_action = self.valid_combinations[top_action_idx]
            top_q = self.q_tables[0][state_idx, top_action_idx]
            logging.info(f"State {state_idx}: Top action {top_action}, Q-value {top_q}")

class LinearSystemEnv:
    def __init__(self, solver, train_data, max_iter=10, tol=1e-6, c1=2.5, c2=1):
        self.solver = solver
        self.train_data = train_data
        self.current_matrix_idx = 0
        self.max_iter = max_iter
        self.tol = tol
        self.c1 = c1
        self.c2 = c2
        self.failure_penalty = 5.0

    def reset(self):
        A, b, x_true, metadata = self.train_data[self.current_matrix_idx]
        A_torch = scipy_to_torch_sparse(A).to(dtype=torch.float64, device=self.solver.device).coalesce()
        b_torch = torch.tensor(b, dtype=torch.float64, device=self.solver.device).view(-1, 1)
        x_true_torch = torch.tensor(x_true, dtype=torch.float64, device=self.solver.device).view(-1, 1)
        self.current_A = A
        self.current_b = b
        self.current_x_true = x_true
        self.current_A_torch = A_torch
        self.current_b_torch = b_torch
        self.current_x_true_torch = x_true_torch
        self.metadata = metadata
        state = self.solver.agent.get_state(self.current_A, self.current_b)
        info = {'matrix_id': self.current_matrix_idx, 'matrix_size': A.shape[0]}
        self.current_matrix_idx = (self.current_matrix_idx + 1) % len(self.train_data)
        return state, info

    def compute_reward(self, x_rl, x_fp64, final_ferr, final_rho, final_nbe, final_cbe, precisions, gmres_iterations_sum, lu_failed, init_failed, fp64_rho, A_norm, b_norm):
        x_fp64_norm = torch.norm(x_fp64, p=float('inf')) + 1e-10
        ferr_distance = torch.norm(x_rl - x_fp64, p=float('inf')) / x_fp64_norm
        norm_factor = A_norm * x_fp64_norm + b_norm + 1e-10
        ferr_normalized = final_ferr / norm_factor
        rho_normalized = final_rho / (b_norm + 1e-10)

        if ferr_distance > 1.0 or ferr_normalized > 1.0:
            accuracy_reward = -self.c1 * 5.0
        else:
            accuracy_reward = -self.c1 * (np.log10(max(ferr_distance, 1e-10)) + np.log10(max(ferr_normalized, 1e-10)))

        convergence_bonus = 2.0 if rho_normalized <= self.tol or abs(rho_normalized - fp64_rho) / (fp64_rho + 1e-10) < 0.1 else 0.0

        cond_number = self.metadata.get('condition_number', 1e6)
        precision_weights = {
            'e5m2': precision_types_errors['e5m2'] / (1 + np.log10(max(cond_number, 1))),
            'e4m3': precision_types_errors['e4m3'] / (1 + np.log10(max(cond_number, 1))),
            'bf16': precision_types_errors['bf16'] / (1 + np.log10(max(cond_number, 1))),
            'fp16': precision_types_errors['fp16'] / (1 + np.log10(max(cond_number, 1))),
            'tf32': precision_types_errors['tf32'] / (1 + np.log10(max(cond_number, 1))),
            'fp32': precision_types_errors['fp32'] / (1 + np.log10(max(cond_number, 1))),
            'fp64': precision_types_errors['fp64'] / (1 + np.log10(max(cond_number, 1)))
        }
        precision_reward = self.c2 * sum(precision_weights[p] for p in precisions) 

        iteration_penalty = np.log2(max(gmres_iterations_sum, 1)) if gmres_iterations_sum > 0 else 5.0

        reward = accuracy_reward + precision_reward 
        logging.info(f"Reward components: accuracy={accuracy_reward:.4f}, precision={precision_reward:.4f}, convergence_bonus={convergence_bonus:.4f}, iteration_penalty={iteration_penalty:.4f}, total={reward:.4f}")
        return reward

    def step(self, action):
        precisions = self.solver.agent.valid_combinations[action]
        x_rl, iters, gmres_iterations, errors, precision_log, final_nbe, final_ferr, final_rho, final_cbe, converged, stagnation, converged_gmres, lu_failed, init_failed = \
            self.solver.ir5_gmres_mixed_precision(self.current_A_torch, self.current_b_torch, self.current_x_true_torch, 
                                                 matrix_id=self.current_matrix_idx, phase="train", precisions=precisions)
        _, _, _, _, final_nbe_fp64, final_ferr_fp64, final_rho_fp64, final_cbe_fp64, _, _, _, _, _  = \
            self.solver.ir5_gmres_fp64(self.current_A, self.current_b_torch, self.current_x_true_torch, matrix_id=self.current_matrix_idx)
        
        A_norm = spla.norm(self.current_A, ord=float('inf'))
        b_norm = torch.norm(self.current_b_torch, p=2).item()
        reward = self.compute_reward(x_rl, self.current_x_true_torch, final_ferr, final_rho, final_nbe, final_cbe, 
                                    precisions, sum(gmres_iterations), lu_failed, init_failed, 
                                    final_rho_fp64, A_norm, b_norm)
        next_state = self.solver.agent.get_state(self.current_A, self.current_b)
        terminated = converged or iters >= self.max_iter
        truncated = not converged_gmres or lu_failed or init_failed
        info = {
            'matrix_id': self.current_matrix_idx,
            'matrix_size': self.current_A.shape[0],
            'converged': converged,
            'iterations': iters,
            'gmres_iterations': sum(gmres_iterations),
            'final_nbe': final_nbe,
            'final_ferr': final_ferr,
            'final_rho': final_rho,
            'final_cbe': final_cbe,
            'precisions': precisions
        }
        return (next_state, info), reward, terminated, truncated, info, precisions

class LOW_PREC_LU:
    def __init__(self, lu, precision, phase, device):
        self.precision = precision
        self.phase = phase
        self.device = device
        L = lu.L
        L.data = rounding(torch.tensor(L.data, dtype=torch.float64, device=device), precision, phase).cpu().numpy()
        U = lu.U
        U.data = rounding(torch.tensor(U.data, dtype=torch.float64, device=device), precision, phase).cpu().numpy()
        self.L = L.tocsr()
        self.U = U.tocsr()
        self.perm_r = lu.perm_r
        self.perm_c = lu.perm_c

    def solve(self, b):
        b = rounding(b, self.precision, self.phase)
        if b is None:
            return None
        b = b.cpu().numpy()
        b_perm = np.empty_like(b)
        b_perm[self.perm_r] = b
        y = spla.spsolve_triangular(self.L, b_perm, lower=True, unit_diagonal=True)
        z = spla.spsolve_triangular(self.U, y, lower=False)
        x_solve = z[self.perm_c]
        return x_solve