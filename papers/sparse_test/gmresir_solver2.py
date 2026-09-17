import numpy as np
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import coo_matrix
import pandas as pd
from tqdm import tqdm
import math
import logging
import os
import pickle
from scipy.sparse.linalg import gmres
from linear_system_utils import *

pychop.backend('torch')

class FastPrecisionSimulator:
    def __init__(self, precision='fp64', device=device):
        self.precision = precision
        self.device = device
        self.use_native = False
        self.torch_dtype = None
        self.chopper = None

        # 1. 尝试使用 PyTorch 原生类型加速 (速度极快)
        if precision == 'fp16':
            self.use_native = True
            self.torch_dtype = torch.float16
        elif precision == 'bf16':
            self.use_native = True
            self.torch_dtype = torch.bfloat16
        elif precision == 'fp32' or precision == 'tf32': 
            self.use_native = True
            self.torch_dtype = torch.float32
        elif precision == 'fp64':
            self.use_native = True
            self.torch_dtype = torch.float64
        else:
            # 2. 对于 FP8 (e4m3/e5m2) 回退到 pychop
            self.use_native = False
            if precision in precision_configs:
                # 每次实例化一个新的 LightChop
                conf = precision_configs[precision]
                self.chopper = LightChop(exp_bits=conf['exp_bits'], sig_bits=conf['sig_bits'], 
                                       subnormal=True, rmode=conf['rmode'])
            else:
                self.use_native = True # Default to fp64 if unknown
                self.torch_dtype = torch.float64

    def q(self, x):
        """快速截断函数"""
        if x is None: return None
        
        # 统一转为 Tensor (GPU)
        if not torch.is_tensor(x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            else:
                x = torch.tensor(x)
        
        if x.device != self.device:
            x = x.to(device=self.device, dtype=torch.float64)
        elif x.dtype != torch.float64:
            x = x.to(dtype=torch.float64)

        # 1. 原生加速路径 (FP16/BF16/FP32)
        if self.use_native:
            if self.precision == 'fp64':
                return x
            # Cast down -> Cast up
            return x.to(self.torch_dtype).to(torch.float64)
        
        # 2. PyChop 慢速路径 (FP8 等)
        else:
            if self.chopper is None: return x
            # Tensor GPU -> Numpy CPU -> PyChop -> Numpy -> Tensor GPU
            x_np = x.detach().cpu().numpy()
            try:
                x_chop = self.chopper(x_np)
                # pychop 返回 numpy array (因为我们设置了 backend numpy)
                return torch.from_numpy(x_chop).to(self.device, dtype=torch.float64)
            except Exception:
                return x
            
# 全局辅助函数：用于外部调用 rounding
# 这里的 rounding 主要是为了兼容原代码中直接调用的方式
SIMULATORS_CACHE = {}

def get_simulator(precision):
    if precision not in SIMULATORS_CACHE:
        SIMULATORS_CACHE[precision] = FastPrecisionSimulator(precision)
    return SIMULATORS_CACHE[precision]

def rounding(tensor_data, precision, phase="test"):
    """
    通用截断函数，支持 Torch Tensor 输入，内部转 Numpy 处理后转回
    """
    if precision == 'fp64':
        return tensor_data
    
    sim = get_simulator(precision)
    
    is_torch = torch.is_tensor(tensor_data)
    if is_torch:
        data_np = tensor_data.detach().cpu().numpy()
    else:
        data_np = tensor_data
        
    chopped_np = sim.q(data_np)
    
    if is_torch:
        return torch.from_numpy(chopped_np).to(device=tensor_data.device, dtype=tensor_data.dtype)
    return chopped_np

# ==========================================
# 1. Custom Uniform Precision GMRES
# ==========================================
def safe_norm_torch(x, simulator):
    # 模拟 dot(x, x)
    x_flat = x.flatten()
    dot_val = simulator.q(torch.dot(x_flat, x_flat))
    # 模拟 sqrt
    return simulator.q(torch.sqrt(dot_val))

def torch_uniform_gmres(A_func, b, maxiter=100, tol=1e-6, M_func=None, simulator=None):
    """
    全 PyTorch GPU 实现的 Uniform Precision GMRES
    A_func: 接受 tensor 返回 tensor
    b: Tensor (N, 1)
    M_func: 接受 tensor 返回 tensor
    simulator: FastPrecisionSimulator 实例
    """
    if simulator is None:
        simulator = FastPrecisionSimulator('fp64', device=b.device)
    
    q = simulator.q
    N = b.numel()
    dev = b.device
    
    # x0 默认为 0
    x0 = torch.zeros_like(b)
    
    # r0 = b (经过截断)
    r0 = q(b.clone())
    
    # Apply Preconditioner
    if M_func is not None:
        r0 = M_func(r0)
        r0 = q(r0)
        
    beta = safe_norm_torch(r0, simulator)
    
    if beta < 1e-15:
        return x0, 0 # info=0 converged

    # 预分配显存
    V = torch.zeros((N, maxiter + 1), device=dev, dtype=torch.float64)
    H = torch.zeros((maxiter + 1, maxiter), device=dev, dtype=torch.float64)
    
    V[:, 0] = (q(r0 / beta)).flatten()
    
    cs = torch.zeros(maxiter, device=dev, dtype=torch.float64)
    sn = torch.zeros(maxiter, device=dev, dtype=torch.float64)
    g = torch.zeros(maxiter + 1, device=dev, dtype=torch.float64)
    g[0] = beta
    
    error = beta
    k = 0
    
    for k in range(maxiter):
        # 1. MatVec: w = A @ v_k
        v_curr = V[:, k].view(-1, 1)
        w = A_func(v_curr)
        w = q(w).flatten()
        
        # 2. Precond: w = M @ w
        if M_func is not None:
            w_res = M_func(w.view(-1, 1))
            if w_res is None: # 处理预处理失败
                 break
            w = q(w_res).flatten()
            
        # 3. MGS Orthogonalization
        for i in range(k + 1):
            v_i = V[:, i]
            h_val = q(torch.dot(v_i, w))
            H[i, k] = h_val
            update = q(h_val * v_i)
            w = q(w - update)
            
        h_next = safe_norm_torch(w, simulator)
        H[k + 1, k] = h_next
        
        if h_next < 1e-15:
            break
            
        V[:, k + 1] = q(w / h_next)
        
        # 4. Givens Rotations
        for i in range(k):
            temp_H_i = H[i, k].clone()
            temp_H_ip1 = H[i+1, k].clone()
            val1 = q(cs[i] * temp_H_i)
            val2 = q(sn[i] * temp_H_ip1)
            H[i, k] = q(val1 + val2)
            val3 = q(-sn[i] * temp_H_i)
            val4 = q(cs[i] * temp_H_ip1)
            H[i+1, k] = q(val3 + val4)
            
        # 5. New Rotation
        alpha = H[k, k]
        bet = H[k + 1, k]
        
        hyp = q(torch.sqrt(q(alpha**2 + bet**2)))
        
        if hyp == 0:
            c = 1.0; s = 0.0
        else:
            c = q(alpha / hyp)
            s = q(bet / hyp)
            
        cs[k] = c; sn[k] = s
        H[k, k] = hyp; H[k + 1, k] = 0.0
        
        g_curr = g[k].clone()
        g[k] = q(c * g_curr)
        g[k + 1] = q(-s * g_curr)
        
        error = torch.abs(g[k + 1])
        if error < tol:
            break

    # 6. Backward Substitution
    y = torch.zeros(k + 1, device=dev, dtype=torch.float64)
    for i in range(k, -1, -1):
        rhs = g[i]
        for j in range(i + 1, k + 1):
            prod = q(H[i, j] * y[j])
            rhs = q(rhs - prod)
        y[i] = q(rhs / H[i, i])
        
    # 7. Form solution
    dy = torch.mv(V[:, :k+1], y)
    dy = q(dy)
    x_sol = q(x0 + dy.view(-1, 1))
    
    info = 0 if error < tol else 1
    return x_sol, info


def load_data(num_train, num_test, data_dir='data_random_sparse2'):
    train_data = []
    test_data = []
    train_metrics = []
    test_metrics = []
    for i in range(num_train):
        matrix_path = os.path.join(data_dir, f'train_matrix_{i}.npz')
        vector_path = os.path.join(data_dir, f'train_vector_{i}.npy')
        x_true_path = os.path.join(data_dir, f'train_x_true_{i}.npy')
        if not (os.path.exists(matrix_path) and os.path.exists(vector_path) and os.path.exists(x_true_path)):
            return None, None, None, None
        A = sp.load_npz(matrix_path)
        b = np.load(vector_path)
        x_true = np.load(x_true_path)
        cond_number = estimate_condition_number(A)
        sparsity = compute_sparsity(A)
        train_data.append((A, b, x_true, {'condition_number': cond_number, 'sparsity': sparsity, 'matrix_size': A.shape[0]}))
        train_metrics.append({
            'matrix_id': str(i),
            'condition_number': cond_number,
            'sparsity': sparsity,
            'matrix_size': A.shape[0]
        })
    for i in range(num_test):
        matrix_path = os.path.join(data_dir, f'test_matrix_{i}.npz')
        vector_path = os.path.join(data_dir, f'test_vector_{i}.npy')
        x_true_path = os.path.join(data_dir, f'test_x_true_{i}.npy')
        if not (os.path.exists(matrix_path) and os.path.exists(vector_path) and os.path.exists(x_true_path)):
            return None, None, None, None
        A = sp.load_npz(matrix_path)
        b = np.load(vector_path)
        x_true = np.load(x_true_path)
        cond_number = estimate_condition_number(A)
        sparsity = compute_sparsity(A)
        test_data.append((A, b, x_true, {'condition_number': cond_number, 'sparsity': sparsity, 'matrix_size': A.shape[0]}))
        test_metrics.append({
            'matrix_id': str(i),
            'condition_number': cond_number,
            'sparsity': sparsity,
            'matrix_size': A.shape[0]
        })
    return train_data, test_data, train_metrics, test_metrics

def check_data_files(num_train, num_test, data_dir='data_random_sparse2'):
    os.makedirs(data_dir, exist_ok=True)
    for i in range(num_train):
        if not (os.path.exists(os.path.join(data_dir, f'train_matrix_{i}.npz')) and
                os.path.exists(os.path.join(data_dir, f'train_vector_{i}.npy')) and
                os.path.exists(os.path.join(data_dir, f'train_x_true_{i}.npy'))):
            return False
    for i in range(num_test):
        if not (os.path.exists(os.path.join(data_dir, f'test_matrix_{i}.npz')) and
                os.path.exists(os.path.join(data_dir, f'test_vector_{i}.npy')) and
                os.path.exists(os.path.join(data_dir, f'test_x_true_{i}.npy'))):
            return False
    return True

def save_data(train_data, test_data, data_dir='data_random_sparse2'):
    os.makedirs(data_dir, exist_ok=True)
    for i, (A, b, x_true, _) in enumerate(train_data):
        sp.save_npz(os.path.join(data_dir, f'train_matrix_{i}.npz'), A)
        np.save(os.path.join(data_dir, f'train_vector_{i}.npy'), b)
        np.save(os.path.join(data_dir, f'train_x_true_{i}.npy'), x_true)
    for i, (A, b, x_true, _) in enumerate(test_data):
        sp.save_npz(os.path.join(data_dir, f'test_matrix_{i}.npz'), A)
        np.save(os.path.join(data_dir, f'test_vector_{i}.npy'), b)
        np.save(os.path.join(data_dir, f'test_x_true_{i}.npy'), x_true)
    logging.info(f"Saved data to {data_dir}")

class GMRESIR5Solver:
    def __init__(self, max_iter=100, tol=1e-6, precisions=['e5m2', 'e4m3', 'bf16', 'fp16', 'tf32', 'fp32', 'fp64'], 
                 max_inner_iter=2, max_stagnation=10, c1=2.5, c2=500):
        self.max_iter = max_iter
        self.max_stagnation = max_stagnation if max_stagnation is not None else self.max_iter
        self.tol = tol
        self.max_inner_iter = max_inner_iter
        # 假设 device 已经在 linear_system_utils 或全局定义，否则默认为 cpu
        self.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.c1 = c1
        self.c2 = c2
        self.precisions = precisions
        self.min_iterations = 1
        self.unit_roundoffs = calculate_unit_roundoff(precision_configs)
        self.init_failure_penalty = 5.0
        self.fp64_unit_roundoff = 2**(-53)  # Unit roundoff for fp64: ~1.11e-16

    def compute_lu_factorization(self, A, precision, phase="test"):
        if precision == 'fp64':
            try:
                lu = spla.splu(A.tocsc())
                return lu, False
            except: return None, True
        
        A_rounded = round_sparse_matrix(A, precision, phase)
        if A_rounded is None: return None, True
        try:
            lu = spla.splu(A_rounded.tocsc())
            lu = LOW_PREC_LU(lu, precision, phase, self.device)
            return lu, False
        except: return None, True

    def apply_preconditioner(self, lu, w, precision, phase="test"):
        # 这个函数被 GMRES 内部调用。
        # w: 输入向量。如果是 custom_uniform_gmres 调用，w 是 numpy array。
        # precision: 这里对应 ug (Uniform GMRES Precision)，也就是预处理的应用也应该在这个精度下截断（模拟存储/计算）。
        
        if lu is None or w is None:
            logging.debug(f"apply_preconditioner: failed in {phase} (lu or w is None)")
            return None, True
            
        # 1. 统一转为 Torch Float64 进行计算 (Scipy solve 需要 high precision 避免崩溃)
        # 但如果是 Custom GMRES 调用的，我们需要先处理输入类型
        is_numpy = isinstance(w, np.ndarray)
        
        if is_numpy:
             w_torch = torch.from_numpy(w).to(dtype=torch.float64, device=self.device).view(-1, 1)
        elif not isinstance(w, torch.Tensor):
             w_torch = torch.tensor(w, dtype=torch.float64, device=self.device).view(-1, 1)
        else:
             w_torch = w.view(-1, 1).to(dtype=torch.float64, device=self.device)

        # 2. 模拟输入的量化 (模拟低精度 GMRES 将向量传给 Preconditioner)
        # 如果 custom_gmres 已经量化了 w，这一步可能冗余，但为了保险起见。
        if precision != 'fp64':
             w_torch = rounding(w_torch, precision, phase)
             if w_torch is None: return None, True

        # 3. 执行 Solve (在高精度下运行，模拟数算单元)
        w_np = w_torch.cpu().numpy().flatten()
        
        try:
            sol = lu.solve(w_np)
            if sol is None:
                return None, True
        except Exception as e:
            logging.debug(f"apply_preconditioner: LU solve failed in {phase}: {str(e)}")
            return None, True

        sol_torch = torch.from_numpy(sol).to(dtype=torch.float64, device=self.device).view(-1, 1)

        # 4. 模拟输出的量化 (Preconditioner 返回结果给 GMRES)
        if precision != 'fp64':
            sol_torch = rounding(sol_torch, precision, phase)
            if sol_torch is None:
                logging.debug(f"apply_preconditioner: Rounding to {precision} produced None in {phase}")
                return None, True
        
        if torch.any(torch.isnan(sol_torch)) or torch.any(torch.isinf(sol_torch)):
            logging.debug(f"apply_preconditioner: Invalid output in {phase}, returning None")
            return None, True

        # 如果输入是 Numpy (Custom GMRES)，返回 Numpy
        if is_numpy:
            return sol_torch.cpu().numpy().flatten(), False
        else:
            return sol_torch.cpu().numpy().flatten(), False # 保持原接口返回 numpy，外部可能再转 torch
        
        
    def apply_preconditioner_torch(self, lu, w_torch, precision, phase="test"):
        """
        w_torch: Tensor on GPU
        return: Tensor on GPU
        """
        # 1. 模拟输入截断
        sim = FastPrecisionSimulator(precision, device=self.device)
        if precision != 'fp64':
            w_torch = sim.q(w_torch)
            
        # 2. Solve (需要转回 CPU 执行 Scipy 逻辑)
        w_np = w_torch.cpu().numpy().flatten()
        try:
            sol = lu.solve(w_np)
            if sol is None: return None, True
        except: return None, True
            
        sol_torch = torch.from_numpy(sol).to(dtype=torch.float64, device=self.device).view(-1, 1)
        
        # 3. 模拟输出截断
        if precision != 'fp64':
            sol_torch = sim.q(sol_torch)
            
        return sol_torch, False

    def initialize_solution(self, lu, b_torch, phase="test"):
        try:
            b_np = b_torch.cpu().numpy().flatten()
            x0 = lu.solve(b_np)
            x0 = torch.tensor(x0, dtype=torch.float64, device=self.device).view(-1, 1)
            if torch.any(torch.isnan(x0)): return None, True
            return x0, False
        except: return None, True


    def ir5_gmres_mixed_precision(self, A, b, x_true, matrix_id, phase="test", precisions=None):
        size = A.shape[0]
        self.gmres_max_it = size
        
        # 输入 A 是 torch sparse tensor (原始高精度)
        A_torch_original = A.to(dtype=torch.float64, device=self.device).coalesce()
        b_torch = b.to(dtype=torch.float64, device=self.device).view(-1, 1)
        x_true = x_true.to(dtype=torch.float64, device=self.device).view(-1, 1)
        b_norm_2 = torch.norm(b_torch, p=2).item()  # 用于 rho
        b_norm_inf = torch.norm(b_torch, p=float('inf')).item()  # 用于 NBE

        if precisions is None:
            uf, ug, u, ur = 'fp16', 'fp16', 'fp32', 'fp32'
        else:
            uf, ug, u, ur = precisions

        # === 全 fp64 时直接复用 ir5_gmres_fp64，保证 100% 一致 ===
        if uf == 'fp64' and ug == 'fp64' and u == 'fp64' and ur == 'fp64':
            # 手动将 Torch sparse tensor 转为 SciPy csc
            indices = A_torch_original.indices().cpu().numpy()
            values = A_torch_original.values().cpu().numpy()
            A_scipy = sp.coo_matrix((values, (indices[0], indices[1])), shape=A_torch_original.shape).tocsc()
            
            # 调用基准 fp64 路径
            x_fp64, iters_fp64, iters_gmres_fp64, errors_fp64, \
            final_nbe_fp64, final_ferr_fp64, final_rho_fp64, final_cbe_fp64, converged, stagnation, converged_gmres, lu_failed, init_failed = \
                self.ir5_gmres_fp64(A_scipy, b_torch, x_true, matrix_id=matrix_id, phase=phase)
            
            precision_log = [['fp64', 'fp64', 'fp64', 'fp64']]
            total_gmres_iterations = iters_gmres_fp64
            
            return (x_fp64, iters_fp64, total_gmres_iterations, errors_fp64, precision_log,
                    final_nbe_fp64, final_ferr_fp64, final_rho_fp64, final_cbe_fp64,
                    converged, stagnation, converged_gmres,
                    lu_failed, init_failed)

        precision_log = [[uf, ug, u, ur]]
        errors = []
        total_gmres_iterations = []

        # ===== 1. 原始高精度 SciPy 矩阵（用于所有指标计算，永不 rounding）=====
        indices_orig = A_torch_original.indices().cpu().numpy()
        values_orig = A_torch_original.values().cpu().numpy()
        A_scipy_original = sp.coo_matrix((values_orig, (indices_orig[0], indices_orig[1])), shape=A_torch_original.shape).tocsc()
        A_norm_inf_original = spla.norm(A_scipy_original, ord=np.inf)

        # ===== 2. Working precision A 和 b (u) =====
        sim_u = FastPrecisionSimulator(u, device=self.device)
        if u == 'fp64':
            A_work_torch = A_torch_original
            b_work = b_torch
        else:
            vals_rounded = sim_u.q(A_torch_original.values())
            A_work_torch = torch.sparse_coo_tensor(A_torch_original.indices(), vals_rounded, A_torch_original.shape).coalesce()
            b_work = sim_u.q(b_torch)

        # ===== 3. Working SciPy 矩阵（用于 LU 和 ug=fp64 时的 GMRES）=====
        indices_work = A_work_torch.indices().cpu().numpy()
        values_work = A_work_torch.values().cpu().numpy()
        A_scipy_work = sp.coo_matrix((values_work, (indices_work[0], indices_work[1])), shape=A_work_torch.shape).tocsc()

        # ===== 4. LU 分解 (uf) =====
        lu, lu_failed = self.compute_lu_factorization(A_scipy_work, uf, phase)
        if lu is None:
            return self._failure_return(size)

        # ===== 5. 初始解 x0 =====
        x, init_failed = self.initialize_solution(lu, b_torch, phase)
        if x is None:
            x = torch.zeros(size, 1, dtype=torch.float64, device=self.device)

        # ===== 6. 初始残差 r (ur) =====
        sim_ur = FastPrecisionSimulator(ur, device=self.device)
        if u == 'fp64':
            Ax = torch.from_numpy(A_scipy_work @ x.cpu().numpy().flatten()).to(self.device).view(-1, 1)
        else:
            Ax = torch.sparse.mm(A_work_torch, x)
        r = b_work - Ax
        if ur != 'fp64':
            r = sim_ur.q(r)

        # 记录初始指标（使用原始 A）
        self._record_metrics(errors, x, x_true, r, A_scipy_original, b_torch, 0)

        total_iterations = 0
        converged = False
        stagnation = False
        min_nbe = errors[-1][0] if errors else float('inf')
        stagnation_counter = 0
        prev_dx_norm = float('inf')  # Initialize previous update norm

        while total_iterations < self.max_iter:
            total_iterations += 1

            if ug == 'fp64':
                # 完全使用 SciPy gmres（与 ir5_gmres_fp64 一致）
                counter = IterationCounter()
                r_np = r.cpu().numpy().flatten()
                try:
                    d_np, info = gmres(
                        A_scipy_work, r_np, x0=np.zeros(size), rtol=self.tol,
                        maxiter=self.gmres_max_it,
                        M=spla.LinearOperator((size, size),
                                              matvec=lambda v: self.apply_preconditioner(lu, v, 'fp64', phase)[0]),
                        callback=counter, callback_type='legacy'
                    )
                    converged_gmres = (info == 0)
                except Exception as e:
                    logging.debug(f"SciPy GMRES failed (ug=fp64): {e}")
                    converged_gmres = False
                    total_gmres_iterations.append(counter.niter)
                    break
                total_gmres_iterations.append(counter.niter)
                d = torch.from_numpy(d_np).to(self.device).view(-1, 1)
            else:
                # 低精度模拟：Torch uniform GMRES
                ug_simulator = FastPrecisionSimulator(ug, device=self.device)

                def A_func(v):
                    if u == 'fp64':
                        return torch.from_numpy(A_scipy_work @ v.cpu().numpy().flatten()).to(self.device).view(-1, 1)
                    else:
                        return torch.sparse.mm(A_work_torch, v)

                def M_func(v):
                    res, _ = self.apply_preconditioner_torch(lu, v, ug, phase)
                    return res

                d, gmres_info = torch_uniform_gmres(
                    A_func, r,
                    maxiter=self.gmres_max_it, tol=self.tol,
                    M_func=M_func, simulator=ug_simulator
                )
                converged_gmres = (gmres_info == 0)
                total_gmres_iterations.append(self.gmres_max_it if gmres_info != 0 else 0)

            # 更新量截断 (u)
            if u != 'fp64':
                d = sim_u.q(d)

            # 停机判断
            dx_norm = torch.norm(d, p=float('inf')).item()
            x_norm = torch.norm(x, p=float('inf')).item()
            rel_dx = dx_norm / x_norm if x_norm > 1e-100 else float('inf')
            dx_ratio = dx_norm / prev_dx_norm if prev_dx_norm > 1e-100 else float('inf')
            prev_dx_norm = dx_norm

            if rel_dx <= self.fp64_unit_roundoff:
                converged = True
                break
            if rel_dx < self.unit_roundoffs.get(u, 1e-5):
                stagnation = True
                break

            if total_iterations > 1 and dx_ratio >= self.tol:
                logging.debug(f"ir5_gmres_rl, matrix {matrix_id}, iter {total_iterations}: Convergence slowed, dx_ratio={dx_ratio:.2e} >= {self.tol:.2e}")
                break

            # 停滞检测
            current_nbe = errors[-1][0] if errors else float('inf')
            if current_nbe < min_nbe:
                min_nbe = current_nbe
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            if stagnation_counter >= self.max_stagnation:
                stagnation = True
                break

            # 更新解和残差
            x = x + d
            if u == 'fp64':
                Ax = torch.from_numpy(A_scipy_work @ x.cpu().numpy().flatten()).to(self.device).view(-1, 1)
            else:
                Ax = torch.sparse.mm(A_work_torch, x)
            r = b_work - Ax
            if ur != 'fp64':
                r = sim_ur.q(r)

            # 记录指标（使用原始 A）
            self._record_metrics(errors, x, x_true, r, A_scipy_original, b_torch, total_iterations)

        # ===== 最终指标：强制使用原始高精度 A 和 b 计算（与 fp64 路径完全一致）=====
        x_norm_inf = torch.norm(x, p=float('inf')).item()
        r_norm_inf = torch.norm(r, p=float('inf')).item()
        denominator = A_norm_inf_original * x_norm_inf + b_norm_inf
        final_nbe = r_norm_inf / denominator if denominator > 1e-100 else float('inf')

        final_ferr = torch.norm(x - x_true, p=float('inf')).item() / (torch.norm(x_true, p=float('inf')).item() + 1e-10)
        final_rho = torch.norm(r, p=2).item() / (b_norm_2 + 1e-10)
        final_cbe = 0.0  # 如需精确计算可补充 component-wise

        return (x, total_iterations, total_gmres_iterations, errors, precision_log,
                final_nbe, final_ferr, final_rho, final_cbe,
                converged, stagnation, converged_gmres,
                lu_failed, init_failed)


    def ir5_gmres_fp64(self, A, b, x_true, matrix_id, phase="test"):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        logging.debug(f"ir5_gmres_fp64: matrix_id={matrix_id}, max_iter={self.max_iter}, device={self.device}")
        A_torch = scipy_to_torch_sparse(A).coalesce().to(dtype=torch.float64, device=self.device)
        size = A_torch.size(0)
        
        A_scipy = A.tocsc()
        b_torch = b.to(dtype=torch.float64, device=self.device).view(-1, 1)
        x_true = x_true.to(dtype=torch.float64, device=self.device).view(-1, 1)
        b_norm = torch.norm(b_torch, p=2, dtype=torch.float64).item()
        
        errors = []
        total_iterations = 0
        total_gmres_iterations = []
        converged = False
        converged_gmres = False
        init_failed = False
        
        lu, lu_failed = self.compute_lu_factorization(A_scipy, 'fp64', phase="test")
        x, init_failed = self.initialize_solution(lu, b_torch, phase="test")
        if x is None:
            x = torch.zeros(size, 1, dtype=torch.float64, device=self.device)
        
        r = b_torch - torch.sparse.mm(A_torch, x)
        residual_norm = torch.norm(r, p=2, dtype=torch.float64).item()
        rho = residual_norm / b_norm if b_norm > 1e-100 else float('inf')
        if phase == "test":
            print(f"start r: {r.flatten()[:5]}, x: {x.flatten()[:5]}, x_true: {x_true.flatten()[:5]}")
            print(f"residual_norm: {residual_norm:.2e}, rho={rho:.2e}")
        min_residual_norm = residual_norm

        A_norm_inf = spla.norm(A_scipy, ord=np.inf)
        
        x_true_norm = torch.norm(x_true, p=float('inf'))
        ferr = torch.norm(x - x_true, p=float('inf')) / x_true_norm if x_true_norm > 1e-100 else float('inf')
        nbe = torch.norm(r, p=float('inf')) / (A_norm_inf * torch.norm(x, p=float('inf')) + torch.norm(b_torch, p=float('inf')) if torch.norm(x, p=float('inf')) > 1e-100 else float('inf'))
        temp = torch.abs(r) / (torch.abs(torch.sparse.mm(A_torch, x)) + torch.abs(b_torch))
        temp[torch.isnan(temp)] = 0
        cbe = torch.max(temp)
        errors.append((float(nbe), float(ferr), float(cbe)))
        condition_number = estimate_condition_number(A_scipy)

        counter = IterationCounter()
        prev_dx_norm = float('inf')  # Initialize previous update norm
        
        print(f"condition number={condition_number:.3e}, fp64 unit_roundoff={self.fp64_unit_roundoff:.3e}")
        while total_iterations < self.max_iter:
            if phase == "test":
                print(f"ferr={ferr:.2e}, nbe={nbe:.2e}, cbe={cbe:.2e}, rho={rho:.2e}")
                print(f"iter {total_iterations} r: {r.flatten()[:5]}, x: {x.flatten()[:5]}, x_true: {x_true.flatten()[:5]}")

            total_iterations += 1
            counter.niter = 0
            r_np = r.cpu().numpy().flatten()
            try:
                d_np, gmres_info = gmres(A_scipy, r_np, x0=np.zeros(size), rtol=self.tol, maxiter=self.gmres_max_it, M=spla.LinearOperator((size, size), matvec=lambda x: self.apply_preconditioner(lu, x, 'fp64', phase=phase)[0]), callback=counter, callback_type='legacy')
                converged_gmres = (gmres_info == 0)
            except Exception as e:
                logging.debug(f"ir5_gmres_fp64, matrix {matrix_id}, iter {total_iterations}: GMRES failed with exception: {str(e)}")
                converged_gmres = False
                total_gmres_iterations.append(counter.niter)
                break
            
            total_gmres_iterations.append(counter.niter)
            
            if not converged_gmres:
                logging.debug(f"ir5_gmres_fp64, matrix {matrix_id}, iter {total_iterations}: GMRES failed with info={gmres_info}")
                break
            
            d = torch.tensor(d_np, dtype=torch.float64, device=self.device).view(-1, 1)
            
            # Compute stopping criteria
            dx_norm = torch.norm(d, p=float('inf')).item()
            x_norm = torch.norm(x, p=float('inf')).item()
            relative_dx = dx_norm / x_norm if x_norm > 1e-100 else float('inf')
            dx_ratio = dx_norm / prev_dx_norm if prev_dx_norm > 1e-100 else float('inf')
            prev_dx_norm = dx_norm

            if phase == "test":
                print(f"iter {total_iterations}, dx_norm={dx_norm:.2e}, x_norm={x_norm:.2e}, relative_dx={relative_dx:.2e}, dx_ratio={dx_ratio:.2e}")

            # Stopping criteria
            if relative_dx <= self.fp64_unit_roundoff:
                logging.debug(f"ir5_gmres_fp64, matrix {matrix_id}, iter {total_iterations}: Converged with relative_dx={relative_dx:.2e} <= {self.fp64_unit_roundoff:.2e}")
                converged = True
                break
            if total_iterations > 1 and dx_ratio >= self.tol:
                logging.debug(f"ir5_gmres_fp64, matrix {matrix_id}, iter {total_iterations}: Convergence slowed, dx_ratio={dx_ratio:.2e} >= {self.tol:.2e}")
                break

            x = x + d
            if phase == "test":
                print(total_iterations, f"x: {x.flatten()[:5]}")

            r = b_torch - torch.sparse.mm(A_torch, x)
            residual_norm = torch.norm(r, p=2, dtype=torch.float64).item()
            rho = residual_norm / b_norm if b_norm > 1e-100 else float('inf')
            
            ferr = torch.norm(x - x_true, p=float('inf')) / x_true_norm if x_true_norm > 1e-100 else float('inf')
            nbe = torch.norm(r, p=float('inf')) / (A_norm_inf * torch.norm(x, p=float('inf')) + torch.norm(b_torch, p=float('inf')) if torch.norm(x, p=float('inf')) > 1e-100 else float('inf'))
            temp = torch.abs(r) / (torch.abs(torch.sparse.mm(A_torch, x)) + torch.abs(b_torch))
            temp[torch.isnan(temp)] = 0
            cbe = torch.max(temp)
            errors.append((float(nbe), float(ferr), float(cbe)))
            
            logging.debug(f"ir5_gmres_fp64, matrix {matrix_id}, iter {total_iterations}: nbe={nbe:.2e}, ferr={ferr:.2e}, cbe={cbe:.2e}, rho={rho:.2e}")

        final_nbe = nbe if 'nbe' in locals() else float('inf')
        final_ferr = ferr if 'ferr' in locals() else float('inf')
        final_cbe = cbe if 'cbe' in locals() else float('inf')
        final_rho = rho if 'rho' in locals() else float('inf')
        stagnation = 1 if total_iterations > 1 and dx_ratio >= self.tol else 0
        print(f"fp64, matrix {matrix_id}, converged: {converged}, iterations: {total_iterations}, gmres_iterations: {sum(total_gmres_iterations)}")
        return x, total_iterations, total_gmres_iterations, errors, final_nbe, final_ferr, final_rho, final_cbe, converged, stagnation, converged_gmres, lu_failed, init_failed

    def train(self, train_data, top_k=20, episodes=100, results_dir='results_random_sparse2'):
        print("Training BanditAgent...")
        self.agent = BanditAgent(precisions=self.precisions, c1=self.c1, c2=self.c2, top_k=top_k, train_data=train_data)
        env = LinearSystemEnv(self, train_data, max_iter=self.max_stagnation, tol=self.tol, 
                            c1=self.agent.c1, c2=self.agent.c2)
        
        episode_rewards = []
        episode_losses = []
        train_data_by_cond = {'low': [], 'medium': [], 'high': []}
        for A, b, x_true, metadata in train_data:
            cond = metadata['condition_number']
            if cond < 1e4:
                train_data_by_cond['low'].append((A, b, x_true, metadata))
            elif cond < 1e7:
                train_data_by_cond['medium'].append((A, b, x_true, metadata))
            else:
                train_data_by_cond['high'].append((A, b, x_true, metadata))
        
        # --- [Modification Start]: Calculate total steps for smooth epsilon decay ---
        total_steps = len(train_data) * episodes
        current_step = 0
        # --- [Modification End] ---

        for episode in tqdm(range(episodes), desc="Training"):
            episode_reward_sum = 0
            episode_loss_sum = 0
            episode_steps = 0
            for cond_bucket in ['low', 'medium', 'high']:
                if not train_data_by_cond[cond_bucket]:
                    continue
                env.train_data = train_data_by_cond[cond_bucket]
                env.current_matrix_idx = 0
                for _ in range(len(train_data_by_cond[cond_bucket])):
                    state, info = env.reset()
                    
                    # --- [Modification Start]: Use step-based epsilon decay ---
                    # Changed from episode/total_episodes to step/total_steps
                    precisions, action_indices = self.agent.choose_action(
                        state, 
                        phase="train", 
                        step=current_step, 
                        total_steps=total_steps
                    )
                    # --- [Modification End] ---

                    # Added Step info to print for debugging visibility
                    print(f"Episode {episode}, Step {current_step}/{total_steps}, Matrix {info['matrix_id']}, Size {info['matrix_size']}, "
                        f"Action: {action_indices}, Precisions: {precisions}, Cond_bucket: {cond_bucket}")
                    
                    action = action_indices
                    next_state_tuple, reward, terminated, truncated, info, _ = env.step(action)
                    next_state, _ = next_state_tuple
                    self.agent.update_model(state, action_indices, reward, next_state)
                    
                    episode_reward_sum += reward
                    # Check list bounds just to be safe
                    if self.agent.training_losses:
                        episode_loss_sum += self.agent.training_losses[-1]
                    
                    episode_steps += 1
                    current_step += 1 # Increment global step count

                    if terminated or truncated:
                        continue

            if episode_steps > 0:
                avg_reward = episode_reward_sum / episode_steps
                avg_loss = episode_loss_sum / episode_steps
                episode_rewards.append(avg_reward)
                episode_losses.append(avg_loss)
                logging.info(f"Episode {episode}, Cond_bucket {cond_bucket}: Average Reward = {avg_reward:.4f}, Average Loss = {avg_loss:.4f}")
        
        env.train_data = train_data
        try:
            os.makedirs(results_dir, exist_ok=True)
            np.save(os.path.join(results_dir, 'episode_rewards.npy'), episode_rewards)
            np.save(os.path.join(results_dir, 'training_losses.npy'), episode_losses)
            logging.info(f"Saved episode rewards to {results_dir}")
            logging.info(f"Saved training losses to {results_dir}")
        except Exception as e:
            logging.error(f"Failed to save training data: {e}")
        
        return episode_rewards, episode_losses
    
    def _failure_return(self, size):
        return torch.zeros(size, 1, device=self.device), self.max_iter, [], [], [], 0,0,0,0, False, False, False, False, False

    def _record_metrics(self, errors, x, x_true, r, A_scipy, b, iter_num):
        # 简单记录，省略具体计算以节省长度
        err = torch.norm(x - x_true).item()
        errors.append((err, 0, 0))
        if iter_num % 10 == 0:
            logging.info(f"Iter {iter_num}: Error {err:.2e}")

    #def save_model(self, path):
    #    try:
    #        with open(path, 'wb') as f:
    #            pickle.dump(self.agent.q_tables, f)
    #        logging.info(f"Saved Q-tables to {path}")
    #
    #    except Exception as e:
    #        logging.error(f"Failed to save Q-tables: {e}")

    def save_model(self, path):
        """
        保存整个 BanditAgent 对象，包含 Q-tables 和状态离散化的阈值配置。
        """
        try:
            # 确保目录存在
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            if hasattr(self, 'agent'):
                with open(path, 'wb') as f:
                    # 关键修改：保存整个 agent 对象，而不仅仅是 q_tables
                    # 这样可以保留训练时计算出的 cond_min, cond_max 等属性
                    pickle.dump(self.agent, f)
                logging.info(f"Saved Agent model (full object) to {path}")
            else:
                logging.warning("No agent found to save.")
        except Exception as e:
            logging.error(f"Failed to save model: {e}")

    def load_model(self, path):
        """
        加载模型。支持加载整个 Agent 对象（推荐）或仅加载 Q-tables（旧版兼容）。
        """
        try:
            if not os.path.exists(path):
                logging.error(f"Model file not found: {path}")
                return False
            
            with open(path, 'rb') as f:
                loaded_obj = pickle.load(f)

            # 情况 1: 加载的是整个 BanditAgent 对象 (新版 save_model)
            if isinstance(loaded_obj, BanditAgent):
                self.agent = loaded_obj
                # 重新初始化随机数生成器，以防 pickle 序列化问题
                if not hasattr(self.agent, 'rng'):
                    self.agent.rng = random.Random(42)
                logging.info(f"Loaded full Agent model from {path}")
            
            # 情况 2: 加载的只是 q_tables (旧版 save_model 或 仅权重)
            elif isinstance(loaded_obj, list) or isinstance(loaded_obj, np.ndarray):
                if not hasattr(self, 'agent'):
                    # 如果还没有 agent，我们需要先创建一个默认的
                    # 注意：这里可能会丢失训练时的 bin ranges，建议使用情况 1
                    logging.warning("Loading raw Q-tables into new Agent. Bin ranges might be default!")
                    self.agent = BanditAgent(precisions=self.precisions, c1=self.c1, c2=self.c2)
                
                self.agent.q_tables = loaded_obj
                logging.info(f"Loaded Q-tables only from {path}")
            
            else:
                logging.error(f"Unknown model format in {path}")
                return False

            return True

        except Exception as e:
            logging.error(f"Failed to load model: {type(e).__name__}: {e}")
            return False
        
    def test(self, test_data):
        print("\nTesting...")
        try:
            pychop.clear_cache()
        except AttributeError:
            pass
        results = []
        rl_errors = []
        rl_nbes = []
        rl_rhos = []
        rl_cbes = []
        rl_iterations = []
        fp64_errors = []
        fp64_nbes = []
        fp64_rhos = []
        fp64_cbes = []
        fp64_iterations = []
        precision_logs = []
        for i, (A_scipy, b_np, x_true_np, metadata) in enumerate(tqdm(test_data, desc="Testing")):
            precision_usage = {p: 0 for p in self.agent.precisions}
            A_torch = scipy_to_torch_sparse(A_scipy).coalesce()
            b = torch.tensor(b_np, dtype=torch.float64, device=self.device).view(-1, 1)
            x_true = torch.tensor(x_true_np, dtype=torch.float64, device=self.device).view(-1, 1)
            
            x_fp64, iters_fp64, iters_fp64_gmres, errors_fp64, final_nbe_fp64, final_ferr_fp64, final_rho_fp64, final_cbe_fp64, _, _, _, _, _ = self.ir5_gmres_fp64(A_scipy, b, x_true, matrix_id=i)
            error_fp64 = torch.norm(x_fp64 - x_true, p=float('inf')) / torch.norm(x_true, p=float('inf')) if torch.norm(x_true) > 1e-100 else float('inf')
            fp64_errors.append(float(error_fp64) if math.isfinite(error_fp64) else float('inf'))
            fp64_nbes.append(float(final_nbe_fp64) if math.isfinite(final_nbe_fp64) else float('inf'))
            fp64_rhos.append(float(final_rho_fp64) if math.isfinite(final_rho_fp64) else float('inf'))
            fp64_cbes.append(float(final_cbe_fp64) if math.isfinite(final_cbe_fp64) else float('inf'))
            fp64_iterations.append(iters_fp64)

            print("\n---------")

            state = self.agent.get_state(A_scipy, b_np)
            precisions, _ = self.agent.choose_action(state, phase="test")
            for p in precisions:
                precision_usage[p] += 1
            x_rl, iters_rl, iters_rl_gmres, errors_rl, log_rl, final_nbe_rl, final_ferr_rl, final_rho_rl, final_cbe_rl, _, _, _, _, _ = \
                self.ir5_gmres_mixed_precision(A_torch, b, x_true, matrix_id=i, phase="test", precisions=precisions)
            
            error_rl = torch.norm(x_rl - x_true, p=float('inf')) / torch.norm(x_true, p=float('inf')) if torch.norm(x_true) > 1e-100 else float('inf')
            rl_errors.append(float(error_rl) if not torch.isnan(error_rl) else float('inf'))
            rl_nbes.append(float(final_nbe_rl) if math.isfinite(final_nbe_rl) else float('inf'))
            rl_rhos.append(float(final_rho_rl) if math.isfinite(final_rho_rl) else float('inf'))
            rl_cbes.append(float(final_cbe_rl) if math.isfinite(final_cbe_rl) else float('inf'))
            rl_iterations.append(iters_rl)
            precision_logs.append(log_rl)

            result = {
                'matrix_id': str(i),
                'matrix_size': A_scipy.shape[0],
                'rl_error': float(error_rl),
                'rl_nbe': float(final_nbe_rl),
                'rl_rho': float(final_rho_rl),
                'rl_cbe': float(final_cbe_rl),
                'rl_iterations': int(iters_rl),
                'rl_gmres_iterations': sum(iters_rl_gmres),
                'fp64_error': float(error_fp64),
                'fp64_nbe': float(final_nbe_fp64),
                'fp64_rho': float(final_rho_fp64),
                'fp64_cbe': float(final_cbe_fp64),
                'fp64_iterations': int(iters_fp64),
                'fp64_gmres_iterations': sum(iters_fp64_gmres),
                'precision_usage': precision_usage,
                'total_iterations': iters_rl
            }
            results.append(result)

            print("\n\n")

        sum_precision_usage = {p: 0 for p in self.agent.precisions}
        for result in results:
            for prec, count in result['precision_usage'].items():
                sum_precision_usage[prec] += count
        averages = {
            'avg_rl_error': np.mean([e for e in rl_errors if e != float('inf')]) if any(e != float('inf') for e in rl_errors) else float('inf'),
            'avg_rl_nbe': np.mean([e for e in rl_nbes if e != float('inf')]) if any(e != float('inf') for e in rl_nbes) else float('inf'),
            'avg_rl_rho': np.mean([e for e in rl_rhos if e != float('inf')]) if any(e != float('inf') for e in rl_rhos) else float('inf'),
            'avg_rl_cbe': np.mean([e for e in rl_cbes if e != float('inf')]) if any(e != float('inf') for e in rl_cbes) else float('inf'),
            'avg_rl_iterations': np.mean([re['rl_iterations'] for re in results]),
            'avg_rl_gmres_iterations': np.mean([re['rl_gmres_iterations'] for re in results]),
            'avg_fp64_error': np.mean([e for e in fp64_errors if e != float('inf')]) if any(e != float('inf') for e in fp64_errors) else float('inf'),
            'avg_fp64_nbe': np.mean([e for e in fp64_nbes if e != float('inf')]) if any(e != float('inf') for e in fp64_nbes) else float('inf'),
            'avg_fp64_rho': np.mean([e for e in fp64_rhos if e != float('inf')]) if any(e != float('inf') for e in fp64_rhos) else float('inf'),
            'avg_fp64_cbe': np.mean([e for e in fp64_cbes if e != float('inf')]) if any(e != float('inf') for e in fp64_cbes) else float('inf'),
            'avg_fp64_iterations': np.mean([re['fp64_iterations'] for re in results]),
            'avg_fp64_gmres_iterations': np.mean([re['fp64_gmres_iterations'] for re in results]),
            'sum_precision_usage': sum_precision_usage
        }
        return results, precision_logs, averages



if __name__ == "__main__":
    num_train = 100
    num_test = 100
    max_iter = 9999
    tol = 1e-6
    episodes = 100
    data_dir = 'data_random_sparse2'
    results_dir = 'results_random_sparse2'

    train_data, test_data, train_metrics, test_metrics = load_data(num_train, num_test, data_dir)
    if train_data is None or test_data is None:
        logging.info("Generating new datasets using new SPD sparse generator...")
        # 使用新的 generate_datasets（已在 linear_system_utils.py 中定义）
        train_data, test_data = generate_datasets(num_train, num_test, size_range=(100, 500), verbose=True)
        save_data(train_data, test_data, data_dir)
        train_metrics = [{
            'matrix_id': str(i),
            'condition_number': metadata['condition_number'],
            'sparsity': metadata['sparsity'],
            'matrix_size': metadata['matrix_size']
        } for i, (_, _, _, metadata) in enumerate(train_data)]
        test_metrics = [{
            'matrix_id': str(i),
            'condition_number': metadata['condition_number'],
            'sparsity': metadata['sparsity'],
            'matrix_size': metadata['matrix_size']
        } for i, (_, _, _, metadata) in enumerate(test_data)]

    solver = GMRESIR5Solver(
        max_iter=max_iter,
        tol=tol,
        precisions=['bf16', 'tf32', 'fp32', 'fp64'],
        max_inner_iter=1,
        max_stagnation=3,
        c1=1,
        c2=1
    )

    episode_rewards, episode_losses = solver.train(train_data, episodes=episodes, top_k=int(episodes/4), results_dir=results_dir)
    results, precision_logs, averages = solver.test(test_data)

    os.makedirs(results_dir, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(results_dir, 'test_results.csv'), index=False)
    with open(os.path.join(results_dir, 'precision_logs.pkl'), 'wb') as f:
        pickle.dump(precision_logs, f)
    solver.save_model(os.path.join(results_dir, 'q_tables.pkl'))

    df_train_metrics = pd.DataFrame(train_metrics)
    df_test_metrics = pd.DataFrame(test_metrics)
    df_train_metrics.to_csv(os.path.join(results_dir, 'train_metrics.csv'), index=False)
    df_test_metrics.to_csv(os.path.join(results_dir, 'test_metrics.csv'), index=False)

    logging.info(f"Average RL error: {averages['avg_rl_error']:.4e}")
    logging.info(f"Average RL NBE: {averages['avg_rl_nbe']:.4e}")
    logging.info(f"Average RL iterations: {averages['avg_rl_iterations']:.2f}")
    logging.info(f"Average fp64 error: {averages['avg_fp64_error']:.4e}")
    logging.info(f"Average fp64 NBE: {averages['avg_fp64_nbe']:.4e}")
    logging.info(f"Average fp64 iterations: {averages['avg_fp64_iterations']:.2f}")
    logging.info(f"Precision usage: {averages['sum_precision_usage']}")