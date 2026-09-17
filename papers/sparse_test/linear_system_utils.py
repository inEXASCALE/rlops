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
from pychop import LightChop
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

# ==================== NEW DATASET GENERATION ====================

def generate_random_linear_system(
    size=100, d='normal', beta=None, solution_return=False, 
    ood=False, sparsity_lambda=1e-2, random_state=0, verbose=True,
    lower=0.2, upper=1.2, lower_beta=1e-9, upper_beta=1e-7, max_retries=10, retry_count=0
):  
    from scipy.sparse import coo_matrix
    from scipy.sparse import identity
    
    rng = np.random.RandomState(random_state)
    if beta is None:
        beta = rng.uniform(lower_beta, upper_beta)
    if ood:
        sparsity_lambda = rng.uniform(lower, upper) * sparsity_lambda 

    nnz = max(int(sparsity_lambda * (size ** 2)), 1)
    rows = rng.choice(size, nnz, replace=True)
    cols = rng.choice(size, nnz, replace=True)
    if d == 'normal':
        vals = rng.standard_normal(nnz)
    else:
        vals = rng.uniform(-1.0, 1.0, nnz)
    
    S = coo_matrix((vals, (rows, cols)), shape=(size, size))
    A = (S @ S.T) + beta * identity(size, format='coo')
    A = A.tocoo()
    A_csc = A.tocsc()

    # ---- singularity / SPD check with retry ----
    if check_singularity(A_csc):
        if retry_count >= max_retries:
            logging.warning(f"Max retries reached. Accepting singular matrix (seed={random_state}).")
        else:
            if verbose:
                print("Matrix singular, retrying...")
            return generate_random_linear_system(
                size, d, beta, solution_return, ood, sparsity_lambda, 
                random_state + 1, verbose, lower, upper, lower_beta, 
                upper_beta, max_retries, retry_count + 1
            )

    try:
        eigenvalues = np.linalg.eigvalsh(A.toarray())
        if not np.all(eigenvalues > 1e-12):
            if retry_count >= max_retries:
                logging.warning(f"Matrix not SPD. Accepting (seed={random_state}).")
            else:
                if verbose:
                    print("Matrix not positive definite, retrying...")
                return generate_random_linear_system(
                    size, d, beta, solution_return, ood, sparsity_lambda, 
                    random_state + 1, verbose, lower, upper, lower_beta, 
                    upper_beta, max_retries, retry_count + 1
                )
        cond_number = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))
    except Exception as e:
        if verbose:
            logging.debug(f"Eigenvalue failed: {e}, retrying...")
        return generate_random_linear_system(
            size, d, beta, solution_return, ood, sparsity_lambda, 
            random_state + 1, verbose, lower, upper, lower_beta, 
            upper_beta, max_retries, retry_count + 1
        )

    if verbose:
        sparsity_val = compute_sparsity(A)
        print(f"Matrix: {size}x{size}, {100 * sparsity_val:.3f}% nnz, cond={cond_number:.2e}")

    x_true = np.random.randn(size)
    b = A_csc @ x_true

    metadata = {
        'condition_number': cond_number,
        'matrix_size': size,
        'kappa': cond_number,
        'sparsity': compute_sparsity(A),
        'sparsity_lambda': sparsity_lambda
    }

    return A, b, x_true, metadata


def generate_datasets(num_train, num_test, size_range=(100, 500), 
                      random_state=42, verbose=True):
    np.random.seed(random_state)
    random.seed(random_state)
    train_set, test_set = [], []
    
    for i in range(num_train + num_test):
        n = np.random.randint(size_range[0], size_range[1] + 1)

        try:
            A, b, x_true, metadata = generate_random_linear_system(
                size=n,
                d='normal',
                beta=None,
                solution_return=False,
                ood=False,
                sparsity_lambda=1e-2,
                random_state=random_state + i,
                verbose=verbose
            )
        except Exception as e:
            print(f"Error generating system {i}: {e}")
            continue

        if check_singularity(A):
            print(f"System {i}: Singular matrix detected, skipping")
            continue

        if verbose:
            print(f"System {i}: {n}x{n}, sparsity {metadata['sparsity']:.3f}, "
                  f"cond {metadata['condition_number']:.3e}")

        if len(train_set) < num_train:
            train_set.append((A, b, x_true, metadata))
        else:
            test_set.append((A, b, x_true, metadata))

    return train_set, test_set

# ==================== REST OF THE ORIGINAL CODE (unchanged) ====================

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
    
    ch = LightChop(**precision_configs[precision])

    if precision in ['e5m2', 'e4m3', 'fp16']:
        scale = torch.max(torch.abs(x))
        
        if scale != 0:
            op_input = x / scale
        else:
            op_input = x
            
        result = ch(op_input)
        
        if isinstance(result, np.ndarray):
            result = torch.from_numpy(result).to(device)
            
        if scale != 0:
            result = result * scale
    else:
        result = ch(x)
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
    data = torch.tensor(A_coo.data, dtype=torch.float64, device=device)
    
    ch = LightChop(**precision_configs[precision])
    try:
        rounded_data = ch(data)
        
        if isinstance(rounded_data, np.ndarray):
            nan_count = np.sum(np.isnan(rounded_data))
            inf_count = np.sum(np.isinf(rounded_data))
            final_data_np = rounded_data
        else:
            nan_count = torch.sum(torch.isnan(rounded_data)).item()
            inf_count = torch.sum(torch.isinf(rounded_data)).item()
            final_data_np = rounded_data.cpu().numpy()

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
        self.c1 = c1
        self.c2 = c2
        self.training_losses = []
        self.rng = random.Random(seed)

        self.cond_min, self.cond_max = 0, 10
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
        cond_log = np.log10(max(condition_number, 1))
        norm_log = np.log10(max(A_norm, 1))
        cond_range = self.cond_max - self.cond_min if self.cond_max > self.cond_min else 1
        norm_range = self.norm_max - self.norm_min if self.norm_max > self.norm_min else 1
        cond_idx = np.clip(int((cond_log - self.cond_min) / cond_range * (self.cond_bins - 1)), 0, self.cond_bins - 1)
        norm_idx = np.clip(int((norm_log - self.norm_min) / norm_range * (self.norm_bins - 1)), 0, self.norm_bins - 1)
        return cond_idx * self.norm_bins + norm_idx

    def choose_action(self, state, phase="train", step=0, total_steps=1000):
        epsilon = max(0.1, 1.0 - step / total_steps)
        random_num = self.rng.random()
        
        if phase == "train" and random_num < epsilon:
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
        action_idx = action
        current_q = self.q_tables[0][state_idx, action_idx]
        new_q = current_q + self.alpha * (reward - current_q) 
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

        reward = accuracy_reward + precision_reward  - 0.5 * iteration_penalty
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