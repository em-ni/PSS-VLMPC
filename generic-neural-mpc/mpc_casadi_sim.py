# mpc_casadi_sim.py
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd.functional import jacobian, hessian
import casadi as ca
import joblib
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

# Import exactly as requested
from model.train_sim import StatePredictor, TrainingConfig as TrainConfig

# --- Configuration ---
class MPCConfig:
    MODEL_PATH = TrainConfig.MODEL_PATH
    INPUT_SCALER_PATH = TrainConfig.INPUT_SCALER_PATH
    OUTPUT_SCALER_PATH = TrainConfig.OUTPUT_SCALER_PATH
    SIM_DATASET_PATH = TrainConfig.SIM_DATASET_PATH
    N = 15
    DT = 0.020
    SIM_TIME = 10.0
    q_pos = 10.0 #30.0
    q_vel = 0.0 #3.0
    Q_diag = [q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]
    r_diag = 100.0 #100.0
    R_diag = [r_diag, r_diag, r_diag, r_diag]  
    r_rate_diag = 150000.0 #3000.0
    R_rate_diag = [r_rate_diag, r_rate_diag, r_rate_diag, r_rate_diag] 
    LAMBDA = 50.0 #10
    max_torque = 9e-2
    U_MIN = [-max_torque, -max_torque, -max_torque, -max_torque]
    U_MAX = [max_torque, max_torque, max_torque, max_torque]
    
    def stability_check(self):
        """
        Check if the MPC configuration is stable (based on Seel et. al., "Neural Network-Based Model Predictive Control with Input-to-State Stability")
        - A1 and A2 are satisfied by choosing the stabilizing control law and the terminal cost as
            k_f(x - x_ref) = K^T (x - x_ref)  + u_ref
        where K is the feedback gain matrix and u_ref is the reference control input.
            V_f(x - x_ref) = (x - x_ref)^T P (x - x_ref)
        where K, and P come from the solution of the discrete algebraic Riccati equation (DARE).
        And by choosing the stage cost as
            l(x, u) = (x - x_ref)^T Q (x - x_ref) + (u - u_ref)^T R (u - u_ref)
        with Q, R positive definite matrices.
        So that the MPC cost function is given by:
            J = sum_{k=0}^{N-1} l(x_k, u_k) + lambda*V_f(x_N - x_ref)
        Note: it is required to know a priori u_ref, it can be done by computing the optimal control input for a constant reference, or for example collecting data and using a lookup table.
        
        - A3 is assumed for the case of full NN (and confirmed by inspecting the test error accross many point of the workspace)
            |y - y^| <= mu 
        where y^ is the output of the NN and y is the true system output.
        While for the case of Taylor approximation of the NN, the approximation of the NN around a point (x_i, u_i) is given by:
            f_NN(x, u) = f_NN(x_i, u_i) + J_NN(x, u) * [x - x_i; u - u_i] + 0.5 * [x - x_i; u - u_i] * H_NN(x, u) * [x - x_i; u - u_i]  + o(||x - x_i; u - u_i||^3)
        and
            o(||x - x_i; u - u_i||^3) < epsilon
        Then
            |y - y^| <= mu + o(||x - x_i; u - u_i||^3) < = mu + epsilon
        
        - A4 is satisfied by designing an uniformly continuous NN, so choosing uniformly continuous activation functions (e.g. tanh, sigmoid).

        The discussion above guarantees ISS inside a region of attraction around the reference, the size of this can be regulated by tuning lambda (Limon et al., "On the stability of constrained MPC without terminal constraint")

        In conclusion, if the cost is chosen such that A1 and A2 hold, the NN is chosen and trained such that A3 and A4 holds, the stability check requires to verify that for the parameters Q and R, LAMBDA is large enough to ensure x_0 is inside the region of attraction

        """
        # Placeholder for stability check logic
        # For now, we assume the configuration is stable
        return True

class MPCController:
    def __init__(self, nn_approximation_order=1):
        """
        Initialize MPC Controller
        
        Args:
            nn_approximation_order (int): Order of neural network approximation
                                         0 = No approximation (use NN directly)
                                         1 = First-order (linear) approximation
                                         2 = Second-order (quadratic) approximation
        """
        self.nn_approximation_order = nn_approximation_order
        self.n_states = 6
        self.n_controls = 4
        self.state_cols = ['tip_position_x', 'tip_position_y', 'tip_position_z', 
                          'tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z']
        
        # Initialize model and scalers
        self._load_assets()
        self._setup_optimization_problem()
        
        # Initialize simulation variables
        self.history_x = []
        self.history_u = []
        self.u_guess = np.zeros((self.n_controls, MPCConfig.N))
        self.last_u_optimal = np.zeros(self.n_controls)
        
        # Initialize matrices for terminal cost computation
        self.Q_np = np.diag(MPCConfig.Q_diag)
        self.R_np = np.diag(MPCConfig.R_diag)
        
        # Print initialization info
        print(f"Initialized MPCController with NN approximation order: {self.nn_approximation_order}")
        print(f"  Number of states: {self.n_states}, Number of controls: {self.n_controls}")
        print(f"  MPC horizon N: {MPCConfig.N}, Time step DT: {MPCConfig.DT}")
        print(f"  Input bounds: {MPCConfig.U_MIN} to {MPCConfig.U_MAX}")
        print(f"  Cost matrices Q: {MPCConfig.Q_diag}, R: {MPCConfig.R_diag}, R_rate: {MPCConfig.R_rate_diag}")
        print(f"  Terminal cost scaling factor: {MPCConfig.LAMBDA}")
        
    def _load_assets(self):
        """Load simulation model assets"""
        print("\nLoading simulation model assets")
        try:
            self.df = pd.read_csv(MPCConfig.SIM_DATASET_PATH)
            self.df.columns = self.df.columns.str.strip()
            
            self.model = StatePredictor(input_dim=10, output_dim=6)
            self.model.load_state_dict(torch.load(MPCConfig.MODEL_PATH))
            self.model.eval()  # Set to evaluation mode
            
            # Don't JIT compile for now to avoid functorch issues
            print("Model loaded in standard PyTorch mode (no JIT compilation for derivative compatibility).")
            
            self.input_scaler = joblib.load(MPCConfig.INPUT_SCALER_PATH)
            self.output_scaler = joblib.load(MPCConfig.OUTPUT_SCALER_PATH)
            
            # Setup scaling tensors
            self.input_scale = torch.tensor(self.input_scaler.scale_, dtype=torch.float32)
            self.input_mean = torch.tensor(self.input_scaler.mean_, dtype=torch.float32)
            self.output_scale = torch.tensor(self.output_scaler.scale_, dtype=torch.float32)
            self.output_mean = torch.tensor(self.output_scaler.mean_, dtype=torch.float32)
            
        except FileNotFoundError as e: 
            print(f"Error: A required file was not found: {e.filename}")
            sys.exit()
    
    def _setup_optimization_problem(self):
        """Define the Optimization Problem (OCP)"""
        print(f"Setting up OCP with NN approximation order {self.nn_approximation_order}")
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(self.n_states, MPCConfig.N + 1)
        self.U = self.opti.variable(self.n_controls, MPCConfig.N)
        
        # Parameters
        self.x0 = self.opti.parameter(self.n_states, 1)
        self.x_ref = self.opti.parameter(self.n_states, 1)
        self.u_prev = self.opti.parameter(self.n_controls, 1)
        
        if self.nn_approximation_order > 0:
            # Linear approximation parameters
            self.A_params = [self.opti.parameter(self.n_states, self.n_states) for _ in range(MPCConfig.N)]
            self.B_params = [self.opti.parameter(self.n_states, self.n_controls) for _ in range(MPCConfig.N)]
            self.C_params = [self.opti.parameter(self.n_states, 1) for _ in range(MPCConfig.N)]
            
            if self.nn_approximation_order == 2:
                # Second-order approximation parameters (Hessian terms)
                input_dim = self.n_controls + self.n_states  # u and x dimensions
                self.H_params = [self.opti.parameter(self.n_states, input_dim * input_dim) for _ in range(MPCConfig.N)]
        
        # Terminal cost matrix parameter
        self.P_terminal = self.opti.parameter(self.n_states, self.n_states)

        # Setup cost function
        self._setup_cost_function()
        
        # Setup constraints
        self._setup_constraints()
        
        # Setup solver
        solver_opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.sb': 'yes',
            'ipopt.acceptable_tol': 1e-3
        }
        self.opti.solver('ipopt', solver_opts)
    
    def _setup_cost_function(self):
        """Setup the cost function for the MPC"""
        cost = 0
        Q = ca.diag(MPCConfig.Q_diag)
        R = ca.diag(MPCConfig.R_diag)
        R_rate = ca.diag(MPCConfig.R_rate_diag)
        
        # Stage costs
        for k in range(MPCConfig.N):
            cost += (self.X[:, k] - self.x_ref).T @ Q @ (self.X[:, k] - self.x_ref)
            cost += self.U[:, k].T @ R @ self.U[:, k]
            if k == 0: 
                cost += (self.U[:, k] - self.u_prev).T @ R_rate @ (self.U[:, k] - self.u_prev)
            else: 
                cost += (self.U[:, k] - self.U[:, k-1]).T @ R_rate @ (self.U[:, k] - self.U[:, k-1])

        # Terminal cost
        x_terminal_error = self.X[:, MPCConfig.N] - self.x_ref
        cost += x_terminal_error.T @ self.P_terminal @ x_terminal_error
        
        self.opti.minimize(cost)
    
    def _setup_constraints(self):
        """Setup constraints for the MPC"""
        # Initial condition constraint
        self.opti.subject_to(self.X[:, 0] == self.x0)
        
        # Dynamics constraints
        if self.nn_approximation_order == 0:
            # No approximation - this would require implementing the full NN in CasADi
            # For now, we'll fall back to first-order approximation
            print("Warning: Zero-order approximation not implemented, using first-order")
            self.nn_approximation_order = 1
            
        if self.nn_approximation_order == 1:
            # First-order (linear) approximation
            for k in range(MPCConfig.N):
                self.opti.subject_to(self.X[:, k+1] == self.A_params[k] @ self.X[:, k] + self.B_params[k] @ self.U[:, k] + self.C_params[k])
        
        elif self.nn_approximation_order == 2:
            # Second-order (quadratic) approximation
            for k in range(MPCConfig.N):
                # Linear terms
                linear_dynamics = self.A_params[k] @ self.X[:, k] + self.B_params[k] @ self.U[:, k] + self.C_params[k]
                
                # Quadratic terms
                input_vec = ca.vertcat(self.U[:, k], self.X[:, k])
                quadratic_terms = ca.MX.zeros(self.n_states, 1)
                
                for i in range(self.n_states):
                    H_i = ca.reshape(self.H_params[k][i, :], self.n_controls + self.n_states, self.n_controls + self.n_states)
                    quadratic_terms[i] = 0.5 * input_vec.T @ H_i @ input_vec
                
                self.opti.subject_to(self.X[:, k+1] == linear_dynamics + quadratic_terms)
        
        # Input constraints
        for k in range(MPCConfig.N):
            self.opti.subject_to(self.opti.bounded(MPCConfig.U_MIN, self.U[:, k], MPCConfig.U_MAX))
    
    def full_pytorch_model(self, x_and_u_torch):
        """Full PyTorch model with scaling"""
        # Ensure input requires grad for derivative computation
        if not x_and_u_torch.requires_grad:
            x_and_u_torch = x_and_u_torch.requires_grad_(True)
            
        scaled_input = (x_and_u_torch - self.input_mean) / self.input_scale
        scaled_output = self.model(scaled_input)
        return scaled_output * self.output_scale + self.output_mean
    
    def compute_finite_differences(self, x_and_u_np, h=1e-6):
        """Compute Jacobian and Hessian using finite differences as fallback"""
        input_dim = len(x_and_u_np)
        x_and_u_tensor = torch.tensor(x_and_u_np, dtype=torch.float32, requires_grad=False)
        
        # Forward pass for baseline
        with torch.no_grad():
            f0 = self.full_pytorch_model(x_and_u_tensor).numpy()
        
        # Compute Jacobian using finite differences
        J = np.zeros((self.n_states, input_dim))
        for i in range(input_dim):
            x_plus = x_and_u_np.copy()
            x_minus = x_and_u_np.copy()
            x_plus[i] += h
            x_minus[i] -= h
            
            x_plus_tensor = torch.tensor(x_plus, dtype=torch.float32, requires_grad=False)
            x_minus_tensor = torch.tensor(x_minus, dtype=torch.float32, requires_grad=False)
            
            with torch.no_grad():
                f_plus = self.full_pytorch_model(x_plus_tensor).numpy()
                f_minus = self.full_pytorch_model(x_minus_tensor).numpy()
            
            J[:, i] = (f_plus - f_minus) / (2 * h)
        
        # Compute Hessian using finite differences (if needed)
        H = None
        if self.nn_approximation_order == 2:
            H = np.zeros((self.n_states, input_dim, input_dim))
            for i in range(input_dim):
                for j in range(input_dim):
                    # Second-order finite difference formula
                    x_pp = x_and_u_np.copy()
                    x_pm = x_and_u_np.copy()
                    x_mp = x_and_u_np.copy()
                    x_mm = x_and_u_np.copy()
                    
                    x_pp[i] += h; x_pp[j] += h
                    x_pm[i] += h; x_pm[j] -= h
                    x_mp[i] -= h; x_mp[j] += h
                    x_mm[i] -= h; x_mm[j] -= h
                    
                    with torch.no_grad():
                        f_pp = self.full_pytorch_model(torch.tensor(x_pp, dtype=torch.float32)).numpy()
                        f_pm = self.full_pytorch_model(torch.tensor(x_pm, dtype=torch.float32)).numpy()
                        f_mp = self.full_pytorch_model(torch.tensor(x_mp, dtype=torch.float32)).numpy()
                        f_mm = self.full_pytorch_model(torch.tensor(x_mm, dtype=torch.float32)).numpy()
                    
                    H[:, i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
        
        return f0, J, H
    
    def get_batch_predictions_and_derivatives(self, x_traj_np, u_traj_np):
        """Get predictions and derivatives using robust computation methods"""
        self.model.to('cpu').eval()

        # Combine state and control trajectories into a single input batch
        x_and_u_traj_np = np.hstack([u_traj_np, x_traj_np])
        
        batch_size = x_and_u_traj_np.shape[0]
        
        # Initialize output arrays
        y_pred_batch = np.zeros((batch_size, self.n_states))
        
        if self.nn_approximation_order >= 1:
            J_batch = np.zeros((batch_size, self.n_states, x_and_u_traj_np.shape[1]))
            
        if self.nn_approximation_order == 2:
            H_batch = np.zeros((batch_size, self.n_states, x_and_u_traj_np.shape[1], x_and_u_traj_np.shape[1]))

        # Process each sample individually
        for i in range(batch_size):
            x_and_u_sample_np = x_and_u_traj_np[i]
            
            try:
                # First, try using torch.autograd.functional
                x_and_u_sample = torch.tensor(x_and_u_sample_np, dtype=torch.float32, requires_grad=True)
                
                # Forward pass
                y_pred = self.full_pytorch_model(x_and_u_sample)
                y_pred_batch[i] = y_pred.detach().numpy()
                
                if self.nn_approximation_order >= 1:
                    try:
                        # Use torch.autograd.functional.jacobian instead of functorch
                        def model_func(x):
                            return self.full_pytorch_model(x)
                        
                        jac = jacobian(model_func, x_and_u_sample)
                        
                        # Check for NaN/Inf in jacobian
                        if torch.isnan(jac).any() or torch.isinf(jac).any():
                            raise ValueError("NaN/Inf in Jacobian")
                        
                        # Clip jacobian values to prevent explosion
                        jac = torch.clamp(jac, -1e3, 1e3)
                        J_batch[i] = jac.detach().numpy()
                        
                    except Exception as jac_e:
                        print(f"Warning: Jacobian computation failed at sample {i}: {jac_e}, using finite differences")
                        # Fallback to finite differences
                        _, J_fd, H_fd = self.compute_finite_differences(x_and_u_sample_np)
                        J_batch[i] = J_fd
                        if self.nn_approximation_order == 2 and H_fd is not None:
                            H_batch[i] = H_fd
                        continue
                    
                    if self.nn_approximation_order == 2:
                        try:
                            # Use torch.autograd.functional.hessian
                            def model_func_hess(x):
                                # For Hessian, we need a scalar output, so we'll compute for each output dimension
                                return self.full_pytorch_model(x)
                            
                            # Compute Hessian for each output dimension
                            for out_dim in range(self.n_states):
                                def scalar_model_func(x):
                                    return model_func_hess(x)[out_dim]
                                
                                hess_out = hessian(scalar_model_func, x_and_u_sample)
                                
                                # Check for NaN/Inf
                                if torch.isnan(hess_out).any() or torch.isinf(hess_out).any():
                                    hess_out = torch.zeros_like(hess_out)
                                
                                # Apply regularization for numerical stability
                                hessian_regularization = 1e-6
                                input_dim = x_and_u_sample.shape[0]
                                hess_out_reg = hess_out + hessian_regularization * torch.eye(input_dim)
                                
                                # Check condition number
                                try:
                                    eigenvals = torch.linalg.eigvals(hess_out_reg)
                                    max_eigval = torch.max(torch.real(eigenvals))
                                    min_eigval = torch.min(torch.real(eigenvals))
                                    
                                    if min_eigval > 0:
                                        condition_number = max_eigval / min_eigval
                                        if condition_number > 1e6:  # Too ill-conditioned
                                            hess_out = torch.zeros_like(hess_out)
                                        else:
                                            hess_out = hess_out_reg
                                    else:
                                        hess_out = torch.zeros_like(hess_out)
                                except:
                                    hess_out = torch.zeros_like(hess_out)
                                
                                # Final clipping
                                hess_out = torch.clamp(hess_out, -1e3, 1e3)
                                H_batch[i, out_dim] = hess_out.detach().numpy()
                                
                        except Exception as hess_e:
                            print(f"Warning: Hessian computation failed at sample {i}: {hess_e}, using finite differences")
                            # Fallback to finite differences
                            _, _, H_fd = self.compute_finite_differences(x_and_u_sample_np)
                            if H_fd is not None:
                                H_batch[i] = H_fd
                            
            except Exception as e:
                print(f"Warning: All derivative computations failed at sample {i}: {e}, using finite differences")
                # Ultimate fallback to finite differences
                y_pred_fd, J_fd, H_fd = self.compute_finite_differences(x_and_u_sample_np)
                y_pred_batch[i] = y_pred_fd
                if self.nn_approximation_order >= 1:
                    J_batch[i] = J_fd
                if self.nn_approximation_order == 2 and H_fd is not None:
                    H_batch[i] = H_fd

        if self.nn_approximation_order == 2:
            return y_pred_batch, J_batch, H_batch
        elif self.nn_approximation_order == 1:
            return y_pred_batch, J_batch, None
        else:
            return y_pred_batch, None, None
    
    def compute_terminal_cost_matrix(self, A, B, Q, R):
        """Compute the terminal cost matrix P by solving DARE"""
        try:
            # Solve the discrete algebraic Riccati equation
            P = solve_discrete_are(A, B, Q, R)
            
            # Compute the optimal feedback gain
            K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
            
            return P, K
        except Exception as e:
            print(f"Warning: Failed to solve DARE, using Q as terminal cost: {e}")
            # Fallback to using Q as terminal cost
            return Q, np.zeros((B.shape[1], A.shape[0]))
    
    def step(self, x_ref, x_current):
        """Solve MPC and return optimal control input with robust derivative handling"""
        # Prediction and Linearization (BATCHED)
        x_guess_np = np.zeros((MPCConfig.N, self.n_states))
        x_guess_np[0, :] = x_current
        
        # Sequentially roll out the nominal trajectory
        for k in range(MPCConfig.N - 1):
            model_input_k = np.concatenate([self.u_guess[:, k], x_guess_np[k, :]])
            with torch.no_grad():
                model_input_torch = torch.from_numpy(model_input_k).float()
                x_next = self.full_pytorch_model(model_input_torch).numpy()
                
                # Safety check for the rollout
                if np.isnan(x_next).any() or np.isinf(x_next).any():
                    print(f"Warning: NaN/Inf in trajectory rollout at step {k}, using previous state")
                    x_next = x_guess_np[k, :]
                elif np.abs(x_next).max() > 1e4:
                    print(f"Warning: Explosive values in trajectory rollout at step {k}, clipping")
                    x_next = np.clip(x_next, -1e4, 1e4)
                
                x_guess_np[k+1, :] = x_next

        # Get predictions and derivatives based on approximation order
        if self.nn_approximation_order == 2:
            y_pred_batch, J_batch, H_batch = self.get_batch_predictions_and_derivatives(x_guess_np, self.u_guess.T)
        else:
            y_pred_batch, J_batch, _ = self.get_batch_predictions_and_derivatives(x_guess_np, self.u_guess.T)
        
        # Set the parameters for the optimizer with additional safety checks
        for k in range(MPCConfig.N):
            if self.nn_approximation_order >= 1:
                J_k = J_batch[k]
                B_k = J_k[:, :self.n_controls]  # First 4 columns for torque inputs
                A_k = J_k[:, self.n_controls:self.n_controls+self.n_states]  # Next 6 columns for states
                
                # Safety check for linearization matrices
                if np.isnan(A_k).any() or np.isnan(B_k).any() or np.isinf(A_k).any() or np.isinf(B_k).any():
                    print(f"Warning: NaN/Inf in linearization matrices at step {k}, using identity")
                    A_k = np.eye(self.n_states)
                    B_k = np.zeros((self.n_states, self.n_controls))
                
                # Additional check for explosive values
                if np.abs(A_k).max() > 1e3 or np.abs(B_k).max() > 1e3:
                    print(f"Warning: Explosive values in linearization matrices at step {k}, clipping")
                    A_k = np.clip(A_k, -1e3, 1e3)
                    B_k = np.clip(B_k, -1e3, 1e3)
                
                C_k = y_pred_batch[k] - A_k @ x_guess_np[k] - B_k @ self.u_guess[:, k]
                
                # Safety check for C_k
                if np.isnan(C_k).any() or np.isinf(C_k).any():
                    print(f"Warning: NaN/Inf in C matrix at step {k}, using zero")
                    C_k = np.zeros_like(C_k)
                elif np.abs(C_k).max() > 1e3:
                    print(f"Warning: Explosive values in C matrix at step {k}, clipping")
                    C_k = np.clip(C_k, -1e3, 1e3)
                
                self.opti.set_value(self.A_params[k], A_k)
                self.opti.set_value(self.B_params[k], B_k)
                self.opti.set_value(self.C_params[k], C_k.reshape(-1, 1))
                
                if self.nn_approximation_order == 2 and 'H_batch' in locals():
                    # Set Hessian parameters with safety checks
                    H_k = H_batch[k]
                    
                    # Safety check for Hessian
                    if np.isnan(H_k).any() or np.isinf(H_k).any():
                        print(f"Warning: NaN/Inf in Hessian at step {k}, using zero")
                        H_k = np.zeros_like(H_k)
                    elif np.abs(H_k).max() > 1e3:
                        print(f"Warning: Explosive values in Hessian at step {k}, clipping")
                        H_k = np.clip(H_k, -1e3, 1e3)
                    
                    # Reshape Hessian for each output dimension
                    H_k_reshaped = H_k[:, :self.n_controls+self.n_states, :self.n_controls+self.n_states]  # Only u,x dimensions
                    H_k_flat = H_k_reshaped.reshape(self.n_states, -1)
                    self.opti.set_value(self.H_params[k], H_k_flat)
        
        # Compute Terminal Cost Matrix with safety checks
        if self.nn_approximation_order >= 1:
            A_terminal = J_batch[-1][:, self.n_controls:self.n_controls+self.n_states]
            B_terminal = J_batch[-1][:, :self.n_controls]
            
            # Safety check for terminal matrices
            if np.isnan(A_terminal).any() or np.isnan(B_terminal).any() or np.isinf(A_terminal).any() or np.isinf(B_terminal).any():
                print("Warning: NaN/Inf in terminal matrices, using identity")
                A_terminal = np.eye(self.n_states)
                B_terminal = np.zeros((self.n_states, self.n_controls))
        else:
            # Fallback to identity matrices if no linearization
            A_terminal = np.eye(self.n_states)
            B_terminal = np.zeros((self.n_states, self.n_controls))
        
        # Compute terminal cost matrix P by solving DARE with safety
        try:
            P_matrix, _ = self.compute_terminal_cost_matrix(A_terminal, B_terminal, self.Q_np, self.R_np)
            
            # Safety check for P matrix
            if np.isnan(P_matrix).any() or np.isinf(P_matrix).any():
                print("Warning: NaN/Inf in terminal cost matrix, using Q")
                P_matrix = self.Q_np
            elif np.abs(P_matrix).max() > 1e6:
                print("Warning: Explosive values in terminal cost matrix, scaling down")
                P_matrix = P_matrix / np.abs(P_matrix).max() * 1e3
        except Exception as e:
            print(f"Warning: Exception in terminal cost computation: {e}, using Q")
            P_matrix = self.Q_np
        
        # Multiply by a gain
        P_matrix *= MPCConfig.LAMBDA 
        self.opti.set_value(self.P_terminal, P_matrix)
        
        # Set Current Values and Solve
        self.opti.set_value(self.x0, x_current)
        self.opti.set_value(self.x_ref, x_ref)
        self.opti.set_value(self.u_prev, self.last_u_optimal)
        self.opti.set_initial(self.U, self.u_guess)
        self.opti.set_initial(self.X, np.hstack([x_current.reshape(-1,1), x_guess_np.T]))

        try:
            sol = self.opti.solve()
            u_optimal_all = sol.value(self.U)
            
            # Safety check for optimal control
            if np.isnan(u_optimal_all).any() or np.isinf(u_optimal_all).any():
                print("Warning: NaN/Inf in optimal control, using previous control")
                return self.last_u_optimal
            
            self.last_u_optimal = u_optimal_all[:, 0]
            self.u_guess = np.roll(u_optimal_all, -1, axis=1)
            self.u_guess[:, -1] = self.last_u_optimal
            return self.last_u_optimal
        except Exception as e:
            print(f"\nSolver failed: {e}")
            # Return the last known good control or zero control as fallback
            return self.last_u_optimal if hasattr(self, 'last_u_optimal') else np.zeros(self.n_controls)
    
    def simulate_system(self, x_current, u_control):
        """Simulate the system for one step using the control input"""
        model_input_sim = np.concatenate([u_control, x_current])
        with torch.no_grad():
             x_next = self.full_pytorch_model(torch.from_numpy(model_input_sim).float()).numpy()
        return x_next
    
    def plot_results(self, history_x_target):
        """Plot the MPC results"""
        history_x = np.array(self.history_x)
        history_u = np.array(self.history_u)
        history_x_target = np.array(history_x_target)
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        time_axis = np.arange(history_x.shape[0]) * MPCConfig.DT
        
        # Position plot
        axs[0].plot(time_axis, history_x[:, 0], label='Tip X')
        axs[0].plot(time_axis, history_x[:, 1], label='Tip Y')
        axs[0].plot(time_axis, history_x[:, 2], label='Tip Z')
        # axs[0].axhline(y=x_target[0], color='r', linestyle='--', label='Target X')
        # axs[0].axhline(y=x_target[1], color='g', linestyle='--', label='Target Y')
        # axs[0].axhline(y=x_target[2], color='b', linestyle='--', label='Target Z')
        axs[0].plot(time_axis, history_x_target[:, 0], 'r--', label='Target X')
        axs[0].plot(time_axis, history_x_target[:, 1], 'g--', label='Target Y')
        axs[0].plot(time_axis, history_x_target[:, 2], 'b--', label='Target Z')
        axs[0].set_ylabel('Position')
        title_suffix = f"(NN Approximation Order: {self.nn_approximation_order})"
        axs[0].set_title(f'MPC Trajectory with Terminal Cost {title_suffix}')
        axs[0].legend()
        axs[0].grid(True)
        
        # Velocity plot
        axs[1].plot(time_axis, history_x[:, 3:6])
        axs[1].axhline(y=0, color='k', linestyle='--')
        axs[1].set_ylabel('Velocity')
        axs[1].grid(True)
        
        # Control input plot
        if history_u.size > 0:
            time_axis_u = np.arange(history_u.shape[0]) * MPCConfig.DT
            axs[2].step(time_axis_u, history_u[:, 0], where='post', label='Rod1 Torque X')
            axs[2].step(time_axis_u, history_u[:, 1], where='post', label='Rod1 Torque Y')
            axs[2].step(time_axis_u, history_u[:, 2], where='post', label='Rod2 Torque X')
            axs[2].step(time_axis_u, history_u[:, 3], where='post', label='Rod2 Torque Y')
        axs[2].set_ylabel('Torque Input')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_title('MPC Control Inputs')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        plot_path = f'results/mpc_casadi_sim_order_{self.nn_approximation_order}.png'
        plt.savefig(plot_path)
        print(f"\nMPC trajectory plot saved as '{plot_path}'.")
    
    def print_timing_stats(self, start, end, sim_times, n_steps):
        """Print timing statistics"""
        total_sim_time = sum(sim_times)
        mpc_time = end - start - total_sim_time
        print(f"\nSimulated {MPCConfig.SIM_TIME:.1f}s in {end - start:.2f} seconds")
        print(f"Total simulation time: {total_sim_time:.2f} seconds.")
        print(f"Avg simulation time per step: {1000 * total_sim_time / n_steps:.2f} ms.")
        print(f"Total MPC time: {mpc_time:.2f} seconds")
        print(f"Avg MPC time per step: {1000 * mpc_time / n_steps:.2f} ms.")
    
def run_mpc_simulation(mode = 'spr', nn_approximation_order=1):
    """Main function to run MPC simulation"""
    # Initialize MPC controller
    mpc = MPCController(nn_approximation_order=nn_approximation_order)
    
    # --- Phase 3: The Simulation Loop ---
    print("--- Starting MPC simulation ---")
    
    # Sample initial and target states from the simulation data
    sample = mpc.df[mpc.state_cols].dropna().sample(2, random_state=42)
    x_current = sample.iloc[0].values
    # x_target = sample.iloc[1].values
    x_target = np.array([0.01, -0.42, -0.62, 0.0, 0.0, 0.0])
    
    history_x, history_u = [x_current], []
    history_x_target = [x_target]
    n_steps = int(MPCConfig.SIM_TIME / MPCConfig.DT)
    sim_times = []
    
    if mode == 'tt':
        # Generate a reference trajectory of final_time/simulation_params['mpc_dt'] steps
        reference_trajectory = np.linspace(x_current, x_target, num=int(MPCConfig.SIM_TIME / MPCConfig.DT))
        ref_index = 0
        x_target = reference_trajectory[ref_index]

    start = time.time()
    for i in range(n_steps):

        if mode == 'tt':
            # Get next target state from the reference trajectory
            if ref_index < len(reference_trajectory):
                x_target = reference_trajectory[ref_index]
                ref_index += 1
            else:
                x_target = reference_trajectory[-1]

        # Get MPC control input
        u_mpc = mpc.step(x_target, x_current)
        if u_mpc is None:
            print(f"MPC failed at step {i}")
            break
            
        # Step the simulation by applying the control input to the model
        start_sim = time.time()
        x_current = mpc.simulate_system(x_current, u_mpc)
        sim_times.append(time.time() - start_sim)
        
        # Store history
        history_x.append(x_current)
        history_u.append(u_mpc)
        history_x_target.append(x_target)
        
        if i % 10 == 0 or i == 0:
            dist_to_target = np.linalg.norm(x_current[:3] - x_target[:3])
            print(f"Step {i+1}/{n_steps}, Pos. Distance to target: {dist_to_target:.4f}")
    
    end = time.time()
    
    # Print timing stats
    total_sim_time = sum(sim_times)
    mpc_time = end - start - total_sim_time
    print(f"\nSimulated {MPCConfig.SIM_TIME:.1f}s in {end - start:.2f} seconds")
    print(f"Total simulation time: {total_sim_time:.2f} seconds.")
    print(f"Avg simulation time per step: {1000 * total_sim_time / n_steps:.2f} ms.")
    print(f"Total MPC time: {mpc_time:.2f} seconds")
    print(f"Avg MPC time per step: {1000 * mpc_time / n_steps:.2f} ms.")
    
    # Plot results
    mpc.history_x = history_x  # Set for plotting
    mpc.history_u = history_u  # Set for plotting
    mpc.plot_results(history_x_target=history_x_target)

if __name__ == "__main__":
    mode = 'spr' # set point regulation
    # mode = 'tt' # trajectory tracking
    run_mpc_simulation(mode=mode, nn_approximation_order=1)