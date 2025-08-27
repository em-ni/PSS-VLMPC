# mpc_casadi_sim.py
import sys
import time
import pandas as pd
import numpy as np
import torch
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
    q_pos = 10.0 #10.0
    q_vel = 0.0 #0.0
    Q_diag = [q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]
    r_diag = 100.0 #100.0
    R_diag = [r_diag, r_diag, r_diag, r_diag]  
    r_rate_diag = 150000.0 #150000.0
    R_rate_diag = [r_rate_diag, r_rate_diag, r_rate_diag, r_rate_diag] 
    LAMBDA = 50.0 #50
    max_torque = 9e-2
    U_MIN = [-max_torque, -max_torque, -max_torque, -max_torque]
    U_MAX = [max_torque, max_torque, max_torque, max_torque]
    # Workspace box constraints: x, y, z, x_dot, y_dot, z_dot
    X_MAX = [0.6, 0.6, 0.8, 0.2, 0.2, 0.2]
    X_MIN = [-0.6, -0.6, 0.0, -0.2, -0.2, -0.2]


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
        This is done in find_stabilizing_lambda.
        """
        # Placeholder for stability check logic
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
        self.LAMBDA = MPCConfig.LAMBDA

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
            
            # try:
            #     self.model.compile()
            #     print("Model successfully compiled")
            # except Exception as e:
            #     print(f"Warning: torch.compile failed ({e}), using standard model")
            
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

    def find_stabilizing_lambda(self, A_terminal, B_terminal, P_matrix, K_matrix, mu=1e-4, safety=1.1, x_ref=None, u_ref=None):
        """
        Compute a robust lambda following the derivation discussed in the thread.
        Args:
            A_terminal (np.ndarray): nominal A (n_states x n_states) at terminal step
            B_terminal (np.ndarray): nominal B (n_states x n_controls) at terminal step
            P_matrix (np.ndarray): terminal cost matrix (n_states x n_states), assumed PD
            K_matrix (np.ndarray): terminal feedback gain matrix (n_controls x n_states)
            mu (float): uniform bound on model prediction/state-map error
            x_ref (np.ndarray or None): reference state (n_states,). If None assumed zero
            u_ref (np.ndarray or None): reference input (n_controls,). If None assumed zero
        Returns:
            float: computed robust lambda (np.inf if infeasible / denominator <= 0)
        """
        # short-hands / safety
        N = MPCConfig.N
        Q = self.Q_np
        R = self.R_np
        n_x = self.n_states
        n_u = self.n_controls

        # defaults for refs
        if x_ref is None:
            x_ref = np.zeros(n_x)
        if u_ref is None:
            u_ref = np.zeros(n_u)

        # Compute radii R_e and R_v from box constraints in MPCConfig ---
        # produce list of corner points for X box and compute max ||x - x_ref||
        X_min = np.array(MPCConfig.X_MIN)
        X_max = np.array(MPCConfig.X_MAX)
        U_min = np.array(MPCConfig.U_MIN)
        U_max = np.array(MPCConfig.U_MAX)

        # enumerate corners for states
        corners = []
        for mask in range(1 << n_x):
            corner = np.zeros(n_x)
            for i in range(n_x):
                corner[i] = X_max[i] if ((mask >> i) & 1) else X_min[i]
            corners.append(corner)
        corners = np.array(corners)
        # Max distance from x_ref to corners
        R_e = float(np.max(np.linalg.norm(corners - x_ref.reshape(1, -1), axis=1)))

        # enumerate corners for inputs
        corners_u = []
        for mask in range(1 << n_u):
            corner = np.zeros(n_u)
            for i in range(n_u):
                corner[i] = U_max[i] if ((mask >> i) & 1) else U_min[i]
            corners_u.append(corner)
        corners_u = np.array(corners_u)
        # Max distance from u_ref to corners
        R_v = float(np.max(np.linalg.norm(corners_u - u_ref.reshape(1, -1), axis=1)))

        # Max eigenvalues
        q_eigs = np.linalg.eigvalsh(Q)
        qmax = float(np.max(q_eigs))

        r_eigs = np.linalg.eigvalsh(R)
        rmax = float(np.max(r_eigs))

        t_eigs = np.linalg.eigvalsh(P_matrix)
        tmax = float(np.max(t_eigs))
        
        # Max cost values
        l_max = R_e * R_e * qmax + R_v * R_v * rmax
        V_max = R_e * R_e * tmax
        D = l_max
        L_N = N * D
        alpha = V_max

        # Lipschitz linear constants
        L_f = float(np.linalg.norm(A_terminal, 2))
        L_l = 2.0 * qmax * R_e
        L_V = 2.0 * tmax * R_e

        # Delta terms
        Delta_L_N = L_l * mu * (L_f ** N - 1)/(L_f - 1)
        Delta_V = L_V * L_f**N * mu

        # Find rho from linearized closed loop system to bound the contraction of the terminal controller around the ref
        # rho = lambda_max( P^{-1/2} Acl^P P Acl P^{-1/2} ) with Acl = A + B K
        Acl = A_terminal + B_terminal @ K_matrix
        # get P^{-1/2} via cholesky: P = L L^T -> P^{-1/2} = inv(L.T)
        Lchol = np.linalg.cholesky(P_matrix)  # P = L L^T
        P_inv_sqrt = np.linalg.inv(Lchol.T)
        M_rho = P_inv_sqrt @ (Acl.T @ (P_matrix @ Acl)) @ P_inv_sqrt
        try:
            rho = float(np.max(np.real(np.linalg.eigvals(M_rho))))
        except Exception as e:
            print(f"Error computing rho: {e}")
            rho = 0.9

        # This can be done smartly, higher d -> lower lambda, worst case is d tends 0
        d = 0.001

        # Feasibility check
        denom = (1.0 - rho) * alpha - Delta_V
        if denom <= 0:
            print(f"Warning: Infeasible robust lambda (denominator = {denom:.6f} <= 0)")
            print(f"  rho = {rho:.6f}, alpha = {alpha:.6f}, Delta_V = {Delta_V:.6f}")
            print(f"  Consider relaxing constraints or adjusting tuning parameters")
            return float('inf')  # infeasible: no finite lambda under worst-case assumptions

        # Compute robust lambda
        lambda_robust = (L_N + Delta_L_N - N*d) / denom

        # Add safety margin
        lambda_robust *= safety
        lambda_robust = max(lambda_robust, 0.0)
        
        # Safety check for extremely large values
        if lambda_robust > 1e6:
            print(f"Warning: Computed lambda ({lambda_robust:.2e}) is very large, may indicate numerical issues")
            
        return float(lambda_robust)


    def full_pytorch_model(self, x_and_u_torch):
        """Full PyTorch model with scaling"""
        scaled_input = (x_and_u_torch - self.input_mean) / self.input_scale
        scaled_output = self.model(scaled_input)
        return scaled_output * self.output_scale + self.output_mean
    
    def get_batch_predictions_and_derivatives(self, x_traj_np, u_traj_np):
        """Get predictions and derivatives using fully batched computation for maximum speed"""
        self.model.to('cpu').eval()

        # Combine state and control trajectories into a single input batch
        x_and_u_traj_np = np.hstack([u_traj_np, x_traj_np])
        
        # Convert to tensor with gradient computation enabled
        x_and_u_batch = torch.tensor(x_and_u_traj_np, dtype=torch.float32, requires_grad=True)
        
        # Define a function for the model to use with vmap
        def model_func(x_and_u_single):
            return self.full_pytorch_model(x_and_u_single.unsqueeze(0)).squeeze(0)

        # Vectorized forward pass
        y_pred_batch = torch.vmap(model_func)(x_and_u_batch)
        y_pred_batch = y_pred_batch.detach().numpy()
        try:
            if self.nn_approximation_order >= 1:
                # TODO: in place of clipping everytime maybe I can check if they explode 
                # Vectorized Jacobian
                jacobian_fn = torch.vmap(torch.func.jacrev(model_func))
                J_batch = jacobian_fn(x_and_u_batch)
                J_batch = torch.clamp(J_batch, -1e3, 1e3)  # Clip to prevent explosion
                J_batch = J_batch.detach().numpy()
                
                if self.nn_approximation_order == 2:
                    # Vectorized Hessian
                    hessian_fn = torch.vmap(torch.func.hessian(model_func))
                    H_batch = hessian_fn(x_and_u_batch)
                    H_batch = torch.clamp(H_batch, -1e3, 1e3) # Clip to prevent explosion
                    H_batch = H_batch.detach().numpy()
                else:
                    H_batch = None
            else:
                J_batch = None
                H_batch = None
                
        except Exception as e:
            print(f"Error during batch computation: {e}")

        return y_pred_batch, J_batch, H_batch

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
                x_guess_np[k+1, :] = x_next

        # Get predictions and derivatives based on approximation order
        y_pred_batch, J_batch, H_batch = self.get_batch_predictions_and_derivatives(x_guess_np, self.u_guess.T)
        
        # Set the parameters for the optimizer
        for k in range(MPCConfig.N):
            if self.nn_approximation_order >= 1:
                J_k = J_batch[k]
                B_k = J_k[:, :self.n_controls]  # First 4 columns for torque inputs
                A_k = J_k[:, self.n_controls:self.n_controls+self.n_states]  # Next 6 columns for states
                C_k = y_pred_batch[k] - A_k @ x_guess_np[k] - B_k @ self.u_guess[:, k]
                
                # Safety checks
                if np.isnan(A_k).any() or np.isinf(A_k).any():
                    print(f"Warning: NaN/Inf in A matrix at step {k}, using identity")
                    A_k = np.eye(self.n_states)
                elif np.abs(A_k).max() > 1e3:
                    print(f"Warning: Explosive values in A matrix at step {k}, clipping")
                    A_k = np.clip(A_k, -1e3, 1e3)       
                if np.isnan(B_k).any() or np.isinf(B_k).any():
                    print(f"Warning: NaN/Inf in B matrix at step {k}, using zero")
                    B_k = np.zeros_like(B_k)
                elif np.abs(B_k).max() > 1e3:
                    print(f"Warning: Explosive values in B matrix at step {k}, clipping")
                    B_k = np.clip(B_k, -1e3, 1e3)
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
        
        # Compute terminal cost matrix P by solving DARE with safety
        P_matrix, K_matrix = self.compute_terminal_cost_matrix(A_terminal, B_terminal, self.Q_np, self.R_np)

        # Find stabilizing lambda
        lambda_robust = self.find_stabilizing_lambda(A_terminal, B_terminal, P_matrix, K_matrix, mu=1e-3, x_ref=x_ref.flatten(), u_ref=None)
        lambda_robust = -1 # Temp use default value
        print(f"Computed robust lambda: {lambda_robust}")

        # Use robust lambda if finite, otherwise fall back to configured lambda
        if np.isfinite(lambda_robust) and lambda_robust > 0:
            self.LAMBDA = lambda_robust
        else:
            self.LAMBDA = MPCConfig.LAMBDA

        # Multiply by the chosen scaling factor
        P_matrix *= self.LAMBDA

        # Ensure P_matrix is finite and positive definite
        if not np.all(np.isfinite(P_matrix)):
            print("Warning: P_matrix contains non-finite values, using Q as fallback")
            P_matrix = self.Q_np * self.LAMBDA

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
        
def run_mpc_simulation(mode = 'spr', nn_approximation_order=1):
    """Main function to run MPC simulation"""
    # Initialize MPC controller
    mpc = MPCController(nn_approximation_order=nn_approximation_order)
    
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