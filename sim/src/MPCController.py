# mpc_controller.py
import importlib
import sys
import os
import numpy as np
import torch
from torch.func import vmap, jacrev
import casadi as ca
import joblib
import pandas as pd
from scipy.linalg import solve_discrete_are

# Get the absolute path to train_sim.py
current_dir = os.path.dirname(os.path.abspath(__file__))  # sim/src/
project_root = os.path.dirname(os.path.dirname(current_dir))  # sorolearn/
nmpc_path = os.path.join(project_root, 'generic-neural-mpc', 'mpc_casadi_sim.py')
train_sim_path = os.path.join(project_root, 'generic-neural-mpc', 'model', 'train_sim.py')

# Import train_sim module directly
spec = importlib.util.spec_from_file_location("train_sim", train_sim_path)
train_sim_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_sim_module)

# Import mpc_casadi_sim
spec_2 = importlib.util.spec_from_file_location("mpc_casadi_sim", nmpc_path)
mpc_casadi_sim_module = importlib.util.module_from_spec(spec_2)
spec_2.loader.exec_module(mpc_casadi_sim_module)

# Extract the classes we need
StatePredictor = train_sim_module.StatePredictor
TrainConfig = train_sim_module.TrainingConfig
MPCConfig = mpc_casadi_sim_module.MPCConfig
print(TrainConfig.MODEL_PATH)


class MPCController:
    """
    Real-time MPC controller for continuum robot control during simulation.
    """
    
    def __init__(self, config=None):
        self.config = config if config else self._default_config()
        
        # Load model and scalers
        self._load_model_assets()
        
        # Setup optimization problem
        self._setup_optimization_problem()
        
        # Initialize state tracking
        self.last_u_optimal = np.zeros(self.n_controls)
        self.u_guess = np.zeros((self.n_controls, self.config['N']))
        self.initialized = False
        
        print(f"MPC Controller initialized with horizon N={self.config['N']}")
    
    def _default_config(self):
        """Default MPC configuration"""
        return {
            'MODEL_PATH': TrainConfig.MODEL_PATH,
            'INPUT_SCALER_PATH': TrainConfig.INPUT_SCALER_PATH,
            'OUTPUT_SCALER_PATH': TrainConfig.OUTPUT_SCALER_PATH,
            'N': MPCConfig.N,
            'DT': MPCConfig.DT,
            'q_pos': MPCConfig.q_pos,
            'q_vel': MPCConfig.q_vel,
            'r_diag': MPCConfig.r_diag,
            'r_rate_diag': MPCConfig.r_rate_diag,
            'terminal_weight': MPCConfig.LAMBDA,
            'max_torque': MPCConfig.max_torque,
            'solver_tolerance': 1e-3,
            'solver_max_iter': 100
        }
        
    def _load_model_assets(self):
        """Load the trained neural network model and scalers"""
        try:
            # Load and JIT compile the model
            self.model = StatePredictor(input_dim=11, output_dim=6)
            self.model.load_state_dict(torch.load(self.config['MODEL_PATH']))
            self.model.eval()
            
            # JIT compile for speed
            dummy_input = torch.randn(1, 11)
            self.model = torch.jit.trace(self.model, dummy_input)
            
            # Load scalers
            self.input_scaler = joblib.load(self.config['INPUT_SCALER_PATH'])
            self.output_scaler = joblib.load(self.config['OUTPUT_SCALER_PATH'])
            
            # Convert scaler parameters to tensors
            self.input_scale = torch.tensor(self.input_scaler.scale_, dtype=torch.float32)
            self.input_mean = torch.tensor(self.input_scaler.mean_, dtype=torch.float32)
            self.output_scale = torch.tensor(self.output_scaler.scale_, dtype=torch.float32)
            self.output_mean = torch.tensor(self.output_scaler.mean_, dtype=torch.float32)
            
            print("MPC model assets loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MPC model assets: {e}")
    
    def _setup_optimization_problem(self):
        """Setup the CasADi optimization problem"""
        self.n_states = 6  # tip position (3) + tip velocity (3)
        self.n_controls = 4  # torques for 2 rods (2 each)
        
        # Create optimizer
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(self.n_states, self.config['N'] + 1)  # States
        self.U = self.opti.variable(self.n_controls, self.config['N'])     # Controls
        
        # Parameters
        self.x0 = self.opti.parameter(self.n_states, 1)        # Initial state
        self.x_ref = self.opti.parameter(self.n_states, 1)     # Reference state
        self.u_prev = self.opti.parameter(self.n_controls, 1)  # Previous control
        
        # Linearization parameters
        self.A_params = [self.opti.parameter(self.n_states, self.n_states) 
                        for _ in range(self.config['N'])]
        self.B_params = [self.opti.parameter(self.n_states, self.n_controls) 
                        for _ in range(self.config['N'])]
        self.C_params = [self.opti.parameter(self.n_states, 1) 
                        for _ in range(self.config['N'])]
        
        # Terminal cost matrix parameter
        self.P_terminal = self.opti.parameter(self.n_states, self.n_states)
        
        # Setup cost function
        self._setup_cost_function()
        
        # Setup constraints
        self._setup_constraints()
        
        # Configure solver
        solver_opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes',
            'ipopt.acceptable_tol': self.config['solver_tolerance'],
            'ipopt.max_iter': self.config['solver_max_iter']
        }
        self.opti.solver('ipopt', solver_opts)
        
        print("MPC optimization problem setup complete")
    
    def _setup_cost_function(self):
        """Setup the MPC cost function with terminal cost"""
        # Weight matrices
        Q_diag = [self.config['q_pos']] * 3 + [self.config['q_vel']] * 3
        Q = ca.diag(Q_diag)
        R = ca.diag([self.config['r_diag']] * self.n_controls)
        R_rate = ca.diag([self.config['r_rate_diag']] * self.n_controls)
        
        cost = 0
        
        # Stage costs
        for k in range(self.config['N']):
            # State tracking cost
            cost += (self.X[:, k] - self.x_ref).T @ Q @ (self.X[:, k] - self.x_ref)
            # Control effort cost
            cost += self.U[:, k].T @ R @ self.U[:, k]
            # Control rate cost
            if k == 0:
                cost += (self.U[:, k] - self.u_prev).T @ R_rate @ (self.U[:, k] - self.u_prev)
            else:
                cost += (self.U[:, k] - self.U[:, k-1]).T @ R_rate @ (self.U[:, k] - self.U[:, k-1])
        
        # Terminal cost: Vf(x̃_N) = x̃_N^T * P * x̃_N, where x̃_N = X[:, N] - x_ref
        x_terminal_error = self.X[:, self.config['N']] - self.x_ref
        cost += x_terminal_error.T @ self.P_terminal @ x_terminal_error
        
        self.opti.minimize(cost)
    
    def _setup_constraints(self):
        """Setup MPC constraints"""
        # Initial condition constraint
        self.opti.subject_to(self.X[:, 0] == self.x0)
        
        # Dynamics constraints (linearized)
        for k in range(self.config['N']):
            self.opti.subject_to(
                self.X[:, k+1] == self.A_params[k] @ self.X[:, k] + 
                                  self.B_params[k] @ self.U[:, k] + 
                                  self.C_params[k]
            )
        
        # Control input bounds
        u_min = [-self.config['max_torque']] * self.n_controls
        u_max = [self.config['max_torque']] * self.n_controls
        
        for k in range(self.config['N']):
            self.opti.subject_to(self.opti.bounded(u_min, self.U[:, k], u_max))
    
    def _compute_terminal_cost_matrix(self, A, B, Q, R):
        """
        Compute the terminal cost matrix P by solving the discrete algebraic Riccati equation (DARE).
        
        Args:
            A: State transition matrix (n_states x n_states)
            B: Control input matrix (n_states x n_controls)
            Q: State cost matrix (n_states x n_states)
            R: Control cost matrix (n_controls x n_controls)
        
        Returns:
            P: Terminal cost matrix (n_states x n_states)
            K: Optimal feedback gain matrix (n_controls x n_states)
        """
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
    
    def _full_pytorch_model(self, x_and_u_torch):
        """Neural network model with scaling"""
        scaled_input = (x_and_u_torch - self.input_mean) / self.input_scale
        scaled_output = self.model(scaled_input)
        return scaled_output * self.output_scale + self.output_mean
    
    def _get_batch_predictions_and_jacobians(self, x_traj_np, u_traj_np):
        """Get batched predictions and Jacobians using vmap"""
        def get_jac_for_sample(x_and_u_sample):
            return jacrev(self._full_pytorch_model)(x_and_u_sample)
        
        # Combine state and control trajectories
        dt_col = np.full((u_traj_np.shape[0], 1), self.config['DT'])
        x_and_u_traj_np = np.hstack([u_traj_np, x_traj_np, dt_col])
        x_and_u_torch = torch.tensor(x_and_u_traj_np, dtype=torch.float32)
        
        # Use vmap for parallel Jacobian computation
        J_batch = vmap(get_jac_for_sample)(x_and_u_torch)
        
        # Get forward predictions
        with torch.no_grad():
            y_pred_batch = self._full_pytorch_model(x_and_u_torch)
        
        return y_pred_batch.detach().numpy(), J_batch.detach().numpy()
    
    def _get_tip_state(self, rods_list):
        """Extract tip position and velocity from elastica rods"""
        # For double rod system, use the tip of the second rod
        rod = rods_list[-1]  # Last rod (tip rod)
        
        # Tip position (last element)
        tip_pos = rod.position_collection[:, -1]
        
        # Tip velocity (last element)  
        tip_vel = rod.velocity_collection[:, -1]
        
        return np.concatenate([tip_pos, tip_vel])
    
    def compute_control(self, rods_list, target_state):
        """
        Compute MPC control input given current robot state and target.
        
        Args:
            rods_list: List of elastica rods
            target_state: Target tip state [x, y, z, vx, vy, vz]
            
        Returns:
            tuple: (rod1_torque, rod2_torque) as numpy arrays
        """
        try:
            # Get current state
            current_state = self._get_tip_state(rods_list)
            
            if not self.initialized:
                print(f"MPC: Initial state = {current_state}")
                print(f"MPC: Target state = {target_state}")
                self.initialized = True
            
            # Generate nominal trajectory using current guess
            x_guess_np = np.zeros((self.config['N'], self.n_states))
            x_guess_np[0, :] = current_state
            
            # Roll out nominal trajectory
            for k in range(self.config['N'] - 1):
                model_input_k = np.concatenate([
                    self.u_guess[:, k], 
                    x_guess_np[k, :], 
                    [self.config['DT']]
                ])
                
                with torch.no_grad():
                    model_input_torch = torch.from_numpy(model_input_k).float()
                    x_guess_np[k+1, :] = self._full_pytorch_model(model_input_torch).numpy()
            
            # Get linearization
            y_pred_batch, J_batch = self._get_batch_predictions_and_jacobians(
                x_guess_np, self.u_guess.T
            )
            
            # Set linearization parameters
            for k in range(self.config['N']):
                J_k = J_batch[k]
                B_k = J_k[:, :self.n_controls]  # Control Jacobian
                A_k = J_k[:, self.n_controls:self.n_controls+self.n_states]  # State Jacobian
                C_k = y_pred_batch[k] - A_k @ x_guess_np[k] - B_k @ self.u_guess[:, k]
                
                self.opti.set_value(self.A_params[k], A_k)
                self.opti.set_value(self.B_params[k], B_k)
                self.opti.set_value(self.C_params[k], C_k.reshape(-1, 1))
            
            # Compute terminal cost matrix using linearization at final horizon step
            A_terminal = J_batch[-1][:, self.n_controls:self.n_controls+self.n_states]
            B_terminal = J_batch[-1][:, :self.n_controls]
            
            # Weight matrices for DARE
            Q_diag = [self.config['q_pos']] * 3 + [self.config['q_vel']] * 3
            Q_np = np.diag(Q_diag)
            R_np = np.diag([self.config['r_diag']] * self.n_controls)
            
            # Compute terminal cost matrix P by solving DARE
            P_matrix, _ = self._compute_terminal_cost_matrix(A_terminal, B_terminal, Q_np, R_np)
            
            # Apply terminal weight
            P_matrix *= self.config['terminal_weight']
            self.opti.set_value(self.P_terminal, P_matrix)
            
            # Set current values
            self.opti.set_value(self.x0, current_state)
            self.opti.set_value(self.x_ref, target_state)
            self.opti.set_value(self.u_prev, self.last_u_optimal)
            
            # Set initial guess
            self.opti.set_initial(self.U, self.u_guess)
            self.opti.set_initial(self.X, np.hstack([
                current_state.reshape(-1, 1), 
                x_guess_np.T
            ]))
            
            # Solve
            sol = self.opti.solve()
            u_optimal_all = sol.value(self.U)
            
            # Extract first control input
            self.last_u_optimal = u_optimal_all[:, 0]
            
            # Update guess for next iteration (warm start)
            self.u_guess = np.roll(u_optimal_all, -1, axis=1)
            self.u_guess[:, -1] = self.last_u_optimal
            
            # Split control into rod torques
            # Assuming: [rod1_torque_x, rod1_torque_y, rod2_torque_x, rod2_torque_y]
            rod1_torque = np.array([self.last_u_optimal[0], self.last_u_optimal[1], 0.0])
            rod2_torque = np.array([self.last_u_optimal[2], self.last_u_optimal[3], 0.0])
            
            return rod1_torque, rod2_torque
            
        except Exception as e:
            print(f"MPC solve failed: {e}")
            # Return zero torques as fallback
            return np.zeros(3), np.zeros(3)
    
    def set_target(self, target_position, target_velocity=None):
        """
        Set MPC target state.
        
        Args:
            target_position: Target tip position [x, y, z]
            target_velocity: Target tip velocity [vx, vy, vz] (optional, defaults to zero)
        """
        if target_velocity is None:
            target_velocity = np.zeros(3)
        
        return np.concatenate([target_position, target_velocity])
    
    def reset(self):
        """Reset MPC controller state"""
        self.last_u_optimal = np.zeros(self.n_controls)
        self.u_guess = np.zeros((self.n_controls, self.config['N']))
        self.initialized = False
        print("MPC controller reset")