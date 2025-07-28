import torch
from torch.func import vmap, jacrev
import numpy as np

from config import TrainingConfig
from model.train_example import NeuralNetwork
from system_x_dot.system_optimizer import SystemOptimizer

class SystemMPC:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"SystemMPC is using device: {self.device}")
        
        # Load model and normalization stats
        payload = torch.load(TrainingConfig.EX_MODEL_SAVE_PATH, map_location=self.device)
        self.model = NeuralNetwork().to(self.device)
        self.model.load_state_dict(payload['state_dict'])
        self.model.eval()

        self.input_mean = payload['input_mean'].to(self.device)
        self.input_std = payload['input_std'].to(self.device)
        self.output_mean = payload['output_mean'].to(self.device)
        self.output_std = payload['output_std'].to(self.device)
        
        self.optimizer = SystemOptimizer()
        self.N = self.optimizer.N
        self.nx = self.optimizer.nx
        self.nu = self.optimizer.nu

        self.x_prev_solution = np.zeros((self.N + 1, self.nx))
        self.u_prev_solution = np.zeros((self.N, self.nu))
        
    def _predict_and_denormalize(self, xu_tensor):
        """ Handles prediction and denormalization for a batch of inputs. """
        norm_input = (xu_tensor - self.input_mean) / self.input_std
        norm_output = self.model(norm_input)
        return (norm_output * self.output_std) + self.output_mean

    def solve(self, current_state, x_ref_trajectory):
        # Set initial state constraint
        self.optimizer.solver.set(0, 'lbx', current_state)
        self.optimizer.solver.set(0, 'ubx', current_state)

        # Use the previous solution as the linearization trajectory
        x_lin_traj = self.x_prev_solution
        u_lin_traj = self.u_prev_solution

        # --- PARALLELIZED JACOBIAN CALCULATION (as per paper's methodology) ---
        # This section replaces the sequential for-loop with a batched computation
        # using torch.func.vmap for significant performance gains.

        # 1. Create batches from the linearization trajectory
        x_lin_batch = torch.from_numpy(x_lin_traj[:-1]).float().to(self.device) # Shape: (N, nx)
        u_lin_batch = torch.from_numpy(u_lin_traj).float().to(self.device)       # Shape: (N, nu)
        
        with torch.no_grad():
            # 2. Get model predictions for the entire batch in one go.
            xu_lin_batch = torch.cat([x_lin_batch, u_lin_batch], dim=1)
            f_d_pred_batch = self._predict_and_denormalize(xu_lin_batch).cpu().numpy() # Shape: (N, nx)

            # 3. Define a function to compute Jacobians for a *single* sample (x, u).
            #    This is the function that `vmap` will parallelize.
            def get_jacobians_for_sample(x_sample, u_sample):
                # Helper functions that accept only the variable we're differentiating with respect to.
                # `jacrev` requires the function to have a single tensor input.
                def dynamics_wrt_x(x_in):
                    xu_in = torch.cat([x_in, u_sample], dim=0).unsqueeze(0)
                    return self._predict_and_denormalize(xu_in).squeeze(0)

                def dynamics_wrt_u(u_in):
                    xu_in = torch.cat([x_sample, u_in], dim=0).unsqueeze(0)
                    return self._predict_and_denormalize(xu_in).squeeze(0)

                # Compute Jacobians for the single sample
                Jx = jacrev(dynamics_wrt_x)(x_sample)
                Ju = jacrev(dynamics_wrt_u)(u_sample)
                return Jx, Ju

            # 4. Use vmap to apply `get_jacobians_for_sample` across the batch dimension.
            #    This computes all Jacobians for the horizon in parallel.
            Jxs, Jus = vmap(get_jacobians_for_sample)(x_lin_batch, u_lin_batch)
            
            # Move results to CPU as numpy arrays
            A_batch = Jxs.cpu().numpy() # Shape: (N, nx, nx)
            B_batch = Jus.cpu().numpy() # Shape: (N, nx, nu)

        # --- End of Parallelized Section ---
        
        # Loop to set parameters in the acados solver for each stage k
        for k in range(self.N):
            # Retrieve pre-computed matrices for stage k
            A_k = A_batch[k]
            B_k = B_batch[k]
            f_d_pred_k = f_d_pred_batch[k]
            
            x_k = x_lin_traj[k]
            u_k = u_lin_traj[k]

            # Calculate the offset term 'c' for the affine model: x_dot = A*x + B*u + c
            # c = f(x_lin, u_lin) - A*x_lin - B*u_lin
            c_k = f_d_pred_k - A_k @ x_k - B_k @ u_k
            
            # Pack parameters [A_flat, B_flat, c] for the solver
            # Use 'F' (Fortran) order for flattening to match acados C-style memory layout
            params_k = np.concatenate([
                A_k.flatten(order='F'),
                B_k.flatten(order='F'),
                c_k
            ])
            self.optimizer.solver.set(k, 'p', params_k)

            # Set references for the cost function at stage k
            y_ref_k = np.concatenate([x_ref_trajectory[k], np.zeros(self.nu)])
            self.optimizer.solver.set(k, 'yref', y_ref_k)

        # Set terminal cost reference
        y_ref_e = x_ref_trajectory[self.N]
        self.optimizer.solver.set(self.N, 'yref', y_ref_e)

        # Warm start the solver with the previous solution
        for k in range(self.N):
            self.optimizer.solver.set(k, 'x', self.x_prev_solution[k])
            self.optimizer.solver.set(k, 'u', self.u_prev_solution[k])
        self.optimizer.solver.set(self.N, 'x', self.x_prev_solution[self.N])

        # Solve the OCP
        status = self.optimizer.solver.solve()
        if status != 0:
            print(f"Warning: Acados solver returned status {status} at time step.")

        # Store the new solution to be used for the next iteration's warm start
        for k in range(self.N):
            self.x_prev_solution[k] = self.optimizer.solver.get(k, 'x')
            self.u_prev_solution[k] = self.optimizer.solver.get(k, 'u')
        self.x_prev_solution[self.N] = self.optimizer.solver.get(self.N, 'x')

        # Return the first control input in the optimal sequence
        return self.u_prev_solution[0]