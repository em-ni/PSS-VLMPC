from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import SX
import numpy as np

from model.train_example import SystemConfig

# --- MPC Configuration ---
class MPCConfig:
    T = 10.0  # Total simulation time (seconds)
    DT = 0.05  # Time step for simulation
    N_HORIZON = 20  # Number of steps in the prediction horizon
    T_HORIZON = 1.0  # Total time for the prediction horizon (seconds)
    
    # --- MPC TUNING: COST MATRICES ---

    # Q: State Cost Matrix. How much do we care about tracking error?
    # Penalize position error more than velocity error.
    Q_pos = 100.0  # Penalty on position error (x0, x1, x2)
    Q_vel = 1.0    # Lower penalty on velocity error (x3, x4, x5)
    Q = np.diag([Q_pos, Q_pos, Q_pos, Q_vel, Q_vel, Q_vel])

    # R: Control Cost Matrix. How much do we care about control effort?
    # This is the most important parameter for stability.
    # Start with a significantly higher value to prevent chattering.
    R_val = 0.5  # INCREASED SIGNIFICANTLY (was 0.01 or 0.1)
    R = np.diag([R_val] * SystemConfig.CONTROL_DIM)
    
    # Control constraints
    U_MIN = [-1.0] * SystemConfig.CONTROL_DIM
    U_MAX = [1.0] * SystemConfig.CONTROL_DIM

class SystemOptimizer:
    def __init__(self):
        self.ocp = AcadosOcp()
        self.nx = SystemConfig.STATE_DIM
        self.nu = SystemConfig.CONTROL_DIM
        self.N = MPCConfig.N_HORIZON
        
        self._configure_ocp()
        
        # Free the solver if it exists to allow re-creation
        if hasattr(self, 'solver') and self.solver is not None:
             self.solver.free()
             del self.solver

        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')

    def _configure_ocp(self):
        # Model
        self.ocp.model.name = 'generic_system_model'
        
        x = SX.sym('x', self.nx)
        u = SX.sym('u', self.nu)
        
        # Parameters: [A_flat, B_flat, c]
        p_size = self.nx * self.nx + self.nx * self.nu + self.nx
        p = SX.sym('p', p_size)
        
        A = p[:self.nx * self.nx].reshape((self.nx, self.nx))
        B = p[self.nx * self.nx : self.nx * self.nx + self.nx * self.nu].reshape((self.nx, self.nu))
        c = p[self.nx * self.nx + self.nx * self.nu:]

        # Dynamics: x_dot = A*x + B*u + c
        self.ocp.model.f_expl_expr = A @ x + B @ u + c
        self.ocp.model.x = x
        self.ocp.model.u = u
        self.ocp.model.p = p

        # Re-initialize `ocp.parameter_values`. It is mandatory for the
        # consistency check during code generation. Its shape must match `p`.
        self.ocp.parameter_values = np.zeros(p_size)

        # Cost
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        
        self.ocp.cost.W = np.block([[MPCConfig.Q, np.zeros((self.nx, self.nu))],[np.zeros((self.nu, self.nx)), MPCConfig.R]])
        self.ocp.cost.W_e = MPCConfig.Q
        
        self.ocp.cost.Vx = np.vstack([np.eye(self.nx), np.zeros((self.nu, self.nx))])
        self.ocp.cost.Vu = np.vstack([np.zeros((self.nx, self.nu)), np.eye(self.nu)])
        self.ocp.cost.Vx_e = np.eye(self.nx)
        
        self.ocp.cost.yref = np.zeros(self.nx + self.nu)
        self.ocp.cost.yref_e = np.zeros(self.nx)

        # Constraints
        self.ocp.constraints.lbu = np.array(MPCConfig.U_MIN)
        self.ocp.constraints.ubu = np.array(MPCConfig.U_MAX)
        self.ocp.constraints.idxbu = np.arange(self.nu)
        self.ocp.constraints.x0 = np.zeros(self.nx)

        # Solver options
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.N_horizon = self.N 
        self.ocp.solver_options.tf = MPCConfig.T_HORIZON