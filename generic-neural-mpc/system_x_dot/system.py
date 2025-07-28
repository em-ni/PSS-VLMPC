import numpy as np

class GenericSystem:
    """
    Represents the generic system being controlled.
    The dynamics are defined by x_dot = f(x, u), which is approximated by a neural network.
    """
    def __init__(self, initial_state):
        self.state = np.array(initial_state)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = np.array(state)

    def rk4_step(self, dynamics_func, u, dt):
        """
        Performs a single Runge-Kutta 4th order integration step.
        This is how the paper integrates the dynamics.
        """
        k1 = dynamics_func(self.state, u)
        k2 = dynamics_func(self.state + 0.5 * dt * k1, u)
        k3 = dynamics_func(self.state + 0.5 * dt * k2, u)
        k4 = dynamics_func(self.state + dt * k3, u)
        self.state = self.state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.state
    