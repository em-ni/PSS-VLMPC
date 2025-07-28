import numpy as np

class GenericSystem:
    """
    Represents the generic system being controlled.
    The dynamics are defined by x_next = f(x, u, a), which is approximated by a neural network.
    """
    def __init__(self, initial_state):
        self.state = np.array(initial_state)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = np.array(state)

    def step(self, dynamics_func, u, dt):
        """
        Performs a single integration step. A single inference of the neural network.
        """
        self.state = dynamics_func(self.state, u, dt)
        return self.state