import elastica as ea

class DataCallback(ea.CallBackBaseClass):
    """
    Callback function to collect data from the simulation.
    """
    def __init__(self, step_skip: int, callback_params: dict):
        ea.CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["forces"].append(system.external_forces.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            self.callback_params["torques"].append(system.external_torques.copy())
            return