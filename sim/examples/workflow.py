import elastica as ea
import numpy as np

class SystemSimulator(
    ea.BaseSystemCollection,
    ea.Constraints, # Enabled to use boundary conditions 'OneEndFixedBC'
    ea.Forcing,     # Enabled to use forcing 'GravityForces'
    ea.Connections, # Enabled to use FixedJoint
    ea.CallBacks,   # Enabled to use callback
    ea.Damping,     # Enabled to use damping models on systems.
):
    pass

# Instantiate the simulator
sim = SystemSimulator()

# Create rod
direction = np.array([-1.0, 0.0, 0.0])
normal = np.array([0.0, 1.0, 0.0])
rod1 = ea.CosseratRod.straight_rod(
    n_elements=50,                                # number of elements
    start=np.array([1.0, 0.0, 0.0]),             # Starting position of first node in rod
    direction=direction,                          # Direction the rod extends
    normal=normal,                                # normal vector of rod
    base_length=0.5,                              # original length of rod (m)
    base_radius=10e-2,                            # original radius of rod (m)
    density=1e3,                                  # density of rod (kg/m^3)
    youngs_modulus=1e7,                           # Elastic Modulus (Pa)
    shear_modulus=1e7/(2* (1+0.5)),               # Shear Modulus (Pa)
)
rod2 = ea.CosseratRod.straight_rod(
    n_elements=50,                                # number of elements
    start=np.array([0.5, 0.0, 0.0]),              # Starting position of first node in rod
    direction=direction,                          # Direction the rod extends
    normal=normal,                                # normal vector of rod
    base_length=0.5,                              # original length of rod (m)
    base_radius=10e-2,                            # original radius of rod (m)
    density=1e3,                                  # density of rod (kg/m^3)
    youngs_modulus=1e7,                           # Elastic Modulus (Pa)
    shear_modulus=1e7/(2* (1+0.5)),               # Shear Modulus (Pa)
)


# Add rod to sim
sim.append(rod1)
sim.append(rod2)

# Add boundary condition
sim.constrain(rod1).using(
    ea.OneEndFixedBC,                  # Displacement BC being applied
    constrained_position_idx=(0,),  # Node number to apply BC
    constrained_director_idx=(0,)   # Element number to apply BC
)

nu = 1e-3   # Damping constant of the rod
dt = 1e-5   # Time-step of simulation in seconds

sim.dampen(rod1).using(
    ea.AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)

sim.dampen(rod2).using(
    ea.AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)

# Connect rod 1 and rod 2. '_connect_idx' specifies the node number that
# the connection should be applied to. You are specifying the index of a
# list so you can use -1 to access the last node.
sim.connect(
    first_rod  = rod1,
    second_rod = rod2,
    first_connect_idx  = -1, # Connect to the last node of the first rod.
    second_connect_idx =  0  # Connect to first node of the second rod.
    ).using(
        ea.FixedJoint,  # Type of connection between rods
        k  = 1e5,    # Spring constant of force holding rods together (F = k*x)
        nu = 0,      # Energy dissipation of joint
        kt = 5e3     # Rotational stiffness of rod to avoid rods twisting
        )



# MyCallBack class is derived from the base call back class.
class MyCallBack(ea.CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params):
        ea.CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    # This function is called every time step
    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            # Save time, step number, position, orientation and velocity
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position" ].append(system.position_collection.copy())
            self.callback_params["directors"].append(system.director_collection.copy())
            self.callback_params["velocity" ].append(system.velocity_collection.copy())
            return

# Create dictionary to hold data from callback function
callback_data_rod1, callback_data_rod2 = ea.defaultdict(list), ea.defaultdict(list)

# Add MyCallBack to sim for each rod telling it how often to save data (step_skip)
sim.collect_diagnostics(rod1).using(
    MyCallBack, step_skip=1000, callback_params=callback_data_rod1)
sim.collect_diagnostics(rod2).using(
    MyCallBack, step_skip=1000, callback_params=callback_data_rod2)

sim.finalize()

timestepper = ea.PositionVerlet()
final_time = 1   # seconds
total_steps = int(final_time / dt)
ea.integrate(timestepper, sim, final_time, total_steps)
# Save data for rendering
import pickle
save_data = {
    "time": callback_data_rod1["time"],
    "position": callback_data_rod1["position"],
}
with open("workflow_simulation.dat", "wb") as f:
    pickle.dump(save_data, f)