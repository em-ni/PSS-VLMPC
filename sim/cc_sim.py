import numpy as np
import elastica as ea
import matplotlib
import imageio_ffmpeg

# Elastica Imports
from elastica.timestepper import extend_stepper_interface, tqdm
from elastica._calculus import _isnan_check
from examples.MuscularSnake.post_processing import plot_video_with_surface
from elastica.boundary_conditions import (
    FixedConstraint,
)
from elastica.joint import get_relative_rotation_two_systems

# Set path for FFMPEG for saving video animations
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path


class ConstantCurvature(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
    ea.Connections,
):
    pass


class CCCallback(ea.CallBackBaseClass):
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

# Initialize the simulation
SAVE_RESULTS = True
double_rod = True
step_skip = 500 # Save results every step_skip steps
min_torque = 0
max_torque = 5e-2
cc_sim = ConstantCurvature()

# rods parameters
n_elem = 30
direction = np.array([0.0, 0.0, -1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 0.4
base_radius = 0.02
base_area = np.pi * base_radius ** 2
density = 1000
E = 1e5
poisson_ratio = 0.5
shear_modulus = E / (2.0 * (1.0 + poisson_ratio)) 
damping_constant = 1e-1
dt = 1e-4

# rod 1
start_rod_1 = np.array([0.0, 0.0, 0.0]) 
rod_1 = ea.CosseratRod.straight_rod(
    n_elem,
    start_rod_1,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)
cc_sim.append(rod_1)
cc_sim.constrain(rod_1).using(
    FixedConstraint,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
)
cc_sim.add_forcing_to(rod_1).using(
    # initial torque is zero
    ea.UniformTorques, torque=max_torque, direction=np.array([0, 1, 0])
)
cc_sim.dampen(rod_1).using(
    ea.dissipation.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)
pp_list_rod1 = ea.defaultdict(list)
cc_sim.collect_diagnostics(rod_1).using(
    CCCallback, step_skip=step_skip, callback_params=pp_list_rod1
)

if double_rod:
    # Start at the end of rod 1
    start_rod_2 = np.array([0.0, 0.0, -base_length])  
    rod_2 = ea.CosseratRod.straight_rod(
        n_elem,
        start_rod_2,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=E,
        shear_modulus=shear_modulus,
    )
    cc_sim.append(rod_2)
    cc_sim.dampen(rod_2).using(
        ea.dissipation.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=dt,
    )
    cc_sim.add_forcing_to(rod_2).using(
        # inital torque is zero
        ea.UniformTorques, torque=max_torque, direction=np.array([0, 1, 0])
    )
    pp_list_rod2 = ea.defaultdict(list)
    cc_sim.collect_diagnostics(rod_2).using(
        CCCallback, step_skip=step_skip, callback_params=pp_list_rod2
    )
    
    # Connect rod 2 to rod 1
    cc_sim.connect(
        first_rod  = rod_1,
        second_rod = rod_2,
        first_connect_idx  = -1, # Connect to the last node of the first rod.
        second_connect_idx =  0, # Connect to first node of the second rod.
        ).using(
            ea.FixedJoint,  # Type of connection between rods
            k  = 1e5,    # Spring constant of force holding rods together (F = k*x)
            nu = 0,      # Energy dissipation of joint
            kt = 1,    # Rotational stiffness of rod to avoid rods twisting
            nut= 0,      # Rotational damping of joint
            rest_rotation_matrix=get_relative_rotation_two_systems(rod_1, -1, rod_2, 0),
            )

# Finalize and Run Simulation
cc_sim.finalize()

# Get references to the torque objects from _ext_forces_torques
rod1_torque_obj = None
rod2_torque_obj = None

# The structure is: _ext_forces_torques contains tuples of (system_index, torque_object)
for system_idx, torque_obj in cc_sim._ext_forces_torques:
    if system_idx == 0:  # First system (rod_1)
        rod1_torque_obj = torque_obj
        print(f"Found Rod 1 torque object: {torque_obj.torque}")
    elif system_idx == 1:  # Second system (rod_2)
        rod2_torque_obj = torque_obj
        print(f"Found Rod 2 torque object: {torque_obj.torque}")

timestepper = ea.PositionVerlet()
final_time = 10
dl = base_length / n_elem
total_steps = int(final_time / dt)
print(f"Total steps: {total_steps}")
do_step, stages_and_updates = extend_stepper_interface(timestepper, cc_sim)

time = 0.0

for i in tqdm(range(total_steps)):
    current_time = i * dt
    
    # # DYNAMIC TORQUE CONTROL: Directly modify the torque attribute
    # if rod1_torque_obj is not None:
    #     if current_time < 2.0:
    #         # Initial phase - keep original torques
    #         rod1_torque_obj.torque = np.array([0.0, -2e-2, 0.0])
    #     elif current_time < 5.0:
    #         # Phase 2 - modify torques
    #         rod1_torque_obj.torque = np.array([0.0, -5e-2, 0.0])  # Increase rod 1 torque
    #     elif current_time < 8.0:
    #         # Phase 3 - reverse torques
    #         rod1_torque_obj.torque = np.array([0.0, 3e-2, 0.0])   # Reverse torque
    #     else:
    #         # Final phase - turn off torques
    #         rod1_torque_obj.torque = np.array([0.0, 0.0, 0.0])    # No torque
    
    # if double_rod and rod2_torque_obj is not None:
    #     if current_time < 2.0:
    #         # Initial phase
    #         rod2_torque_obj.torque = np.array([0.0, 7e-3, 0.0])
    #     elif current_time < 5.0:
    #         # Phase 2 - modify torques
    #         rod2_torque_obj.torque = np.array([0.0, 1e-2, 0.0])   # Modified torque
    #     elif current_time < 8.0:
    #         # Phase 3 - reverse torques
    #         rod2_torque_obj.torque = np.array([0.0, -8e-3, 0.0])  # Reverse torque
    #     else:
    #         # Final phase - turn off torques
    #         rod2_torque_obj.torque = np.array([0.0, 0.0, 0.0])    # No torque
    
    # Step the simulation
    time = do_step(timestepper, stages_and_updates, cc_sim, time, dt)
    
    # Check for numerical instability in either rod
    if _isnan_check(rod_1.position_collection):
        print("System broke")
        break
    if double_rod and _isnan_check(rod_2.position_collection):
        print("System broke")
        break

# Post-processing and Visualization
matplotlib.use('Qt5Agg')
post_processing_dict_list = [pp_list_rod1]
if double_rod:
    post_processing_dict_list.append(pp_list_rod2)

rendering_fps = 40
plot_video_with_surface(
    post_processing_dict_list, 
    video_name="cc_video_dynamic_control.mp4", 
    fps=rendering_fps,
    step=1,
    x_limits=(-1.5, 1.5),  
    y_limits=(-1.5, 1.5),
    z_limits=(-1.5, 1.5),
    dpi=100,
    vis3D=False,
    vis2D=True,
)

if SAVE_RESULTS:
    import pickle as pickle
    import os
    import shutil

    # Clean results folder
    if not os.path.exists("results"):
        os.makedirs("results")
    else:
        for file in os.listdir("results"):
            file_path = os.path.join("results", file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    # Save rod 1 data
    filename = "results/rod1.dat"
    with open(filename, "wb") as file:
        pickle.dump(pp_list_rod1, file)
    print(f"Saved rod 1 data to {filename}")

    # Save rod 2 data if it exists
    if double_rod:
        filename = "results/rod2.dat"
        with open(filename, "wb") as file:
            pickle.dump(pp_list_rod2, file)
        print(f"Saved rod 2 data to {filename}")