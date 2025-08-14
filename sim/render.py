cc_povray_script = """ Rendering Script for CC simulation using POVray

This script reads simulation data file to render POVray animation movie.
The data file should contain a dictionary of position vectors and times.

The script supports multiple camera positions where a video is generated
for each camera view.

Notes
-----
    The module requires POVray installed.
"""

import multiprocessing
import os
from functools import partial
from multiprocessing import Pool
import shutil
import glob

import numpy as np
from scipy import interpolate
from tqdm import tqdm

# Import your povray macros and rendering utilities
from utils._povmacros import Stages, pyelastica_rod, pyelastica_sphere, render

# Setup (USER DEFINE)
DATA_PATH = "results"  # Path to the simulation data folder
SAVE_PICKLE = True

# Rendering Configuration (USER DEFINE)
OUTPUT_FILENAME = "render"
OUTPUT_IMAGES_DIR = "frames"
FPS = 30.0
WIDTH = 1920
HEIGHT = 1080
DISPLAY_FRAMES = "Off"  # ['On', 'Off']
ROTATE_VIDEO = False  # Rotate final video 180 degrees after stitching

# Delete all frames in the output directory if it exists
if os.path.exists(OUTPUT_IMAGES_DIR):
    for file in os.listdir(OUTPUT_IMAGES_DIR):
        file_path = os.path.join(OUTPUT_IMAGES_DIR, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}") 

# Camera/Light Configuration (USER DEFINE)
stages = Stages()
stages.add_camera(
    location=[0.6, 1.2, 0.6],  
    angle=0,
    look_at=[0.0, 0.0, -0.4],  
    name="diag",
    sky=[0.0, 0.0, 0.1]  
)
stages.add_light(
    position=[1500, 2500, 1000],
    color="White",
    camera_id=-1
)
stages.add_light(
    position=[2.0, 2.0, 2.0],
    color=[0.09, 0.09, 0.1],
    camera_id=0
)

stage_scripts = stages.generate_scripts()

# Externally Including Files (USER DEFINE)
included = ["utils/default.inc"]

# Multiprocessing Configuration (USER DEFINE)
MULTIPROCESSING = True
THREAD_PER_AGENT = 4
NUM_AGENT = multiprocessing.cpu_count() // 2

if __name__ == "__main__":
    # Load Data
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(f"Data folder '{DATA_PATH}' not found.")

    all_rod_data_files = glob.glob(os.path.join(DATA_PATH, "rod*.dat"))
    all_target_data_files = glob.glob(os.path.join(DATA_PATH, "target*.dat"))
    all_obstacle_data_files = glob.glob(os.path.join(DATA_PATH, "obstacle*.dat"))
    
    if not all_rod_data_files:
        raise FileNotFoundError(f"No rod*.dat files found in '{DATA_PATH}'")

    all_rod_data = []
    all_target_data = []
    all_obstacle_data = []
    max_runtime = 0.0

    print(f"Found {len(all_rod_data_files)} rod data files, {len(all_target_data_files)} target data files, and {len(all_obstacle_data_files)} obstacle data files in '{DATA_PATH}'. Loading all...")

    # Load rod data
    for data_file_path in all_rod_data_files:
        try:
            if SAVE_PICKLE:
                import pickle as pk
                with open(data_file_path, "rb") as fptr:
                    data = pk.load(fptr)
            else:
                raise NotImplementedError("Only pickled data is supported")
        except OSError as err:
            print(f"Cannot open the datafile {data_file_path}")
            print(str(err))
            raise
        
        all_rod_data.append(data)
        if "time" in data and len(data["time"]) > 0:
            max_runtime = max(max_runtime, np.array(data["time"]).max())
        else:
            print(f"Warning: Data file {data_file_path} has no 'time' key or empty time data.")
    
    # Load target data
    for data_file_path in all_target_data_files:
        try:
            if SAVE_PICKLE:
                import pickle as pk
                with open(data_file_path, "rb") as fptr:
                    data = pk.load(fptr)
            else:
                raise NotImplementedError("Only pickled data is supported")
        except OSError as err:
            print(f"Cannot open the datafile {data_file_path}")
            print(str(err))
            raise
        
        all_target_data.append(data)
        if "time" in data and len(data["time"]) > 0:
            max_runtime = max(max_runtime, np.array(data["time"]).max())
        else:
            print(f"Warning: Data file {data_file_path} has no 'time' key or empty time data.")
    
    # Load obstacle data
    for data_file_path in all_obstacle_data_files:
        try:
            if SAVE_PICKLE:
                import pickle as pk
                with open(data_file_path, "rb") as fptr:
                    data = pk.load(fptr)
            else:
                raise NotImplementedError("Only pickled data is supported")
        except OSError as err:
            print(f"Cannot open the datafile {data_file_path}")
            print(str(err))
            raise
        
        all_obstacle_data.append(data)
        if "time" in data and len(data["time"]) > 0:
            max_runtime = max(max_runtime, np.array(data["time"]).max())
        else:
            print(f"Warning: Data file {data_file_path} has no 'time' key or empty time data.")

    # Interpolate Data
    total_frame = int(max_runtime * FPS)
    if total_frame == 0:
        print("No simulation data with valid time found to render. Exiting.")
        exit()

    times_true = np.linspace(0, max_runtime, total_frame)

    # Interpolate rod data
    all_interpolated_xs = []
    all_base_radii_for_rods = []

    for data_entry in all_rod_data:
        current_times = np.array(data_entry["time"])
        current_xs = np.array(data_entry["position"])

        if len(current_times) == 0 or current_xs.shape[0] == 0:
            print(f"Skipping interpolation for a rod with empty time or position data.")
            all_interpolated_xs.append(np.zeros((total_frame, 3, 0)))
            all_base_radii_for_rods.append(np.zeros((total_frame, 0)))
            continue

        f_interp = interpolate.interp1d(current_times, current_xs, axis=0,
                                        fill_value=(current_xs[0], current_xs[-1]),
                                        bounds_error=False)
        interpolated_xs_for_rod = f_interp(times_true)
        all_interpolated_xs.append(interpolated_xs_for_rod)

        # Use the actual radius from the simulation data
        if "radius" in data_entry and len(data_entry["radius"]) > 0:
            current_radii = np.array(data_entry["radius"])
            f_radius_interp = interpolate.interp1d(current_times, current_radii, axis=0,
                                                   fill_value=(current_radii[0], current_radii[-1]),
                                                   bounds_error=False)
            interpolated_radii = f_radius_interp(times_true)
            all_base_radii_for_rods.append(interpolated_radii)
        else:
            # Fallback to constant radius
            n_elem_current_rod = current_xs.shape[2]
            base_radii_for_current_rod = np.ones((total_frame, n_elem_current_rod)) * 0.006
            all_base_radii_for_rods.append(base_radii_for_current_rod)

    # Interpolate target data
    all_interpolated_target_positions = []
    target_colors = ['red', 'blue', 'orange', 'purple']

    for i, data_entry in enumerate(all_target_data):
        current_times = np.array(data_entry["time"])
        current_positions = np.array(data_entry["position"])

        if len(current_times) == 0 or current_positions.shape[0] == 0:
            print(f"Skipping interpolation for target {i+1} with empty time or position data.")
            # Create static target at origin if no data
            static_position = np.zeros((total_frame, 3))
            all_interpolated_target_positions.append(static_position)
            continue

        # Target position is usually a single point (center of sphere)
        if current_positions.ndim == 2 and current_positions.shape[1] == 3:
            # Position data is [time, 3] format
            f_interp = interpolate.interp1d(current_times, current_positions, axis=0,
                                            fill_value=(current_positions[0], current_positions[-1]),
                                            bounds_error=False)
            interpolated_target_position = f_interp(times_true)
        elif current_positions.ndim == 3 and current_positions.shape[1] == 3:
            # Position data is [time, 3, 1] format (single point)
            positions_2d = current_positions[:, :, 0]  # Extract [time, 3]
            f_interp = interpolate.interp1d(current_times, positions_2d, axis=0,
                                            fill_value=(positions_2d[0], positions_2d[-1]),
                                            bounds_error=False)
            interpolated_target_position = f_interp(times_true)
        else:
            print(f"Warning: Unexpected target position data shape: {current_positions.shape}")
            # Create static target at first position
            first_pos = current_positions[0] if current_positions.ndim == 2 else current_positions[0, :, 0]
            interpolated_target_position = np.tile(first_pos, (total_frame, 1))

        all_interpolated_target_positions.append(interpolated_target_position)

    # Interpolate obstacle data
    all_interpolated_obstacle_positions = []

    for i, data_entry in enumerate(all_obstacle_data):
        current_times = np.array(data_entry["time"])
        current_positions = np.array(data_entry["position"])

        if len(current_times) == 0 or current_positions.shape[0] == 0:
            print(f"Skipping interpolation for obstacle {i+1} with empty time or position data.")
            # Create static obstacle at origin if no data
            static_position = np.zeros((total_frame, 3))
            all_interpolated_obstacle_positions.append(static_position)
            continue

        # Obstacle position is usually a single point (center of cylinder)
        if current_positions.ndim == 2 and current_positions.shape[1] == 3:
            # Position data is [time, 3] format
            f_interp = interpolate.interp1d(current_times, current_positions, axis=0,
                                            fill_value=(current_positions[0], current_positions[-1]),
                                            bounds_error=False)
            interpolated_obstacle_position = f_interp(times_true)
        elif current_positions.ndim == 3 and current_positions.shape[1] == 3:
            # Position data is [time, 3, 1] format (single point)
            positions_2d = current_positions[:, :, 0]  # Extract [time, 3]
            f_interp = interpolate.interp1d(current_times, positions_2d, axis=0,
                                            fill_value=(positions_2d[0], positions_2d[-1]),
                                            bounds_error=False)
            interpolated_obstacle_position = f_interp(times_true)
        else:
            print(f"Warning: Unexpected obstacle position data shape: {current_positions.shape}")
            # Create static obstacle at first position
            first_pos = current_positions[0] if current_positions.ndim == 2 else current_positions[0, :, 0]
            interpolated_obstacle_position = np.tile(first_pos, (total_frame, 1))

        all_interpolated_obstacle_positions.append(interpolated_obstacle_position)

    # Rendering
    batch = []
    view_name = "diag"
    output_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)
    os.makedirs(output_path, exist_ok=True)
    stage_script = stage_scripts[view_name]

    print(f"Generating POV-Ray scripts for {total_frame} frames...")
    for frame_number in tqdm(range(total_frame), desc="Scripting"):
        script = []
        script.extend([f'#include "{s}"' for s in included])
        script.append(stage_script)

        # Light blue color for rods
        rod_color = "rgb<0.7,0.9,1.0>"  # Light blue
        
        # Render rods
        for rod_idx, (interpolated_xs_for_rod, base_radius_for_rod) in enumerate(zip(all_interpolated_xs, all_base_radii_for_rods)):
            if interpolated_xs_for_rod.shape[2] == 0:
                continue

            rod_object = pyelastica_rod(
                x=interpolated_xs_for_rod[frame_number],
                r=base_radius_for_rod[frame_number],
                color=rod_color,
            )
            script.append(rod_object)

        # Render targets with their specific colors
        pov_target_colors = {
            'red': 'rgb<1.0,0.0,0.0>',
            'blue': 'rgb<0.0,0.0,1.0>',
            'orange': 'rgb<1.0,0.5,0.0>',
            'purple': 'rgb<0.5,0.0,0.5>'
        }
        
        for target_idx, target_position in enumerate(all_interpolated_target_positions):
            if target_idx < len(target_colors):
                color_name = target_colors[target_idx]
                pov_color = pov_target_colors[color_name]
            else:
                pov_color = 'rgb<1.0,0.5,0.0>'  # Default orange
            
            # Convert coordinates from (x,y,z) to (-y,x,z)
            original_position = target_position[frame_number]
            converted_position = np.array([original_position[1], -original_position[0], original_position[2]])
            
            target_object = pyelastica_sphere(
                center=converted_position,
                radius=0.05,  # Match the radius from Sim.py
                color=pov_color,
            )
            script.append(target_object)

        # Render obstacles as gray cylinders
        obstacle_color = 'rgb<0.5,0.5,0.5>'  # Gray color
        
        for obstacle_idx, obstacle_position in enumerate(all_interpolated_obstacle_positions):
            # Convert coordinates from (x,y,z) to (-y,x,z)
            original_position = obstacle_position[frame_number]
            converted_position = np.array([original_position[1], -original_position[0], original_position[2]])
            
            # Create cylinder from base to top (length 0.4 as defined in Sim.py)
            cylinder_length = 0.4
            base_position = converted_position
            top_position = base_position + np.array([0, 0, cylinder_length])
            
            # POV-Ray cylinder syntax: cylinder { <base>, <top>, radius texture { ... } }
            cylinder_object = f"""cylinder {{
    <{base_position[0]:.6f}, {base_position[1]:.6f}, {base_position[2]:.6f}>,
    <{top_position[0]:.6f}, {top_position[1]:.6f}, {top_position[2]:.6f}>,
    0.05
    texture {{
        pigment {{ color {obstacle_color} }}
        finish {{ 
            ambient 0.3
            diffuse 0.7
            specular 0.2
            roughness 0.1
        }}
    }}
}}"""
            script.append(cylinder_object)

        pov_script = "\n".join(script)

        file_path = os.path.join(output_path, f"frame_{frame_number:04d}")
        with open(file_path + ".pov", "w+") as f:
            f.write(pov_script)
        batch.append(file_path)

    # Process POVray
    pbar = tqdm(total=len(batch), desc="Rendering")
    if MULTIPROCESSING:
        func = partial(
            render,
            width=WIDTH,
            height=HEIGHT,
            display=DISPLAY_FRAMES,
            pov_thread=THREAD_PER_AGENT,
        )
        with Pool(NUM_AGENT) as p:
            for _ in p.imap_unordered(func, batch):
                pbar.update()
    else:
        for filename in batch:
            render(
                filename,
                width=WIDTH,
                height=HEIGHT,
                display=DISPLAY_FRAMES,
                pov_thread=multiprocessing.cpu_count(),
            )
            pbar.update()

    # Create video
    imageset_path = os.path.join(OUTPUT_IMAGES_DIR, "diag")
    filename = OUTPUT_FILENAME + "_diag.mp4"
    print(f"Stitching frames into video: {filename}")
    os.system(f"ffmpeg -r {FPS} -i {imageset_path}/frame_%04d.png {filename}")
    if ROTATE_VIDEO:
        # Rotate final video 180 degrees after stitching
        os.system(f"ffmpeg -i {filename} -vf 'rotate=180*PI/180' {filename}_rotated.mp4")
        # delete the original video file
        os.remove(filename)
        os.rename(f"{filename}_rotated.mp4", filename)
    print("Video generation complete.")