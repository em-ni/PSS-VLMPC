""" Rendering Script for flexible_swinging_pendulum.py simulation using POVray

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

import numpy as np
from scipy import interpolate
from tqdm import tqdm

# Import your povray macros and rendering utilities
from _povmacros import Stages, pyelastica_rod, render

# Setup (USER DEFINE)
DATA_PATH = "flexible_swinging_pendulum.dat"  # Path to the simulation data
SAVE_PICKLE = True

# Rendering Configuration (USER DEFINE)
OUTPUT_FILENAME = "pov_flexible_pendulum"
OUTPUT_IMAGES_DIR = "frames"
FPS = 20.0
WIDTH = 1920
HEIGHT = 1080
DISPLAY_FRAMES = "Off"  # ['On', 'Off']

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
    location=[2.0, 2.0, -4.0],
    angle=30,
    look_at=[0.0, 0.0, 0.5],
    name="diag",
)
stages.add_camera(
    location=[0, 5, 0.5],
    angle=30,
    look_at=[0.0, 0.0, 0.5],
    sky=[-1, 0, 0],
    name="top",
)
stages.add_light(
    position=[1500, 2500, -1000],
    color="White",
    camera_id=-1,
)
stages.add_light(
    position=[2.0, 2.0, -4.0],
    color=[0.09, 0.09, 0.1],
    camera_id=0,
)
stages.add_light(
    position=[0.0, 4.0, 1.0],
    color=[0.09, 0.09, 0.1],
    camera_id=1,
)
stage_scripts = stages.generate_scripts()

# Externally Including Files (USER DEFINE)
included = ["default.inc"]

# Multiprocessing Configuration (USER DEFINE)
MULTIPROCESSING = True
THREAD_PER_AGENT = 4
NUM_AGENT = multiprocessing.cpu_count() // 2

if __name__ == "__main__":
    # Load Data
    assert os.path.exists(DATA_PATH), "File does not exist"
    try:
        if SAVE_PICKLE:
            import pickle as pk
            with open(DATA_PATH, "rb") as fptr:
                data = pk.load(fptr)
        else:
            raise NotImplementedError("Only pickled data is supported")
    except OSError as err:
        print(f"Cannot open the datafile {DATA_PATH}")
        print(str(err))
        raise

    # Convert data to numpy array
    times = np.array(data["time"])  # shape: (timelength)
    xs = np.array(data["position"])  # shape: (timelength, 3, n_elem)

    # Interpolate Data
    runtime = times.max()
    total_frame = int(runtime * FPS)
    times_true = np.linspace(0, runtime, total_frame)

    xs = interpolate.interp1d(times, xs, axis=0)(times_true)
    times = interpolate.interp1d(times, times, axis=0)(times_true)
    base_radius = np.ones_like(xs[:, 0, :]) * 0.005  # Use the same as in simulation

    # Rendering
    batch = []
    for view_name in stage_scripts.keys():
        output_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)
        os.makedirs(output_path, exist_ok=True)
    for frame_number in tqdm(range(total_frame), desc="Scripting"):
        for view_name, stage_script in stage_scripts.items():
            output_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)

            script = []
            script.extend([f'#include "{s}"' for s in included])
            script.append(stage_script)

            rod_object = pyelastica_rod(
                x=xs[frame_number],
                r=base_radius[frame_number],
                color="rgb<0.2,0.6,0.8>",
            )
            script.append(rod_object)
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

    # Create Video using ffmpeg
    for view_name in stage_scripts.keys():
        imageset_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)
        filename = OUTPUT_FILENAME + "_" + view_name + ".mp4"
        os.system(f"ffmpeg -r {FPS} -i {imageset_path}/frame_%04d.png {filename}")
