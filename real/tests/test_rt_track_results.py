import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to your CSV file
csv_path = r"C:\Users\dogro\Desktop\Emanuele\github\sorolearn\data\exp_2025-07-22_12-23-07\output_exp_2025-07-22_12-23-07.csv"

# Read the CSV
df = pd.read_csv(csv_path)

# Get unique trajectories
trajectories = df['trajectory'].unique()

for traj in trajectories:
    df_traj = df[df['trajectory'] == traj]
    t = df_traj['T (ms)'].values / 1000.0  # convert ms to s

    # Extract columns
    vol1 = df_traj['volume_1 (mm)'].values
    vol2 = df_traj['volume_2 (mm)'].values
    vol3 = df_traj['volume_3 (mm)'].values
    tip_x = df_traj['tip_x (cm)'].values
    tip_y = df_traj['tip_y (cm)'].values
    tip_z = df_traj['tip_z (cm)'].values

    # Tip velocity norm
    vx = df_traj['tip_velocity_x (cm/s)'].fillna(0).values
    vy = df_traj['tip_velocity_y (cm/s)'].fillna(0).values
    vz = df_traj['tip_velocity_z (cm/s)'].fillna(0).values
    vnorm = np.sqrt(vx**2 + vy**2 + vz**2)

    # Tip acceleration norm
    ax = df_traj['tip_acceleration_x (cm/ss)'].fillna(0).values
    ay = df_traj['tip_acceleration_y (cm/ss)'].fillna(0).values
    az = df_traj['tip_acceleration_z (cm/ss)'].fillna(0).values
    anorm = np.sqrt(ax**2 + ay**2 + az**2)

    fig, axs = plt.subplots(8, 1, figsize=(10, 16), sharex=True)
    fig.suptitle(f"Trajectory {traj}")

    axs[0].plot(t, vol1)
    axs[0].set_ylabel("Vol 1 (mm)")

    axs[1].plot(t, vol2)
    axs[1].set_ylabel("Vol 2 (mm)")

    axs[2].plot(t, vol3)
    axs[2].set_ylabel("Vol 3 (mm)")

    axs[3].plot(t, tip_x)
    axs[3].set_ylabel("Tip X (cm)")

    axs[4].plot(t, tip_y)
    axs[4].set_ylabel("Tip Y (cm)")

    axs[5].plot(t, tip_z)
    axs[5].set_ylabel("Tip Z (cm)")

    axs[6].plot(t, vnorm)
    axs[6].set_ylabel("||v|| (cm/s)")

    axs[7].plot(t, anorm)
    axs[7].set_ylabel("||a|| (cm/s²)")
    axs[7].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
