from math import ceil

def workspace_preview(w_max_stroke, w_steps, max_stroke):
    print("Input parameters:")
    print(f"w_max_stroke: {w_max_stroke} mm")
    print(f"w_steps: {w_steps} steps")
    print(f"max_stroke: {max_stroke} mm")
    print()

    # To convert from mm to steps
    steps_ratio = w_steps / w_max_stroke # 1/mm

    # Number of windows
    n_windows = ceil(max_stroke / w_max_stroke) 

    # Overlap between windows
    # tot_overlap = n_windows * w_overlap = n_windows * w_max_stroke - max_stroke
    # Single window overlap
    w_overlap = (n_windows * w_max_stroke - max_stroke) / n_windows # mm

    # Elongation step and steps (how much to move the window)
    elongation_step = w_max_stroke - w_overlap # mm
    elongation_steps = steps_ratio * elongation_step 

    # Total steps for the entire system
    tot_steps = int(steps_ratio * max_stroke)

    print("Calculated parameters:")
    print(f"n_windows: {n_windows}")
    print(f"elongation_step: {elongation_step} mm")
    print(f"elongation_steps: {elongation_steps} steps")
    print(f"w_overlap: {w_overlap} mm")
    print(f"tot_steps: {tot_steps}")

    # Initialize grid
    print("Grid representation:")
    print("Each 'o' represents a step, '|' represents window boundaries, '*' represents overlap.")
    grid = [['o' for _ in range(tot_steps)] for _ in range(3)]

    # Mark window boundaries and overlaps
    for window in range(n_windows):
        # Calculate window start and end positions in steps
        start_step = int(window * elongation_steps)
        end_step = min(start_step + w_steps - 1, tot_steps - 1)
        
        # Mark window boundaries with '|'
        if start_step < tot_steps:
            for i in range(3):
                grid[i][start_step] = '|'
        
        if end_step < tot_steps:
            for i in range(3):
                grid[i][end_step] = '|'
        
        # Mark overlap with previous window
        if window > 0:
            overlap_start = start_step
            overlap_end = min(int(start_step + (w_overlap * steps_ratio) - 1), tot_steps - 1)
            
            for i in range(3):
                for j in range(overlap_start, overlap_end + 1):
                    if j < tot_steps and grid[i][j] != '|':
                        grid[i][j] = '*'  # Emphasize overlap

    # Print grid
    for i in range(3):
        for j in range(tot_steps):
            print(grid[i][j], end=' ')
        print()

if __name__ == "__main__":

    # Maximum stroke of the single window
    w_max_stroke = 2.4 # mm

    # Number of steps inside the window
    w_steps = 21

    # Maximum stroke of the entire system
    max_stroke = 6 # mm

    workspace_preview(w_max_stroke, w_steps, max_stroke)


