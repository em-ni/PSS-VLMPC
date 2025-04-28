import time

from src import config

def test_sliding_window_exploration():
 # Configuration values
    initial_pos = 0
    windowsteps = 5  # Total steps in each window
    stepSize = 1  # Step size for entire exploration Maximum stroke / steps
    elongationstepSize = 5  # Step size for elongation
    elongationstep = 0
    print("Testing sliding window exploration")
    i = 0
    j = 0
    k = 0
    k_flipFlag = 1
    j_flipFlag = 1
    stepCounter = 0
    covered_positions = []
    while elongationstep <= 10:
        print("\r", elongationstep)
        # i += elongationstepSize
        # j += elongationstepSize
        # k += elongationstepSize
        while i <= windowsteps + elongationstep:
            while j <= windowsteps + elongationstep  :
                while k <= windowsteps + elongationstep :
                    if k == elongationstep and k_flipFlag == -1:
                        k_flipFlag = -k_flipFlag
                        print("\r", i, j, k)
                        # self.step_and_save(position_i, position_j, position_k)
                        position_k = initial_pos + k * stepSize
                        stepCounter += 1
                        break
                    if k == windowsteps+elongationstep and k_flipFlag == 1:
                        k_flipFlag = -k_flipFlag
                        print("\r", i, j, k)
                        # self.step_and_save(position_i, position_j, position_k)
                        position_k = initial_pos + k * stepSize
                        stepCounter += 1
                        break
                    print("\r", i, j, k)
                    # self.step_and_save(position_i, position_j, position_k)  
                    position_k = initial_pos + k * stepSize
                    k = k + k_flipFlag
                    stepCounter += 1
                if j == elongationstep and j_flipFlag == -1:
                    j_flipFlag = -j_flipFlag
                    position_j = initial_pos + j * stepSize
                    break
                if j == windowsteps+elongationstep and j_flipFlag == 1:
                    j_flipFlag = -j_flipFlag
                    position_j = initial_pos + j * stepSize
                    break
                j = j + j_flipFlag
                position_j = initial_pos + j * stepSize
            position_i = initial_pos + i * stepSize
            i = i + 1
        j += elongationstepSize
        k += elongationstepSize
        elongationstep +=  windowsteps

    print(f"\nTotal positions covered: {len(covered_positions)}")
    return covered_positions
# Run the test
test_sliding_window_exploration()