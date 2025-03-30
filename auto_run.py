import numpy as np
import os

import prob_density_advanced
import fun

############################################################################################################
# This script is used to automatically run 'calculate_probs_advanced' for multiple board states (boards_to_run)
############################################################################################################
if __name__ == "__main__":

    ############################################################################################################
    # Determine (boards_to_run)
    ############################################################################################################
    # Find board states for which results are available in montecarlo results library (prob_maps_mc_100chars_dir),
    # Filter only those for which:
    # 1. advanced times_results are not yet calculated
    # 2. results are not obvious (fun.check_if_obvious(known_board))
    # 3. board state contains desired number of free fields (eg. c >= 15 and c < 60:)

    # RadioTelegraphic Phonebook (RTP) directory, calculated by MC algorithm:
    prob_maps_mc_100chars_dir = "prob_density_maps_mc_100chars\\"
    prob_maps_mc_library = os.listdir(prob_maps_mc_100chars_dir)
    n_files = len(prob_maps_mc_library)

    # Numpy file containing data about advanced algorythm calculation time per board state
    times_adv_dir = "times_adv\\"
    times_adv_file = "times_adv_file.npy"
    times_adv_path = os.path.join(times_adv_dir, times_adv_file)
    times_results = np.load(times_adv_path, allow_pickle=True)

    n_app_files = 0
    boards_to_run = []
    for filename in prob_maps_mc_library:
        board_state_100chars_code = filename.removesuffix('.npy')
        if board_state_100chars_code not in times_results:
            print(board_state_100chars_code)
            c = filename.count('0')
            if c >= 15 and c < 60:
                matrix = np.zeros(100)
                for n in range(0,100):
                    matrix[n] = str(board_state_100chars_code[n])
                known_board = np.array(matrix.reshape(10,10), dtype='int16')
                # Check if there is any field that is obvious as a next target
                obvious_field = fun.check_if_obvious(known_board)
                if obvious_field is None:
                    boards_to_run.append(known_board)
                    n_app_files += 1
    print(n_app_files)
    print(len(boards_to_run))

    ############################################################################################################
    # Run (boards_to_run)
    ############################################################################################################
    for files_done, known_board in enumerate(boards_to_run):
        print(known_board)
        occurances, best_field, best_prob, total_unique, time_adv, board_state_100chars_code = prob_density_advanced.\
            calculate_probs_advanced(known_board)
        fun.save_adv_times(board_state_100chars_code, time_adv)

        print(f'Files done: {files_done + 1} out of {n_app_files}')

# tu jakis komentarz sie dodal
