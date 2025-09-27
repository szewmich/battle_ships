import numpy as np
import copy
import random
from collections import Counter
import math
import time
import os
import pandas as pd

import fun
import random_board_generator
import reverse_validator
import prob_map_advanced
import prob_map_montecarlo
import RTP_lmdb as rtp

###########################################################################
# LOCAL DATA
# Only valid if this script is run independently with predefined board
###########################################################################
start_time = time.time()

param_sens_study_dir = "param_sens_study\\L03"
param_sens_study_file = "param_sens_study_file_mc_or_adv.csv"
param_sens_study_path = os.path.join(param_sens_study_dir, param_sens_study_file)
print (param_sens_study_path)

# Initialize lmdb enviroment (Data storage system for RTP library - more explained later)
SHARD_COUNT = 100  # Number of LMDB database shards
LMDB_PATH_TEMPLATE = "prob_maps_adv_lmdb\\shard_{:02d}.lmdb"  # LMDB file path pattern
MAP_SIZE = 10 ** 8  # 100MB per shard (can be increased at any time)
shard_envs = rtp. initialize_RTP_lmdb(SHARD_COUNT, LMDB_PATH_TEMPLATE, MAP_SIZE)

n_repetitions = 1

#                     75%,  80%,  85%,  90%,  95%,  97%
# conf_level_values = [1.15, 1.28, 1.44, 1.65, 1.96, 2.17]
# margin_estim_values = [0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04]
# margin_highest_values = [0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04]

#                     95%,  97%,  99%, 99.9%
# conf_level_values = [1.96, 2.17, 2.58, 3.29]
# margin_estim_values = [1.00]
# margin_highest_values = [0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06]

#                    99.9%, 99%,  97%,  95%
conf_level_values = [3.29]
margin_estim_values = [1.00]
margin_highest_values = [0.06]

# parameters = [
#     [1.960, 1.000, 0.060],
#     [1.960, 1.000, 0.080],
#     [1.960, 1.000, 0.100],
#     [1.960, 1.000, 0.120],
#     [2.170, 1.000, 0.060],
#     [2.170, 1.000, 0.080],
#     [2.170, 1.000, 0.100],
#     [2.170, 1.000, 0.120],
#     [2.170, 1.000, 0.140],
#     [2.580, 1.000, 0.080],
#     [2.580, 1.000, 0.100],
#     [2.580, 1.000, 0.120],
#     [2.580, 1.000, 0.140],
#     [2.580, 1.000, 0.160],
#     [2.580, 1.000, 0.180],
#     [3.290, 1.000, 0.140],
#     [3.290, 1.000, 0.160],
#     [3.290, 1.000, 0.180],
#     [3.290, 1.000, 0.200],
#     [3.290, 1.000, 0.220],
#     [3.290, 1.000, 0.240]
# ]
# parameters = [
#     [2.170, 1.000, 0.020]
# ]

row = 10
col = 10

known_board = np.zeros((100), dtype="int16")
known_board = known_board.reshape(10, 10)

known_board[0] = [0, 7, 3, 7, 7, 0, 0, 0, 0, 0]
known_board[1] = [0, 7, 3, 7, 0, 7, 0, 0, 0, 0]
known_board[2] = [0, 7, 3, 7, 0, 0, 7, 7, 7, 7]
known_board[3] = [0, 7, 7, 7, 0, 0, 7, 2, 2, 7]
known_board[4] = [7, 0, 0, 0, 0, 0, 7, 7, 7, 7]
known_board[5] = [0, 0, 0, 0, 7, 0, 0, 0, 0, 0]
known_board[6] = [0, 0, 0, 0, 0, 7, 0, 0, 0, 0]
known_board[7] = [0, 0, 0, 0, 0, 0, 7, 0, 0, 0]
known_board[8] = [7, 7, 7, 7, 7, 7, 0, 0, 0, 0]
known_board[9] = [7, 4, 4, 4, 4, 7, 0, 0, 0, 0]

boards_to_run = [known_board]

number_of_unknown_fields = np.count_nonzero(known_board == 0)

# n_files = len(prob_map_library)



###########################################################################
# Load results file or create new one
if param_sens_study_file in os.listdir(param_sens_study_dir):
    results = pd.read_csv(param_sens_study_path)
    print('loaded existing results dataframe')
else:
    results = None
    print('created new results dataframe')

codes_done = results['board_state_100chars_code'].to_list()
###########################################################################
# Optional - use to run all games from prob_maps_adv_reduced_dir folder
###########################################################################
prob_map_adv_100chars_dir = "prob_maps_adv_100chars\\"
prob_map_adv_100chars_library = os.listdir(prob_map_adv_100chars_dir)
boards_to_run = []
for filename in prob_map_adv_100chars_library:
    board_state_100chars_code = filename.removesuffix('.npy')
    zero_count = board_state_100chars_code.count('0')
    if zero_count >= 25:
        matrix = np.zeros(100)
        for n in range(0,100):
            matrix[n] = str(board_state_100chars_code[n])
        prob_map = np.array(matrix.reshape(10,10), dtype='int16')
        if board_state_100chars_code not in codes_done:
            boards_to_run.append(prob_map)
n_files = len(boards_to_run)
print(n_files)
# exit()
###########################################################################
m = 0
for known_board in boards_to_run:

    board_state_100chars_code = fun.update_board_state_100chars_code(known_board)

    occurances_adv = rtp.load_array_from_RTP_lmdb(known_board, LMDB_PATH_TEMPLATE)
    if occurances_adv is not None:
        good_fields, best_field, total_unique_adv, best_prob_adv = fun.find_best_fields(occurances_adv, margin=0.002)
        print('advanced results read from library')
        print(occurances_adv)
    else:
        occurances_adv, best_field_adv, best_prob_adv, total_unique_adv, time_adv, game_reduced_code =\
            prob_map_advanced.calculate_probs_advanced(known_board)

    relative_error_avg_game = 0
    time_mc_avg_game = 0

    id = 0

    parameters = []
    for conf_level in conf_level_values:
        for margin_estim in margin_estim_values:
            for margin_highest in margin_highest_values:
                parameters.append([conf_level, margin_estim, margin_highest])
    n_p_sets = len(parameters)
    p = 0
    for param_set in parameters:
        conf_level = param_set[0]
        margin_estim = param_set[1]
        margin_highest = param_set[2]

        new_data_batch = None
        # new_data_batch = pd.DataFrame(columns=
        #                               ['board_state_100chars_code', 'conf_level', 'margin_est', 'margin_highest',
        #                                'rel_error', 'time_mc'])
        for k in range (0,n_repetitions):

            best_field_mc, time_mc, occurances_mc, best_prob_mc = prob_map_montecarlo.\
                calculate_probs_montecarlo (known_board, conf_level, margin_estim, margin_highest)

            x = best_field_mc[0]
            y = best_field_mc[1]
            actual_prob = occurances_adv[x][y] / total_unique_adv
            relative_error = (best_prob_adv - actual_prob) / best_prob_adv

            #print(occurances_adv)
            print(occurances_mc)
            print(f'best_field_mc is: {best_field_mc}')
            print(f'best_prob_mc is: {best_prob_mc}')
            print(f'best_prob_adv is: {best_prob_adv}')
            print(f'relative error is: {relative_error}')
            relative_error_avg_game = relative_error_avg_game + relative_error
            time_mc_avg_game = time_mc_avg_game + time_mc

            a0 = board_state_100chars_code
            a1 = round(conf_level, 2)
            a2 = round(margin_estim, 2)
            a3 = round(margin_highest, 2)
            a4 = round(relative_error, 5)
            a5 = round(time_mc, 2)

            new_row = [a0, a1, a2, a3, a4, a5]
            new_row = pd.DataFrame([new_row], columns=
            ['board_state_100chars_code', 'conf_level', 'margin_est', 'margin_highest', 'rel_error', 'time_mc'])

            if new_data_batch is not None:
                new_data_batch = pd.concat([new_data_batch, new_row], ignore_index=True)
            else:
                new_data_batch = new_row.copy()


            print(f'Finished {m} out of {n_files} files')
            print(f'Finished {p} out of {n_p_sets} parameter sets')
            print(f'Currently at: {param_set}')
            print(f'{k+1} out of {n_repetitions}')

        # Add batch of new rows (series of repetitive runs for the same board_state and parameters) to results dataframe
        if results is not None:
            results = pd.concat([results, new_data_batch], ignore_index=True)
        else:
            results = new_data_batch.copy()

        p += 1

        # Save as a .csv file after each completed parameter set
        #results.to_csv(param_sens_study_path, index=False)

    # Save as a .csv file after each completed board_state_100chars_code
    results.to_csv(param_sens_study_path, index=False)

    m += 1
    print (results)
    print("--- %s seconds ---" % (time.time() - start_time))




