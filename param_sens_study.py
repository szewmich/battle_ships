import numpy as np
import copy
import random
from collections import Counter
import math
import time
import os

import fun
import random_board_generator
import reverse_validator
import prob_density_advanced
import prob_density_montecarlo

###########################################################################
# LOCAL DATA
# Only valid if this script is run independently with predefined board
###########################################################################
start_time = time.time()

prob_maps_adv_reduced_dir = "prob_density_maps_adv_reduced\\"
param_sens_study_dir = "param_sens_study\\"
param_sens_study_file = "param_sens_study_file_temp3.npy"
param_sens_study_path = os.path.join(param_sens_study_dir, param_sens_study_file)
print (param_sens_study_path)
#                     75%,  80%,  85%,  90%,  95%,  97%
# conf_level_values = [1.15, 1.28, 1.44, 1.65, 1.96, 2.17]
# margin_estim_values = [0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04]
# margin_highest_values = [0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04]

#                     95%,  97%,  99%, 99.9%
# conf_level_values = [1.96, 2.17, 2.58, 3.29]
# margin_estim_values = [1.00]
# margin_highest_values = [0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02]

parameters = [
    [1.960, 1.000, 0.060],
    [1.960, 1.000, 0.080],
    [1.960, 1.000, 0.100],
    [1.960, 1.000, 0.120],
    [2.170, 1.000, 0.060],
    [2.170, 1.000, 0.080],
    [2.170, 1.000, 0.100],
    [2.170, 1.000, 0.120],
    [2.170, 1.000, 0.140],
    [2.580, 1.000, 0.080],
    [2.580, 1.000, 0.100],
    [2.580, 1.000, 0.120],
    [2.580, 1.000, 0.140],
    [2.580, 1.000, 0.160],
    [2.580, 1.000, 0.180],
    [3.290, 1.000, 0.140],
    [3.290, 1.000, 0.160],
    [3.290, 1.000, 0.180],
    [3.290, 1.000, 0.200],
    [3.290, 1.000, 0.220],
    [3.290, 1.000, 0.240]
]

row = 10
col = 10

prob_dens = np.array([[0] * col for i in range(row)])

prob_dens[0] = [0, 0, 7, 4, 7, 0, 7, 2, 7, 0]
prob_dens[1] = [0, 0, 7, 4, 7, 0, 7, 2, 7, 0]
prob_dens[2] = [0, 0, 7, 4, 7, 0, 7, 7, 7, 0]
prob_dens[3] = [0, 0, 7, 4, 7, 0, 7, 0, 7, 7]
prob_dens[4] = [0, 0, 7, 7, 7, 0, 0, 0, 7, 2]
prob_dens[5] = [0, 7, 7, 7, 7, 7, 0, 0, 7, 2]
prob_dens[6] = [7, 0, 7, 2, 2, 7, 0, 0, 7, 7]
prob_dens[7] = [0, 0, 7, 7, 7, 7, 0, 0, 0, 0]
prob_dens[8] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
prob_dens[9] = [0, 0, 0, 0, 7, 0, 7, 0, 0, 0]

boards_to_run = [prob_dens]
number_of_unknown_fields = 0
for i in prob_dens:
    for j in i:
        if j == 0:
            number_of_unknown_fields += 1
#print(number_of_unknown_fields)

# game_id = str(number_of_unknown_fields)
prob_map_library = os.listdir(prob_maps_adv_reduced_dir)
n_files = len(prob_map_library)
n_p_sets = len(parameters)

###########################################################################
# Load results file or create new one
if param_sens_study_file in os.listdir(param_sens_study_dir):
    results = np.load(param_sens_study_path, allow_pickle=True)
    print('loaded existing results array')
else:
    results = np.empty((0, 6), float)
    print('created new results array')

###########################################################################
# Optional - use to run all games from prob_maps_adv_reduced_dir folder
###########################################################################
boards_to_run = []
for filename in prob_map_library:
    game_reduced_code = filename.removesuffix('.npy')
    matrix = np.zeros(100)
    for n in range(0,100):
        matrix[n] = str(game_reduced_code[n])
    prob_dens = np.array(matrix.reshape(10,10), dtype='int16')
    if game_reduced_code not in results:
        boards_to_run.append(prob_dens)
print(len(boards_to_run))
###########################################################################
m = 0
for prob_dens in boards_to_run:

    game_reduced_code = fun.update_board_state_100chars_code(prob_dens)
    prob_map_adv_reduced_file_name = game_reduced_code + '.npy'

    if prob_map_adv_reduced_file_name in prob_map_library:
        full_path = prob_maps_adv_reduced_dir + prob_map_adv_reduced_file_name
        good_fields_adv, best_prob_adv, occurances_adv, total_unique_adv = fun.load_data(full_path, margin=0.002)
        print('advanced results read from library')
    else:
        occurances_adv, best_field_adv, best_prob_adv, total_unique_adv, time_adv, game_reduced_code =\
            prob_density_advanced.calculate_probs_advanced(prob_dens)
    relative_error_avg_game = 0
    time_mc_avg_game = 0



    id = 0

    # for conf_level in conf_level_values:
    #     for margin_estim in margin_estim_values:
    #         for margin_highest in margin_highest_values:
    p = 0
    for param_set in parameters:
        conf_level = param_set[0]
        margin_estim = param_set[1]
        margin_highest = param_set[2]

        for k in range (0,3):
            best_field_mc, time_mc, occurances_mc, best_prob_mc = prob_density_montecarlo.\
                calculate_probs_montecarlo (prob_dens, conf_level, margin_estim, margin_highest)

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

            a0 = game_reduced_code
            a1 = "{:.3f}".format(float(conf_level))
            a2 = "{:.3f}".format(float(margin_estim))
            a3 = "{:.3f}".format(float(margin_highest))
            a4 = "{:.5f}".format(float(relative_error))
            a5 = "{:.2f}".format(float(time_mc))

            results = np.append(results, [[a0, a1, a2, a3, a4, a5]], axis=0)
            print([a0, a1, a2, a3, a4, a5])

            print(f'Finished {m} out of {n_files} files')
            print(f'Finished {p} out of {n_p_sets} parameter sets')
            print(f'{k} out of 3')

        # # Save as a .npy file after each completed loop of parameter change
        # np.save(param_sens_study_path, results)
        p += 1
    # Save as a .npy file after each completed game code
    np.save(param_sens_study_path, results)
    m += 1
    print (results)
    #print(f'Finished {m} out of {n_files} files')
    print("--- %s seconds ---" % (time.time() - start_time))
    #print(f'relative error average per game is: {relative_error_avg_game/10}')



