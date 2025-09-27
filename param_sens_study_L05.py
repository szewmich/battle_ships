import pandas as pd
import time
import os
import numpy as np

import prob_map_montecarlo

###########################################################################
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1700)
pd.set_option('display.max_columns', 20)
pd.set_option('max_colwidth', 2000)
###########################################################################

param_sens_study_dir = "param_sens_study\\L05"

adv_time_data_file = "adv_time_data.csv"
param_sens_study_file = "adv_time_data_with_mc.csv"

param_sens_study_path = os.path.join(param_sens_study_dir, param_sens_study_file)

n = 3

###########################################################################
# df = pd.read_csv(param_sens_study_path)
# adv_times_df = pd.read_csv(adv_time_data_file)

# adv_times_df.rename(columns={'calc_time': 'time_adv'}, inplace = True)

# # print(adv_times_df.sample(10))
# # exit()
# # From adv_times_df, find board_state_100chars_code values which are not in df and add thosse rows to df
# missing_board_states = adv_times_df[~adv_times_df['board_state_100chars_code'].isin(df['board_state_100chars_code'])]
# df = pd.concat([df, missing_board_states], ignore_index=True)


# print(df.describe())
# print(df.sample(100))
# # exit()

# # Add column 'time_mc' to the dataframe
# # df['time_mc'] = None
# # df = df.query('calc_time >= 1.0')

# df_red = df.query('time_mc.isnull()')
# # print (df_red)

# tot = len(df_red)
# print(tot)
# # exit()

# # df_red.to_csv(param_sens_study_path, index=False)

# # exit()

# if not df_red.empty:
#     print(f'Found {len(df_red)} rows with time_mc == None')
#     # exit()

#     c = 1
#     for index, row in df_red.iterrows():
        

#         board_state_100chars_code = row['board_state_100chars_code']
#         matrix = np.zeros(100)
#         for x in range(100):
#             matrix[x] = str(board_state_100chars_code[x])
#         known_board = np.array(matrix.reshape(10,10), dtype='int16')

#         # Calculate time_mc
#         time_mc_sum = 0
#         for k in range (n):
#             best_field_mc, time_mc, occurances_mc, best_prob_mc = prob_map_montecarlo.\
#                 calculate_probs_montecarlo(known_board, conf_level=2.58, margin_estim=1.00, margin_highest=0.12)
#             time_mc_sum = time_mc_sum + time_mc
#         time_mc_avg = time_mc_sum / n

#         # Update the dataframe
#         df.at[index, 'time_mc'] = time_mc_avg

#         # Save the updated dataframe
#         df.to_csv(param_sens_study_path, index=False)

#         print(f'Done {c} out of {tot}')
#         c += 1

# else:
#     print('No rows with time_mc == None found, nothing to update.')

import time
import fun
import random_board_generator

n_seg_for_lvl = {
    1: 5,
    2: 4,
    3: 3,
    4: 3,
    5: 2,
    6: 2,
    7: 2
}

df = pd.read_csv(param_sens_study_path)

start_time = time.time()
c = 1
for index, row in df.iterrows():

    board_state_100chars_code = row['board_state_100chars_code']
    matrix = np.zeros(100)
    for x in range(100):
        matrix[x] = str(board_state_100chars_code[x])
    known_board = np.array(matrix.reshape(10,10), dtype='int16')

    free_fields, hit_unsunk = fun.find_free_and_hit_fields(known_board)

    configs = random_board_generator.create_configs(known_board, free_fields, n_seg_for_lvl)

    # print(known_board)
    # print(configs)
    # print(configs[7])

    df.at[index, 'cnf_5'] = len(configs[1])
    df.at[index, 'cnf_4'] = len(configs[2])
    df.at[index, 'cnf_3'] = len(configs[3])
    df.at[index, 'cnf_2'] = len(configs[5])

    c += 1
    if c % 100 == 0:
        print(f'Processed {c} rows')
    # df.at[index, 'cnf_5'] = len(configs[1])**((5 - df.at[index, 'n_5']) / 5)
    # df.at[index, 'cnf_4'] = len(configs[2])**((4 - df.at[index, 'n_4']) / 4)
    # df.at[index, 'cnf_3'] = len(configs[3])**((6 - df.at[index, 'n_3']) / 3)
    # df.at[index, 'cnf_2'] = len(configs[5])**((6 - df.at[index, 'n_2']) / 2)


    # print(df.at[index, 'cnf_5'])
    # print(df.at[index, 'cnf_4'])
    # print(df.at[index, 'cnf_3'])
    # print(df.at[index, 'cnf_2'])

    # df.at[index, 'cnf_4'] = configs[1]
    # df.at[index, 'cnf_3'] = configs[2]
    # df.at[index, 'cnf_3'] = configs[2]

total_time = time.time() - start_time
print(f'Total time: {total_time} seconds')
print(f'Avergae time: {total_time / c} seconds')

df.to_csv(param_sens_study_path, index=False)