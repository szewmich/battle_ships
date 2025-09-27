import numpy as np
import os
import pandas as pd

import fun

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 20)
pd.set_option('max_colwidth', 2000)

prob_maps_adv_reduced_dir = "prob_maps_adv_reduced\\"
prob_map_library = os.listdir(prob_maps_adv_reduced_dir)

boards_to_run = []
game_codes_to_run = []
number_of_0s = []
number_of_1s = []

for filename in prob_map_library:
    game_reduced_code = filename.removesuffix('.npy')
    matrix = np.zeros(100)
    for n in range(0,100):
        matrix[n] = str(game_reduced_code[n])
    prob_map = np.array(matrix.reshape(10,10), dtype='int16')
    if fun.check_if_obvious(prob_map) is None:
        boards_to_run.append(prob_map)
        game_codes_to_run.append(game_reduced_code)
        number_of_0s.append(game_reduced_code.count('0'))
        c_ones = game_reduced_code.count('1')
        # if c_ones == 0:
        #     c = 0
        # if c_ones == 1:
        #     c = 1
        # if c_ones > 1:
        #     c = 2
        number_of_1s.append(c_ones)
print(len(boards_to_run))
# print(number_of_0s)

data_df = pd.DataFrame(list(zip(number_of_0s, number_of_1s)), columns = ['number_of_0s', 'number_of_1s'])
# print(data_df)

df_2 = data_df.groupby(['number_of_0s', 'number_of_1s']).agg(
    count = ('number_of_0s', 'count'),
)

print(df_2)


