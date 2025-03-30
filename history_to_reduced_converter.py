import numpy as np
import copy
import time
import random
import os
import shutil

import random_board_generator
import prob_density_montecarlo
import fun


prob_maps_dir = "prob_density_maps\\"
prob_maps_100chars_dir = "prob_density_maps_100chars\\"

prob_map_library = os.listdir(prob_maps_dir)
n = 0
tot = len(prob_map_library)
unique = 0
duplicates = []

for prob_map_file_name in prob_map_library:

    prob_map_library = os.listdir(prob_maps_dir)

    # Board state known to the attacking player. Empty at the beginning.
    known_board = np.array([[0] * 10 for i in range(10)])

    game_history_code = prob_map_file_name.replace('000-', '')
    game_history_code = game_history_code.replace('.npy', '')

    while len(game_history_code) > 0:

        move_code = game_history_code[0:4]

        x = int(move_code[0])
        y = int(move_code[1])
        res = int(move_code[2])

        hit = False

        if res == 1 or res == 7:
            known_board[x][y] = res
        else:
            known_board[x][y] = 1
            groups = fun.group_adjacent_symbols(known_board, 1)
            for gr in groups:
                if [x, y] in gr:
                    fields_to_sink = gr
                    break
            known_board = fun.fill_adjacent(known_board, fields_to_sink, res)

        game_history_code = game_history_code[4:]
        game_100chars_code = fun.update_board_state_100chars_code(board_known)


    prob_map_100chars_file_name = game_100chars_code + ".npy"

    source = os.path.join(prob_maps_dir, prob_map_file_name)
    dest = os.path.join(prob_maps_100chars_dir, prob_map_100chars_file_name)

    prob_map_100chars_library = os.listdir(prob_maps_100chars_dir)
    if prob_map_100chars_file_name not in prob_map_100chars_library:
        shutil.copy(source, dest)
        unique +=1
    else:
        duplicates.append(prob_map_100chars_file_name)
    n = n + 1
    print (f'Processed {n} out of {tot} files. Total {unique} unique found.')

print (f'Duplicates list: {duplicates}')
