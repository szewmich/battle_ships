import numpy as np
import pandas as pd
import copy
import time
import random
import os
import joblib

import random_board_generator_dim as random_board_generator
import prob_map_montecarlo_dim as prob_map_montecarlo
import prob_map_advanced_dim as prob_map_advanced
import fun_dim as fun
import RTP_lmdb as rtp
import bias_metrics
import bias_detection

###########################################################################
# INITIAL DATA SHARED ACROSS ALL GAMES (NOT MODIFIED ONCE CREATED)
###########################################################################
n_games = 1000
moves_till_break = 100
dim = 6

# RadioTelegraphic Phonebook (RTP) directory:
prob_maps_mc_history_dir = "prob_maps_mc_history\\"
prob_maps_mc_100chars_dir = "prob_maps_mc_100chars\\"
prob_maps_adv_100chars_dir = "prob_maps_adv_100chars\\"
all_games_results_file_path = "all_games_results.csv"

# Data for param_sens_study_L05
adv_time_data_file_path = "adv_time_data.csv"                       

# Initialize lmdb enviroment (Data storage system for RTP library - more explained later)
# SHARD_COUNT = 100  # Number of LMDB database shards
# LMDB_PATH_TEMPLATE = "prob_maps_mc_lmdb\\shard_{:02d}.lmdb"  # LMDB file path pattern
# MAP_SIZE = 10 ** 8  # 100MB per shard (can be increased at any time)
# shard_envs = rtp. initialize_RTP_lmdb(SHARD_COUNT, LMDB_PATH_TEMPLATE, MAP_SIZE)

# # Load all_games_results file or create new one
# if all_games_results_file_path in os.listdir(os.getcwd()):
#     all_games_results = pd.read_csv(all_games_results_file_path)
#     print('loaded existing all_games_results dataframe')
# else:
#     all_games_results = None
#     print('created new all_games_results dataframe')

all_games_results = None
print('created new all_games_results dataframe')

# # Data for param_sens_study_L05
# # Load adv_time_data_file or create new one
# if adv_time_data_file_path in os.listdir(os.getcwd()):
#     adv_time_data = pd.read_csv(adv_time_data_file_path)
#     print('loaded existing adv_time_data dataframe')
# else:
#     adv_time_data = None
#     print('created new adv_time_data dataframe')


# GLOBAL COUNTERS
moves_all_games = 0
time_all_games = 0
sum_probs = 0
sum_hits = 0

# Create 1D list of dim x dim fields and convert to tuple (0 to 35)
fields = tuple([(x, y) for x in range(dim) for y in range(dim)])

# 7 levels of hypothetical board creation, starting from placement of a 5-seg ship, ending with 2-segs.
# Each level is assigned a length of a ship to be placed
n_seg_for_lvl = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 3,
    6: 2,
    7: 2
}

game_rules_n_ships = {
    5:0,
    4:0,
    3:1,
    2:2
    }

# These are dummy arguments for clear board, explained in detail in further modules where they are needed.
seg5_done = 0
seg4_done = 0
seg3_done = 0
seg2_done = 0
hit_unsunk_lvl_0 = []
minimum_lengths = []

# "Free fields" are those containing either 0 or 1 value. In clear board every field is free
free_fields = fields

board_clear = np.zeros((dim, dim), dtype = "int16")

# Configs_levels = all valid (compliant with game rules) combinations of a free field and orientation (hor/ver)
# for each of 7 ships.
configs_levels = random_board_generator.create_configs(board_clear, free_fields, n_seg_for_lvl)

###########################################################################
# MULTIPLE GAMES LOOP
###########################################################################

all_boards_df = pd.read_csv("bias_checks\\all_boards_with_metrics.csv")
# get only first 10500 rows of df
red_boards_df = all_boards_df[10500:10600]

for g_num in range(1, n_games + 1):
    print('Starting game number', g_num)
    zero_time = time.time()
    time_at_switch_set = True
    time_at_switch = -1
    # List available probability maps, update after each game
    prob_map_library = os.listdir(prob_maps_mc_100chars_dir)

    #############################################
    # BOARD SETUP
    #############################################

    bias_detected = True
    if bias_detected:
        # read classifier and scaler using joblib
        clf = joblib.load("bias_checks\\logistic_regression_classifier.pkl")
        scaler = joblib.load("bias_checks\\robust_scaler.pkl")

        # Pre-load once
        center = scaler.center_
        scale  = scaler.scale_
        # feature_order = scaler.feature_names_in_   # if using sklearn 1.3+
        

        pi_R = 10000 / (11000)
        pi_B = 1000 / (11000)
        class_weights = (pi_R, pi_B)

        dummy_metrics = bias_metrics.make_metrics_dict()
        feature_order = list(dummy_metrics.keys()) 
        scaler_params = (center, scale, feature_order)

    if False:
        # Generate initial board from clear board
        board_initial = None
        while board_initial is None:
            board_initial = random_board_generator.\
                            generate_random_board(board_clear, configs_levels, n_seg_for_lvl,
                                                seg5_done = 0, seg4_done = 0, seg3_done = 0, seg2_done = 0,
                                                hit_unsunk_lvl_0 = [], minimum_lengths = [])

    # board_initial_code = n'th element of red_boards_df 'board' column as a string of 36 characters (0-9)
    board_initial_code = red_boards_df.iloc[g_num - 1]['board']
    
    # board_initial_code = red_boards_df[g_num - 1]['board']
    board_initial = np.array([int(x) for x in board_initial_code]).reshape((6,6))

    # Change board setup manually here if needed
    # board_initial[0] = [0, 0, 7, 7, 7, 7, 7, 0, 0, 0]
    # board_initial[1] = [0, 0, 7, 4, 4, 4, 4, 7, 7, 7]
    # board_initial[2] = [7, 7, 7, 7, 7, 7, 7, 7, 7, 5]
    # board_initial[3] = [2, 2, 7, 0, 0, 0, 7, 7, 7, 5]
    # board_initial[4] = [7, 7, 7, 0, 0, 0, 7, 2, 7, 5]
    # board_initial[5] = [0, 7, 7, 7, 7, 7, 7, 2, 7, 5]
    # board_initial[6] = [0, 7, 3, 3, 3, 7, 7, 7, 7, 5]
    # board_initial[7] = [0, 7, 7, 7, 7, 7, 0, 0, 7, 7]
    # board_initial[8] = [0, 0, 0, 7, 2, 7, 7, 7, 7, 7]
    # board_initial[9] = [0, 0, 0, 7, 2, 7, 3, 3, 3, 7]

    # Replace '7' fields with '0'. 7s are there because of the way random board generator works (to avoid touching
    # ships and rule out certain fields). For initial board generation 7s shall be removed
    for field in fields:
        x = field[0]
        y = field[1]
        if board_initial[x][y] == 7:
            board_initial[x][y] = 0

    # 100chars_code encodes initial board state in a string of 100 characters (0-9)
    board_initial_100chars_code = fun.update_board_state_100chars_code(board_initial)

    # Detect and group all ships that are present on the initial board in a list.
    # There should be 7 items in the list, 1 per each ship. Each ship contains list of fields (X, Y) coord. it occupies
    ship_fields_initial = []
    for n in range (2,6):
        g = fun.group_adjacent_symbols(board_initial, n)
        ship_fields_initial = ship_fields_initial + g
    # Create deepcopy of this list. It will be later needed to store info about which ships are left to sink.
    ship_fields_left = copy.deepcopy (ship_fields_initial)

    print (board_initial)

    # Board state known to the attacking player. Empty at the beginning.
    known_board = np.zeros((dim, dim), dtype = "int16")

    #############################################
    # SHOOTING SEQUENCE
    #############################################

    moves = 0
    hit_count = 0

    # number of board states retrieved from database or calculated using certain method - for statistics
    n_retrieved = 0
    n_obvious = 0
    n_calc_mc = 0
    n_calc_adv = 0
    target_methods = ''

    # All game scenarios start with "000" code
    game_history_code = "000"

    # 21 is sum of all ship segments (5+4+3+3+2+2+2)
    # Scoring 21st hit means game over
    while hit_count < 7:

        # Earlier game finish for testing/benchmarking purposes
        if moves == moves_till_break:
            break

        # After each move update game history code by adding "-" sign and 3 digits encoding the last hit as follows:
        # 1st digit: X coordinate of shot field, 2nd: Y coordinate, 3rd: value of this field sent to the known board
        # More details on how these are determined will come later.
        # game_history_code precisely defines the course and current state of the game.

        if moves > 0:
            previous_bf = str(x) + str(y) + str(known_board[x][y])
            game_history_code = game_history_code + "-" + previous_bf

        # 100chars_code encodes current known board state, without recording actual game history
        board_state_100chars_code = fun.update_board_state_100chars_code(known_board)

        hit = False

        # Determine best hit probability field
        print ('looking for best probability field...')

        # A precalculated probability map map file for this known board state would have this name:
        prob_map_history_file_name = game_history_code + ".npy"
        prob_map_100chars_file_name = board_state_100chars_code + ".npy"

        #######################################
        # FIND TARGET AND SAVE CALCULATION DATA
        #######################################

        # Check if the calculation data already exist in the RadioTelegraphic Phonebook (RTP) Library.
        # If so, load the data, find highest probability fields list (with given margin) and pick on e randomly.
        # Margin of 0.05 means that only fields where calculated probability values are above 95% of the
        # maximum found probability qualify as "good fields" to randomly pick from.
        # Choose low margin for more linearized gameplay or higher margin for more variation.

        # occurances = rtp.load_array_from_RTP_lmdb(known_board, LMDB_PATH_TEMPLATE)
        occurances = None
        if occurances is not None:
            good_fields, best_field, total_samples, best_prob = fun.find_best_fields(occurances, margin = 0.002)
            best_field = random.choice(good_fields)
            n_retrieved +=1
            target_methods = target_methods + 'r'


        # If no data found in RTP, check if there is any field that is obvious as a next target
        #obvious_field = fun.check_if_obvious(known_board)
        elif (obvious_field := fun.check_if_obvious(known_board)) is not None:
            best_field = list(obvious_field)
            print("Only one option possible")
            n_obvious +=1
            target_methods = target_methods + 'o'

        # If there is no obvious field, calculate probabilities with montecarlo or advanced approach for current game state
        # That's the key part of this whole project
        else:
            method_choice = None
            n_zeros = board_state_100chars_code.count("0")
            n_2 = board_state_100chars_code.count("2")

            if n_zeros > 0:
                method_choice = "mc"
            else:
                method_choice = "adv"

            if method_choice == "mc":
                print(f'Did not find precalculated board_state_100chars_code: {board_state_100chars_code} - Calculating using MC method...')
                
                if bias_detected:
                    best_field, calc_time, occurances, best_prob = prob_map_montecarlo.\
                        calculate_probs_montecarlo(known_board, conf_level = 2.58, margin_estim = 1.00, margin_highest = 0.12, bias_detected = True, class_weights = class_weights, clf = clf, scaler_params = scaler_params)
                else:
                    best_field, calc_time, occurances, best_prob = prob_map_montecarlo.\
                        calculate_probs_montecarlo(known_board, conf_level = 2.58, margin_estim = 1.00, margin_highest = 0.12)
                n_calc_mc +=1
                target_methods = target_methods + 'm' 

            # if method_choice == "adv":
            #     if not time_at_switch_set:
            #         time_at_switch = round(time.time() - zero_time)
            #         print(f'Time at switch: {time_at_switch} sec')
            #         time_at_switch_set = True
            #     print(f'Did not find precalculated board_state_100chars_code: {board_state_100chars_code} - Calculating using ADV method...')
            #     occurances, best_field, best_prob, total_unique, calc_time, game_reduced_code =\
            #         prob_map_advanced.calculate_probs_advanced(known_board)
            #     n_calc_adv +=1
            #     target_methods = target_methods + 'a'



                # # Save data for param_sens_study_L05
                # if n_zeros <= 50 and n_zeros > 30:
                #     adv_time_new_row_list = [board_state_100chars_code, best_field, best_prob, calc_time]
                #     adv_time_new_row_df = pd.DataFrame([adv_time_new_row_list], columns=['board_state_100chars_code', 'best_field', 'best_prob', 'calc_time'])
                #     adv_time_data = pd.concat([adv_time_data, adv_time_new_row_df], ignore_index=True)
                #     adv_time_data.to_csv(adv_time_data_file_path, index=False)

            # # Add calculated data to the library. Do it only if calculation took > 0.1s
            # # (to avoid overloading the library with lots of data that could be calculated fast in place)
            # if calc_time > 0.1:
            #     # fun.write_to_library (prob_maps_mc_history_dir, occurances, prob_map_history_file_name)
            #     # fun.write_to_library (prob_maps_mc_100chars_dir, occurances, prob_map_100chars_file_name)
            #     rtp.save_to_RTP_lmdb(known_board, occurances, shard_envs)

        ###############################################
        # TARGET SET - NOW SHOOT AND UPDATE KNOWN BOARD
        ###############################################

        print ('Target set to field: ', best_field)
        x = best_field[0]
        y = best_field[1]

        if board_initial[x][y] > 1 and board_initial[x][y] < 6:
            hit = True
            # If true board value contains any ship on the hit field, mark it as "1" in the board know by the shooter
            known_board[x][y] = 1
            hit_count +=1

            # Find the ship that the hit field belongs to. Remove that field from the remaining fields list of that ship
            for ship in ship_fields_left:
                if list(best_field) in ship:
                    ship.remove(list(best_field))
                    # If this was the last field removed from the remaining fields for that ship, it gets sunk.
                    # Then the attacker is informed about the ship's length, so the board_known is updated with
                    # ship's length values in place of '1's and the neighbouring fields become forbidden zone ('7')
                    if not ship:
                        known_board = fun.sink_ship(known_board, ship_fields_initial, best_field)
                    break
        else:
            # If the hit field did not contain any ship segment, mark the field as a miss (7) on the known board
            known_board[x][y] = 7

        moves +=1   # Move completed
        sum_probs = sum_probs + best_prob   # Gathering statistics

        # Move summary
        print(f'hit = {hit}')
        print(f"Known board after shot:\n {known_board}")

        # PROCEED TO NEXT SHOT
        ###########################################

    ###############################################
    # GAME OVER - DISPLAY STATISTICS
    ###############################################

    time_game = round(time.time() - zero_time)
    print('Finished game number', (g_num))
    print ('total moves: ', moves)
    print ('total time: ', time_game)


    new_row_data = [board_initial_100chars_code, game_history_code, moves, time_game, n_retrieved, n_obvious, n_calc_mc, n_calc_adv, target_methods, time_at_switch]
    new_row = pd.DataFrame([new_row_data], columns=['board_initial_100chars_code', 'game_history_code', 'moves', 'time_game', 'n_retrieved', 'n_obvious', 'n_calc_mc', 'n_calc_adv', 'target_methods', 'time_at_switch'])

    if all_games_results is not None:
        all_games_results = pd.concat([all_games_results, new_row], ignore_index=True)
    else:
        all_games_results = new_row.copy()

    # # Save as a .csv file after each completed game
    # all_games_results.to_csv(all_games_results_file_path, index=False)


    moves_avg = all_games_results['moves'].mean()
    time_avg = all_games_results['time_game'].mean()
 
    print(all_games_results)
    print('average moves per game: ', moves_avg)
    print('average time per game: ', time_avg)



    ###################
    # OLD SUMMARY
    ###################
    # moves_all_games = moves_all_games + moves
    # time_all_games = time_all_games + time_game

    # moves_avg = (moves_all_games)/(g_num)
    # time_avg = (time_all_games)/(g_num)

    # print('average moves per game: ', moves_avg)
    # print('average time per game: ', time_avg)

    # sum_hits = sum_hits + hit_count

    # # Calculated average probability of hit per one shot (calculated)
    # prob_avg = sum_probs / moves_all_games * 100

    # # Actual scored hit ratio (percentage)
    # score_perc = sum_hits / moves_all_games * 100

    # print('prob_avg: ', prob_avg)
    # print('score_perc: ', score_perc)