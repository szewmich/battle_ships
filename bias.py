import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import copy

import fun_8x8 as fun
import random_board_generator_8x8 as random_board_generator

def get_avg_orients (groups):
    avg_2seg_orient = 0
    avg_3seg_orient = 0
    for ship in groups:
        if len(ship) == 2:
            (x1,y1), (x2,y2) = ship
            if x1 == x2:
                avg_2seg_orient += 1    # positive = vertical
            elif y1 == y2:
                avg_2seg_orient -= 1    # negative = horizontal
            else:
                raise ValueError("Invalid ship coordinates")
        if len(ship) == 3:
            (x1,y1), (x2,y2), (x3,y3) = ship
            if x1 == x2:
                avg_3seg_orient += 1
            elif y1 == y2:
                avg_3seg_orient -= 1
            else:
                raise ValueError("Invalid ship coordinates")
            
    if abs(avg_2seg_orient) == 2:
        seg2_same_orient = True
    else:
        seg2_same_orient = False

    if avg_2seg_orient * avg_3seg_orient > 0:
        seg2_3_same_orient = True
    else:
        seg2_3_same_orient = False

    return avg_2seg_orient, avg_3seg_orient, seg2_same_orient, seg2_3_same_orient


def get_avg_distance_to_edges (groups, size):
    
    seg2_dist_to_left = 0
    seg2_dist_to_right = 0
    seg2_dist_to_upper = 0
    seg2_dist_to_bottom = 0

    seg3_dist_to_left = 0
    seg3_dist_to_right = 0
    seg3_dist_to_upper = 0
    seg3_dist_to_bottom = 0

    for ship in groups:
        if len(ship) == 2:
            (x1,y1), (x2,y2) = ship

            seg2_dist_to_left += min(y1, y2)
            seg2_dist_to_right += min(size - 1 - y1, size - 1 - y2)

            seg2_dist_to_upper += min(x1, x2)
            seg2_dist_to_bottom += min(size - 1 - x1, size - 1 - x2)
            
        if len(ship) == 3:
            (x1,y1), (x2,y2), (x3,y3) = ship

            seg3_dist_to_left += min(y1, y2, y3)
            seg3_dist_to_right += min(size - 1 - y1, size - 1 - y2, size - 1 - y3)

            seg3_dist_to_upper += min(x1, x2, x3)
            seg3_dist_to_bottom += min(size - 1 - x1, size - 1 - x2, size - 1 - x3)
    
    return (seg2_dist_to_left / 2, seg2_dist_to_right / 2, seg2_dist_to_upper / 2, seg2_dist_to_bottom / 2,
            seg3_dist_to_left, seg3_dist_to_right, seg3_dist_to_upper, seg3_dist_to_bottom)


def get_avg_spreads (groups):

    def get_centroid (ship):
        xs = [x for x,y in ship]
        ys = [y for x,y in ship]
        return (sum(xs)/len(xs), sum(ys)/len(ys))

    def distance (p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
    
    spread_2_2 = 0
    spread_2_3 = 0
    spread_all = 0

    #find pair of centroids of seg2 ships
    seg2_centroids = [get_centroid(ship) for ship in groups if len(ship) == 2]
    seg3_centroid = [get_centroid(ship) for ship in groups if len(ship) == 3]

    spread_2_2 += distance(seg2_centroids[0], seg2_centroids[1])
    spread_2_3 += distance(seg2_centroids[0], seg3_centroid[0])
    spread_2_3 += distance(seg2_centroids[1], seg3_centroid[0])
    spread_all = spread_2_2 + spread_2_3

    avg_spread_2_2 = spread_2_2
    avg_spread_2_3 = spread_2_3 / 2
    avg_spread_all = spread_all / 3

    return avg_spread_2_2, avg_spread_2_3, avg_spread_all


def get_ships_at_edges (groups, size):
    segments_of_ships_at_edges = 0

    for ship in groups:
        (x1, y1) = ship[0]
        (x2, y2) = ship[1]
            
        if x1 == x2:
            if x1 == 0 or x1 == size - 1:
                segments_of_ships_at_edges += len(ship)
        elif y1 == y2:
            if y1 == 0 or y1 == size - 1:
                segments_of_ships_at_edges += len(ship)
    
    return segments_of_ships_at_edges


def calculate_bias_metrics(known_board):
    binary_mask = np.logical_or(known_board == 3, known_board == 2).astype(int)

    groups = fun.group_adjacent_symbols(binary_mask, 1)

    avg_2seg_orient, avg_3seg_orient, seg2_same_orient, seg2_3_same_orient = get_avg_orients (groups)

    (seg2_dist_to_left, seg2_dist_to_right, seg2_dist_to_upper, seg2_dist_to_bottom, seg3_dist_to_left, seg3_dist_to_right, seg3_dist_to_upper, seg3_dist_to_bottom) = get_avg_distance_to_edges (groups, known_board.shape[0])

    avg_spread_2_2, avg_spread_2_3, avg_spread_all = get_avg_spreads (groups)

    segments_of_ships_at_edges = get_ships_at_edges (groups, known_board.shape[0])

    return {
        "avg_2seg_orient": avg_2seg_orient,
        "avg_3seg_orient": avg_3seg_orient,
        "seg2_same_orient": seg2_same_orient,
        "seg2_3_same_orient": seg2_3_same_orient,
        "seg2_dist_to_left": seg2_dist_to_left,
        "seg2_dist_to_right": seg2_dist_to_right,
        "seg2_dist_to_upper": seg2_dist_to_upper,
        "seg2_dist_to_bottom": seg2_dist_to_bottom,
        "seg3_dist_to_left": seg3_dist_to_left,
        "seg3_dist_to_right": seg3_dist_to_right,
        "seg3_dist_to_upper": seg3_dist_to_upper,
        "seg3_dist_to_bottom": seg3_dist_to_bottom,
        "avg_spread_2_2": avg_spread_2_2,
        "avg_spread_2_3": avg_spread_2_3,
        "avg_spread_all": avg_spread_all,
        "segments_of_ships_at_edges": segments_of_ships_at_edges
    }


def generate_board_from_empty(dim):
    board_hyp_lvl_7 = None
    while board_hyp_lvl_7 is None:
        known_board = np.array([[0] * dim for i in range(dim)])

        free_fields, hit_unsunk = fun.find_free_and_hit_fields(known_board)

        # Find different groups of adjacent '1' fields
        # (rare but possible scenario when we have multiple, non-adjacent ships hit but not sunk)
        groups = fun.group_adjacent_symbols(known_board, 1)

        minimum_lengths = fun.find_minimum_lengths(hit_unsunk, groups)

        # Counter of how many ships of each type are already sunk
        seg5_done = fun.n_segments(known_board, 5) / 5
        seg4_done = fun.n_segments(known_board, 4) / 4
        seg3_done = fun.n_segments(known_board, 3) / 3
        seg2_done = fun.n_segments(known_board, 2) / 2

        segments_done = {
            5: seg5_done,
            4: seg4_done,
            3: seg3_done,
            2: seg2_done
        }


        # 7 levels of hypothetical board, starting from placement of a 5-seg ship, ending with 2-segs
        n_seg_for_lvl = {
            1: 5,
            2: 4,
            3: 3,
            4: 3,
            5: 2,
            6: 2,
            7: 2
        }

        # Configs_levels = all valid combinations of a free field and orientation (hor/ver) for each ship length.
        configs_levels = random_board_generator.create_configs (known_board, free_fields, n_seg_for_lvl)

        board_hyp_lvl_7 = random_board_generator.generate_random_board(known_board, configs_levels, n_seg_for_lvl,
                                                                                seg5_done, seg4_done, seg3_done, seg2_done,
                                                                                hit_unsunk, minimum_lengths, game_rules_n_ships={5:0,4:0,3:1,2:2})
    return board_hyp_lvl_7


def calculate_confidence_interval(data, confidence=0.95):
    
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    h = sem * stats.t.ppf((1 + confidence) / 2., n-1)  # Margin of error

    return (mean - h, mean + h)


def confidence_intervals_overlap(cA, cB):
    # cA and cB are tuples (lower_bound, upper_bound)
    return not (cA[1] < cB[0] or cB[1] < cA[0])


if __name__ == "__main__":
    size = 6
    # known_board = np.zeros((size, size), dtype = "int16")

    # known_board[0] = [0, 0, 0, 2, 2, 0]
    # known_board[1] = [0, 3, 0, 0, 0, 0]
    # known_board[2] = [0, 3, 0, 0, 0, 0]
    # known_board[3] = [0, 3, 0, 2, 2, 0]
    # known_board[4] = [0, 0, 0, 0, 0, 0]
    # known_board[5] = [0, 0, 0, 0, 0, 0]

    # metrics = calculate_bias_metrics(known_board)

    # print("avg_2seg_orient:", avg_2seg_orient)
    # print("avg_3seg_orient:", avg_3seg_orient)

    # print("seg2_same_orient:", seg2_same_orient)
    # print("seg2_3_same_orient:", seg2_3_same_orient)

    # ################################################
    # avg_metrics = {
    #     "avg_2seg_orient":              0,
    #     "avg_3seg_orient":              0,
    #     "seg2_same_orient":             0,
    #     "seg2_3_same_orient":           0,
    #     "seg2_dist_to_left":            0,
    #     "seg2_dist_to_right":           0,
    #     "seg2_dist_to_upper":           0,
    #     "seg2_dist_to_bottom":          0,
    #     "seg3_dist_to_left":            0,
    #     "seg3_dist_to_right":           0,
    #     "seg3_dist_to_upper":           0,
    #     "seg3_dist_to_bottom":          0,
    #     "avg_spread_2_2":               0,
    #     "avg_spread_2_3":               0,
    #     "avg_spread_all":               0,
    #     "segments_of_ships_at_edges":   0
    # }

    # avg_metrics_for_history = copy.deepcopy(avg_metrics)
    # metrics_data = {key: [] for key in avg_metrics}
    # metrics_CI = {}

    # n_samples = 1000
    # history = np.zeros((n_samples, len(avg_metrics)))

    # for k in range(n_samples):
    #     board = generate_board_from_empty(size)
    #     current_metrics = calculate_bias_metrics(board)
    #     for key in current_metrics:
    #         metrics_data[key].append(current_metrics[key])
    #         avg_metrics[key] += current_metrics[key]
    #         avg_metrics_for_history[key] = avg_metrics[key] / (k+1)
    #     history[k] = [avg_metrics_for_history[key] for key in avg_metrics]

    # for key in avg_metrics:
    #     avg_metrics[key] /= n_samples   

    # for key in metrics_data:
    #     metrics_CI[key] = calculate_confidence_interval(metrics_data[key], confidence=0.95)
    
    # df = pd.DataFrame([metrics_CI])

    # metrics_CI_1 = copy.deepcopy(metrics_CI)

    # print(avg_metrics) 
    # print(df.T)
    # #print(history)
    # ############################################
    # ################################################
    # avg_metrics = {
    #     "avg_2seg_orient":              0,
    #     "avg_3seg_orient":              0,
    #     "seg2_same_orient":             0,
    #     "seg2_3_same_orient":           0,
    #     "seg2_dist_to_left":            0,
    #     "seg2_dist_to_right":           0,
    #     "seg2_dist_to_upper":           0,
    #     "seg2_dist_to_bottom":          0,
    #     "seg3_dist_to_left":            0,
    #     "seg3_dist_to_right":           0,
    #     "seg3_dist_to_upper":           0,
    #     "seg3_dist_to_bottom":          0,
    #     "avg_spread_2_2":               0,
    #     "avg_spread_2_3":               0,
    #     "avg_spread_all":               0,
    #     "segments_of_ships_at_edges":   0
    # }

    # avg_metrics_for_history = copy.deepcopy(avg_metrics)
    # metrics_data = {key: [] for key in avg_metrics}
    # metrics_CI = {}

    # n_samples = 1000
    # history = np.zeros((n_samples, len(avg_metrics)))

    # for k in range(n_samples):
    #     board = generate_board_from_empty(size)
    #     current_metrics = calculate_bias_metrics(board)
    #     for key in current_metrics:
    #         metrics_data[key].append(current_metrics[key])
    #         avg_metrics[key] += current_metrics[key]
    #         avg_metrics_for_history[key] = avg_metrics[key] / (k+1)
    #     history[k] = [avg_metrics_for_history[key] for key in avg_metrics]

    # for key in avg_metrics:
    #     avg_metrics[key] /= n_samples   

    # for key in metrics_data:
    #     metrics_CI[key] = calculate_confidence_interval(metrics_data[key], confidence=0.95)
    
    # df = pd.DataFrame([metrics_CI])

    # metrics_CI_2 = copy.deepcopy(metrics_CI)
    # print(avg_metrics) 
    # print(df.T)
    # #print(history)
    # ############################################

    # overlaps = {}

    # for key in metrics_CI_1:
    #     overlaps[key] = confidence_intervals_overlap(metrics_CI_1[key], metrics_CI_2[key])

    # print(overlaps)

    # plt.figure(figsize=(16, 12))
    # for i, key in enumerate(avg_metrics):
    #     plt.subplot(4, 4, i + 1)
    #     plt.plot(history[:, i], marker='o')
    #     plt.title(key, fontsize = 9)
    #     plt.title
    #     plt.xlabel('Sample Number')
    #     plt.ylabel('Value')
    #     plt.grid(True)
    
    # plt.show()

    random_boards = pd.read_csv("bias_checks\\random_boards.csv")['board'].tolist()
    biased_boards = pd.read_csv("bias_checks\\biased_boards.csv")['board'].tolist()

    #print(biased_boards)


    random_boards_metrics = {
        "avg_2seg_orient":              [],
        "avg_3seg_orient":              [],
        "seg2_same_orient":             [],
        "seg2_3_same_orient":           [],
        "seg2_dist_to_left":            [],
        "seg2_dist_to_right":           [],
        "seg2_dist_to_upper":           [],
        "seg2_dist_to_bottom":          [],
        "seg3_dist_to_left":            [],
        "seg3_dist_to_right":           [],
        "seg3_dist_to_upper":           [],
        "seg3_dist_to_bottom":          [],
        "avg_spread_2_2":               [],
        "avg_spread_2_3":               [],
        "avg_spread_all":               [],
        "segments_of_ships_at_edges":   []
    }
    biased_boards_metrics = copy.deepcopy(random_boards_metrics)

    random_boards_CI = {}
    biased_boards_CI = {}

    ###################
    for board_code in random_boards:
        board_array = np.array([int(x) for x in board_code]).reshape((6,6))
        #print(board_array)
        current_metrics = calculate_bias_metrics(board_array)

        for key in current_metrics:
            random_boards_metrics[key].append(current_metrics[key])

    for key in random_boards_metrics:
        random_boards_CI[key] = calculate_confidence_interval(random_boards_metrics[key], confidence=0.95)

    print("random boards:")
    print(random_boards_CI)
    ###################

    ###################
    for board_code in biased_boards:
        board_array = np.array([int(x) for x in board_code]).reshape((6,6))
        # print(board_array)
        current_metrics = calculate_bias_metrics(board_array)

        for key in current_metrics:
            biased_boards_metrics[key].append(current_metrics[key])

    for key in biased_boards_metrics:
        biased_boards_CI[key] = calculate_confidence_interval(biased_boards_metrics[key], confidence=0.95)

    print("biased boards:")
    print(biased_boards_CI)
    ###################

    overlaps = {}
    for key in random_boards_CI:
        overlaps[key] = confidence_intervals_overlap(random_boards_CI[key], biased_boards_CI[key])

    overlaps_df = pd.DataFrame([overlaps]).T
    print(overlaps_df)

    detected_bias = overlaps_df[overlaps_df[0] == False]

    #in detected_bias dataframe, add columns with mean metrics values from random_boards_metrics and biased_boards_metrics
    detected_bias['random_boards_mean'] = detected_bias.index.map(lambda key: np.mean(random_boards_metrics[key]))
    detected_bias['biased_boards_mean'] = detected_bias.index.map(lambda key: np.mean(biased_boards_metrics[key]))

    print(detected_bias)