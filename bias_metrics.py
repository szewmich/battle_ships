import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import copy
from collections import Counter
import matplotlib as mpl
from math import sqrt, acos, pi, log2

import fun_dim as fun
import random_board_generator_dim as random_board_generator
import prob_map_montecarlo_dim

#########################################################################################
# HELPER FUNCTIONS

def get_centroid (ship):
    xs = [x for x,y in ship]
    ys = [y for x,y in ship]
    return (sum(xs)/len(xs), sum(ys)/len(ys))

def distance (p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def angle(a, b, c):
    """
    Compute angle ABC where A, B, C are (x,y).
    Returns angle in radians.
    """
    BA = np.array(a) - np.array(b)
    BC = np.array(c) - np.array(b)
    denom = np.linalg.norm(BA) * np.linalg.norm(BC)
    if denom == 0:
        return 0.0
    cosv = np.clip(np.dot(BA, BC) / denom, -1.0, 1.0)
    return acos(cosv)

#########################################################################################
# METRICS 

def get_avg_orients (groups, board):
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
    
    seg2_dist_to_left /= 2
    seg2_dist_to_right /= 2
    seg2_dist_to_upper /= 2
    seg2_dist_to_bottom /= 2
    
    return (seg2_dist_to_left, seg2_dist_to_right, seg2_dist_to_upper, seg2_dist_to_bottom,
            seg3_dist_to_left, seg3_dist_to_right, seg3_dist_to_upper, seg3_dist_to_bottom)

def get_avg_spreads (groups):
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
    seg2_ships_whole_at_edge = 0
    seg3_ships_whole_at_edge = 0
    total_ships_whole_at_edge = 0

    seg2_ships_touching_edge = 0
    seg3_ships_touching_edge = 0
    total_ships_touching_edge = 0

    for ship in groups:
        (x1, y1) = ship[0]
        (x2, y2) = ship[1]
        if len(ship) == 3:
            (x3, y3) = ship[2]
            
        # if horizontal
        if x1 == x2:   
            if len(ship) == 2:  
                # if whole ship is on board's edge
                if x1 == 0 or x1 == size - 1:
                    seg2_ships_whole_at_edge += 1
                # if ship is only touching board's edge
                elif y1 == 0 or y1 == size - 1 or y2 == 0 or y2 == size - 1:
                    seg2_ships_touching_edge += 1

            if len(ship) == 3:                         
                # if whole ship is on board's edge
                if x1 == 0 or x1 == size - 1:
                    seg3_ships_whole_at_edge += 1
                # if ship is only touching board's edge
                elif y1 == 0 or y1 == size - 1 or y3 == 0 or y3 == size - 1:
                    seg3_ships_touching_edge += 1

        # if vertical
        elif y1 == y2:                              
            if len(ship) == 2:  
                # if whole ship is on board's edge
                if y1 == 0 or y1 == size - 1:
                    seg2_ships_whole_at_edge += 1
                # if ship is only touching board's edge
                elif x1 == 0 or x1 == size - 1 or x2 == 0 or x2 == size - 1:
                    seg2_ships_touching_edge += 1

            if len(ship) == 3:                         
                # if whole ship is on board's edge
                if y1 == 0 or y1 == size - 1:
                    seg3_ships_whole_at_edge += 1
                # if ship is only touching board's edge
                elif x1 == 0 or x1 == size - 1 or x3 == 0 or x3 == size - 1:
                    seg3_ships_touching_edge += 1

    total_ships_whole_at_edge = seg2_ships_whole_at_edge + seg3_ships_whole_at_edge
    total_ships_touching_edge = seg2_ships_touching_edge + seg3_ships_touching_edge

    return (seg2_ships_whole_at_edge, seg3_ships_whole_at_edge, total_ships_whole_at_edge,
            seg2_ships_touching_edge, seg3_ships_touching_edge, total_ships_touching_edge)


def triangle_configuration(groups):
    groups_sorted = sorted(groups, key=lambda ship: len(ship))
    
    centroids = [get_centroid (ship) for ship in groups_sorted]
    A, B, C = centroids # c is seg3 ship
    angle_C = np.abs(angle(A, C, B))

    # calculate area of triangle:
    a = distance(B, C)
    b = distance(A, C)
    c = distance(A, B)
    s = (a + b + c) / 2
    area = sqrt(s * (s - a) * (s - b) * (s - c))

    circumference = a + b + c

    return angle_C, area, circumference


def mutual_exclusion(groups):
    clear_board = np.zeros((6,6))
    shadow_area = np.zeros((6,6))
    for ship in groups:
        forbidden_area = fun.fill_adjacent(clear_board, ship, len(ship))
        forbidden_area = (forbidden_area == 7).astype(int)
        shadow_area += forbidden_area
    
    n_cells = np.sum(shadow_area > 1)
    return n_cells
    

def symmetry_score(board):

    board = np.array(board)
    hor = np.mean(board == np.flip(board, axis=1))
    ver = np.mean(board == np.flip(board, axis=0))

    # center point symmetry
    cent = np.mean(board == np.flip(np.flip(board, axis=1), axis=0))

    return hor, ver, cent


def free_cells(board):
    board = np.array(board)
    n_free = np.sum(board == 0)
    return n_free




#########################################################################################
# PIPELINE

def calculate_bias_metrics(known_board):
    binary_mask = np.logical_or(known_board == 3, known_board == 2).astype(int)

    groups = fun.group_adjacent_symbols(binary_mask, 1)

    avg_2seg_orient, avg_3seg_orient, seg2_same_orient, seg2_3_same_orient = get_avg_orients (groups, binary_mask)

    (seg2_dist_to_left, seg2_dist_to_right, seg2_dist_to_upper, seg2_dist_to_bottom, seg3_dist_to_left, seg3_dist_to_right, seg3_dist_to_upper, seg3_dist_to_bottom) = get_avg_distance_to_edges (groups, known_board.shape[0])

    avg_spread_2_2, avg_spread_2_3, avg_spread_all = get_avg_spreads (groups)

    (seg2_ships_whole_at_edge, seg3_ships_whole_at_edge, total_ships_whole_at_edge,
     seg2_ships_touching_edge, seg3_ships_touching_edge, total_ships_touching_edge) = get_ships_at_edges (groups, known_board.shape[0])

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
        "seg2_ships_whole_at_edge": seg2_ships_whole_at_edge,
        "seg3_ships_whole_at_edge": seg3_ships_whole_at_edge,
        "total_ships_whole_at_edge": total_ships_whole_at_edge,
        "seg2_ships_touching_edge": seg2_ships_touching_edge,
        "seg3_ships_touching_edge": seg3_ships_touching_edge,
        "total_ships_touching_edge": total_ships_touching_edge
    }


def calculate_confidence_interval(data, confidence=0.95):
    
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    h = sem * stats.t.ppf((1 + confidence) / 2., n-1)  # Margin of error

    return (mean - h, mean + h)


def confidence_intervals_overlap(cA, cB):
    # cA and cB are tuples (lower_bound, upper_bound)
    return not (cA[1] < cB[0] or cB[1] < cA[0])


def make_metrics_dict():
    metrics = {
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
        "seg2_ships_whole_at_edge":     [],
        "seg3_ships_whole_at_edge":     [],
        "total_ships_whole_at_edge":    [],
        "seg2_ships_touching_edge":     [],
        "seg3_ships_touching_edge":     [],
        "total_ships_touching_edge":    [],
    }
    return metrics


def get_metrics_CI (boards_list):
    temp_boards_metrics = make_metrics_dict()
    temp_boards_CI = {}
    for board_code in boards_list:
        board_array = np.array([int(x) for x in board_code]).reshape((6,6))
        current_metrics = calculate_bias_metrics(board_array)

        for key in current_metrics:
            temp_boards_metrics[key].append(current_metrics[key])

    for key in temp_boards_metrics:
        temp_boards_CI[key] = calculate_confidence_interval(temp_boards_metrics[key], confidence=0.95)

    return temp_boards_metrics, temp_boards_CI


def get_overlaps_df(random_boards_CI, biased_boards_CI, random_boards_metrics, biased_boards_metrics):
    overlaps = {}
    for key in random_boards_CI:
        overlaps[key] = confidence_intervals_overlap(random_boards_CI[key], biased_boards_CI[key])

    overlaps_df = pd.DataFrame([overlaps]).T
    print("Metrics overlap between random boards and biased boards:")
    print(overlaps_df)

    detected_bias = overlaps_df[overlaps_df[0] == False]

    #in detected_bias dataframe, add columns with mean metrics values from random_boards_metrics and biased_boards_metrics
    detected_bias['random_boards_mean'] = detected_bias.index.map(lambda key: np.mean(random_boards_metrics[key]))
    detected_bias['biased_boards_mean'] = detected_bias.index.map(lambda key: np.mean(biased_boards_metrics[key]))

    print("Mean values of metrics with detected bias")
    print(detected_bias)


def ship_heatmaps(board_set):
    seg3_heatmap = np.zeros((6,6))
    seg2_heatmap = np.zeros((6,6))
    for board_code in board_set:
        board_array = np.array([int(x) for x in board_code]).reshape((6,6))
        for x in range(6):
            for y in range(6):
                if board_array[x,y] == 3:
                    seg3_heatmap[x,y] += 1
                if board_array[x,y] == 2:
                    seg2_heatmap[x,y] += 1

    return seg3_heatmap, seg2_heatmap


def heatmap_conf_intervals(ship_heatmap, n_samples, conf_level=1.96):

    # Convert occurances array to a Counter object (key = 1D field id, value = occurance count) for better visualisation
    counts = Counter()
    flat_temp = ship_heatmap.flatten()
    flat = flat_temp.tolist()
    for id, val in enumerate(flat):
        counts [id] = val

    ship_heatmap_CI = prob_map_montecarlo_dim.calculate_confidence(counts, n_samples, conf_level=1.96)

    return ship_heatmap_CI




def plot_heatmaps_with_CI_overlaps(seg3_heatmap_random, seg2_heatmap_random,
                                   seg3_heatmap_biased, seg2_heatmap_biased,
                                   seg3_heatmap_random_CI, seg2_heatmap_random_CI,
                                   seg3_heatmap_biased_CI, seg2_heatmap_biased_CI,
                                   n1 = 1000, n2 = 1000, n3 = 1000,
                                   first_title = "Random Boards",
                                   second_title = "Biased Boards",
                                   third_title = "CI Overlap (Red=No Overlap)",
                                   fourth_title = "CI Overlap (Red=No Overlap)"):

    seg3_heatmap_fr_random = seg3_heatmap_random / n1
    seg2_heatmap_fr_random = seg2_heatmap_random / n1

    seg3_heatmap_fr_biased = seg3_heatmap_biased / n2
    seg2_heatmap_fr_biased = seg2_heatmap_biased / n2

    seg3_overlap_array = np.zeros((6,6), dtype=bool)
    seg2_overlap_array = np.zeros((6,6), dtype=bool)

    for x in range(6):
        for y in range(6):
            field_id = x * 6 + y
            seg3_overlap_array[x,y] = confidence_intervals_overlap(seg3_heatmap_random_CI[field_id], seg3_heatmap_biased_CI[field_id])
            seg2_overlap_array[x,y] = confidence_intervals_overlap(seg2_heatmap_random_CI[field_id], seg2_heatmap_biased_CI[field_id])
    
    # compute common color limits and norm
    vmin = min(d.min() for d in (seg3_heatmap_fr_random, seg3_heatmap_fr_biased))
    vmax = max(d.max() for d in (seg3_heatmap_fr_random, seg3_heatmap_fr_biased))
    norm_seg3 = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # compute common color limits and norm
    vmin = min(d.min() for d in (seg2_heatmap_fr_random, seg2_heatmap_fr_biased))
    vmax = max(d.max() for d in (seg2_heatmap_fr_random, seg2_heatmap_fr_biased))
    norm_seg2 = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    norm_true_false = mpl.colors.Normalize(0, 1)

    #plot heatmaps side by side for comparison
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    axs[0, 0].imshow(seg3_heatmap_fr_random, cmap='hot', interpolation='nearest', norm=norm_seg3)
    axs[0, 0].set_title(f'3-seg ships - {first_title}')
    axs[0, 1].imshow(seg3_heatmap_fr_biased, cmap='hot', interpolation='nearest', norm=norm_seg3)
    axs[0, 1].set_title(f'3-seg ships - {second_title}')
    axs[1, 0].imshow(seg2_heatmap_fr_random, cmap='hot', interpolation='nearest', norm=norm_seg2)
    axs[1, 0].set_title(f'2-seg ships - {first_title}')
    axs[1, 1].imshow(seg2_heatmap_fr_biased, cmap='hot', interpolation='nearest', norm=norm_seg2)
    axs[1, 1].set_title(f'2-seg ships - {second_title}')
    axs[0, 2].imshow(seg3_overlap_array, cmap='RdYlGn', interpolation='nearest', norm = norm_true_false)
    axs[0, 2].set_title('3-seg ships - CI Overlap (Red=No Overlap)')
    axs[1, 2].imshow(seg2_overlap_array, cmap='RdYlGn', interpolation='nearest', norm = norm_true_false)
    axs[1, 2].set_title('2-seg ships - CI Overlap (Red=No Overlap)')

    # add colorbar for seg3 heatmaps based on norm_seg3
    cbar_seg3 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_seg3, cmap='hot'), ax=axs[0, :2], orientation='vertical', fraction = 0.5)
    cbar_seg3.set_label('3-seg Ship fraction')

    # add colorbar for seg3 heatmaps based on norm_seg2
    cbar_seg2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_seg2, cmap='hot'), ax=axs[1, :2], orientation='vertical', fraction = 0.5)
    cbar_seg2.set_label('2-seg Ship fraction')

    plt.tight_layout()
    plt.show()


def detect_bias(random_boards, biased_boards):
    random_boards_metrics = make_metrics_dict()
    biased_boards_metrics = make_metrics_dict()
    temp_boards_metrics = make_metrics_dict()

    random_boards_CI = {}
    biased_boards_CI = {}
    # temp_boards_CI = {}

    random_boards_metrics, random_boards_CI = get_metrics_CI(random_boards)
    biased_boards_metrics, biased_boards_CI = get_metrics_CI(biased_boards)

    get_overlaps_df(random_boards_CI, biased_boards_CI, random_boards_metrics, biased_boards_metrics)

    (seg3_heatmap_random, seg2_heatmap_random) = ship_heatmaps(random_boards)
    (seg3_heatmap_biased, seg2_heatmap_biased) = ship_heatmaps(biased_boards)

    seg3_heatmap_random_CI = heatmap_conf_intervals(seg3_heatmap_random, len(random_boards), conf_level=1.96)
    seg2_heatmap_random_CI = heatmap_conf_intervals(seg2_heatmap_random, len(random_boards), conf_level=1.96)

    seg3_heatmap_biased_CI = heatmap_conf_intervals(seg3_heatmap_biased, len(biased_boards), conf_level=1.96)
    seg2_heatmap_biased_CI = heatmap_conf_intervals(seg2_heatmap_biased, len(biased_boards), conf_level=1.96)


    plot_heatmaps_with_CI_overlaps(seg3_heatmap_random, seg2_heatmap_random,
                                    seg3_heatmap_biased, seg2_heatmap_biased,
                                    seg3_heatmap_random_CI, seg2_heatmap_random_CI,
                                    seg3_heatmap_biased_CI, seg2_heatmap_biased_CI,
                                    n1 = len(random_boards), n2 = len(biased_boards), n3 = 1000,)
    
#########################################################################################

