import numpy as np
import copy
import random
from collections import Counter
import math
import time

import fun
import random_board_generator
import reverse_validator
import utilize_ones

###########################################################################
# SUB - FUNCTIONS valid for this script only
###########################################################################

def count_occurances(board: np.ndarray, free_fields: list, hit_unsunk: list, occurances: np.ndarray)\
                    -> np.ndarray:
    """
    Update 'occurances' array by adding +1 to those fields which contain any type of ship segment (values 2-5) in
    respective fields in 'board' array
    :param board: generated board state to count ship occurances on
    :param free_fields: initial list of free fields on the board state before generation (lvl_0)
    :param hit_unsunk: initial list of hit unsunk fields on the board state before generation (lvl_0)
    :param occurances: array used to store number of ship occurances per field, updated each time the function is called
    :return: updated occurances array
    """
    # Ships could be added only on free fields, so it makes sense to search those only
    for field in free_fields:
        x = field[0]
        y = field[1]
        # Do not consider fields containing '1' (members of hit_unsunk)
        if field not in hit_unsunk:
            if board[x][y] > 1 and board[x][y] < 6:
                occurances[x][y] +=1
    return occurances


def calculate_confidence(counts: dict, generated_games: int, conf_level: float) -> dict:
    """
    Calculate confidence intervals for the estimated fraction of each number in 'counts' object in total population,
    with given confidence level

    :param counts: Counter object - dictionary that binds field number (ID 0-99) with ship occurance count (int)
    :param generated_games: number of generated boards = sample size for statistical calculation
    :return: dictionary binding field number (ID 0-99) with its confidence interval of the estimated population fraction
    (list of lower and upper bounds)
    """
    confidence_intervals = {}
    for num, count in counts.items():
        fraction = count / generated_games
        if fraction == 0:  # Avoid division by zero in margin calculation
            margin_of_error = 0
        else:
            margin_of_error = conf_level * math.sqrt(fraction * (1 - fraction) / generated_games)
        confidence_intervals[num] = (fraction - margin_of_error, fraction + margin_of_error)
    return confidence_intervals


def calculate_required_samples(fraction: float, conf_level: float, margin_estim: float) -> float:
    """
    Dynamically calculate the required number of samples for given confidence level and allowed margin of error.

    :param fraction: fraction in sample achieved so far
    :param margin_estim: allowed margin of error in estimation of fraction in population
    :param conf_level: confidence level of the input fraction in sample
    :return: required number of samples
    """

    # For known estimated fraction (estimated fraction in the population = currently achieved fraction in sample)
    if fraction == 0:
        return float('inf')  # Edge case - infinite samples needed (avoid division by zero in the following equation)
    required_samples = math.ceil((conf_level ** 2 * fraction * (1 - fraction)) / (margin_estim ** 2))

    # For unknown estimated fraction (this always yields higher value than in the upper case):
    # required_samples = math.ceil(confidence_level ** 2 / (4 * (margin ** 2)))

    return required_samples


def find_highest_probability_number(counts, generated_games, conf_level, margin_estim, margin_highest, limiter="dummy"):
    done = False
    # Calculate confidence intervals
    confidence_intervals = calculate_confidence(counts, generated_games, conf_level)

    # Identify the field with the highest observed number of occurances and calculate its fraction in generated games
    most_common = counts.most_common(1)[0][0]
    highest_fraction = counts[most_common] / generated_games

    # Calculate the required samples dynamically for statistical significance
    required_samples = calculate_required_samples(highest_fraction, conf_level, margin_estim)

    # TODO graphical explanation for this
    # k = Lower boundary of confidence interval for highest occurances number
    k = confidence_intervals[most_common][0]
    # For each field calculate relative error between k and the upper estimate of occurances number in this field
    # For all fields this relative error must be >= (- margin_highest) to accept the calculation as statistically valid
    is_confident = all(
        (k - confidence_intervals[n][1]) / k >= (- margin_highest)
        for n in counts
    )

    if is_confident and generated_games >= required_samples and generated_games >= 100:
        done = True
        #print(confidence_intervals)
        #print(confidence_intervals[most_common][0])
    if not is_confident and generated_games >= required_samples and generated_games >= 100:
        limiter = "confidence"
    if is_confident and not generated_games >= required_samples and generated_games >= 100:
        limiter = "req_samples"
    if is_confident and generated_games >= required_samples and generated_games < 100:
        limiter = "100_samples"

    return most_common, counts, required_samples, done, limiter

###########################################################################
# MAIN FUNCTION
###########################################################################
def calculate_probs_montecarlo (known_board, conf_level = 2.58, margin_estim = 1.00, margin_highest = 0.12):
    start_time = time.time()
    orientations = ('hor', 'ver')
    generated_games = 0
    limiter = "dummy"

    # Create 1D list of 100 fields and convert to tuple (0 to 99)
    fields = tuple([(x, y) for x in range(10) for y in range(10)])

    # Create 2D np.array of 100 fields to count occurances for each field
    occurances = np.zeros((10, 10), dtype = int)

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

    segments_max = {
        5: 1,
        4: 1,
        3: 2,
        2: 3
    }

    # Free fields = containing either 0 or 1. Only in these fields hypothetical ships can be placed
    # hit_unsunk fields = containing 1.
    free_fields, hit_unsunk = fun.find_free_and_hit_fields(known_board)

    # Find different groups of adjacent '1' fields
    # (rare but possible scenario when we have multiple, non-adjacent ships hit but not sunk)
    groups = fun.group_adjacent_symbols(known_board, 1)

    minimum_lengths = fun.find_minimum_lengths(hit_unsunk, groups)


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

    # All initial status data determined - begin generating random boards fitting to it
    done = False
    all_tries = 0

    while not done:
        all_tries +=1
        known_board = copy.deepcopy(known_board)
        board_hyp_lvl_7 = random_board_generator.generate_random_board(known_board, configs_levels, n_seg_for_lvl,
                                                                       seg5_done, seg4_done, seg3_done, seg2_done,
                                                                       hit_unsunk, minimum_lengths)

        if board_hyp_lvl_7 is None:
            continue

        # REVERSE VALIDATION - optional
        # if reverse_validator.validate(board_hyp_lvl_7) == False:
        #     print("FAILED VALIDATION")
        #     break

        generated_games +=1

        # Count ship occurances for each field (+1 if any ship segment is placed on the field)
        count_occurances(board_hyp_lvl_7, free_fields, hit_unsunk, occurances)

        # Convert occurances array to a Counter object (key = 1D field id, value = occurance count) for better visualisation
        counts = Counter()
        flat_temp = occurances.flatten()
        flat = flat_temp.tolist()
        for id, val in enumerate(flat):
            counts [id] = val

        # Run statistical functions to determine highest probability field
        # If this is possible with desired level of confidence, return done = True and break the loop.
        # result - highest probability field, counts - Counter object,
        # required_samples - required number of samples for statistical confidence
        result, counts, required_samples, done, limiter =\
            find_highest_probability_number(counts, generated_games, conf_level, margin_estim, margin_highest, limiter)
        if done:
            break

    best_field = fields[result]  # 2D array address of best probability field
    print(f"Field with the highest probability: {best_field}")
    print(f"Counts: {counts}")
    print(f"Total games generated: {generated_games}")
    print(f"Total required samples: {required_samples}")

    print(occurances)
    print('all tries = ', all_tries)
    print(f'limiter is: {limiter}')

    print("--- %s seconds ---" % (time.time() - start_time))
    time_mc = round(time.time() - start_time, 2)

    x = best_field[0]
    y = best_field[1]
    best_prob = occurances[x][y] / generated_games

    return best_field, time_mc, occurances, best_prob

###########################################################################
# Only valid if this script is run as __main__ with predefined board
###########################################################################

if __name__ == "__main__":

    known_board = np.zeros((10, 10), dtype = "int16")

    known_board[0] = [5, 5, 5, 5, 5, 7, 7, 3, 3, 3]
    known_board[1] = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    known_board[2] = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    known_board[3] = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    known_board[4] = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    known_board[5] = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    known_board[6] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    known_board[7] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    known_board[8] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    known_board[9] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    time_avg = 0
    n_runs = 1
    for n in range (0, n_runs):
        best_field, time_mc, occurances, best_prob = calculate_probs_montecarlo (known_board)
        time_avg = time_avg + time_mc
        print(f'{n+1} finished out of {n_runs}')
    print (f'average time out of {n_runs} runs: {time_avg / n_runs}')
