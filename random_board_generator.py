import random
import numpy as np
import copy

import fun


def create_configs(board_hyp_lvl_0: np.ndarray, free_fields_lvl_0: list, n_seg_for_lvl: dict) -> dict:
    """Creates all valid (compliant with game rules) combinations of a free field and orientation (hor/ver)
    for each hypothetical ship on currently considered board setup.

    Returns dictionary with hypothesis levels (1-7) as keys and their valid configs lists as items"""

    # Config = combination of a free field and orientation (hor/ver).
    configs_levels = {}
    orientations = ('hor', 'ver')

    # Create list of all configs for each level
    for lvl in range(1, 8):
        configs_levels[lvl] = []
        for f in free_fields_lvl_0:
            for ori in orientations:
                configs_levels[lvl].append((f, ori))

        # Loop through all configs and rule out those which upon placing the ship do not produce a valid board
        # (because of conflicts with other fields)
        # configs_temp - auxiliary copy to use in loop while items are being removed from the original configs list
        configs_temp = copy.deepcopy(configs_levels[lvl])
        for config in configs_temp:
            field = config[0]
            orient = config[1]
            # Try placing a hypothetical ship of given size in given config on current board setup
            board_hyp_lvl_n = fun.place_ship(board_hyp_lvl_0, n_seg_for_lvl[lvl], orient, field)
            # If placing ship with this config does not produce valid board, remove this config from configs_levels
            if board_hyp_lvl_n is None:
                configs_levels[lvl].remove(config)

    return configs_levels


def generate_random_board(board_hyp_lvl_0: np.ndarray, configs_levels: dict, n_seg_for_lvl: dict,
                          seg5_done: int, seg4_done: int, seg3_done: int, seg2_done: int,
                          hit_unsunk_lvl_0: list, minimum_lengths: list) -> np.ndarray or None:
    """
    Generates random board setup from given initial board state
    :param board_hyp_lvl_0: initial board state
    :param configs_levels: valid configs for each level
    :param n_seg_for_lvl: ship lengths to be placed at each level
    :param seg5_done: number of seg5 ships present on initial board
    :param seg4_done: number of seg4 ships present on initial board
    :param seg3_done: number of seg2 ships present on initial board
    :param seg2_done: number of seg2 ships present on initial board
    :param hit_unsunk_lvl_0: list of fields containing '1' value on initial board
    :param minimum_lengths: list of minimum values that fields of corresponding indices in hit_unsunk_lvl_0 can turn into
    :return: hypothetical board with all ships placed that is compliant with the initial state and game rules
             OR None if validation fails at any stage
    """

    # Place ships on each level sequentially by picking random configurations available per level.
    # If at any point the returned board is None, then scrap whole attempt and return None.

    # Retrying ship placement at the failed level would lead to statistically incorrect outcome! That is because it
    # would enforce the function to generate board setup that might be very unlikely due to previous level choices
    # (they might occupy lots of space so that there are very few places left to fit remaining ships) and thus
    # artificially increase probability of such rare configurations.

    # BEGIN LVL 1
    lvl = 1
    if seg5_done == 1:  # if initial board had more than 0 seg5 ships done, pass unmodified board into next level
        board_hyp_lvl_1 = board_hyp_lvl_0
    else:
        config = random.choice(configs_levels[lvl])
        field = config[0]
        orient = config[1]
        board_hyp_lvl_1 = fun.place_ship(board_hyp_lvl_0, n_seg_for_lvl[lvl], orient, field)
        if board_hyp_lvl_1 is None:
            return None
    # BEGIN LVL 2
    lvl = 2
    if seg4_done == 1:  # if initial board had more than 0 seg4 ships done, pass unmodified board into next level
        board_hyp_lvl_2 = board_hyp_lvl_1
    else:
        config = random.choice(configs_levels[lvl])
        field = config[0]
        orient = config[1]
        board_hyp_lvl_2 = fun.place_ship(board_hyp_lvl_1, n_seg_for_lvl[lvl], orient, field)
        if board_hyp_lvl_2 is None:
            return None
    # BEGIN LVL 3
    lvl = 3
    if seg3_done == 2:  # if initial board had more than 1 seg3 ships done, pass unmodified board into next level
        board_hyp_lvl_3 = board_hyp_lvl_2
    else:
        config = random.choice(configs_levels[lvl])
        field = config[0]
        orient = config[1]
        board_hyp_lvl_3 = fun.place_ship(board_hyp_lvl_2, n_seg_for_lvl[lvl], orient, field)
        if board_hyp_lvl_3 is None:
            return None
    # BEGIN LVL 4
    lvl = 4
    if seg3_done > 0:  # if initial board had more than 0 seg3 ships done, pass unmodified board into next level
        board_hyp_lvl_4 = board_hyp_lvl_3
    else:
        config = random.choice(configs_levels[lvl])
        field = config[0]
        orient = config[1]
        board_hyp_lvl_4 = fun.place_ship(board_hyp_lvl_3, n_seg_for_lvl[lvl], orient, field)
        if board_hyp_lvl_4 is None:
            return None
    # BEGIN LVL 5
    lvl = 5
    if seg2_done > 2:   # if initial board had more than 2 seg2 ships done, pass unmodified board into next level
        board_hyp_lvl_5 = board_hyp_lvl_4
    else:
        config = random.choice(configs_levels[lvl])
        field = config[0]
        orient = config[1]
        board_hyp_lvl_5 = fun.place_ship(board_hyp_lvl_4, n_seg_for_lvl[lvl], orient, field)
        if board_hyp_lvl_5 is None:
            return None
    # BEGIN LVL 6
    lvl = 6
    if seg2_done > 1:   # if initial board had more than 1 seg2 ships done, pass unmodified board into next level
        board_hyp_lvl_6 = board_hyp_lvl_5
    else:
        config = random.choice(configs_levels[lvl])
        field = config[0]
        orient = config[1]
        board_hyp_lvl_6 = fun.place_ship(board_hyp_lvl_5, n_seg_for_lvl[lvl], orient, field)
        if board_hyp_lvl_6 is None:
            return None
    # BEGIN LVL 7
    lvl = 7
    if seg2_done > 0:   # if initial board had more than 0 seg2 ships done, treat the received board as final
        board_hyp_lvl_7 = board_hyp_lvl_6
    else:
        config = random.choice(configs_levels[lvl])
        field = config[0]
        orient = config[1]
        board_hyp_lvl_7 = fun.place_ship(board_hyp_lvl_6, n_seg_for_lvl[lvl], orient, field)
    if board_hyp_lvl_7 is None:
        return None

    # Check if all '1's turned into either 2,3,4 or 5 (unknown hit ship becoming a certain ship)
    # If any of these fields ends up being 0 or 7, or remains as 1, then this generated board is not valid
    # Also check if ships put in place of '1's reach their minimum length (at least 1 segment longer than
    # initial length of '1' segments). If not, board is not valid - scrap it and return None

    # Remark - enforcing the utilization of '1' fields at start would lead to statistically incorrect results!

    if board_hyp_lvl_7 is not None:
        for id, p in enumerate(hit_unsunk_lvl_0):
            px = p[0]
            py = p[1]
            g = board_hyp_lvl_7[px][py]
            if g == 0 or g == 7 or g < minimum_lengths[id]:
                board_hyp_lvl_7 = None
                break

    return board_hyp_lvl_7


# Not used - left from previous test, may be utilized in future
# def generate_random_board_with_combs(board_hyp_lvl_0: np.ndarray, configs_combs: list, n_seg_for_lvl: dict,
#                                     seg5_done: int, seg4_done: int, seg3_done: int, seg2_done: int,
#                                     hit_unsunk_lvl_0: list, minimum_lengths: list) -> np.ndarray or None:
#     """
#     Generates random board setup from given initial board state
#     :param board_hyp_lvl_0: initial board state
#     :param configs_levels: valid configs for each level
#     :param n_seg_for_lvl: ship lengths to be placed at each level
#     :param seg5_done: number of seg5 ships present on initial board
#     :param seg4_done: number of seg4 ships present on initial board
#     :param seg3_done: number of seg2 ships present on initial board
#     :param seg2_done: number of seg2 ships present on initial board
#     :param hit_unsunk_lvl_0: list of fields containing '1' value on initial board
#     :param minimum_lengths: list of minimum values that fields of corresponding indices in hit_unsunk_lvl_0 can turn into
#     :return: hypothetical board with all ships placed that is compliant with the initial state and game rules
#              OR None if validation fails at any stage
#     """
#
#     # Place ships on each level sequentially by picking random configurations available per level.
#     # If at any point the returned board is None, then scrap whole attempt and return None.
#
#     # Retrying ship placement at the failed level would lead to statistically incorrect outcome! That is because it
#     # would enforce the function to generate board setup that might be very unlikely due to previous level choices
#     # (they might occupy lots of space so that there are very few places left to fit remaining ships) and thus
#     # artificially increase probability of such rare configurations.
#
#     # BEGIN LVL 1
#     lvl = 1
#     if seg5_done == 1:  # if initial board had more than 01 seg5 ships done, pass unmodified board into next level
#         board_hyp_lvl_1 = board_hyp_lvl_0
#     else:
#         config = configs_combs[lvl]
#         field = config[0]
#         orient = config[1]
#         board_hyp_lvl_1 = fun.place_ship(board_hyp_lvl_0, n_seg_for_lvl[lvl], orient, field)
#         if board_hyp_lvl_1 is None:
#             return None
#     # BEGIN LVL 2
#     lvl = 2
#     if seg4_done == 1:  # if initial board had more than 0 seg4 ships done, pass unmodified board into next level
#         board_hyp_lvl_2 = board_hyp_lvl_1
#     else:
#         config = configs_combs[lvl]
#         field = config[0]
#         orient = config[1]
#         board_hyp_lvl_2 = fun.place_ship(board_hyp_lvl_1, n_seg_for_lvl[lvl], orient, field)
#         if board_hyp_lvl_2 is None:
#             return None
#     # BEGIN LVL 3
#     lvl = 3
#     if seg3_done == 2:  # if initial board had more than 1 seg3 ships done, pass unmodified board into next level
#         board_hyp_lvl_3 = board_hyp_lvl_2
#     else:
#         config = configs_combs[lvl]
#         field = config[0]
#         orient = config[1]
#         board_hyp_lvl_3 = fun.place_ship(board_hyp_lvl_2, n_seg_for_lvl[lvl], orient, field)
#         if board_hyp_lvl_3 is None:
#             return None
#     # BEGIN LVL 4
#     lvl = 4
#     if seg3_done > 0:  # if initial board had more than 0 seg3 ships done, pass unmodified board into next level
#         board_hyp_lvl_4 = board_hyp_lvl_3
#     else:
#         config = configs_combs[lvl]
#         field = config[0]
#         orient = config[1]
#         board_hyp_lvl_4 = fun.place_ship(board_hyp_lvl_3, n_seg_for_lvl[lvl], orient, field)
#         if board_hyp_lvl_4 is None:
#             return None
#     # BEGIN LVL 5
#     lvl = 5
#     if seg2_done > 2:   # if initial board had more than 2 seg2 ships done, pass unmodified board into next level
#         board_hyp_lvl_5 = board_hyp_lvl_4
#     else:
#         config = configs_combs[lvl]
#         field = config[0]
#         orient = config[1]
#         board_hyp_lvl_5 = fun.place_ship(board_hyp_lvl_4, n_seg_for_lvl[lvl], orient, field)
#         if board_hyp_lvl_5 is None:
#             return None
#     # BEGIN LVL 6
#     lvl = 6
#     if seg2_done > 1:   # if initial board had more than 1 seg2 ships done, pass unmodified board into next level
#         board_hyp_lvl_6 = board_hyp_lvl_5
#     else:
#         config = configs_combs[lvl]
#         field = config[0]
#         orient = config[1]
#         board_hyp_lvl_6 = fun.place_ship(board_hyp_lvl_5, n_seg_for_lvl[lvl], orient, field)
#         if board_hyp_lvl_6 is None:
#             return None
#     # BEGIN LVL 7
#     lvl = 7
#     if seg2_done > 0:   # if initial board had more than 0 seg2 ships done, treat the received board as final
#         board_hyp_lvl_7 = board_hyp_lvl_6
#     else:
#         config = configs_combs[lvl]
#         field = config[0]
#         orient = config[1]
#         board_hyp_lvl_7 = fun.place_ship(board_hyp_lvl_6, n_seg_for_lvl[lvl], orient, field)
#     if board_hyp_lvl_7 is None:
#         return None
#
#
#
#     # Check if all '1's turned into either 2,3,4 or 5 (unknown hit ship becoming a certain ship)
#     # If any of these fields ends up being 0 or 7, or remains as 1, then this generated board is not valid
#     # Also check if ships put in place of '1's reach their minimum length (at least 1 segment longer than
#     # initial length of '1' segments). If not, board is not valid - scrap it and return None
#
#     # Remark - enforcing the utilization of '1' fields at start would lead to statistically incorrect results!
#
#     if board_hyp_lvl_7 is not None:
#         for id, p in enumerate(hit_unsunk_lvl_0):
#             px = p[0]
#             py = p[1]
#             g = board_hyp_lvl_7[px][py]
#             if g == 0 or g == 7 or g < minimum_lengths[id]:
#                 board_hyp_lvl_7 = None
#                 break
#
#
#     return board_hyp_lvl_7