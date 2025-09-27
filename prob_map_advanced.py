import numpy as np
import copy
import time
import os
import gc
import shutil
import random
from tabulate import tabulate
from scipy.ndimage import label

import fun

###########################################################################
# SUB - FUNCTIONS valid for this script only
###########################################################################
def save_chunk(chunk: list([np.ndarray]), chunk_dir, chunk_id: int):
    """
    Save a chunk of generated unique ndarrays to an external numpy file.
    :param chunk: list of ndarrays
    :param chunk_id: ID of current chunk (each different placement of 4seg ship is a new chunk)
    :return:
    """
    filename = os.path.join(chunk_dir, f"chunk_{chunk_id}.npy")
    np.save(filename, chunk)  # Save as a .npy file for efficient storage
    print(f"Saved chunk {chunk_id} with {len(chunk)} unique arrays.")

###########################################################################
def remove_duplicates(chunk: list) -> list:
    """
    Remove duplicate arrays from list. Each unique array is converted to bytes and stored in "seen" set.
    """
    # Doing this: "unique_arrays = list(dict.fromkeys(chunk))" is not possible because ndarrays are not hashable
    unique_arrays = []
    seen = set()
    for arr in chunk:
        arr_bytes = arr.tobytes()
        if arr_bytes not in seen:
            unique_arrays.append(arr)
            seen.add(arr_bytes)
    print (f'Duplicates removed from current chunk. Length reduced from {len(chunk)} to: {len(unique_arrays)}')
    return unique_arrays
###########################################################################

###########################################################################
# MAIN FUNCTION
###########################################################################

def calculate_probs_advanced (known_board):
    # INITIAL DATA
    start_time = time.time()
    orientations = ('hor', 'ver')

    prob_maps_adv_100chars_dir = "prob_maps_adv_100chars\\"

    CHUNK_SIZE = 1000000  # Number of arrays per chunk
    chunk_count = 0  # Track the number of chunks saved

    # Directory to save chunks
    chunk_dir = "array_chunks_statki_temp_chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    board = known_board
    board_hyp_lvl_0 = copy.deepcopy(known_board)

    # Temporary in-memory list of arrays
    current_chunk = []

    # Create 1D list of 100 fields and convert to tuple (0 to 99)
    fields = tuple([(x, y) for x in range(10) for y in range(10)])

    # Progress trackers
    seg_5_count = 0
    total = 0

    # These will be switched to True if certain ship is already placed in the given board.
    # It will then allow to skip some levels of hypothetical ship loops
    seg5_already_placed = False
    seg4_already_placed = False
    seg31_already_placed = False
    seg32_already_placed = False
    seg21_already_placed = False
    seg22_already_placed = False
    seg23_already_placed = False

    # Initially take the original board
    last_completed_board = copy.deepcopy(known_board)

    free_fields_lvl_0, hit_unsunk = fun.find_free_and_hit_fields(board_hyp_lvl_0)

    # Find different groups of adjacent '1' fields
    groups = fun.group_adjacent_symbols(board, 1)

    minimum_lengths = fun.find_minimum_lengths(hit_unsunk, groups)

    ###############################################################################################
    # MONSTER LOOP
    ###############################################################################################
    for field5 in free_fields_lvl_0:
        # if one loop was finished with 5seg ship already placed in initial board - break (no point to go through
        # all other fields again as the output will be the same)
        if seg5_already_placed:
            break
        for orient in orientations:
            if seg5_already_placed:
                break
            # Reset 'already_placed' variables when trying out another 5seg position (so we don't break the lower loops
            # immediately without going through them at least once)
            seg4_already_placed = False
            seg31_already_placed = False
            seg32_already_placed = False
            seg21_already_placed = False
            seg22_already_placed = False
            seg23_already_placed = False
            # If board already contains all possible '5' fields -> go on and use unmodified lvl_0 board
            if fun.n_segments(board_hyp_lvl_0, 5) == 5:
                mat = True
                board_hyp_lvl_1 = board_hyp_lvl_0
                seg5_already_placed = True
            # If not, place the ship
            else:
                mat = False
                board_hyp_lvl_1 = fun.place_ship(board_hyp_lvl_0, 5, orient, field5)

            # If board was not generated (ship could not be placed), continue the loop (try next field)
            if board_hyp_lvl_1 is None:
                continue

            seg_5_count = seg_5_count + 1
            print('successfully placed a new 5-segment. Progress: ', seg_5_count)
            print('so far total is ', total)
            print("%.1f seconds" % (time.time() - start_time))

            for field4 in free_fields_lvl_0:
                if seg4_already_placed:
                    break
                for orient in orientations:
                    if seg4_already_placed:
                        break
                    # If board already contains all possible '4' fields -> go on and use unmodified lvl_1 board
                    if fun.n_segments(board_hyp_lvl_1, 4) == 4:
                        mat = True
                        board_hyp_lvl_2 = board_hyp_lvl_1
                        seg4_already_placed = True
                    # If not, place the ship
                    else:
                        mat = False
                        board_hyp_lvl_2 = fun.place_ship(board_hyp_lvl_1, 4, orient, field4)
                    # If board was not generated (ship could not be placed), continue the loop (try next field)
                    if board_hyp_lvl_2 is None:
                        continue
                    print('successfully placed a new 4-segment')
                    seg31_already_placed = False
                    seg32_already_placed = False
                    for field31 in free_fields_lvl_0:
                        # if one loop was finished with 31seg ship already placed in initial board - break (no point to go through all other fields again as the output will be the same)
                        if seg31_already_placed:
                            break
                        for orient in orientations:
                            if seg31_already_placed:
                                break
                            # If board already contains all possible '3' fields -> go on and use unmodified previous level board
                            if fun.n_segments(board_hyp_lvl_2, 3) >= 3:
                                mat = True
                                board_hyp_lvl_3 = board_hyp_lvl_2
                                seg31_already_placed = True
                            # If not, place the ship
                            else:
                                mat = False
                                board_hyp_lvl_3 = fun.place_ship(board_hyp_lvl_2, 3, orient, field31)
                            # If board was not generated (ship could not be placed), continue the loop (try next field)
                            if board_hyp_lvl_3 is None:
                                continue
                            for field32 in free_fields_lvl_0:
                                # if one loop was finished with 32seg ship already placed in initial board - break (no point to go through all other fields again as the output will be the same)
                                if seg32_already_placed:
                                    break
                                for orient in orientations:
                                    if seg32_already_placed:
                                        break
                                    # If board already contains all possible '3' fields -> go on and use unmodified previous level board
                                    if fun.n_segments(board_hyp_lvl_3, 3) == 6:
                                        mat = True
                                        board_hyp_lvl_4 = board_hyp_lvl_3
                                        seg32_already_placed = True
                                    # If not, place the ship
                                    else:
                                        mat = False
                                        board_hyp_lvl_4 = fun.place_ship(board_hyp_lvl_3, 3, orient, field32)
                                    if field32 ==[4,2]:
                                        marker = 'here'
                                    # If board was not generated (ship could not be placed), continue the loop (try next field)
                                    if board_hyp_lvl_4 is None:
                                        continue
                                    seg21_already_placed = False
                                    seg22_already_placed = False
                                    seg23_already_placed = False
                                    for field21 in free_fields_lvl_0:
                                        # if one loop was finished with 21seg ship already placed in initial board - break (no point to go through all other fields again as the output will be the same)
                                        if seg21_already_placed:
                                            break
                                        for orient in orientations:
                                            if seg21_already_placed:
                                                break
                                            # If board already contains all possible '2' fields -> go on and use unmodified previous level board
                                            if fun.n_segments(board_hyp_lvl_4, 2) >= 2:
                                                mat = True
                                                board_hyp_lvl_5 = board_hyp_lvl_4
                                                seg21_already_placed = True
                                            # If not, place the ship
                                            else:
                                                mat = False
                                                board_hyp_lvl_5 = fun.place_ship(board_hyp_lvl_4, 2, orient, field21)
                                            # If board was not generated (ship could not be placed), continue the loop (try next field)
                                            if board_hyp_lvl_5 is None:
                                                continue
                                            for field22 in free_fields_lvl_0:
                                                # if one loop was finished with 22seg ship already placed in initial board - break (no point to go through all other fields again as the output will be the same)
                                                if seg22_already_placed:
                                                    break
                                                for orient in orientations:
                                                    if seg22_already_placed:
                                                        break
                                                    # If board already contains all possible '2' fields -> go on and use unmodified previous level board
                                                    if fun.n_segments(board_hyp_lvl_5, 2) >= 4:
                                                        mat = True
                                                        board_hyp_lvl_6 = board_hyp_lvl_5
                                                        seg22_already_placed = True
                                                    # If not, place the ship
                                                    else:
                                                        mat = False
                                                        board_hyp_lvl_6 = fun.place_ship(board_hyp_lvl_5, 2, orient, field22)
                                                    # If board was not generated (ship could not be placed), continue the loop (try next field)
                                                    if board_hyp_lvl_6 is None:
                                                        continue
                                                    for field23 in free_fields_lvl_0:
                                                        # if one loop was finished with 23seg ship already placed in initial board - break (no point to go through all other fields again as the output will be the same)
                                                        if seg23_already_placed:
                                                            break
                                                        for orient in orientations:
                                                            if seg23_already_placed:
                                                                break
                                                            # If board already contains all possible '2' fields -> go on and use unmodified previous level board
                                                            if fun.n_segments(board_hyp_lvl_6, 2) == 6:
                                                                board_hyp_lvl_7 = board_hyp_lvl_6
                                                                seg23_already_placed = True
                                                            # If not, place the ship
                                                            else:
                                                                board_hyp_lvl_7 = fun.place_ship(board_hyp_lvl_6, 2, orient,field23)
                                                            # If board was not generated (ship could not be placed), continue the loop (try next field)
                                                            if board_hyp_lvl_7 is None:
                                                                continue

                                                            # Check if all '1's were used as either 2,3,4 or 5 (unknown hit ship becoming a certain ship)
                                                            # If any of these is 0 or 7 then this generated board is not valid
                                                            if board_hyp_lvl_7 is not None:
                                                                for id, p in enumerate(hit_unsunk):
                                                                    px = p[0]
                                                                    py = p[1]
                                                                    g = board_hyp_lvl_7 [px][py]
                                                                    if g == 0 or g == 7 or g < minimum_lengths[id]:
                                                                        board_hyp_lvl_7 = None
                                                                        break
                                                            if board_hyp_lvl_7 is None:
                                                                continue
                                                            # Finished board generation, append to list of boards
                                                            total = total + 1
                                                            last_completed_board = copy.deepcopy(board_hyp_lvl_7)
                                                            if total % 100000 == 0:
                                                                print (total)
                                                                # print(tabulate(board_hyp_lvl_7))
                                                            current_chunk.append (board_hyp_lvl_7)
                                                            if total % CHUNK_SIZE == 0:
                                                                current_chunk = remove_duplicates(current_chunk)
                                                            #print (board_hyp_lvl_7)
                # Deduplicate and save chunk after each generated seg4 (at the end of seg4 loop)
                # Each chunk with different seg4 or seg5 locations is unique (because there is only one seg5 and seg4)
                if current_chunk:
                    current_chunk = remove_duplicates(current_chunk)
                    save_chunk(current_chunk, chunk_dir, chunk_count)
                    current_chunk = []  # Clear the in-memory chunk to free up memory
                    gc.collect()
                    chunk_count += 1
                    print('Progress: field5 = ', field5, ', field4 =  ', field4)
                else:
                    print('empty chunk, nothing saved')
                    print('Progress: field5 = ', field5, ', field4 =  ', field4)

    # MONSTER LOOP DONE
    ###############################################################################################

    print ("Finished all chunks except the unfinished one, before deduplication")
    # Save any remaining arrays in the last chunk
    if current_chunk:
        current_chunk = remove_duplicates(current_chunk)
        save_chunk(current_chunk, chunk_dir, chunk_count)
    print ("Finished all chunks. Now loading them back individually and counting occurances")

    # Create 2D np.array of 100 fields to count occurances for each field
    occurances = np.zeros((10, 10), dtype = int)

    n_chunks = len(os.listdir(chunk_dir))
    total_unique = 0
    w = 0

    # Load each numpy data file individually
    for filename in os.listdir(chunk_dir):
        if filename.endswith(".npy"):
            chunk = np.load(os.path.join(chunk_dir, filename), allow_pickle=True)
            # Add length of this chunk to total unique arrays' number
            total_unique = total_unique + len(chunk)
            # Loop through initial free fields and add occurances
            for field in free_fields_lvl_0:
                x = field[0]
                y = field[1]
                # Do not consider fields containing '1' (members of hit_unsunk)
                if field not in hit_unsunk:
                    for ar in chunk:
                        if ar[x][y] > 1 and ar[x][y] < 7:
                            occurances[x][y] += 1
        w += 1
        print(f'Counted {w} out of {n_chunks} chunks')

    print(f"Final number of unique arrays: {total_unique}")
    print("--- %s seconds ---" % (time.time() - start_time))

    # Postprocess occurances board - find max value (m), best probability (best_prob) and best probability field (best_field)
    m = occurances.max()
    best_prob = m / total_unique
    best_field = np.argwhere(occurances == m)
    # If there are multiple fields with the same max value, choose the first one
    best_field = best_field[0].tolist()

    print(f'max probability for field: {best_field}')
    print(occurances)
    time_adv = round(time.time() - start_time, 2)

    # Saving generated data in numpy format
    board_state_100chars_code = fun.update_board_state_100chars_code(known_board)
    prob_map_adv_100chars_file_name = board_state_100chars_code + '.npy'
    fun.write_to_library(prob_maps_adv_100chars_dir, occurances, prob_map_adv_100chars_file_name)

    # Remove temporary files containing generated chunks (they are too big to be stored forever)
    shutil.rmtree(chunk_dir)

    return occurances, best_field, best_prob, total_unique, time_adv, board_state_100chars_code


#####################################################################################
# Only valid if this script is run as __main__ with predefined board (singular tests)
#####################################################################################
if __name__ == "__main__":

    known_board = np.zeros((10, 10), dtype = "int16")

    known_board[0] = [7, 2, 7, 0, 0, 0, 7, 3, 3, 3]
    known_board[1] = [7, 2, 7, 0, 0, 7, 7, 7, 7, 7]
    known_board[2] = [7, 7, 7, 0, 7, 0, 0, 0, 7, 0]
    known_board[3] = [0, 0, 0, 7, 7, 7, 7, 7, 7, 7]
    known_board[4] = [0, 0, 7, 7, 5, 5, 5, 5, 5, 7]
    known_board[5] = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    known_board[6] = [7, 4, 7, 7, 7, 0, 7, 1, 0, 0]
    known_board[7] = [7, 4, 7, 3, 7, 0, 0, 0, 7, 7]
    known_board[8] = [7, 4, 7, 3, 7, 7, 0, 7, 7, 2]
    known_board[9] = [7, 4, 7, 3, 7, 7, 0, 0, 7, 2]

    # known_board[0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # known_board[1] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # known_board[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # known_board[3] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # known_board[4] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # known_board[5] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # known_board[6] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # known_board[7] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # known_board[8] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # known_board[9] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    print(known_board)

    number_of_unknown_fields = 0
    for i in known_board:
        for j in i:
            if j == 0:
                number_of_unknown_fields += 1

    occurances, best_field, best_prob, total_unique, time_adv, board_state_100chars_code = calculate_probs_advanced (known_board)
    #fun.save_adv_times(board_state_100chars_code, time_adv)