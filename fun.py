import numpy as np
from scipy.ndimage import label
import os


###########################################################################
# LOCAL DATA
###########################################################################

def n_segments(board: np.ndarray, n_seg: int) -> int:
    """Count number of fields in 2D array containing <n_seg> value. Return this count as int."""
    temp_for_counting = board.copy()
    temp_for_counting.flatten()
    k = np.count_nonzero(temp_for_counting == n_seg)
    return k


def group_adjacent_symbols(board: np.ndarray, symbol: int) -> list:
    """
    Group adjacent fields containing the same given symbol in 2D array. Return list of these groups (nested lists)

    :param board: passed board state
    :param symbol: symbol to group
    :return: list of groups of adjacent fields containing the specified symbol
    """

    # Create a binary mask where fields containing <symbol> value become 1. All other fields become 0.
    binary_mask = (board == symbol).astype(int)

    # labeled_array - array where each group of adjacent '1's is represented by a unique number
    # num_features - number of detected groups
    labeled_array, num_features = label(binary_mask)

    groups = []
    # loop through all found labels
    for label_num in range(1, num_features + 1):
        # Find indices of the current group. group_indices is ndarray type
        group_indices = np.argwhere(labeled_array == label_num)
        # Convert group_indices from ndarray to nested list and append it to groups list
        groups.append(group_indices.tolist())
        # Groups is 3-level nested list. 1st - separate groups, 2nd - fields within group, 3rd - field X,Y coordinates
    return groups


def place_ship(board_passed: np.ndarray, n_seg: int, orient: str, field: list) -> np.ndarray or None:
    """Tries to place ship of n_seg length on given board, with given orientation and starting field, acc. to game rules

    If successful, returns hypothetical board with this ship placed. If not, returns None"""

    hits = locate_hits(board_passed, n_seg, orient, field)
    # If hypothetical ship doesn't collide with anything (hits not empty),
    # fill "hit fields" with 'n_seg' and the surrounding fields with '7's and return updated hypothetical board
    if hits:
        # Auxiliary hypothetical board
        new_board_hyp = board_passed.copy()
        new_board_hyp = fill_adjacent(new_board_hyp, hits, n_seg)
        return new_board_hyp
    # If hits are empty, return None instead of updated board
    else:
        return None


def locate_hits(board_passed: np.ndarray, n_seg: int, orient: str, field: list) -> list:
    """Create list of fields which the hypothetical ship would occupy

    Parameters: considered board, length of ship, orientation, starting field.

    Returns: list of fields (could be empty if any field is unavailable)
    """

    hits = []

    if orient == 'hor':
        # Search for fields ON THE RIGHT from the starting field
        # Only go further if there's enough space before the board's edge.
        if field[1] + (n_seg - 1) < 10:
            for k in range(n_seg):
                checked_field = [field[0], field[1] + k]
                # Only allowed 0 (unknown field) and 1 (hit, not yet sunk ships)
                if board_passed[checked_field[0]][checked_field[1]] < 2:
                    hits.append(checked_field)
                # If a field containing value >=2 gets on the way, clear the hits and break, as this will not be valid.
                else:
                    hits = []
                    break
            return hits

    if orient == 'ver':
        # Search for fields DOWNWARDS from the starting field
        # Only go further if there's enough space before the board's edge.
        if field[0] + (n_seg - 1) < 10:
            for k in range(n_seg):
                checked_field = [field[0] + k, field[1]]
                # Only allowed 0 (unknown field) and 1 (hit, not yet sunk ships)
                if board_passed[checked_field[0]][checked_field[1]] < 2:
                    hits.append(checked_field)
                # If a field containing value >=2 gets on the way, clear the hits and break, as this will not be valid.
                else:
                    hits = []
                    break
            return hits


def fill_adjacent(new_board_hyp: np.ndarray, hits: list, n_seg: int) -> np.ndarray or None:
    """
    Fills the hit fields with <n_seg> value and the surrounding fields with '7's (forbidden space)
    :param new_board_hyp:
    :param hits:
    :param n_seg:
    :return: Updated hypothetical board or None if conflict detected
    """

    edf = (0, 7)  # Editable fields when filling forbidden space (7) - either 0 or 7

    # Fill each hit field with the number symbolizing the ship (n_seg)
    for field in hits:
        x = field[0]
        y = field[1]
        new_board_hyp[x][y] = n_seg

    # For each hit field, find adjacent fields and check their values
    # If adjacent field value is either 0 or 7 -> replace it with 7 (forbiden space)
    # If it's not 0 or 7, check if it is one of the hit fields (in hits list). If so, leave it unchanged.
    # If it's not 0 or 7 and not in hits list (could happen for '1' fields), that means conflict -> return None
    for field in hits:
        x = field[0]
        y = field[1]

        if x > 0:
            if new_board_hyp[x - 1][y] in edf:  # Upper
                new_board_hyp[x - 1][y] = 7
            elif [x - 1, y] not in hits:
                return None
            if y > 0:
                if new_board_hyp[x - 1][y - 1] in edf:  # Upper-left
                    new_board_hyp[x - 1][y - 1] = 7
                elif [x - 1, y - 1] not in hits:
                    return None
        if x < 9:
            if new_board_hyp[x + 1][y] in edf:  # Lower
                new_board_hyp[x + 1][y] = 7
            elif [x + 1, y] not in hits:
                return None
            if y < 9:
                if new_board_hyp[x + 1][y + 1] in edf:  # Lower-right
                    new_board_hyp[x + 1][y + 1] = 7
                elif [x + 1, y + 1] not in hits:
                    return None
        if y > 0:
            if new_board_hyp[x][y - 1] in edf:  # Left
                new_board_hyp[x][y - 1] = 7
            elif [x, y - 1] not in hits:
                return None
            if x < 9:
                if new_board_hyp[x + 1][y - 1] in edf:  # Lower-left
                    new_board_hyp[x + 1][y - 1] = 7
                elif [x + 1, y - 1] not in hits:
                    return None
        if y < 9:
            if new_board_hyp[x][y + 1] in edf:  # Right
                new_board_hyp[x][y + 1] = 7
            elif [x, y + 1] not in hits:
                return None
            if x > 0:
                if new_board_hyp[x - 1][y + 1] in edf:  # Upper-right
                    new_board_hyp[x - 1][y + 1] = 7
                elif [x - 1, y + 1] not in hits:
                    return None

    return new_board_hyp


def find_free_and_hit_fields(board: np.ndarray) -> tuple[tuple, list]:
    """
    Create lists of free and hit fields.

    free_fields = containing either 0 or 1 (possible to place ship there)

    hit_unsunk = containing 1 (already hit but not sunk)
    """
    free_fields = []
    hit_unsunk = []
    for x in range(10):
        for y in range(10):
            field = (x, y)
            if board[x][y] == 0:
                free_fields.append(field)
            # 1 is considered as 'free field' aswell, because it means ships can be placed there.
            # In the end, '1' fields will not be included in hit probability calculation.
            if board[x][y] == 1:
                free_fields.append(field)
                hit_unsunk.append(field)
    free_fields = tuple(free_fields)
    return free_fields, hit_unsunk

def find_minimum_lengths(hit_unsunk: list, groups: list) -> list:
    """
    For each unsunk ship determine its minimal length equal to current length of '1's sequence + 1
    (not yet sunk means there's at least 1 more segment hidden in the fog of war)
    :param hit_unsunk: list of hit unsunk fields
    :param groups: list of hit unsunk ships (groups of fields)
    :return: 'minimum_lengths' list containing one integer per each group in 'groups' list
    """
    minimum_lengths = []
    for f in hit_unsunk:
        for gr in groups:
            if list(f) in gr:
                k = len(gr)
                minimum_lengths.append(k+1)
                break
    minimum_lengths = tuple(minimum_lengths)
    return minimum_lengths

def sink_ship(board_known: np.ndarray, ship_fields_initial: list, best_field: tuple) -> np.ndarray:
    """
    Sinks the ship to which 'best_field' belonged to.

    Board_known is updated with ship's length value in place of '1's, neighbouring fields become forbidden zone ('7')

    :return: Updated known board state
    """
    for ship in ship_fields_initial:
        if best_field in ship:
            hits = ship
            n_seg = len(ship)
            break
    updated_board = fill_adjacent(board_known, hits, n_seg)
    return updated_board


def load_data(data_dir: str, margin: float) -> tuple[list, float, np.ndarray, int]:
    """
    Loads the data from specified probability map file and returns list of highest probability fields and highest
    probability value


    :param data_dir: full path to the probability map file
    :param margin: margin to set threshold for considering probabilities high enough to include in returned list
    :return: list of highest probability fields, highest probability value, occurances array, number of samples
    """
    prob_table = np.load(data_dir, allow_pickle=True)
    print(f"Loaded game setup:  {data_dir}")

    # Threshold set as % of maximum probability found
    max = prob_table.max()
    threshold = (1 - margin) * max

    # Total hits = sum of all values in the array (values in the array represent how many times each field was hit in
    # the process of generating random games from considered state - check prob_density_montecarlo module for details)
    total_hits = 0

    good_fields = []

    for x in range(0, 10):
        for y in range(0, 10):
            total_hits = total_hits + prob_table[x][y]
            # Only if probability of current field is above threshold, add the current field to list
            if prob_table[x][y] > threshold:
                good_fields.append([x, y])

    # In each generated game exactly 21 fields were hit (21 ship segments)
    total_games = total_hits / 21

    # Highest probability can be recalculated from the available data like this (no need to store it in numpy file)
    best_prob = max / total_games

    return good_fields, best_prob, prob_table, total_games


# Create list of fields on which the hypothetical ship is placed - MODIFIED to allow placement on top of existing ships. To be used in special cases only
def locate_hits_modified(board_passed, n_seg, orient, field):
    hits = []
    if orient == 'hor':
        # Only go further if there's enough space before the board's edge.
        if field[1] + (n_seg - 1) < 10:
            for k in range(n_seg):
                checked_field = [field[0], field[1] + k]
                # Only allowed 0 (unknown field) and 1 (hit, not yet sunk ships) and on top of existing ships
                if board_passed[checked_field[0]][checked_field[1]] < 6:
                    hits.append(checked_field)
                # If a field containing value >=2 gets on the way -> clear the hits and break, as this will not be valid.
                else:
                    hits = []
                    break
            return hits
    if orient == 'ver':
        # Only go further if there's enough space before the board's edge.
        if field[0] + (n_seg - 1) < 10:
            for k in range(n_seg):
                checked_field = [field[0] + k, field[1]]
                # Only allowed 0 (unknown field) and 1 (hit, not yet sunk ships) and on top of existing ships
                if board_passed[checked_field[0]][checked_field[1]] < 6:
                    hits.append(checked_field)
                # If a field containing value >=2 gets on the way -> clear the hits and break, as this will not be valid.
                else:
                    hits = []
                    break
            return hits


# Try to place ship of n_seg length, on given board, with given orientation, on given starting field
# - MODIFIED to allow placement on top of existing ships. To be used in special cases only
def place_ship_modified(board_passed, n_seg, orient, field):
    # Auxiliary hypothetical board
    new_board_hyp = board_passed.copy()
    hits = locate_hits_modified(new_board_hyp, n_seg, orient, field)
    # If hypothetical ship doesn't collide with anything (hits not empty),
    # fill "hit fields" with 'n_seg' and the surrounding fields with '7's and return updated hypothetcal board
    if hits:
        new_board_hyp = fill_adjacent(new_board_hyp, hits, n_seg)
        return new_board_hyp
    # If hits are empty, return None instead of updated board
    else:
        return None


def write_to_library(prob_maps_dir: str, prob_dens_map: np.ndarray, prob_map_file_name: str):
    """
    Writes calculated probability density map for given game code to the RadioTelegraphic Phonebook Library.

    :param prob_maps_dir: directory of the library
    :param prob_dens_map: calculated probability density map (ndarray)
    :param prob_map_file_name: name of the file (game code + .npy)
    """
    full_path = os.path.join(prob_maps_dir, prob_map_file_name)
    np.save(full_path, prob_dens_map)  # Save as a .npy file
    print(f"Saved calculation data for current game setup as {prob_map_file_name}")

def update_board_state_100chars_code(board_known: np.ndarray) -> str:
    """
    Create 'board_state_100chars_code' string from known board state.
    It consists of 100 chars, each representing the field value on flattened board array
    """
    board_state_100chars_code = ""
    flat_board = board_known.flatten()
    for field in flat_board:
        board_state_100chars_code = board_state_100chars_code + str(field)
    return board_state_100chars_code

def save_adv_times(board_state_100chars_code: str, time_adv: float) -> None:
    """
    Save calculation time of given board state code by advanced algorythm into the database
    """
    times_adv_dir = "times_adv\\"
    times_adv_file = "times_adv_file.npy"
    times_adv_path = os.path.join(times_adv_dir, times_adv_file)

    new_row = np.array([[board_state_100chars_code, time_adv]])
    if times_adv_file in os.listdir(times_adv_dir):
        results = np.load(times_adv_path, allow_pickle=True)
        e = new_row[0][0]
        if new_row[0][0] not in results:
            results = np.append(results, new_row, axis=0)
    else:
        results = new_row
    np.save(times_adv_path, results)

def check_if_obvious (board: np.ndarray) -> None or tuple[int, int]:
    """
    Check if there is any field that is obvious as a next target.

    For each hit unsunk ship (if there is any) count available neighbouring fields.
    If there's only one such, return its coordinates. If there are more, or no hit unsunk ships present, return None.
    :param board: current known board state
    :return: None or obvious field (X, Y coords) to pick as target
    """
    groups = group_adjacent_symbols(board, 1)
    if not groups:
        return None

    # For each hit unsunk field group, check its length
    # (single field will be checked for 4 neighbours, 2+ fields can only expand into 2 fields depending on orientation)
    for gr in groups:
        zero_neighbours = []
        if len(gr) == 1:                            # single field -> check all 4 neighbours
            x = gr[0][0]
            y = gr[0][1]
            if x > 0:                               # up
                if board [x - 1][y] == 0:
                    zero_neighbours.append((x - 1, y))
            if x < 9:                               # down
                if board [x + 1][y] == 0:
                    zero_neighbours.append((x + 1, y))
            if y > 0:                               # left
                if board [x][y - 1] == 0:
                    zero_neighbours.append((x, y - 1))
            if y < 9:                               # right
                if board [x][y + 1] == 0:
                    zero_neighbours.append((x, y + 1))
            # If there is only one available neighbouring field to shoot at, return its coordinates
            if len(zero_neighbours) == 1:
                obvious_field = zero_neighbours[0]
                return obvious_field

        elif len(gr) > 1 and gr[0][0] == gr[1][0]:    # X coordinates are equal -> ship is horizontal (check left/right)
            x = gr[0][0]
            y_coords = [field[1] for field in gr]
            left_end = min(y_coords)
            right_end = max(y_coords)
            if left_end > 0:                               # left
                if board [x][left_end - 1] == 0:
                    zero_neighbours.append((x, left_end - 1))
            if right_end < 9:                               # right
                if board [x][right_end + 1] == 0:
                    zero_neighbours.append((x, right_end + 1))
            # If there is only one available neighbouring field to shoot at, return its coordinates
            if len(zero_neighbours) == 1:
                obvious_field = zero_neighbours[0]
                return obvious_field

        elif len(gr) > 1 and gr[0][1] == gr[1][1]:    # Y coordinates are equal -> ship is vertical (check up/down)
            y = gr[0][1]
            x_coords = [field[0] for field in gr]
            upper_end = min(x_coords)
            bottom_end = max(x_coords)
            if upper_end > 0:                               # up
                if board [upper_end - 1][y] == 0:
                    zero_neighbours.append((upper_end - 1, y))
            if bottom_end < 9:                               # down
                if board [bottom_end + 1][y] == 0:
                    zero_neighbours.append((bottom_end + 1, y))
            # If there is only one available neighbouring field to shoot at, return its coordinates
            if len(zero_neighbours) == 1:
                obvious_field = zero_neighbours[0]
                return obvious_field



if __name__ == "__main__":
    field_number = 0
    data_dir = "prob_density_maps\\prob_density_map_miss_field_" + str(field_number) + ".npy"
    good_fields = load_data(data_dir, margin=0.10)

    print(good_fields)
